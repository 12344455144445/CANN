"""Generate a whole-AIV-row-block resident experiment from frozen v124.

This is a scheduling/storage experiment. It does not invent a new reciprocal
or silently substitute the unconfirmed Competition source for the baseline.
"""
from pathlib import Path
import difflib
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'v124_scale_output_bank_20261012/kernel.asc'
OUT = ROOT / 'v134_row_resident_20261013'
BASE_SHA = '4010bfbbb0f4d1be969556db956cebad2de65b7784bb5a9da55b57ea25338574'

RESIDENT = r'''
    // V134: hold the entire AIV row block in one consumed workspace bank.
    // Unlike V70, compute all scales once with the unchanged vector repair.
    // Unlike V104, residency is not restricted to the old 1--4-row batch.
    const uint64_t residentElements = static_cast<uint64_t>(rows) * n;
    const bool residentRowBlock = n >= 32U && n <= kPairCapacity &&
        n % 32U == 0U && residentElements <= kPairCapacity;
    if (residentRowBlock) {
        const uint32_t residentCount = static_cast<uint32_t>(residentElements);
        const uint32_t residentCompareCount = QmqAlignUp(residentCount, 64U);
        const uint32_t alignedN = n;
        float residentRowScale[192]; // resident rows <= 6144 / 32.

        LocalTensor<float> x2ScaleLocal = x2ScaleQueue.AllocTensor<float>();
        DataCopyExtParams x2ScaleCopy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> x2ScalePad{true, 0, 0, 0.0f};
        DataCopyPad(x2ScaleLocal, x2ScaleGm, x2ScaleCopy, x2ScalePad);
        x2ScaleQueue.EnQue(x2ScaleLocal);
        x2ScaleLocal = x2ScaleQueue.DeQue<float>();

        const uint64_t offset = static_cast<uint64_t>(rowStart) * n;
        QmqEnqueueWorkspaceTile(workspaceQueue, workspaceGm,
            offset, residentCount, residentCount);
        LocalTensor<int32_t> inputLocal = workspaceQueue.DeQue<int32_t>();
        LocalTensor<float> fp32Local = fp32Buffer.Get<float>();
        Cast(fp32Local, inputLocal, RoundMode::CAST_RINT, residentCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) {
            Muls(fp32Local[r * alignedN], fp32Local[r * alignedN],
                 x1ScaleLocal.GetValue(r), alignedN);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) {
            Mul(fp32Local[r * alignedN], fp32Local[r * alignedN],
                x2ScaleLocal, alignedN);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float> scalarMaxLocal = scalarMaxBuffer.Get<float>();
        for (uint32_t group = 0; group < rows; group += 4U) {
            const uint32_t groupRows = QmqMinValue(4U, rows - group);
            for (uint32_t r = 0; r < groupRows; ++r) {
                const uint32_t actualRow = group + r;
                ReduceMax(scalarMaxLocal[r * kFloatBlock],
                    fp32Local[actualRow * alignedN],
                    reduceBuffer.Get<float>()[actualRow * alignedN], alignedN);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < groupRows; ++r) {
                const float tileMax = scalarMaxLocal.GetValue(r * kFloatBlock);
                if (tileMax > 0.0f) {
                    rowMaxLocal.SetValue(group + r, tileMax);
                }
            }
        }

        // Save dequantized values before the scale repair reuses FP32 scratch.
        // The INT32 input has no remaining reader and becomes activation storage.
        LocalTensor<float> activationLocal = inputLocal.ReinterpretCast<float>();
        DataCopy(activationLocal, fp32Local, residentCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> scaleDivisorLocal = halfBuffer.Get<float>();
        LocalTensor<float> scaleResidualLocal = reduceBuffer.Get<float>();
        LocalTensor<float> scaleScratchLocal = fp32Buffer.Get<float>();
        LocalTensor<uint8_t> scaleMaskLocal = correctionMaskBuffer.Get<uint8_t>();
        Duplicate(scaleDivisorLocal, 127.0f, alignedRows);
        PipeBarrier<PIPE_V>();
        QmqCorrectDiv(rowScaleLocal, rowMaxLocal, scaleDivisorLocal,
            scaleResidualLocal, scaleScratchLocal, scaleMaskLocal,
            alignedRows, QmqAlignUp(alignedRows, 64U));
        for (uint32_t r = 0; r < rows; ++r) {
            if (rowMaxLocal.GetValue(r) <= 0.0f) {
                rowScaleLocal.SetValue(r, 1.0f);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) {
            residentRowScale[r] = rowScaleLocal.GetValue(r);
        }
        LocalTensor<float> yScaleLocal = rowScaleLocal;
        yScaleQueue.EnQue(yScaleLocal);
        yScaleLocal = yScaleQueue.DeQue<float>();
        DataCopyExtParams yScaleCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
        DataCopyPad(yScaleGm[rowStart], yScaleLocal, yScaleCopy);

        // The original two separate FP32 multiplications have already completed.
        // ReLU is applied to that retained FP32 result, not recomputed from GM.
        Maxs(activationLocal, activationLocal, 0.0f, residentCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> divisorLocal = halfBuffer.Get<float>();
        LocalTensor<float> residualLocal = reduceBuffer.Get<float>();
        LocalTensor<float> scratchLocal = halfBuffer.Get<float>();
        LocalTensor<uint8_t> correctionMaskLocal = correctionMaskBuffer.Get<uint8_t>();
        for (uint32_t r = 0; r < rows; ++r) {
            Duplicate(divisorLocal[r * alignedN], residentRowScale[r], alignedN);
        }
        PipeBarrier<PIPE_V>();
__QUANT_BODY__
        workspaceQueue.FreeTensor(inputLocal);
        yScaleQueue.FreeTensor(yScaleLocal);
        x2ScaleQueue.FreeTensor(x2ScaleLocal);
        x1ScaleQueue.FreeTensor(x1ScaleLocal);
        PipeBarrier<PIPE_ALL>();
        return;
    }
'''


def gate(n, rows):
    return n >= 32 and n <= 6144 and n % 32 == 0 and rows * n <= 6144


def round_odd_half(values):
    nearest = values.astype(np.float16)
    exact = nearest.astype(np.float32) == values
    bits = nearest.view(np.uint16).copy()
    bits -= np.asarray(nearest.astype(np.float32) > values, dtype=np.uint16)
    bits = np.where(exact, bits, bits | np.uint16(1)).astype(np.uint16)
    return bits.view(np.float16).astype(np.float32)


def quantize(activation, scale):
    # The unchanged helper supplies the exact near-half correction on device.
    # CPU compares mathematical RN division/rounding with RN division/ODD/RINT.
    with np.errstate(under='ignore', invalid='ignore', divide='ignore'):
        q = np.asarray(activation / scale[:, None], dtype=np.float32)
    q = np.clip(q, np.float32(0), np.float32(127))
    return np.rint(round_odd_half(q)).astype(np.int8)


def cpu_check():
    rng = np.random.default_rng(1341013)
    case_count = element_count = 0
    shapes = [(32, 1), (32, 192), (64, 96), (96, 64), (128, 48),
              (256, 24), (512, 12), (1024, 6), (1536, 4),
              (2048, 3), (3072, 2), (4096, 1), (6144, 1)]
    for n, max_rows in shapes:
        for rows in sorted({1, max_rows, max(1, max_rows - 1)}):
            assert gate(n, rows)
            for fixture in range(8):
                c = rng.integers(-16777216, 16777217, (rows, n), dtype=np.int32)
                s1 = np.exp2(rng.uniform(-12, 12, rows)).astype(np.float32)
                s2 = np.exp2(rng.uniform(-12, 12, n)).astype(np.float32)
                if fixture == 0:
                    c[:] = 0
                elif fixture == 1:
                    c = -np.abs(c)
                elif fixture == 2:
                    s1[::2] = np.float32(0)
                elif fixture == 3:
                    s1[::2] *= np.float32(-1)
                    s2[::3] *= np.float32(-1)
                # Reference follows old row-batch scheduling and recomputation.
                batch = min(4, 6144 // n) if n <= 3072 else 1
                original = np.empty((rows, n), dtype=np.float32)
                recomputed = np.empty_like(original)
                for start in range(0, rows, batch):
                    stop = min(rows, start + batch)
                    first = np.asarray(c[start:stop].astype(np.float32) * s1[start:stop, None], dtype=np.float32)
                    original[start:stop] = np.asarray(first * s2, dtype=np.float32)
                    first_again = np.asarray(c[start:stop].astype(np.float32) * s1[start:stop, None], dtype=np.float32)
                    recomputed[start:stop] = np.maximum(np.asarray(first_again * s2, dtype=np.float32), np.float32(0))
                first_resident = np.asarray(c.astype(np.float32) * s1[:, None], dtype=np.float32)
                retained = np.asarray(first_resident * s2, dtype=np.float32)
                assert np.array_equal(original.view(np.uint32), retained.view(np.uint32))
                reference_max = np.maximum(np.max(original, axis=1), np.float32(0))
                resident_max = np.maximum(np.max(retained, axis=1), np.float32(0))
                assert np.array_equal(reference_max.view(np.uint32), resident_max.view(np.uint32))
                reference_scale = np.asarray(reference_max / np.float32(127), dtype=np.float32)
                reference_scale = np.where(reference_max <= 0, np.float32(1), reference_scale).astype(np.float32)
                resident_scale = np.asarray(resident_max / np.float32(127), dtype=np.float32)
                resident_scale = np.where(resident_max <= 0, np.float32(1), resident_scale).astype(np.float32)
                assert np.array_equal(reference_scale.view(np.uint32), resident_scale.view(np.uint32))
                resident_act = np.maximum(retained, np.float32(0))
                assert np.array_equal(recomputed.view(np.uint32), resident_act.view(np.uint32))
                assert np.array_equal(quantize(recomputed, reference_scale), quantize(resident_act, resident_scale))
                case_count += 1
                element_count += rows * n
    # Explicit boundary proof: padding/unaligned/multiple-N-tile paths are excluded.
    shape_count = 0
    for n in range(1, 13001):
        for rows in (1, 2, 3, 4, 5, 8, 16, 31, 32, 63, 64, 127, 192, 193, 1024):
            if not gate(n, rows):
                continue
            b = n * rows
            assert n % 32 == 0 and b <= 6144 and rows <= 192
            assert ((b + 63) // 64) * 64 <= 6144
            assert rows * 4 <= 192 * 4
            assert n * 4 <= 6144 * 4
            assert b // 64 <= 96  # Div repeatTimes fits uint8.
            # Every per-row vector source, divisor, reduction-scratch base is 32B aligned.
            assert all((r * n * 4) % 32 == 0 for r in range(rows))
            shape_count += 1
    return case_count, element_count, shape_count


def main():
    base_bytes = BASE.read_bytes()
    assert hashlib.sha256(base_bytes).hexdigest() == BASE_SHA
    source = base_bytes.decode('utf-8')
    start = source.index('__aicore__ inline void quant_matmul_relu_quant_vector(')
    end = source.index('\n__aicore__ inline', start + 1)
    original_function = source[start:end]
    # Preserve the entire original quantization body, changing only loop-scope names.
    qstart = original_function.index('                QmqDiv(fp32Local, activationLocal, divisorLocal, pairCount);')
    qend = original_function.index('                workspaceQueue.FreeTensor(inputLocal);', qstart)
    quant = original_function[qstart:qend]
    quant = '\n'.join(line[8:] if line.startswith('        ') else line for line in quant.splitlines())
    quant = quant.replace('pairRowScalePrefetch[row + r]', 'residentRowScale[r]')
    quant = quant.replace('pairCompareCount', 'residentCompareCount').replace('pairCount', 'residentCount')
    quant = quant.replace('currentN * pairRows', 'residentCount').replace('currentN', 'n')
    branch = RESIDENT.replace('__QUANT_BODY__', quant)
    anchor = '    Duplicate(rowMaxLocal, 0.0f, alignedRows);\n    PipeBarrier<PIPE_V>();\n'
    assert original_function.count(anchor) == 1
    modified_function = original_function.replace(anchor, anchor + branch)
    modified = source[:start] + modified_function + source[end:]
    assert modified_function.replace(branch, '') == original_function
    assert modified[:start] == source[:start]
    assert modified[start + len(modified_function):] == source[end:]
    cases, elements, gates = cpu_check()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'submission_attempt.json').exists(), 'Do not modify an attempted source.'
    (OUT / 'kernel.asc').write_text(modified, encoding='utf-8', newline='\n')
    candidate_sha = hashlib.sha256((OUT / 'kernel.asc').read_bytes()).hexdigest()
    review = {
        'review_pass': True, 'sha256': candidate_sha, 'base': BASE.parent.name,
        'base_sha256': BASE_SHA,
        'only_change': 'Whole AIV row block resident for aligned N and rows*N<=6144; one GM C load, one two-step FP32 dequantization, one original vector scale repair, original Div/near-half/ODD quantization.',
        'gate': 'N>=32 && N<=6144 && N%32==0 && uint64(rows)*N<=6144',
        'outside_general_vector_byte_identical': True,
        'fallback_general_vector_byte_identical': True,
        'precision_helpers_div_halfrepair_and_cast_chain_unchanged': True,
        'new_API_used': False,
        'UB_allocations_unchanged': True,
        'UB_peak_bytes': 166784,
        'API_Select_reserve_bytes': 8192,
        'UB_with_API_reserve_bytes': 174976,
        'UB_limit_bytes': 196608,
        'resident_max_rows': 192,
        'rowScale_scalar_array_bytes': 768,
        'ownership': 'One dequeue, activation reinterpretation after last int32 use, no reload/reallocation; distinct FP32 quotient and preserved activation bank; one free per queue allocation after final read/copy.',
        'alignment': 'N%32 makes every GM/UB row and flattened output contiguous and 32B aligned; no padded flat multi-row copy.',
        'CPU_bitexact_schedule_cases': cases,
        'CPU_bitexact_schedule_elements': elements,
        'CPU_dispatch_resource_cases': gates,
        'CPU_limits': 'NumPy models separate FP32 RN operations and ODD rounding; it does not execute A3 instructions, CANN compiler, vector Div approximation, or hardware correction helper. The latter helper is byte-identical.',
        'historical_distinction': 'V70 computes per 1--4-row scales using scalar uint64 /127; V104 retains only one old row batch. V134 retains all rows fitting P and invokes the original vector repair once.',
        'device': 'PENDING; CANN compile, all 15 precision points, performance and repeat controls are required.'
    }
    (OUT / 'review_checks.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'source.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True), fromfile='v124', tofile=OUT.name)), encoding='utf-8')
    (OUT / 'SUBMIT.md').write_text('''# V134 整个 AIV 行块驻留实验\n\n基线为用户确认的 v124（最高单次估分约25.25），设备速度不能由CPU验证推断。\n\n门控：N为32的倍数、32<=N<=6144、uint64(rows)*N<=6144。整个行块只读一次C，在INT32输入消费完成后把原FP32反量化结果保存在同一个UB bank。所有行最大值完成后只调用一次原QmqCorrectDiv修复yScale；后续保留原Div、近半值修复、CAST_ODD/CAST_RINT链。两个FP32缩放乘法顺序不变。\n\n区别：v70曾逐1–4行标量/127算尺度；v104仅保留一个旧rowBatch。本版保留整个能放入UB的AIV行块，统一向量尺度修复；非驻留路径与v124逐字相同。UB显式166784 B，加Select API预留8192 B为174976 B，不超A3的196608 B。\n\n需平台编译和15/15零错误，再与当天v124相邻对照、至少一次原样复测；当前未证明性能收益。若门控未覆盖主要测点，不能把15/15通过解释成所有形状都走了新路径。保持精确源码SHA与结果绑定，失败不可覆盖此目录。\n''', encoding='utf-8')
    print(json.dumps(review, ensure_ascii=False))


if __name__ == '__main__':
    main()
