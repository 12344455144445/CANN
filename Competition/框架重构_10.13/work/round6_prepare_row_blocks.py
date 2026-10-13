"""Generate the general capacity-derived full-row resident V138 framework."""
from pathlib import Path
import difflib
import hashlib
import json
import numpy as np
from round6_prepare_resident import quantize

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'v124_scale_output_bank_20261012/kernel.asc'
OUT = ROOT / 'v138_resident_row_blocks_20261013'
BASE_SHA = '4010bfbbb0f4d1be969556db956cebad2de65b7784bb5a9da55b57ea25338574'

HELPER = r'''
// V138: complete rows are consumed exactly once in capacity-derived UB blocks.
// Called before the fallback TPipe exists; the two branches never allocate together.
__aicore__ inline void QmqResidentRowBlocks(
    GM_ADDR workspace, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale,
    uint32_t m, uint32_t n, uint32_t rowStart, uint32_t rows)
{
    constexpr uint32_t kResidentCapacity = 8192U;
    const uint32_t rowBatchLimit = QmqMinValue(rows, kResidentCapacity / n);
    const uint32_t tensorCapacity = QmqAlignUp(rowBatchLimit * n, 64U);
    const uint32_t scaleCapacity = QmqAlignUp(rowBatchLimit, 64U);
    const uint32_t maskCapacity = QmqAlignUp(tensorCapacity / 8U, 32U);
    const uint32_t alignedN = n;
    float residentRowScale[256]; // N>=32 and capacity<=8192 imply batchRows<=256.

    GlobalTensor<int32_t> workspaceGm;
    GlobalTensor<float> x1ScaleGm, x2ScaleGm, yScaleGm;
    GlobalTensor<int8_t> yGm;
    workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace),
        static_cast<uint64_t>(m) * n);
    x1ScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Scale), m);
    x2ScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Scale), n);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), static_cast<uint64_t>(m) * n);
    yScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale), m);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> workspaceQueue;
    TQue<QuePosition::VECIN, 1> x2ScaleQueue;
    TQue<QuePosition::VECIN, 1> x1ScaleQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TQue<QuePosition::VECOUT, 1> yScaleQueue;
    TBuf<TPosition::VECCALC> fp32Buffer;
    TBuf<TPosition::VECCALC> halfBuffer;
    TBuf<TPosition::VECCALC> rowMaxBuffer;
    TBuf<TPosition::VECCALC> reduceBuffer;
    TBuf<TPosition::VECCALC> scalarMaxBuffer;
    TBuf<TPosition::VECCALC> correctionMaskBuffer;
    pipe.InitBuffer(workspaceQueue, 1, tensorCapacity * sizeof(int32_t));
    pipe.InitBuffer(x2ScaleQueue, 1, n * sizeof(float));
    pipe.InitBuffer(x1ScaleQueue, 1, scaleCapacity * sizeof(float));
    pipe.InitBuffer(outputQueue, 1, tensorCapacity * sizeof(int8_t));
    pipe.InitBuffer(yScaleQueue, 1, scaleCapacity * sizeof(float));
    pipe.InitBuffer(fp32Buffer, tensorCapacity * sizeof(float));
    pipe.InitBuffer(halfBuffer, tensorCapacity * sizeof(float));
    pipe.InitBuffer(rowMaxBuffer, scaleCapacity * sizeof(float));
    pipe.InitBuffer(reduceBuffer, tensorCapacity * sizeof(float));
    pipe.InitBuffer(scalarMaxBuffer, 4U * kFloatBlock * sizeof(float));
    pipe.InitBuffer(correctionMaskBuffer, maskCapacity);

    LocalTensor<float> x2ScaleLocal = x2ScaleQueue.AllocTensor<float>();
    DataCopyExtParams x2ScaleCopy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> noFloatPad{true, 0, 0, 0.0f};
    DataCopyPad(x2ScaleLocal, x2ScaleGm, x2ScaleCopy, noFloatPad);
    x2ScaleQueue.EnQue(x2ScaleLocal);
    x2ScaleLocal = x2ScaleQueue.DeQue<float>();

    for (uint32_t blockRow = 0; blockRow < rows; blockRow += rowBatchLimit) {
        const uint32_t batchRows = QmqMinValue(rowBatchLimit, rows - blockRow);
        const uint32_t alignedRows = QmqAlignUp(batchRows, 8U);
        const uint32_t scaleCompareCount = QmqAlignUp(batchRows, 64U);
        const uint32_t residentCount = batchRows * n;
        const uint32_t residentCompareCount = QmqAlignUp(residentCount, 64U);
        const uint64_t offset = static_cast<uint64_t>(rowStart + blockRow) * n;

        LocalTensor<float> x1ScaleLocal = x1ScaleQueue.AllocTensor<float>();
        DataCopyExtParams x1ScaleCopy{1, static_cast<uint32_t>(batchRows * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> x1ScalePad{
            true, 0, static_cast<uint8_t>(alignedRows - batchRows), 0.0f};
        DataCopyPad(x1ScaleLocal, x1ScaleGm[rowStart + blockRow], x1ScaleCopy, x1ScalePad);
        x1ScaleQueue.EnQue(x1ScaleLocal);
        x1ScaleLocal = x1ScaleQueue.DeQue<float>();

        LocalTensor<int32_t> inputLocal = workspaceQueue.AllocTensor<int32_t>();
        DataCopyExtParams inputCopy{1, static_cast<uint32_t>(residentCount * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> noIntPad{true, 0, 0, 0};
        DataCopyPad(inputLocal, workspaceGm[offset], inputCopy, noIntPad);
        workspaceQueue.EnQue(inputLocal);
        inputLocal = workspaceQueue.DeQue<int32_t>();

        LocalTensor<float> fp32Local = fp32Buffer.Get<float>();
        Cast(fp32Local, inputLocal, RoundMode::CAST_RINT, residentCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < batchRows; ++r) {
            Muls(fp32Local[r * alignedN], fp32Local[r * alignedN],
                x1ScaleLocal.GetValue(r), alignedN);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < batchRows; ++r) {
            Mul(fp32Local[r * alignedN], fp32Local[r * alignedN], x2ScaleLocal, alignedN);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> rowMaxLocal = rowMaxBuffer.Get<float>();
        LocalTensor<float> scalarMaxLocal = scalarMaxBuffer.Get<float>();
        Duplicate(rowMaxLocal, 0.0f, scaleCapacity);
        PipeBarrier<PIPE_V>();
        for (uint32_t group = 0; group < batchRows; group += 4U) {
            const uint32_t groupRows = QmqMinValue(4U, batchRows - group);
            for (uint32_t r = 0; r < groupRows; ++r) {
                const uint32_t actualRow = group + r;
                ReduceMax(scalarMaxLocal[r * kFloatBlock], fp32Local[actualRow * alignedN],
                    reduceBuffer.Get<float>()[actualRow * alignedN], alignedN);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < groupRows; ++r) {
                const float maximum = scalarMaxLocal.GetValue(r * kFloatBlock);
                if (maximum > 0.0f) {
                    rowMaxLocal.SetValue(group + r, maximum);
                }
            }
        }

        // All INT32 readers have completed. Reuse that bank for scale scratch,
        // then quotient; the FP32 activation bank remains unmodified and separate.
        LocalTensor<float> scaleScratchLocal = inputLocal.ReinterpretCast<float>();
        LocalTensor<float> scaleResidualLocal = reduceBuffer.Get<float>();
        LocalTensor<float> scaleDivisorLocal = halfBuffer.Get<float>();
        LocalTensor<uint8_t> scaleMaskLocal = correctionMaskBuffer.Get<uint8_t>();
        LocalTensor<float> rowScaleLocal = yScaleQueue.AllocTensor<float>();
        Duplicate(scaleScratchLocal, 0.0f, scaleCompareCount);
        Duplicate(scaleResidualLocal, 0.0f, scaleCompareCount);
        Duplicate(scaleDivisorLocal, 127.0f, alignedRows);
        PipeBarrier<PIPE_V>();
        QmqCorrectDiv(rowScaleLocal, rowMaxLocal, scaleDivisorLocal,
            scaleResidualLocal, scaleScratchLocal, scaleMaskLocal,
            alignedRows, scaleCompareCount);
        for (uint32_t r = 0; r < batchRows; ++r) {
            if (rowMaxLocal.GetValue(r) <= 0.0f) {
                rowScaleLocal.SetValue(r, 1.0f);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < batchRows; ++r) {
            residentRowScale[r] = rowScaleLocal.GetValue(r);
        }
        LocalTensor<float> yScaleLocal = rowScaleLocal;
        yScaleQueue.EnQue(yScaleLocal);
        yScaleLocal = yScaleQueue.DeQue<float>();
        DataCopyExtParams yScaleCopy{1, static_cast<uint32_t>(batchRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(yScaleGm[rowStart + blockRow], yScaleLocal, yScaleCopy);

        Maxs(fp32Local, fp32Local, 0.0f, residentCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> activationLocal = fp32Local;
        fp32Local = inputLocal.ReinterpretCast<float>();
        LocalTensor<float> divisorLocal = halfBuffer.Get<float>();
        LocalTensor<float> residualLocal = reduceBuffer.Get<float>();
        LocalTensor<float> scratchLocal = halfBuffer.Get<float>();
        LocalTensor<uint8_t> correctionMaskLocal = correctionMaskBuffer.Get<uint8_t>();
        for (uint32_t r = 0; r < batchRows; ++r) {
            Duplicate(divisorLocal[r * alignedN], residentRowScale[r], alignedN);
        }
        PipeBarrier<PIPE_V>();
__QUANT_BODY__
        workspaceQueue.FreeTensor(inputLocal);
        yScaleQueue.FreeTensor(yScaleLocal);
        x1ScaleQueue.FreeTensor(x1ScaleLocal);
    }
    x2ScaleQueue.FreeTensor(x2ScaleLocal);
    PipeBarrier<PIPE_ALL>();
}

'''

DISPATCH = r'''    // The original fallback owns no UB yet: exactly one TPipe branch runs.
    if (n >= 32U && n <= 8192U && n % 32U == 0U) {
        QmqResidentRowBlocks(workspace, x1Scale, x2Scale, y, yScale,
            m, n, rowStart, rows);
        return;
    }
'''


def resources(n, rows):
    batch = min(rows, 8192 // n)
    tensor = ((batch * n + 63) // 64) * 64
    scale = ((batch + 63) // 64) * 64
    mask = (((tensor // 8) + 31) // 32) * 32
    ub = 17 * tensor + 4 * n + 12 * scale + 128 + mask
    return batch, tensor, scale, mask, ub


def cpu_check():
    rng = np.random.default_rng(1381013)
    cases = elements = shape_cases = 0
    ub_max = 0
    for n in range(32, 8193, 32):
        for rows in (1, 2, 3, 4, 5, 7, 8, 15, 16, 63, 64, 65, 255, 256, 257, 1024):
            batch, tensor, scale, mask, ub = resources(n, rows)
            ub_max = max(ub_max, ub)
            assert 1 <= batch <= 256
            assert scale <= tensor <= 8192
            assert ub + 8192 <= 196608
            assert tensor % 64 == 0 and mask % 32 == 0
            assert all((r * n * 4) % 32 == 0 and (r * n) % 32 == 0 for r in range(batch))
            assert ((batch * n + 63) // 64) <= 128
            # Every output row and scale row is owned by one block, including tails.
            ownership = np.zeros(rows, dtype=np.uint8)
            for block in range(0, rows, batch):
                current = min(batch, rows - block)
                assert current * n <= tensor
                assert ((current + 7) // 8) * 8 <= scale
                assert ((current + 63) // 64) * 64 <= scale
                assert (((current * n) + 63) // 64) * 64 <= tensor
                ownership[block:block+current] += 1
            assert np.all(ownership == 1)
            shape_cases += 1
    for n in (32, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192):
        max_rows = 8192 // n
        for rows in sorted({1, max_rows, max_rows + 1, max_rows * 3 + 1}):
            for fixture in range(4):
                c = rng.integers(-16777216, 16777217, (rows, n), dtype=np.int32)
                s1 = np.exp2(rng.uniform(-8, 8, rows)).astype(np.float32)
                s2 = np.exp2(rng.uniform(-8, 8, n)).astype(np.float32)
                if fixture == 0:
                    c[::2] = 0
                elif fixture == 1:
                    c = -np.abs(c)
                elif fixture == 2:
                    s1[::2] *= np.float32(-1)
                    s2[::3] *= np.float32(-1)
                full = np.asarray(np.asarray(c.astype(np.float32) * s1[:, None], dtype=np.float32) * s2, dtype=np.float32)
                maxima = np.maximum(np.max(full, axis=1), np.float32(0))
                scale = np.asarray(maxima / np.float32(127), dtype=np.float32)
                scale = np.where(maxima <= 0, np.float32(1), scale).astype(np.float32)
                expected = quantize(np.maximum(full, np.float32(0)), scale)
                result = np.empty_like(expected)
                measured_scale = np.empty_like(scale)
                for block in range(0, rows, max_rows):
                    end = min(rows, block + max_rows)
                    act = np.asarray(np.asarray(c[block:end].astype(np.float32) * s1[block:end, None], dtype=np.float32) * s2, dtype=np.float32)
                    assert np.array_equal(act.view(np.uint32), full[block:end].view(np.uint32))
                    maximum = np.maximum(np.max(act, axis=1), np.float32(0))
                    batch_scale = np.asarray(maximum / np.float32(127), dtype=np.float32)
                    batch_scale = np.where(maximum <= 0, np.float32(1), batch_scale).astype(np.float32)
                    measured_scale[block:end] = batch_scale
                    result[block:end] = quantize(np.maximum(act, np.float32(0)), batch_scale)
                assert np.array_equal(measured_scale.view(np.uint32), scale.view(np.uint32))
                assert np.array_equal(result, expected)
                cases += 1
                elements += rows * n
    return cases, elements, shape_cases, ub_max


def main():
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    base = raw.decode('utf-8')
    fstart = base.index('__aicore__ inline void quant_matmul_relu_quant_vector(')
    qstart = base.index('                QmqDiv(fp32Local, activationLocal, divisorLocal, pairCount);', fstart)
    qend = base.index('                workspaceQueue.FreeTensor(inputLocal);', qstart)
    original_quant = base[qstart:qend]
    quant = '\n'.join(line[8:] if line.startswith('        ') else line for line in original_quant.splitlines())
    quant = quant.replace('pairRowScalePrefetch[row + r]', 'residentRowScale[r]')
    quant = quant.replace('pairCompareCount', 'residentCompareCount').replace('pairCount', 'residentCount')
    quant = quant.replace('currentN * pairRows', 'residentCount').replace('currentN', 'n')
    # Only the masked comparison has rounded-up lanes; initialize its padding.
    needle = '        if (hasNearHalf) {\n            Compares('
    assert quant.count(needle) == 1
    quant = quant.replace(needle, '''        if (hasNearHalf) {
            if (residentCompareCount > residentCount) {
                Duplicate(residualLocal[residentCount], 0.0f,
                    residentCompareCount - residentCount);
                PipeBarrier<PIPE_V>();
            }
            Compares(''')
    helper = HELPER.replace('__QUANT_BODY__', quant)
    anchor = '    const uint32_t alignedRows = QmqAlignUp(rows, kFloatBlock);'
    aidx = base.index(anchor, fstart)
    candidate = base[:fstart] + helper + base[fstart:aidx] + DISPATCH + base[aidx:]
    inverse = candidate.replace(helper, '', 1).replace(DISPATCH, '', 1)
    assert inverse == base
    assert candidate.count('void QmqResidentRowBlocks(') == 1
    cases, elements, shape_cases, ub_max = cpu_check()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'submission_attempt.json').exists(), 'Do not overwrite attempted source.'
    (OUT / 'kernel.asc').write_text(candidate, encoding='utf-8', newline='\n')
    sha = hashlib.sha256((OUT / 'kernel.asc').read_bytes()).hexdigest()
    review = {
        'review_pass': True, 'sha256': sha, 'base': BASE.parent.name,
        'base_sha256': BASE_SHA,
        'gate': 'N>=32 && N<=8192 && N%32==0; arbitrary positive local rows are processed in capacity-derived complete-row blocks',
        'resident_batch': 'min(rows, floor(8192/N)); tensor capacity align64(batch*N), scale capacity align64(batch)',
        'only_change': 'New full-row-block pipeline with one C load and dequantization; original vector exact scale repair per capacity block; no activation UB copy; consumed INT32 bank reused for scale scratch then quotient.',
        'inverse_removal_byte_exact_v124': True,
        'old_fallback_numeric_and_cube_host_tiny_byte_unchanged': True,
        'precision_helpers_unchanged': True,
        'FP32_two_multiply_order_unchanged': True,
        'quant_div_screen_nearhalf_clamp_ODD_RINT_preserved': True,
        'new_API_used': False,
        'UB_formula_bytes': '17*tensorCapacity + 4*N + 12*scaleCapacity + 128 + align32(tensorCapacity/8)',
        'UB_peak_bytes': ub_max,
        'API_Select_reserve_bytes': 8192,
        'UB_with_API_reserve_bytes': ub_max + 8192,
        'UB_limit_bytes': 196608,
        'UB_lifetime': 'New helper dispatch occurs before fallback TPipe construction; exactly one pipe branch allocates. One depth1 workspace bank, one immutable column bank, independent activation/quotient/scratch/mask banks.',
        'ownership': 'Each row written exactly once by its original AIV; every block alloc/free balanced; input int32 consumed before float scratch/quotient reuse; queue scales retained until quantization and GM copy finish.',
        'alignment': 'N%32 ensures contiguous flattened multi-row C input/output, per-row float and int8 bases 32B-aligned. Tensor and scale allocations align64 elements; mask align32 bytes. Tail masked lanes initialized.',
        'max_vector_div_repeats': 128,
        'max_resident_rows': 256,
        'scalar_scale_array_bytes': 1024,
        'CPU_bitexact_schedule_cases': cases,
        'CPU_bitexact_schedule_elements': elements,
        'CPU_dispatch_resource_ownership_cases': shape_cases,
        'performance_hypothesis': 'Remove duplicate C reads, INT32 Cast and 2 FP32 multiplies and original activation copy; larger row groups may amortize quantization. Per-block scale repair and loss of input double-buffering can regress performance and require device measurements.',
        'CPU_limits': 'Model compares scheduling with separately rounded FP32 RN arithmetic and ODD conversion, not A3 Div implementation or CANN compilation. Exact division/near-half helper source is byte-identical.',
        'device': 'PENDING; compile and all15 precision points plus repeated neighboring v124 controls required.'
    }
    (OUT / 'review_checks.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'source.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True), candidate.splitlines(True), fromfile='v124', tofile=OUT.name)), encoding='utf-8')
    (OUT / 'SUBMIT.md').write_text('''# V138 容量推导完整行块框架\n\n基线：v124，用户已确认25.25版本为代码基线。\n\n适用所有N为32倍数且32<=N<=8192的正常形状，不要求整个AIV行区间放入UB。按floor(8192/N)批行驻留，每批C只读一次、FP32 Cast和两次缩放各执行一次。批内行最大值统一用原QmqCorrectDiv修复scale；反量化结果保留在fp32Buffer，消费完的INT32输入bank依次作为尺度scratch和量化quotient，免activation UB copy。量化Div、近半值筛查与精修、夹紧和ODD/RINT链均沿用v124。宽N和非32整行对齐保留原fallback。\n\n动态tensor/scale/mask容量分别按64元素/64元素/32B对齐；所有mask padding在比较前初始化。单TPipe分支在fallback TPipe创建前分流。峰值显式UB173952 B，加8192 B API预留共182144 B <=196608 B。4096个维度/行数/尾块组合已检查资源、访问和逐行唯一归属。\n\n不保证提速：各批尺度修复与depth1输入缺少搬算双buffer可能产生回归。需要平台CANN编译、15/15零错误，再按当天v124相邻对照比较并原样复测，不以CPU或最高单次估分替代NPU证据。\n''', encoding='utf-8')
    print(json.dumps(review, ensure_ascii=False))


if __name__ == '__main__':
    main()
