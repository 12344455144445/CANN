"""One-launch guarded M/N Cube grid with mixed hardware SyncAll barrier.

Builds from immutable v124. CPU models validate layout and ownership only.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
import ast
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "v124_scale_output_bank_20261012/kernel.asc"
FOLDER = ROOT / "v139_partition_fused_barrier_20261013"
BASE_SHA = "4010bfbbb0f4d1be969556db956cebad2de65b7784bb5a9da55b57ea25338574"
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
source = BASE.read_text(encoding="utf-8")

device = r'''
// [V139] One physical launch: each AIC computes an independent M/N tile.
// All launched AICs and AIVs reach the same mixed hardware barrier once.
// Only after COMPLETE C exists, AIVs own full rows for the original math.
__schedmode__(1) __global__ __mix__(1, 2) void qmq_partition_fused_entry(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR workspace,
    __kfc_workspace__ __gm__ uint8_t* systemWorkspace,
    GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale,
    uint32_t m, uint32_t n, uint32_t k,
    AscendC::tiling::TCubeTiling tiling,
    uint32_t rowGroups, uint32_t columnGroups,
    uint32_t columnWidth, uint32_t blockRows)
{
    AscendC::InitSocState();
    if ASCEND_IS_AIC {
        AscendC::SetSysWorkspace(systemWorkspace);
        const uint32_t core = GetBlockIdx();
        const uint32_t rowGroup = core / columnGroups;
        const uint32_t columnGroup = core % columnGroups;
        const uint32_t rowOffset = rowGroup * blockRows;
        const uint32_t columnOffset = columnGroup * columnWidth;
        const uint32_t currentM = QmqMinValue(blockRows, m - rowOffset);
        const uint32_t currentN = QmqMinValue(columnWidth, n - columnOffset);
        using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
        using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
        using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
        using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
        GlobalTensor<int8_t> aGm, bGm;
        GlobalTensor<int32_t> cGm;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), static_cast<uint64_t>(m) * k);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), static_cast<uint64_t>(n) * k);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), static_cast<uint64_t>(m) * n);
        TPipe pipe;
        matmul::Matmul<AType, BType, CType, BiasType, kInt8MatmulConfig> mm;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
        // Full N is the output C row stride. Physical B is [N,K].
        mm.SetOrgShape(m, n, k, k, n);
        mm.SetTail(static_cast<int32_t>(currentM), static_cast<int32_t>(currentN));
        mm.SetTensorA(aGm[static_cast<uint64_t>(rowOffset) * k], false);
        mm.SetTensorB(bGm[static_cast<uint64_t>(columnOffset) * k], true);
        mm.IterateAll(cGm[static_cast<uint64_t>(rowOffset) * n + columnOffset]);
        mm.End();
    }
    // No return/continue can bypass this barrier, including empty AIV owners.
    // false synchronizes AIC AND AIV in this __mix__(1,2) entry.
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t vectorCores = 2U * rowGroups * columnGroups;
        const uint32_t rowStart = static_cast<uint32_t>(
            static_cast<uint64_t>(m) * aiv / vectorCores);
        const uint32_t rowEnd = static_cast<uint32_t>(
            static_cast<uint64_t>(m) * (aiv + 1U) / vectorCores);
        if (rowStart < rowEnd) {
            // The full row, not a column fragment, defines yScale.
            quant_matmul_relu_quant_vector(
                workspace, x1Scale, x2Scale, y, yScale,
                m, n, rowStart, rowEnd - rowStart);
        }
    }
}

'''

host = r'''    // [V139] Match wider complete M tiles with parallel N slices on small M.
    // Regular ND strides are deliberately gated; all other shapes keep v124.
    if (m <= 384U && n >= 1024U && n % 32U == 0U &&
        k >= 256U && k % 32U == 0U &&
        static_cast<uint64_t>(n) * k >= 1ULL * 1024ULL * 1024ULL) {
        const uint32_t physicalAic = platform->GetCoreNumAic();
        const uint32_t physicalAiv = platform->GetCoreNumAiv();
        const uint32_t gridBudget = std::min<uint32_t>(
            usableCoreNum, std::min<uint32_t>(physicalAic, physicalAiv / 2U));
        const uint32_t blockRows = m <= 192U ? 16U : 32U;
        const uint32_t rowGroups = QmqHostCeilDiv(m, blockRows);
        const uint32_t requestedColumns = std::min<uint32_t>(
            gridBudget / rowGroups, n / 128U);
        if (requestedColumns >= 2U) {
            const uint32_t columnWidth = QmqHostCeilDiv(
                QmqHostCeilDiv(n, requestedColumns), 128U) * 128U;
            const uint32_t columnGroups = QmqHostCeilDiv(n, columnWidth);
            const uint32_t gridCores = rowGroups * columnGroups;
            const uint32_t oldRowGroups = QmqHostCeilDiv(m, 16U);
            const bool usefulGrid = m <= 192U ||
                gridCores >= oldRowGroups + QmqHostCeilDiv(oldRowGroups, 4U);
            matmul_tiling::MultiCoreMatmulTiling partApi(*platform);
            if (columnGroups >= 2U && usefulGrid && gridCores <= gridBudget &&
                partApi.SetDim(1) != -1) {
                partApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8, false);
                partApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8, true);
                partApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT32);
                partApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT32);
                partApi.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
                partApi.SetShape(static_cast<int32_t>(blockRows), static_cast<int32_t>(columnWidth), static_cast<int32_t>(k));
                partApi.EnableMultiCoreSplitK(false);
                partApi.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
                partApi.EnableBias(false);
                partApi.SetBufferSpace(-1, -1, -1);
                if (columnWidth >= 512U && partApi.SetFixSplit(-1, 512, -1) == -1) {
                    partApi.SetFixSplit(-1, -1, -1);
                }
                AscendC::tiling::TCubeTiling partTiling;
                if (partApi.GetTiling(partTiling) != -1 && partTiling.usedCoreNum != 0 &&
                    partTiling.M != 0 && partTiling.N != 0 &&
                    partTiling.Ka != 0 && partTiling.Kb != 0 &&
                    partTiling.singleCoreM == blockRows &&
                    partTiling.singleCoreN == columnWidth) {
                    const uint64_t partElements = static_cast<uint64_t>(m) * n;
                    if (partElements <= SIZE_MAX / sizeof(int32_t)) {
                        uint8_t* partResult = nullptr;
                        uint8_t* partSystem = nullptr;
                        const size_t partSystemSize = std::max<size_t>(
                            static_cast<size_t>(platform->GetLibApiWorkSpaceSize()), 1U);
                        if (QmqAcquireWorkspace(static_cast<size_t>(partElements * sizeof(int32_t)),
                            partSystemSize, partResult, partSystem)) {
                            qmq_partition_fused_entry<<<gridCores, 0, stream>>>(
                                x1, x2, partResult, partSystem, x1Scale, x2Scale, y, yScale,
                                m, n, k, partTiling, rowGroups, columnGroups, columnWidth, blockRows);
                            (void)aclrtSynchronizeStream(stream);
                            return;
                        }
                    }
                }
            }
        }
    }

'''

device_anchor = "// ============================================================================\n// [PATCH-H1] workspace"
host_anchor = "    const uint32_t aivCandidate = static_cast<uint32_t>(std::min<uint64_t>(\n"
assert source.count(device_anchor) == source.count(host_anchor) == 1
candidate = source.replace(device_anchor, device + device_anchor).replace(host_anchor, host + host_anchor)
assert candidate.replace(device, "", 1).replace(host, "", 1) == source
assert "return" not in re.sub(r"//[^\n]*", "", device)
assert "continue" not in re.sub(r"//[^\n]*", "", device)
assert device.count("SyncAll<false>()") == 1
assert "CrossCore" not in device
assert host.count("stream>>>") == 1
assert not re.search(r"partTiling\.\w+\s*=(?!=)", host)
assert device.index("mm.End();") < device.index("SyncAll<false>()")
assert device.index("SyncAll<false>()") < device.index("quant_matmul_relu_quant_vector(")


def ceil_div(a, b):
    return (a + b - 1) // b


def grid(m, n, k, available, aic, aiv):
    if not (0 < m <= 384 and n >= 1024 and n % 32 == 0 and
            k >= 256 and k % 32 == 0 and n * k >= 1 * 1024 * 1024):
        return None
    budget = min(available, aic, aiv // 2)
    block = 16 if m <= 192 else 32
    rows = ceil_div(m, block)
    requested = min(budget // rows, n // 128)
    if requested < 2:
        return None
    width = ceil_div(ceil_div(n, requested), 128) * 128
    columns = ceil_div(n, width)
    old_rows = ceil_div(m, 16)
    if columns < 2 or rows * columns > budget:
        return None
    if m > 192 and rows * columns < old_rows + ceil_div(old_rows, 4):
        return None
    return rows, columns, width, block, budget


geometry_cases = 0
largest_vector_rows = 0
for available in (1, 8, 16, 20, 24, 32, 48, 64, 128):
    for aic, aiv in ((8, 16), (24, 48), (32, 64), (48, 96), (24, 47), (0, 0)):
        for m in range(1, 386):
            for n in (1024, 1025, 1056, 2048, 3072, 4096, 4128, 6144, 8192, 12288, 16384, 2147483616):
                for k in (255, 256, 288, 511, 512, 1024, 4096):
                    g = grid(m, n, k, available, aic, aiv)
                    if g is None:
                        continue
                    rg, cg, width, block, budget = g
                    cores = rg * cg
                    assert 0 < cores <= budget <= aic
                    assert 2 * cores <= aiv
                    assert width % 128 == 0 and width <= 2147483647
                    assert (width * k) % 32 == 0
                    r_edges = [(i * block, min(m, (i + 1) * block)) for i in range(rg)]
                    c_edges = [(i * width, min(n, (i + 1) * width)) for i in range(cg)]
                    assert r_edges[0][0] == c_edges[0][0] == 0
                    assert r_edges[-1][1] == m and c_edges[-1][1] == n
                    assert all(x[1] == y[0] for x, y in zip(r_edges, r_edges[1:]))
                    assert all(x[1] == y[0] for x, y in zip(c_edges, c_edges[1:]))
                    for r0, r1 in r_edges:
                        for c0, c1 in c_edges:
                            assert r0 < r1 <= m and c0 < c1 <= n
                            assert r0 * k + (r1 - r0) * k <= m * k
                            assert c0 * k + (c1 - c0) * k <= n * k
                            assert (r1 - 1) * n + c1 <= m * n
                    v_edges = [(m * i // (2 * cores), m * (i + 1) // (2 * cores)) for i in range(2 * cores)]
                    assert v_edges[0][0] == 0 and v_edges[-1][1] == m
                    assert all(x[1] == y[0] for x, y in zip(v_edges, v_edges[1:]))
                    largest_vector_rows = max(largest_vector_rows, max(b - a for a, b in v_edges))
                    assert largest_vector_rows <= 1024
                    geometry_cases += 1

assert grid(64, 4096, 1024, 24, 24, 48) is not None
assert grid(16, 8192, 512, 24, 24, 48) is not None
assert grid(16, 8192, 256, 24, 24, 48) is not None
assert grid(16, 8192, 255, 24, 24, 48) is None
assert grid(16, 8192, 511, 24, 24, 48) is None
assert grid(16, 1024, 256, 24, 24, 48) is None
assert grid(64, 4097, 1024, 24, 24, 48) is None
assert grid(64, 4096, 1025, 24, 24, 48) is None
assert grid(385, 4096, 4096, 24, 24, 48) is None

# Layout and COMPLETE-row quantization checks with negative scales and zero
# rows. CPU K is reduced while testing the same spatial strides/offsets.
rng = np.random.default_rng(13920261013)
numeric_cases = 0
cpu_elements = 0
for m, n, actual_k, available in (
    (1, 4096, 7, 24), (7, 4128, 3, 24), (31, 4096, 5, 24),
    (64, 4096, 7, 24), (65, 4128, 3, 24), (96, 6144, 5, 24),
    (127, 8192, 7, 24), (191, 4096, 3, 24), (192, 4128, 7, 24),
    (193, 4096, 3, 24), (256, 6144, 5, 24), (319, 4096, 3, 48),
    (383, 4128, 5, 48), (384, 4096, 7, 48),
):
    g = grid(m, n, 4096, available, available, 2 * available)
    if g is None:
        continue
    rg, cg, width, block, _ = g
    a = rng.integers(-128, 128, (m, actual_k), dtype=np.int8)
    b = rng.integers(-128, 128, (n, actual_k), dtype=np.int8)
    expected = a.astype(np.int32) @ b.astype(np.int32).T
    got = np.zeros((m, n), dtype=np.int32)
    writes = np.zeros((m, n), dtype=np.uint8)
    for core in range(rg * cg):
        r0, c0 = (core // cg) * block, (core % cg) * width
        r1, c1 = min(m, r0 + block), min(n, c0 + width)
        tile = a[r0:r1].astype(np.int32) @ b[c0:c1].astype(np.int32).T
        for row in range(r1 - r0):
            offset = (r0 + row) * n + c0
            got.reshape(-1)[offset:offset + c1 - c0] = tile[row]
            writes.reshape(-1)[offset:offset + c1 - c0] += 1
    assert np.all(writes == 1) and np.array_equal(got, expected)
    s1 = rng.uniform(-2, 2, m).astype(np.float32)
    s2 = rng.uniform(-2, 2, n).astype(np.float32)
    s1[0] = 0.0
    reference_act = np.maximum((expected.astype(np.float32) * s1[:, None]) * s2, np.float32(0))
    reference_max = reference_act.max(axis=1)
    reference_scale = np.where(reference_max > 0, reference_max / np.float32(127), np.float32(1))
    reference_y = np.clip(np.rint(reference_act / reference_scale[:, None]), 0, 127).astype(np.int8)
    y = np.empty_like(reference_y)
    scale = np.empty_like(reference_scale)
    row_writes = np.zeros(m, dtype=np.uint8)
    cores = 2 * rg * cg
    for index in range(cores):
        r0, r1 = m * index // cores, m * (index + 1) // cores
        if r0 == r1:
            continue
        act = np.maximum((got[r0:r1].astype(np.float32) * s1[r0:r1, None]) * s2, np.float32(0))
        maximum = act.max(axis=1)
        local_scale = np.where(maximum > 0, maximum / np.float32(127), np.float32(1))
        scale[r0:r1] = local_scale
        y[r0:r1] = np.clip(np.rint(act / local_scale[:, None]), 0, 127).astype(np.int8)
        row_writes[r0:r1] += 1
    assert np.all(row_writes == 1)
    assert np.array_equal(scale.view(np.uint32), reference_scale.view(np.uint32))
    assert np.array_equal(y, reference_y)
    assert np.all(y[0] == 0) and scale[0] == np.float32(1)
    numeric_cases += 1
    cpu_elements += m * n

# Reuse the audited parser without executing that module's historical CLI,
# which would otherwise rewrite the old round5 aggregate audit on import.
audit_path = ROOT / "audit_round5_shapes.py"
audit_tree = ast.parse(audit_path.read_text(encoding="utf-8"))
audit_definitions = ast.Module(body=[node for node in audit_tree.body if isinstance(
    node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)
)], type_ignores=[])
audit_namespace = {}
exec(compile(audit_definitions, str(audit_path), "exec"), audit_namespace)
audit_result = audit_namespace["audit"](candidate)
FOLDER.mkdir(exist_ok=True)
if any((FOLDER / name).exists() for name in ("submission_attempt.json", "submission_info.json")):
    raise SystemExit("Submitted source is immutable; choose a fresh candidate folder")
payload = candidate.encode("utf-8")
(FOLDER / "kernel.asc").write_bytes(payload)
report = {
    "review_pass": True,
    "sha256": hashlib.sha256(payload).hexdigest(),
    "base": BASE.parent.name,
    "base_sha256": BASE_SHA,
    "exact_inverse_restores_baseline": True,
    "precision_tiny_general_vector_original_cube_original_host_fallback_byte_identical": True,
    "only_additions": "One batch-scheduled mixed M/N partition kernel, and guarded host dispatch",
    "dispatch_gate": "M<=384,N>=1024,N%32=0,K>=256,K%32=0,N*K>=1MiB,at least2 N partitions, coherent tiling",
    "Sheldon_proxy_16x8192x512_is_covered_by_requested_grid": True,
    "M_rows": "16 if M<=192, else32",
    "N_slice_width_alignment_elements": 128,
    "grid_budget": "min(availableCoreNum,GetCoreNumAic(),GetCoreNumAiv()/2)",
    "all_cores_once_unconditional_SyncAll_false": True,
    "new_kernel_self_flags_or_busy_wait": False,
    "kernel_launches_in_new_branch": 1,
    "workspace_shape_and_strides": "C INT32[M,N] with full original N row stride; A offset=r*K; B[N,K] offset=c*K",
    "B_template_host_runtime_transpose_consistent": True,
    "GM_bytes": "4*M*N + existing library system workspace; no new global reduction workspace",
    "UB_allocations": "Original vector helper unchanged; same max174976 incl Select reserve; observed vectorRows<=1024",
    "audit_round5_shapes_result": audit_result,
    "geometry_CPU_cases": geometry_cases,
    "largest_vector_rows_per_call": largest_vector_rows,
    "physical_B_NK_C_Nstride_numeric_CPU_cases": numeric_cases,
    "numeric_CPU_elements": cpu_elements,
    "CPU_zero_rows_negative_scales_complete_rowmax": True,
    "CPU_models_limit": "Reduced K numeric tests model exact spatial layout; not CANN compiler, library implementation, or hardware scheduler",
    "official_CANN9_sources": {
        "SyncAll": "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html",
        "schedmode": "https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html",
        "MatmulType": "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0614.html",
        "SetOrgShape": "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0651.html",
        "GetCoreNumAic": "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1030.html",
    },
    "hardware_validation": "PENDING: mixed barrier scheduling, partial-N C stride, CANN compilation,15/15 zero error,actual branch coverage and performance",
    "performance_risks": [
        "Global barrier adds delay and removes old two-Mtile Cube/Vector overlap",
        "Batch scheduling may add resource queue latency",
        "Small M allows fewer active whole-row AIVs even with many AICs",
        "Changes to compiled module layout may affect fallback timings",
        "Actual library returned shape must match requested; otherwise original fallback runs",
    ],
    "concurrency_limit": "Existing baseline static workspace cache retained: serial contest calls assumed; not safe for concurrent calls sharing cache",
}
(FOLDER / "review_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(FOLDER / "source.diff").write_text("".join(difflib.unified_diff(source.splitlines(True), candidate.splitlines(True), fromfile="v124/kernel.asc", tofile="v139/kernel.asc")), encoding="utf-8")
(FOLDER / "SUBMIT.md").write_text(
    "# v139 单launch M/N Cube重构\n\n小M大N的Cube仅沿M划分会浪费补齐到16行的计算通道。"
    "本候选把Cube M块提升至16/32，并沿N建立二维核网格；所有AIC/AIV在同一无条件SyncAll<false>处"
    "完成完整C矩阵可见性，再由唯一AIV负责每个完整行max、scale和quant。只一次launch；"
    "零新增忙等/手工旗；其他形状和数值链为v124原样。\n\n"
    "CPU模型和静态审核通过。CANN编译/混合屏障/NPU15点/分支覆盖/性能尚未验证。"
    "需完整15/15零错误与邻近同源码对照，不能仅凭提交Pass宣称新分支被覆盖。\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
