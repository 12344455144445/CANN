"""Produce immutable single-change Cube candidates from the exact v124 source.

Offline source/layout checks do not establish CANN compilation or NPU correctness.
"""
from pathlib import Path
import difflib
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "v124_scale_output_bank_20261012/kernel.asc"
EXPECTED_BASE_SHA = "4010bfbbb0f4d1be969556db956cebad2de65b7784bb5a9da55b57ea25338574"
base_bytes = BASE.read_bytes()
assert hashlib.sha256(base_bytes).hexdigest() == EXPECTED_BASE_SHA
base = base_bytes.decode("utf-8")
assert "\r" not in base


def ceil_div(n, d):
    return (n + d - 1) // d


# Exhaustive proxy for the unchanged M-group/odd-tail ownership equations.
# The library's actual returned singleCoreM remains a device-tiling dependency.
ownership_cases = 0
for cores in (8, 16, 20, 24, 32, 48, 64):
    for m in range(1, 8193):
        candidates = min(m, 2 * cores)
        balanced = ceil_div(m, candidates)
        span = ceil_div(m, cores)
        floor = max(4, span) if span <= 8 else 16
        block_m = min(m, max(floor, balanced))
        blocks = ceil_div(m, block_m)
        blocks_per_group = min(2, max(1, ceil_div(blocks, cores)))
        groups = ceil_div(blocks, blocks_per_group)
        assert groups <= cores
        assert blocks <= candidates
        previous_end = 0
        for group in range(groups):
            for tile in range(blocks_per_group):
                start = (group * blocks_per_group + tile) * block_m
                if start >= m:
                    continue
                end = min(m, start + block_m)
                half = ceil_div(block_m, 2)
                a0, a1 = start, min(start + half, end)
                b0, b1 = start + half, end
                assert a0 == previous_end
                assert a1 <= end
                if b0 < end:
                    assert a1 == b0 and b0 <= b1
                else:
                    assert a1 == end
                previous_end = end
        assert previous_end == m
        ownership_cases += 1

# Exercise physical B[N,K], including negative INT8 endpoints, odd shapes,
# and B column-partition addresses. This is a layout model, not API emulation.
rng = np.random.default_rng(13520261013)
layout_cases = 0
max_error = 0
for m in (1, 3, 7, 16, 31):
    for n in (1, 3, 17, 32, 65):
        for k in (1, 7, 32, 65):
            a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
            b = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
            expected = a.astype(np.int32) @ b.astype(np.int32).T
            got = np.empty((m, n), dtype=np.int32)
            for column in range(n):
                b_offset = column * k
                physical_row = b.reshape(-1)[b_offset:b_offset + k]
                got[:, column] = np.sum(
                    a.astype(np.int32) * physical_row.astype(np.int32),
                    axis=1,
                    dtype=np.int32,
                )
            assert np.array_equal(got, expected)
            max_error = max(max_error, int(np.max(np.abs(got - expected))))
            layout_cases += 1


def publish(name, candidate, inverse, change, docs, extra):
    assert inverse(candidate) == base
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    for state in ("submission_info.json", "submission_attempt.json"):
        if (folder / state).exists():
            raise SystemExit(f"Submitted source is immutable: {folder.name}")
    payload = candidate.encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    review = {
        "review_pass": True,
        "sha256": sha,
        "base": BASE.parent.name,
        "base_sha256": EXPECTED_BASE_SHA,
        "only_change": change,
        "exact_source_inverse": True,
        "precision_general_vector_tiny_host_flags_byte_identical_except_described_change": True,
        "new_UB_or_GM_allocations": False,
        "additional_kernel_launches": 0,
        "unchanged_M_ownership_proxy_cases": ownership_cases,
        "B_physical_NK_matmul_layout_CPU_cases": layout_cases,
        "CPU_layout_error": max_error,
        "official_CANN9_sources": docs,
        "hardware_validation": "PENDING: compile, 15/15 zero-error, and timing remain required",
        **extra,
    }
    (folder / "kernel.asc").write_bytes(payload)
    (folder / "review_checks.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "source.diff").write_text(
        "".join(difflib.unified_diff(
            base.splitlines(True), candidate.splitlines(True),
            fromfile="v124/kernel.asc", tofile=f"{name}/kernel.asc"
        )), encoding="utf-8"
    )
    (folder / "SUBMIT.md").write_text(
        change + "\n\n源码逆变换精确恢复 v124；CPU 检查不代表 CANN 编译或 NPU 通过。"
        "需 15/15 零误差、来源 hash 绑定及邻近对照再决定是否保留。\n\n"
        + "\n".join(docs) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidate": name, **review}, ensure_ascii=False))


b_old = "using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;"
b_new = "using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;"
assert base.count(b_old) == 1
publish(
    "v135_cube_B_transpose_consistency_20261013",
    base.replace(b_old, b_new),
    lambda x: x.replace(b_new, b_old),
    "显式开启 BType 的 ISTRANS=true，使 INT8 B[N,K] 的模板、host SetBType(true)"
    "及 kernel SetTensorB(true)一致。只改一个 MatmulType 声明。",
    ["https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0614.html"],
    {
        "known_baseline_documentation_mismatch":
            "v124 omitted BType ISTRANS(default false), while host/kernel both request transpose true. "
            "v124 passed hardware despite documented requirement; no universal error claim made.",
        "performance_hypothesis":
            "Library layout specialization may change; no timing improvement is assumed.",
    },
)

org_old = "        matmulObj.SetOrgShape(m, n, k, k);\n"
register = "    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);\n"
org_new = "    matmulObj.SetOrgShape(m, n, k, k);\n"
assert base.count(org_old) == base.count(register) == 1
hoisted = base.replace(org_old, "").replace(register, register + org_new)
publish(
    "v136_cube_original_shape_hoist_20261013",
    hoisted,
    lambda x: x.replace(register + org_new, register).replace(
        "        matmulObj.SetTail(static_cast<int32_t>(currentM), static_cast<int32_t>(n));\n",
        org_old + "        matmulObj.SetTail(static_cast<int32_t>(currentM), static_cast<int32_t>(n));\n"
    ),
    "将不变的 SetOrgShape(m,n,k,k)从最多两个 localTile 的循环内移到 Matmul 注册之后；"
    "SetTail、Tensor A/B地址、IterateAll、旗及原 BType 全部不变。",
    ["https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0651.html"],
    {
        "expected_API_call_count_before": "mBlocksPerGroup (1 or 2)",
        "expected_API_call_count_after": 1,
        "shape_lifetime": "m,n,k remain constant across all localTile iterations; only currentM changes via SetTail.",
        "performance_hypothesis": "Only eliminates duplicate runtime original-shape setup for two-tile groups.",
    },
)
