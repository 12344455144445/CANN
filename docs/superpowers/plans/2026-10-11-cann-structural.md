# CANN v83-v85 Structural Optimization Implementation Plan

> For agentic workers: execute inline using executing-plans; obtain one independent final review.

Goal: build and validate structural candidates from the platform-passed v82 (submission 724207).
Architecture: independent padded-row DMA/quantization batching and tiny host launch simplification, then their composition. Keep numerical helpers and the general Cube protocol unchanged.
Tech Stack: Ascend C, CANN 9.0 / dav-2201, Python/NumPy CPU models, PowerShell delivery.
Spec: ../../../Carl Memory/2026-10-10_工作总结.md and the submitted v82 source; current 15-point evidence in the parent workspace outputs/CANN_724207_评测分析_2026-10-11.md.

## Constraints and review focus

- Preserve 14-argument run_kernel and all three precision helper bodies.
- No CPU result computation in submitted code. No input-content or case-ID specialization.
- CANN DataCopyPad: UB addresses/strides in 32-byte blocks, GM strides in bytes; each side padding at most32 bytes.
- Test unaligned N tails, zero/negative scales, zero rows, half-rounding neighborhoods, odd row counts, group boundaries64 and multiple cores.
- GPU compilation, synchronization and speed are pending platform evaluation; CPU models do not prove them.

## Tasks

- [x] Write source/DMA/row-coverage and CPU numerical checks; run against absent candidate artifacts to establish missing implementation.
- [x] Generate v83: only for64<=N<=3072 and N%32!=0, pad float rows to32 elements, clear skipped UB gaps, use strided multiblock GM->UB and tightly packed UB->GM. Preserve existing aligned path.
- [x] Generate v84 independently: validated tiny shapes launch directly with a bounded host row planner before Cube tiling. Retain result workspace for the device fallback and stream synchronization.
- [x] Generate v85 as composition. Prove transformations invert exactly and unchanged scopes match baseline.
- [x] Exercise adversarial layout/numerical CPU cases, resource and row coverage checks; independent review; repair findings once and rerun.
- [x] Deliver versioned files, hashes, clipboard tools, reading map, test order and failure criteria.

Ruling: user explicitly requested autonomous implementation, so proceed without a plan approval pause. Use a new linked worktree to preserve the verified sources. Retain artifacts and tests for teammate continuation instead of deleting them after review. Main remains the platform baseline until controlled device evaluation chooses a winner.
