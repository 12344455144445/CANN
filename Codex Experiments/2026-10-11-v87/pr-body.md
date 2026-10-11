The user-provided v86 screenshot shows 15/15 Pass with zero reported output errors, but general-path latencies remain close to the prior baseline. This adds a v87 experiment combining the v86 tiny path with the previously reviewed v83 padded multi-row general path.

Static checks prove the tiny/host/Cube functions match v86 and the general Vector/DMA helper functions match v83. CPU layout, queue, mask-tail and resource models pass. Current test shapes are unavailable, so no claim is made that a particular point hits the new gate or that performance improves.

NPU compilation, correctness and timing remain pending. Keep v86 for rollback and compare v86 -> v87 -> v86 within one scoring window.
