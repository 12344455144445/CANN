The user confirmed that v84 slightly improved their competition score. This adds an independent v86 candidate that replaces per-element tiny postprocessing with separate FP32 vector multiplies, ReLU and a value-only reduction for N>=16 and bounded scales.

Out-of-range scales and smaller N retain the v84 scalar path. Precision helpers, general Vector/Cube paths, entry point and host launcher remain unchanged. Four CPU/static check groups pass, including direct interpretation of the generated guards and padding layout. An independent read-only review found no blocking source defect; suggested test linkage and fallback-selection checks were added.

NPU compilation, correctness, synchronization and performance remain pending. This is a draft experiment, not a confirmed score improvement. Compare v84 -> v86 -> v84 in the same scoring window.
