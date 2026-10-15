QuantMatmulReluQuant v82 passed submission724207 but still shows large per-point performance gaps. This adds three reversible structural candidates: v83 extends safe multi-row batching to unaligned small N, v84 prepares a bounded tiny vector launch before Cube tiling, and v85 combines the independent changes. The verified baseline remains available.

Validation: five static/CPU check groups pass, including DMA packing, row coverage, resource bounds, queue boundaries and source change contracts. Four historical CPU arithmetic/layout checks also pass; a read-only independent review found no confirmed new source defect and its verification-gap finding was addressed. Full-source clipboard helpers validate version hashes.

ASC compilation, NPU correctness, synchronization and performance are pending. These are experimental artifacts, not confirmed score improvements. No change to the team's recommended kernel.
