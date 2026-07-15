# Branch Map

Use this split to keep `main` small and make future work explicit.

- `main`: `hsopt`, focused on product-formula T-gate optimization in
  `examples/pfs/` with `$hsopt` and `$hsgepa`.
- `codex/archive-showroom-20260715`: tracked MkDocs showroom source and the
  GitHub Pages deployment workflow removed from `main` during the 2026-07-15
  cleanup.
- `backup/pre-hsopt-cleanup-20260511`: exact backup of the dirty workspace
  before the hsopt cleanup.
- `resource/classical-compute`: branch point for the broader classical compute
  side of Hamiltonian-simulation resource work, including CPU/GPU constraints,
  GPU-memory objectives, database experiments, and non-PFS protocol examples.

Keep campaign histories and generated reports outside the benchmark tree unless
they are deliberately being promoted as checked-in benchmark artifacts.
