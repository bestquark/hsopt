# Branch Map

Use this split to keep `main` small and make future work explicit.

- `main`: `hsopt`, focused on product-formula T-gate optimization in
  `examples/ft_protocols/pfs/` with `$t-autoresearch` and `$t-gepa`.
- `backup/pre-hsopt-cleanup-20260511`: exact backup of the dirty workspace
  before the hsopt cleanup.
- `resource/hamiltonian-simulation-resources`: branch point for the broader
  Hamiltonian-simulation resource-estimation work, including CPU/GPU/QPU
  constraints, GPU-memory objectives, database experiments, and non-PFS protocol
  examples.
- `resource/gpu-memory-constraints`: explicit branch name for continuing the
  GPU-memory/resource-constraint direction without mixing it back into `main`.

Keep campaign histories and generated reports outside the benchmark tree unless
they are deliberately being promoted as checked-in benchmark artifacts.
