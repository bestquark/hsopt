# Branch Map

Use this split to keep `main` small and make future work explicit.

- `main`: exact compiled-unitary HSOpt with the visible 82-Hamiltonian and
  ten-reaction paper protocols under `experiments/`. The former `clean` branch
  was promoted to `main` and deleted.
- `site`: tracked documentation-site source and the GitHub Pages deployment
  workflow removed from `main` during the 2026-07-15 cleanup.
- `codex/safety-pre-cleanup-20260715`: exact tracked snapshot of the July 2026
  pre-cleanup worktree, including the older HamLib policy trajectories. Keep it
  as recovery evidence; do not use it as a campaign runtime or merge it back
  wholesale.
- `backup/pre-hsopt-cleanup-20260511`: exact backup of the dirty workspace
  before the hsopt cleanup.
- `resource/classical-compute`: branch point for the broader classical compute
  side of Hamiltonian-simulation resource work, including CPU/GPU constraints,
  GPU-memory objectives, database experiments, and non-PFS protocol examples.

Keep campaign histories and generated reports outside the benchmark tree unless
they are deliberately being promoted as checked-in benchmark artifacts.
