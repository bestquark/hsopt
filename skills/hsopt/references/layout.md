# hsopt exact-campaign layout

Current organization:

- `skills/hsopt/`: instructions for sequential exact T-gate optimization.
- `src/hsopt/compiled_unitary.py`: fixed compiled-unitary evaluator.
- `src/hsopt/exact_queue.py`: content-addressed evaluation queue.
- `src/hsopt/exact_loop.py`: external lane state and acceptance loop.
- `experiments/abs_energy/`: compact HamLib absolute-energy fixtures.
- `experiments/diff_energy/`: compact energy-difference fixtures and reaction runtime.
- `experiments/generate_initial_pf.py`: immutable baseline-policy generator.
- caller-supplied external `HSOPT_DATA_ROOT`: all campaign workspaces and logs.

Keep mutable campaign histories outside the repository checkout under
`$HSOPT_DATA_ROOT/{paper,development,old_experiments,scratch}`. Paper HamLib-121
runs live below `paper/hamlib121/<campaign-id>/run_01` through `run_05`;
reaction runs live below `paper/reactions10/<campaign-id>`.
