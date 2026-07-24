# hsopt exact-campaign layout

Current organization:

- `skills/hsopt/SKILL.md`: instructions for sequential exact T-gate optimization.
- `src/hsopt/compiled_unitary.py`: fixed compiled-unitary evaluator.
- `src/hsopt/exact_queue.py`: content-addressed evaluation queue.
- `src/hsopt/exact_loop.py`: external lane state and acceptance loop.
- `src/hsopt/openshell_agent.py`: isolated OpenCode/OpenShell mutation adapter.
- `experiments/abs_energy/`: compact HamLib absolute-energy fixtures.
- `experiments/diff_energy/`: ten-reaction definition and compact difference fixture.
- `experiments/generate_initial_pf.py`: immutable baseline-policy generator.
- caller-supplied external `HSOPT_DATA_ROOT`: all campaign workspaces and logs.

Keep mutable campaign histories outside the repository checkout under
`$HSOPT_DATA_ROOT/{exact,development,old_experiments,scratch}`. The campaign
mirrors the two checked-in families under
`exact/<campaign-id>/lanes/abs_energy/` and
`exact/<campaign-id>/lanes/diff_energy/`.
