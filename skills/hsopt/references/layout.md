# hsopt exact-campaign layout

Current organization:

- `skills/hsopt/SKILL.md`: instructions for sequential exact T-gate optimization.
- `cli/src/hsopt/evaluation/compiled_unitary.py`: fixed compiled-unitary evaluator.
- `cli/src/hsopt/campaign/queue.py`: content-addressed evaluation queue.
- `cli/src/hsopt/campaign/loop.py`: external lane state and acceptance loop.
- `cli/src/hsopt/sandbox/openshell.py`: OpenShell host setup, isolated OpenCode
  mutation adapter, and redacted tool/security telemetry.
- `cli/src/hsopt/view/mutation_history.py`: read-only JSON adapter and compact
  mutation-history grid shared with the hosted monitoring site.
- `experiments/abs_energy/`: compact HamLib absolute-energy fixtures.
- `experiments/diff_energy/`: ten-reaction definition and compact difference fixture.
- `experiments/generate_initial_pf.py`: sole loose/tight paper-baseline
  implementation and immutable policy generator.
- `experiments/{abs_energy,diff_energy}/<instance>/initial_pf_{loose,tight}.py`:
  the two generated iteration-0 policies; a campaign copies exactly one into
  its isolated lane.
- caller-supplied external `HSOPT_DATA_ROOT`: all campaign workspaces and logs.

Keep mutable campaign histories outside the repository checkout under
`$HSOPT_DATA_ROOT/{exact,development,old_experiments,scratch}`. The campaign
mirrors the two checked-in families under
`exact/<campaign-id>/lanes/abs_energy/` and
`exact/<campaign-id>/lanes/diff_energy/`.
