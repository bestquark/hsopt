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
- `cli/src/hsopt/campaign/interface.py`: portable task requests and result delivery;
  delivery records the result, applies acceptance, and proposes the next mutation.
- external `hsopt_workdir/data/`: inputs and sealed references/calibrations.
- external `hsopt_workdir/runs/`: lane histories and the scientific task ledger.
- external `hsopt_workdir/runtime/`: user-supplied dispatchers and machine setup.

See `docs/runtime.md` for implementing a runtime. Scientific code and plotting
stay in the checkout; compute-specific code and generated records stay outside.
