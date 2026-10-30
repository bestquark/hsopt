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
- `baseline/molecules.json`: molecular selections and pinned archive checksums.
- `baseline/reactions.json`: reaction definitions, species recipes and historical cache audit.
- `baseline/initial_nested_pf.py`: common grouped PF2 template for shared protocols.
- `baseline/generate_initial_pf.py`: compact Pauli PF2 generator from sealed calibrations.
- `experiments/abs_protocol/`: shared molecular ELOTE workflow.
- `experiments/diff_protocol/`: shared reaction δ-ELOTE workflow.
- `experiments/elote_search/`: search-strategy comparison and frozen selected ELOTE.
- `single_molecule` branch: archived single-instance experiment folders and baseline copies.
- `cli/src/hsopt/campaign/interface.py`: portable task requests and result delivery;
  delivery records the result, applies acceptance, and proposes the next mutation.
- external `hsopt_workdir/data/`: inputs and sealed references/calibrations.
- external `hsopt_workdir/runs/`: lane histories and the scientific task ledger.
- external `hsopt_workdir/runtime/`: user-supplied dispatchers and machine setup.

See `docs/runtime.md` for implementing a runtime. Scientific code and plotting
stay in the checkout; compute-specific code and generated records stay outside.
