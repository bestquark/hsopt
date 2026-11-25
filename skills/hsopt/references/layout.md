# hsopt exact-campaign layout

Current organization:

- `skills/hsopt/SKILL.md`: instructions for sequential exact T-gate optimization.
- `skills/hsopt/references/`: mutation brief plus ELOTE, runtime, reproduction and CLI guides.
- `skills/hsopt/evaluation/compiled_unitary.py`: fixed compiled-unitary evaluator.
- `skills/hsopt/campaign/queue.py`: content-addressed evaluation queue.
- `skills/hsopt/campaign/loop.py`: external lane state and acceptance loop.
- `skills/hsopt/sandbox/openshell.py`: OpenShell host setup, isolated OpenCode
  mutation adapter, and redacted tool/security telemetry.
- `skills/hsopt/view/mutation_history.py`: read-only JSON adapter and compact
  mutation-history grid shared with the hosted monitoring site.
- `experiments/baseline/molecules.json`: molecular selections and pinned archive checksums.
- `experiments/baseline/reactions.json`: reaction definitions, species recipes and historical cache audit.
- `experiments/baseline/initial_nested_pf.py`: common grouped PF2 template for shared protocols.
- `experiments/baseline/generate_initial_pf.py`: compact Pauli PF2 generator from sealed calibrations.
- `experiments/abs_protocol/`: shared molecular ELOTE workflow.
- `experiments/diff_protocol/`: shared reaction δ-ELOTE workflow.
- `experiments/elote_search/`: search-strategy comparison.
- `results/elote/`, `results/delta_elote/`: frozen released ELOTE and δ-ELOTE policies.
- `results/paper/`: frozen manuscript figure data.
- `single_molecule` branch: archived single-instance experiment folders and baseline copies.
- `skills/hsopt/campaign/interface.py`: portable task requests and result delivery;
  delivery records the result, applies acceptance, and proposes the next mutation.
- external `hsopt_workdir/data/`: inputs and sealed references/calibrations.
- external `hsopt_workdir/runs/`: lane histories and the scientific task ledger.
- external `hsopt_workdir/runtime/`: user-supplied dispatchers and machine setup.

See `skills/hsopt/references/runtime.md` for implementing a runtime. Scientific code and plotting
stay in the checkout; compute-specific code and generated records stay outside.
