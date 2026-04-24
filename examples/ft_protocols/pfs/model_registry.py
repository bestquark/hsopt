from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class MoleculeSpec:
    key: str
    display_name: str
    geometry: str
    active_electrons: int
    active_orbitals: int
    aolabels: Tuple[str, ...] | None = None
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"
    target_time: float = 1.0
    target_iteration: int = 100
    max_parallel: int = 1
    localization: str = "pipek_mezey_split_occ_vir"

    @property
    def cas_label(self) -> str:
        return f"CAS({self.active_electrons}e,{self.active_orbitals}o)"

    @property
    def lane_dir(self) -> Path:
        return SCRIPT_DIR / self.key

    @property
    def editable_script(self) -> Path:
        return self.lane_dir / "simple_pf.py"

    @property
    def snapshots_dir(self) -> Path:
        return self.lane_dir / "snapshots"

    @property
    def results_path(self) -> Path:
        return self.snapshots_dir / "results.tsv"

    @property
    def instruction_path(self) -> Path:
        return SCRIPT_DIR / "instructions" / f"{self.key}.md"

    @property
    def config_path(self) -> Path:
        return SCRIPT_DIR / "configs" / f"{self.key}.json"

    def to_config(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "cas_label": self.cas_label,
                "lane_dir": str(self.lane_dir),
                "editable_script": str(self.editable_script),
                "snapshots_dir": str(self.snapshots_dir),
                "results_path": str(self.results_path),
                "instruction_path": str(self.instruction_path),
                "config_path": str(self.config_path),
            }
        )
        return payload


MOLECULE_SPECS: Dict[str, MoleculeSpec] = {
    "h2": MoleculeSpec(
        key="h2",
        display_name="H2 PF1",
        geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
        active_electrons=2,
        active_orbitals=2,
    ),
    "dmds": MoleculeSpec(
        key="dmds",
        display_name="DMDS PF1",
        geometry="""
C  -2.1538   -0.6230    0.3341
S  -0.6470    0.3161    0.7111
S   0.6470   -0.3160   -0.7111
C   2.1538    0.6231   -0.3341
H  -2.9509   -0.3548    1.0631
H  -2.4958   -0.3767   -0.6961
H  -1.9388   -1.7128    0.4040
H   2.9509    0.3549   -1.0631
H   1.9386    1.7128   -0.4047
H   2.4957    0.3775    0.6963
""".strip(),
        active_electrons=6,
        active_orbitals=5,
        aolabels=("S 3p", "S 3s"),
    ),
    "bh": MoleculeSpec(
        key="bh",
        display_name="BH PF1",
        geometry="""
B 0.000000 0.000000 0.000000
H 0.000000 0.000000 1.230000
""".strip(),
        active_electrons=2,
        active_orbitals=3,
    ),
    "lih": MoleculeSpec(
        key="lih",
        display_name="LiH PF1",
        geometry="""
Li 0.000000 0.000000 0.000000
H  0.000000 0.000000 1.570000
""".strip(),
        active_electrons=2,
        active_orbitals=4,
    ),
}

ACTIVE_MOLECULES = tuple(MOLECULE_SPECS)


def get_spec(key: str) -> MoleculeSpec:
    try:
        return MOLECULE_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"unknown molecule key: {key}") from exc


def render_instruction(spec: MoleculeSpec) -> str:
    return f"""Work only on the local `{spec.display_name}` benchmark.

Before doing anything, make sure your current working directory is
`{spec.lane_dir}`. Keep the session in that lane directory.

Edit only:
- `simple_pf.py`

Do not edit:
- `../evaluate.py`
- `../track_iteration.py`
- `../queued_track_iteration.py`
- `../restore_best_iteration.py`
- `../chemistry.py`
- `../model_registry.py`

Fixed evaluation command:
`uv run python ../queued_track_iteration.py --molecule {spec.key} --script {spec.editable_script} --max-parallel {spec.max_parallel} --description "<one-line mutation summary>"`

Restore-best command:
`uv run python ../restore_best_iteration.py --molecule {spec.key} --script {spec.editable_script}`

Goal:
- lower `score`
- in practice this means lowering compiled `t_gate_count` while keeping
  `passes_accuracy_constraint == true`
- there is no wall-time objective or cap; evaluation runtime is diagnostic only
- exact target: a QPE-like ground-state phase estimate from `exp(+i H t)` with `t = {spec.target_time}`
- the constraint is chemical accuracy on the extracted ground-state energy, not state fidelity
- active space is fixed at `{spec.cas_label}` in the default transformed MO basis
- orbital transform is fixed at `{spec.localization}`
- use this lane to optimize synthesis precision for Pauli rotations, not to
  change the molecule or the chemistry setup

Rules:
1. Continue from the current snapshot history. Do not delete, reset, or renumber existing snapshots.
2. Before the first mutation, inspect the current best kept iteration and restore it if `simple_pf.py` is not already that best kept version.
3. Then do exactly {spec.target_iteration} mutated outer iterations beyond the current highest archived iteration.
4. One outer iteration = one code mutation in `simple_pf.py` plus one queued evaluation.
5. Every mutation must be scored and archived with the fixed queued command, even if you expect it to fail the chemical-accuracy energy check or be worse than the current best.
6. Do not skip, overwrite, or suppress failed or worse mutations. `results.tsv` should reflect every scored proposal, including `discard` and `crash`.
7. After each evaluation, inspect the scored JSON result before launching the next scored iteration.
8. If the last result is `discard` or `crash`, run the restore-best command before the next mutation.
9. If the last result is `keep`, continue mutating from that kept file.
10. Valid-but-worse proposals are still useful exploration; archive them and then restore the best valid file before the next mutation if needed.
11. Do not launch batches of queued runs ahead of time.
12. Do not leave behind detached/background controller scripts. The external watchdog handles relaunching sessions.
13. Do not change the molecule, geometry, basis, CAS, target time, localization scheme, evaluator, or product-formula order.
14. Keep `n_steps` fixed unless exact PF1 no longer clears the chemical-accuracy ground-energy requirement; otherwise focus on synthesis precision.
15. Prefer simple precision policies based on Pauli support size, coefficient magnitude, Pauli label structure, or per-step angle magnitude.
16. If you get 2 failed-accuracy runs in a short span, restore the best valid file and change mutation family rather than repeating the same relaxation pattern.
17. If you go 5 scored iterations without a strictly better valid score, switch to a different precision-policy family instead of making cosmetic equal-score tweaks.
18. Equal-score keeps are not enough; keep pushing for strictly lower valid `t_gate_count`.
19. Keep the code simple.
20. The `--description` text passed to `queued_track_iteration.py` is treated as the per-iteration autoresearch note, so make it specific and concrete about the mutation family, labels, and epsilon changes.
21. Avoid long stretches of one-step or one-label micro-tweaks that only touch step 0 or a single diagonal label for tiny wins; after at most 3 such tweaks, restore the best file and switch to a broader mutation family.
"""


def render_lane_readme(spec: MoleculeSpec) -> str:
    return f"""# {spec.display_name}

Fixed chemistry setup:
- Molecule key: `{spec.key}`
- Active space: `{spec.cas_label}`
- Basis: `{spec.basis}`
- Charge: `{spec.charge}`
- Multiplicity: `{spec.multiplicity}`
- Target time: `{spec.target_time}`
- Orbital transform: `{spec.localization}`
- AO labels: `{", ".join(spec.aolabels) if spec.aolabels else "frontier occupied/virtual split"}`

Only `simple_pf.py` is meant to mutate during the outer loop.
"""
