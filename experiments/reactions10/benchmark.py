"""Authoritative ten-reaction benchmark and public policy-evaluation API.

The reaction registry is copied from the archived Omni campaign driver with
SHA-256 ``09b713ebd784614ec774c73c22c0228b084d82bd9a3ea176589caf06918d965e``.
Only the fixed definitions, arithmetic, and policy evaluation remain here;
legacy sweeps, plotting, and command-line orchestration were intentionally
left behind.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.diff_energy.reactions10_runtime import evaluate as reaction_evaluate  # noqa: E402
from experiments.diff_energy.reactions10_runtime.chemistry import MoleculeSpec  # noqa: E402


ARCHIVED_BENCHMARK_SHA256 = "09b713ebd784614ec774c73c22c0228b084d82bd9a3ea176589caf06918d965e"
FAILURE_SCORE_BASE = 1.0e15
CONTRACTS = ("equal_share", "reaction_aware")


@dataclass(frozen=True)
class ReactionDefinition:
    key: str
    label: str
    species: tuple[MoleculeSpec, ...]
    terms: tuple[tuple[float, str], ...]
    rationale: str


# Dict insertion order is the paper order.  This is the sole registry of
# species geometries, active spaces, and stoichiometry for the experiment.
REACTIONS: dict[str, ReactionDefinition] = {
    "hcn_hnc": ReactionDefinition(
        key="hcn_hnc",
        label="HCN -> HNC",
        terms=((-1.0, "hcn"), (1.0, "hnc")),
        species=(
            MoleculeSpec(
                key="hcn",
                display_name="HCN reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.064000
N 0.000000 0.000000 2.220000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="hnc",
                display_name="HNC reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
N 0.000000 0.000000 1.064000
C 0.000000 0.000000 2.220000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Existing three-atom isomerization baseline with matched sto-3g/CAS(6e,5o) setup."
        ),
    ),
    "hco_hoc_cation": ReactionDefinition(
        key="hco_hoc_cation",
        label="HCO+ -> HOC+",
        terms=((-1.0, "hco_plus"), (1.0, "hoc_plus")),
        species=(
            MoleculeSpec(
                key="hco_plus",
                display_name="HCO+ reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.100000
O 0.000000 0.000000 2.228000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
                charge=1,
            ),
            MoleculeSpec(
                key="hoc_plus",
                display_name="HOC+ reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
O 0.000000 0.000000 0.950000
C 0.000000 0.000000 2.078000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
                charge=1,
            ),
        ),
        rationale=(
            "Linear proton-transfer/isomer pair, isoelectronic with HCN/HNC "
            "and same active-space size."
        ),
    ),
    "hno_noh": ReactionDefinition(
        key="hno_noh",
        label="HNO -> NOH",
        terms=((-1.0, "hno"), (1.0, "noh")),
        species=(
            MoleculeSpec(
                key="hno",
                display_name="HNO reaction PF",
                geometry="""
N 0.000000 0.000000 0.000000
O 1.212000 0.000000 0.000000
H -0.328480 1.010985 0.000000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="noh",
                display_name="NOH reaction PF",
                geometry="""
N 0.000000 0.000000 0.000000
O 1.212000 0.000000 0.000000
H 0.960942 0.936917 0.000000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Small H-shift isomerization with identical atoms and the same "
            "closed-shell electron count."
        ),
    ),
    "h2co_hcoh": ReactionDefinition(
        key="h2co_hcoh",
        label="H2CO -> HCOH",
        terms=((-1.0, "h2co"), (1.0, "hcoh")),
        species=(
            MoleculeSpec(
                key="h2co",
                display_name="H2CO reaction PF",
                geometry="""
C 0.000000 0.000000 0.000000
O 1.210000 0.000000 0.000000
H -0.583257 0.932969 0.000000
H -0.583257 -0.932969 0.000000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="hcoh",
                display_name="HCOH reaction PF",
                geometry="""
C 0.000000 0.000000 0.000000
O 1.320000 0.000000 0.000000
H -0.464880 0.996938 0.000000
H 1.571038 0.936917 0.000000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Same-formula carbonyl/enol-like rearrangement with shared C/O/H active chemistry."
        ),
    ),
    "acetylene_vinylidene": ReactionDefinition(
        key="acetylene_vinylidene",
        label="HCCH -> H2CC",
        terms=((-1.0, "acetylene"), (1.0, "vinylidene")),
        species=(
            MoleculeSpec(
                key="acetylene",
                display_name="Acetylene reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.060000
C 0.000000 0.000000 2.263000
H 0.000000 0.000000 3.323000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="vinylidene",
                display_name="Vinylidene reaction PF",
                geometry="""
C 0.000000 0.000000 0.000000
C 1.310000 0.000000 0.000000
H -0.540000 0.935307 0.000000
H -0.540000 -0.935307 0.000000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Same-formula valence isomerization; the two Hamiltonians share "
            "size and many structural features."
        ),
    ),
    "h2_cl2_hcl": ReactionDefinition(
        key="h2_cl2_hcl",
        label="H2 + Cl2 -> 2 HCl",
        terms=((-1.0, "h2"), (-1.0, "cl2"), (2.0, "hcl")),
        species=(
            MoleculeSpec(
                key="h2",
                display_name="H2 reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
                active_electrons=2,
                active_orbitals=2,
            ),
            MoleculeSpec(
                key="cl2",
                display_name="Cl2 reaction PF",
                geometry="""
Cl 0.000000 0.000000 0.000000
Cl 0.000000 0.000000 1.987000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="hcl",
                display_name="HCl reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
Cl 0.000000 0.000000 1.275000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Closed-shell hydrogen-halogen reaction requested by the user; "
            "H2 uses its natural two-electron active space."
        ),
    ),
    "n2_h2_nh3": ReactionDefinition(
        key="n2_h2_nh3",
        label="N2 + 3 H2 -> 2 NH3",
        terms=((-1.0, "n2"), (-3.0, "h2"), (2.0, "nh3")),
        species=(
            MoleculeSpec(
                key="n2",
                display_name="N2 reaction PF",
                geometry="""
N 0.000000 0.000000 0.000000
N 0.000000 0.000000 1.098000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="h2",
                display_name="H2 reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
                active_electrons=2,
                active_orbitals=2,
            ),
            MoleculeSpec(
                key="nh3",
                display_name="NH3 reaction PF",
                geometry="""
N 0.000000 0.000000 0.000000
H 0.000000 0.937000 0.382000
H 0.812000 -0.469000 0.382000
H -0.812000 -0.469000 0.382000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Closed-shell Haber-Bosch stoichiometry requested by the user, "
            "evaluated as an active-space reaction-energy screen."
        ),
    ),
    "co_h2o_co2_h2": ReactionDefinition(
        key="co_h2o_co2_h2",
        label="CO + H2O -> CO2 + H2",
        terms=((-1.0, "co"), (-1.0, "h2o"), (1.0, "co2"), (1.0, "h2")),
        species=(
            MoleculeSpec(
                key="co",
                display_name="CO reaction PF",
                geometry="""
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.128000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="h2o",
                display_name="H2O reaction PF",
                geometry="""
O 0.000000 0.000000 0.000000
H 0.756950 0.000000 0.585882
H -0.756950 0.000000 0.585882
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="co2",
                display_name="CO2 reaction PF",
                geometry="""
O 0.000000 0.000000 -1.160000
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.160000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="h2",
                display_name="H2 reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
                active_electrons=2,
                active_orbitals=2,
            ),
        ),
        rationale="Closed-shell water-gas-shift stoichiometry requested by the user.",
    ),
    "h2_f2_hf": ReactionDefinition(
        key="h2_f2_hf",
        label="H2 + F2 -> 2 HF",
        terms=((-1.0, "h2"), (-1.0, "f2"), (2.0, "hf")),
        species=(
            MoleculeSpec(
                key="h2",
                display_name="H2 reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
                active_electrons=2,
                active_orbitals=2,
            ),
            MoleculeSpec(
                key="f2",
                display_name="F2 reaction PF",
                geometry="""
F 0.000000 0.000000 0.000000
F 0.000000 0.000000 1.412000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="hf",
                display_name="HF reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
F 0.000000 0.000000 0.917000
""".strip(),
                active_electrons=8,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Closed-shell hydrogen-halogen analogue selected to keep the "
            "screen small and comparable to H2 + Cl2 -> 2 HCl."
        ),
    ),
    "co_h2_ch4_h2o": ReactionDefinition(
        key="co_h2_ch4_h2o",
        label="CO + 3 H2 -> CH4 + H2O",
        terms=((-1.0, "co"), (-3.0, "h2"), (1.0, "ch4"), (1.0, "h2o")),
        species=(
            MoleculeSpec(
                key="co",
                display_name="CO reaction PF",
                geometry="""
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.128000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="h2",
                display_name="H2 reaction PF",
                geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
                active_electrons=2,
                active_orbitals=2,
            ),
            MoleculeSpec(
                key="ch4",
                display_name="CH4 reaction PF",
                geometry="""
C 0.000000 0.000000 0.000000
H 0.629000 0.629000 0.629000
H -0.629000 -0.629000 0.629000
H -0.629000 0.629000 -0.629000
H 0.629000 -0.629000 -0.629000
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
            MoleculeSpec(
                key="h2o",
                display_name="H2O reaction PF",
                geometry="""
O 0.000000 0.000000 0.000000
H 0.756950 0.000000 0.585882
H -0.756950 0.000000 0.585882
""".strip(),
                active_electrons=6,
                active_orbitals=5,
            ),
        ),
        rationale=(
            "Closed-shell CO methanation stoichiometry selected as a second "
            "balanced small-molecule reaction."
        ),
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policy(path: Path, policy_hash: str) -> ModuleType:
    module_name = f"hsopt_reaction_policy_{policy_hash[:16]}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load policy {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _reaction(reaction_key: str) -> ReactionDefinition:
    try:
        return REACTIONS[reaction_key]
    except KeyError as exc:
        choices = ", ".join(REACTIONS)
        raise KeyError(f"unknown reaction {reaction_key!r}; choose one of: {choices}") from exc


def evaluate_policy(
    policy_path: Path | str,
    reaction_key: str,
    contract: str,
    cache_root: Path | str,
    iteration: int,
    lane_id: str,
    campaign_id: str = "",
) -> dict[str, object]:
    """Evaluate one policy and return the stable, JSON-safe lane result."""

    started = time.perf_counter()
    candidate = _reaction(str(reaction_key))
    if contract not in CONTRACTS:
        raise ValueError(f"contract must be one of {CONTRACTS}, got {contract!r}")
    if int(iteration) < 0:
        raise ValueError(f"iteration must be nonnegative, got {iteration}")
    if not str(lane_id).strip():
        raise ValueError("lane_id must be nonempty")

    resolved_policy = Path(policy_path).expanduser().resolve()
    if not resolved_policy.is_file():
        raise FileNotFoundError(f"policy does not exist: {resolved_policy}")
    resolved_cache = Path(cache_root).expanduser().resolve()
    policy_hash = _sha256_file(resolved_policy)
    policy = _load_policy(resolved_policy, policy_hash)
    species_keys = tuple(spec.key for spec in candidate.species)
    if len(set(species_keys)) != len(species_keys):
        raise ValueError(f"reaction {candidate.key} repeats a species definition")
    term_species = {species for _coefficient, species in candidate.terms}
    if term_species != set(species_keys):
        raise ValueError(
            f"reaction {candidate.key} terms/species mismatch: "
            f"terms={sorted(term_species)}, species={sorted(species_keys)}"
        )
    reaction_evaluate.validate_policy_module(policy, species_keys)
    backend_name, xp = reaction_evaluate.array_backend()

    species_results: dict[str, dict[str, object]] = {}
    for molecule_spec in candidate.species:
        species_key = molecule_spec.key
        policy.set_active_species(species_key)
        config = policy.get_method_config()
        seed_material = (
            f"reaction-autoresearch-schedule-v1\0{campaign_id}\0{lane_id}\0{species_key}"
        ).encode()
        evaluator_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        species_results[species_key] = reaction_evaluate.evaluate_species(
            molecule_spec,
            policy,
            config,
            resolved_cache,
            evaluator_seed=evaluator_seed,
            backend_name=backend_name,
            xp=xp,
        )

    target = reaction_evaluate.REACTION_TARGET_HARTREE
    stoichiometric_abs_sum = sum(
        abs(float(coefficient)) for coefficient, _species in candidate.terms
    )
    if stoichiometric_abs_sum <= 0.0:
        raise ValueError(f"reaction {candidate.key} has zero stoichiometric weight")
    equal_share_limit = target / stoichiometric_abs_sum

    exact_delta = sum(
        float(coefficient) * float(species_results[species]["exact_energy_hartree"])
        for coefficient, species in candidate.terms
    )
    pf_delta = sum(
        float(coefficient) * float(species_results[species]["pf_energy_estimate_hartree"])
        for coefficient, species in candidate.terms
    )
    signed_reaction_error = sum(
        float(coefficient) * float(species_results[species]["signed_energy_error_hartree"])
        for coefficient, species in candidate.terms
    )
    arithmetic_error = (pf_delta - exact_delta) - signed_reaction_error
    if not math.isclose(arithmetic_error, 0.0, abs_tol=1.0e-12, rel_tol=0.0):
        raise ArithmeticError(
            f"reaction error arithmetic is inconsistent by {arithmetic_error:.3e} Ha"
        )
    triangle_bound = sum(
        abs(float(coefficient))
        * abs(float(species_results[species]["signed_energy_error_hartree"]))
        for coefficient, species in candidate.terms
    )
    triangle_tolerance = 1.0e-12
    triangle_holds = abs(signed_reaction_error) <= triangle_bound + triangle_tolerance
    if not triangle_holds:
        raise ArithmeticError("reaction error violated its triangle-inequality bound")

    max_species_error = max(
        float(species_results[species]["absolute_energy_error_hartree"]) for species in species_keys
    )
    passes_equal_share = all(
        float(species_results[species]["absolute_energy_error_hartree"]) <= equal_share_limit
        for species in species_keys
    )
    passes_reaction = abs(signed_reaction_error) <= target
    passes_contract = passes_equal_share if contract == "equal_share" else passes_reaction
    total_t_gate_count = sum(
        int(species_results[species]["t_gate_count"]) for species in species_keys
    )
    total_scheduled_rotation_count = sum(
        int(species_results[species]["scheduled_rotation_count"]) for species in species_keys
    )
    violation_ratio = (
        max_species_error / equal_share_limit
        if contract == "equal_share"
        else abs(signed_reaction_error) / target
    )
    score = (
        float(total_t_gate_count)
        if passes_contract
        else FAILURE_SCORE_BASE * (1.0 + max(1.0, violation_ratio)) ** 2 + float(total_t_gate_count)
    )

    return {
        "lane_id": str(lane_id),
        "reaction_key": candidate.key,
        "reaction": candidate.label,
        "rationale": candidate.rationale,
        "reaction_terms": [
            {"coefficient": float(coefficient), "species": species}
            for coefficient, species in candidate.terms
        ],
        "species_order": list(species_keys),
        "contract": contract,
        "iteration": int(iteration),
        "status": "ok",
        "policy_path": str(resolved_policy),
        "policy_sha256": policy_hash,
        "evaluation_mode": "compiled_product_formula_energy_estimation",
        "ground_phase_matching": "fixed_ground_state_phase_estimator",
        "compilation_backend": "pygridsynth",
        "array_backend": backend_name,
        "reaction_target_hartree": target,
        "reaction_target_mha": 1000.0 * target,
        "stoichiometric_abs_coefficient_sum": stoichiometric_abs_sum,
        "equal_share_error_limit_hartree": equal_share_limit,
        "equal_share_error_limit_mha": 1000.0 * equal_share_limit,
        "exact_reaction_energy_hartree": exact_delta,
        "pf_reaction_energy_hartree": pf_delta,
        "signed_reaction_error_hartree": signed_reaction_error,
        "signed_reaction_error_mha": 1000.0 * signed_reaction_error,
        "absolute_reaction_error_hartree": abs(signed_reaction_error),
        "absolute_reaction_error_mha": 1000.0 * abs(signed_reaction_error),
        "triangle_bound_hartree": triangle_bound,
        "triangle_bound_mha": 1000.0 * triangle_bound,
        "triangle_inequality_holds": triangle_holds,
        "triangle_inequality_slack_hartree": triangle_bound - abs(signed_reaction_error),
        "max_species_absolute_error_hartree": max_species_error,
        "max_species_absolute_error_mha": 1000.0 * max_species_error,
        "passes_equal_share_constraint": passes_equal_share,
        "passes_reaction_constraint": passes_reaction,
        "passes_contract": passes_contract,
        "total_t_gate_count": total_t_gate_count,
        "total_scheduled_rotation_count": total_scheduled_rotation_count,
        "t_count_aggregation": "sum_each_unique_species_once",
        "score": score,
        "species": species_results,
        "cache_root": str(resolved_cache),
        "source_provenance": {
            "archived_benchmark_sha256": ARCHIVED_BENCHMARK_SHA256,
            "archived_evaluator_sha256": (reaction_evaluate.ARCHIVED_EVALUATOR_SHA256),
        },
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


__all__ = [
    "ARCHIVED_BENCHMARK_SHA256",
    "CONTRACTS",
    "REACTIONS",
    "ReactionDefinition",
    "evaluate_policy",
]
