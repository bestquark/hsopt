#!/usr/bin/env python3
"""Validate and export one complete reactions10 paper campaign.

This is deliberately an offline, fail-closed exporter.  It replays every
accept/reject decision from immutable policy snapshots and evaluator results;
it never trusts the campaign's rolling summary files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-runner-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX


PAPER_PROPOSAL_ITERATIONS = 50
PAPER_EVALUATION_ITERATIONS = tuple(range(PAPER_PROPOSAL_ITERATIONS + 1))
PAPER_LANE_COUNT = 20
PAPER_REACTION_COUNT = 10
PAPER_CANDIDATE_COUNT = len(PAPER_EVALUATION_ITERATIONS) * PAPER_LANE_COUNT
PAPER_MUTATION_COUNT = PAPER_PROPOSAL_ITERATIONS * PAPER_LANE_COUNT
PAPER_TARGET_HARTREE = 1.0 / 627.5094740631
PAPER_MODEL = "gpt-5.5"
PAPER_REASONING_EFFORT = "high"
PAPER_CODEX_CLI_VERSION = "0.144.0"
PAPER_CODEX_PACKAGE = "@openai/codex@0.144.0"
CAMPAIGN_ID_DOMAIN = b"hsopt-reactions10-campaign-v1\0"
EXEC_ACK_PROTOCOL = "codex_jsonl_lifecycle_v1"
ARCHIVED_REACTION_BENCHMARK_SHA256 = (
    "09b713ebd784614ec774c73c22c0228b084d82bd9a3ea176589caf06918d965e"
)
ARCHIVED_REACTION_EVALUATOR_SHA256 = (
    "b81d8e936f71c0cbab76ba8f0b134b4ecbf2c610d95aaca04136b7aeb5309425"
)
FAILURE_SCORE_BASE = 1.0e15
RUNTIME_SNAPSHOT_DIRS = frozenset({".venv", ".pytest_cache", ".ruff_cache"})
SUBMISSION_ID_DOMAIN = b"hsopt-reactions10-submission-v1\0"
HAMILTONIAN_THRESHOLDS = {
    "hermiticity_error": 1.0e-12,
    "pauli_dense_max_abs_residual_hartree": 1.0e-10,
    "pauli_dense_relative_frobenius_residual": 1.0e-10,
    "max_spectrum_delta_hartree": 1.0e-10,
    "cached_ground_vector_residual": 1.0e-8,
    "evaluator_ground_state_outside_sector_norm": 1.0e-10,
    "evaluator_ground_state_residual": 1.0e-8,
}
CAMPAIGN_TOP_FILES = frozenset({"manifest.json", "state.json"})
CAMPAIGN_TOP_DIRECTORIES = frozenset(
    {
        ".locks",
        "baselines",
        "iterations",
        "lanes",
        "logs",
        "problem_cache",
        "receipts",
        "runner",
        "source",
        "transactions",
    }
)
ITERATION_FILES = frozenset({"decisions.json", "validation.json"})
ITERATION_DIRECTORIES = frozenset({"batches", "candidates", "mutations", "results"})
SEED_ABS_TOL = 1.0e-10
SEED_REL_TOL = 1.0e-9
PAPER_INPUT_HASHES = {
    "seed_manifest": "7cd26b6d25117cbebcec93cd9b800c4ab473925adee895d147ddc3e135f45094",
    "seed_validation": "e2bee43bbf79fd5cfa8877c0be166a115443528ca39f2c53ec596b6239555fcc",
    "problem_cache_checksums": ("cf521d0a98c7b9cb3478ef28349200f4ab98b171a79a915edca6459349f9c22f"),
}
CONTRACTS = ("equal_share", "reaction_aware")
REACTION_KEYS = (
    "hcn_hnc",
    "hco_hoc_cation",
    "hno_noh",
    "h2co_hcoh",
    "acetylene_vinylidene",
    "h2_cl2_hcl",
    "n2_h2_nh3",
    "co_h2o_co2_h2",
    "h2_f2_hf",
    "co_h2_ch4_h2o",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ITERATION_DIR_RE = re.compile(r"iter_(\d{4})\Z")
BATCH_ID_RE = re.compile(r"iter(\d{4})_batch(\d+)\Z")
DERIVED_ABS_TOL = 2.0e-12
DERIVED_REL_TOL = 2.0e-12


class ExportValidationError(RuntimeError):
    """Raised when a campaign is not a valid paper campaign."""


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    reaction_key: str
    reaction: str
    contract: str
    species: tuple[str, ...]
    terms: tuple[tuple[str, float], ...]
    seed_policy_sha256: str
    seed_result_expectation: dict[str, Any] | None = None
    campaign_id: str | None = None


@dataclass
class Candidate:
    lane: LaneSpec
    iteration: int
    policy_sha256: str
    policy_path: Path
    result: dict[str, Any]
    total_t_gate_count: int
    signed_reaction_error_hartree: float
    absolute_reaction_error_hartree: float
    triangle_bound_hartree: float
    equal_share_error_limit_hartree: float
    max_species_absolute_error_hartree: float
    passes_equal_share_constraint: bool
    passes_reaction_constraint: bool
    passes_contract: bool
    accepted: bool = False


@dataclass(frozen=True)
class DirectoryEntries:
    files: frozenset[str]
    directories: frozenset[str]


def _duplicate_key_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ExportValidationError(f"missing or non-regular JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_key_guard)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExportValidationError(f"cannot read strict JSON from {path}: {exc}") from exc


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_directory(directory: Path, context: str) -> DirectoryEntries:
    """Classify direct children without accepting symlinks or special entries."""
    if directory.is_symlink() or not directory.is_dir():
        raise ExportValidationError(f"{context} must be a real directory: {directory}")
    files: set[str] = set()
    directories: set[str] = set()
    symlinks: list[str] = []
    special: list[str] = []
    for child in directory.iterdir():
        if child.is_symlink():
            symlinks.append(child.name)
        elif child.is_file():
            files.add(child.name)
        elif child.is_dir():
            directories.add(child.name)
        else:
            special.append(child.name)
    if symlinks:
        raise ExportValidationError(f"{context} contains symlinks: {sorted(symlinks)}")
    if special:
        raise ExportValidationError(f"{context} contains special entries: {sorted(special)}")
    return DirectoryEntries(frozenset(files), frozenset(directories))


def require_exact_directory(
    directory: Path,
    *,
    context: str,
    files: Iterable[str] = (),
    directories: Iterable[str] = (),
    optional_files: Iterable[str] = (),
    optional_directories: Iterable[str] = (),
) -> DirectoryEntries:
    """Require exact direct-child types, including narrowly named optional files."""
    entries = classify_directory(directory, context)
    required_files = frozenset(files)
    required_directories = frozenset(directories)
    allowed_files = required_files | frozenset(optional_files)
    allowed_directories = required_directories | frozenset(optional_directories)
    missing_files = required_files - entries.files
    missing_directories = required_directories - entries.directories
    extra_files = entries.files - allowed_files
    extra_directories = entries.directories - allowed_directories
    if missing_files or missing_directories or extra_files or extra_directories:
        raise ExportValidationError(
            f"{context} has the wrong exact structure: "
            f"missing_files={sorted(missing_files)}, "
            f"missing_directories={sorted(missing_directories)}, "
            f"extra_files={sorted(extra_files)}, "
            f"extra_directories={sorted(extra_directories)}"
        )
    return entries


def require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExportValidationError(f"{context} must be a JSON object")
    return value


def require_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExportValidationError(f"{context} must be a JSON array")
    return value


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportValidationError(f"{context} must be a nonempty string")
    return value


def require_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ExportValidationError(f"{context} must be a JSON boolean")
    return value


def require_int(value: object, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ExportValidationError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ExportValidationError(f"{context} must be >= {minimum}, got {value}")
    return value


def require_optional_int(value: object, context: str) -> int | None:
    return None if value is None else require_int(value, context)


def require_finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportValidationError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ExportValidationError(f"{context} must be finite, got {value!r}")
    return number


def require_sha256(value: object, context: str) -> str:
    digest = require_string(value, context)
    if SHA256_RE.fullmatch(digest) is None:
        raise ExportValidationError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def validate_campaign_identity(manifest: Mapping[str, Any]) -> str:
    nonce = require_sha256(manifest.get("preparation_nonce"), "manifest.json.preparation_nonce")
    campaign_id = require_sha256(manifest.get("campaign_id"), "manifest.json.campaign_id")
    expected = hashlib.sha256(CAMPAIGN_ID_DOMAIN + bytes.fromhex(nonce)).hexdigest()
    if campaign_id != expected:
        raise ExportValidationError(
            "manifest.json.campaign_id does not derive from preparation_nonce"
        )
    return campaign_id


def require_exact_target(value: object, context: str) -> None:
    actual = require_finite(value, context)
    if actual != PAPER_TARGET_HARTREE:
        raise ExportValidationError(
            f"{context} changed: expected exact 1/627.5094740631 "
            f"({PAPER_TARGET_HARTREE!r}), got {actual!r}"
        )


def require_close(actual: object, expected: float, context: str) -> float:
    number = require_finite(actual, context)
    if not math.isclose(
        number,
        expected,
        rel_tol=DERIVED_REL_TOL,
        abs_tol=DERIVED_ABS_TOL,
    ):
        raise ExportValidationError(f"{context}={number!r}, recomputed value={expected!r}")
    return number


def require_optional_close(
    payload: Mapping[str, Any], key: str, expected: float, context: str
) -> None:
    if key in payload:
        require_close(payload[key], expected, f"{context}.{key}")


def normalized_terms(lane: Mapping[str, Any], context: str) -> tuple[tuple[str, float], ...]:
    raw_terms = require_list(lane.get("terms"), f"{context}.terms")
    if not raw_terms:
        raise ExportValidationError(f"{context}.terms must not be empty")
    terms: list[tuple[str, float]] = []
    for index, raw in enumerate(raw_terms):
        term = require_mapping(raw, f"{context}.terms[{index}]")
        species = require_string(term.get("species"), f"{context}.terms[{index}].species")
        coefficient = require_finite(
            term.get("coefficient"), f"{context}.terms[{index}].coefficient"
        )
        if coefficient == 0.0:
            raise ExportValidationError(f"{context}.terms[{index}].coefficient must be nonzero")
        terms.append((species, coefficient))
    term_species = [species for species, _coefficient in terms]
    if len(term_species) != len(set(term_species)):
        raise ExportValidationError(f"{context}.terms contains duplicate species")
    return tuple(terms)


def normalized_seed_expectation(
    value: object,
    *,
    species: Sequence[str],
    context: str,
) -> dict[str, Any]:
    payload = require_mapping(value, context)
    numeric_fields = (
        "signed_reaction_error_hartree",
        "absolute_reaction_error_hartree",
        "triangle_bound_hartree",
        "max_species_absolute_error_hartree",
        "equal_share_error_limit_hartree",
    )
    normalized: dict[str, Any] = {
        field: require_finite(payload.get(field), f"{context}.{field}") for field in numeric_fields
    }
    normalized["formula_order"] = require_int(
        payload.get("formula_order"), f"{context}.formula_order", minimum=1
    )
    if normalized["formula_order"] not in (1, 2):
        raise ExportValidationError(f"{context}.formula_order must be 1 or 2")
    normalized["n_steps"] = require_int(payload.get("n_steps"), f"{context}.n_steps", minimum=1)
    normalized["total_t_gate_count"] = require_int(
        payload.get("total_t_gate_count"), f"{context}.total_t_gate_count", minimum=0
    )
    raw_species = require_mapping(payload.get("species"), f"{context}.species")
    if set(raw_species) != set(species):
        raise ExportValidationError(f"{context}.species does not match the lane")
    normalized["species"] = {}
    for name in species:
        row = require_mapping(raw_species[name], f"{context}.species.{name}")
        normalized["species"][name] = {
            "signed_energy_error_hartree": require_finite(
                row.get("signed_energy_error_hartree"),
                f"{context}.species.{name}.signed_energy_error_hartree",
            ),
            "t_gate_count": require_int(
                row.get("t_gate_count"),
                f"{context}.species.{name}.t_gate_count",
                minimum=0,
            ),
        }
    if (
        sum(row["t_gate_count"] for row in normalized["species"].values())
        != normalized["total_t_gate_count"]
    ):
        raise ExportValidationError(f"{context} species T counts do not sum to the seed total")
    return normalized


def validate_manifest(manifest: object) -> dict[str, LaneSpec]:
    payload = require_mapping(manifest, "manifest.json")
    if payload.get("experiment") != "reactions10_autoresearch":
        raise ExportValidationError(
            "manifest.json.experiment must be 'reactions10_autoresearch' for a paper export"
        )
    if require_int(payload.get("schema_version"), "manifest.json.schema_version") != 1:
        raise ExportValidationError("manifest.json.schema_version must be exactly 1")
    execution = require_mapping(payload.get("execution"), "manifest.json.execution")
    if set(execution) != {"cluster_root", "cluster_python", "batch_size"}:
        raise ExportValidationError("manifest.json.execution has an unexpected schema")
    _require_absolute_posix_path(
        execution.get("cluster_root"), "manifest.json.execution.cluster_root"
    )
    _require_absolute_posix_path(
        execution.get("cluster_python"), "manifest.json.execution.cluster_python"
    )
    batch_size = require_int(
        execution.get("batch_size"), "manifest.json.execution.batch_size", minimum=1
    )
    if batch_size > PAPER_LANE_COUNT:
        raise ExportValidationError("manifest.json.execution.batch_size exceeds lane count")
    if (
        require_int(
            payload.get("proposal_iterations"), "manifest.json.proposal_iterations", minimum=0
        )
        != PAPER_PROPOSAL_ITERATIONS
    ):
        raise ExportValidationError("manifest proposal_iterations must be exactly 50")
    expected_evaluations = list(PAPER_EVALUATION_ITERATIONS)
    raw_evaluations = payload.get("evaluation_iterations")
    if isinstance(raw_evaluations, list):
        valid_evaluations = raw_evaluations == expected_evaluations
    elif isinstance(raw_evaluations, dict):
        valid_evaluations = raw_evaluations == {
            "first": expected_evaluations[0],
            "last": expected_evaluations[-1],
        }
    else:
        valid_evaluations = False
    if not valid_evaluations:
        raise ExportValidationError(
            "manifest evaluation_iterations must describe exactly the inclusive range 0..50"
        )
    if require_int(payload.get("lane_count"), "manifest.json.lane_count") != PAPER_LANE_COUNT:
        raise ExportValidationError("manifest lane_count must be exactly 20")
    if (
        require_int(payload.get("reaction_count"), "manifest.json.reaction_count")
        != PAPER_REACTION_COUNT
    ):
        raise ExportValidationError("manifest reaction_count must be exactly 10")
    campaign_id = validate_campaign_identity(payload)

    target_key = next(
        (key for key in ("chemical_accuracy_hartree", "reaction_target_hartree") if key in payload),
        None,
    )
    if target_key is None:
        raise ExportValidationError("manifest is missing the exact chemical-accuracy target")
    require_exact_target(payload[target_key], f"manifest.json.{target_key}")

    raw_lanes = require_list(payload.get("lanes"), "manifest.json.lanes")
    if len(raw_lanes) != PAPER_LANE_COUNT:
        raise ExportValidationError(
            f"manifest must contain exactly {PAPER_LANE_COUNT} lanes, got {len(raw_lanes)}"
        )
    lanes: dict[str, LaneSpec] = {}
    pair_set: set[tuple[str, str]] = set()
    by_reaction: dict[str, list[LaneSpec]] = {}
    for index, raw in enumerate(raw_lanes):
        context = f"manifest.json.lanes[{index}]"
        lane = require_mapping(raw, context)
        lane_id = require_string(lane.get("lane_id"), f"{context}.lane_id")
        reaction_key = require_string(lane.get("reaction_key"), f"{context}.reaction_key")
        reaction = require_string(lane.get("reaction"), f"{context}.reaction")
        contract = require_string(lane.get("contract"), f"{context}.contract")
        if reaction_key not in REACTION_KEYS:
            raise ExportValidationError(f"{context}.reaction_key is not in the fixed paper set")
        if contract not in CONTRACTS:
            raise ExportValidationError(f"{context}.contract must be one of {CONTRACTS}")
        expected_lane_id = f"{reaction_key}__{contract}"
        if lane_id != expected_lane_id:
            raise ExportValidationError(
                f"{context}.lane_id must be {expected_lane_id!r}, got {lane_id!r}"
            )
        if lane_id in lanes or (reaction_key, contract) in pair_set:
            raise ExportValidationError(f"duplicate lane or reaction/contract pair: {lane_id}")
        raw_species = require_list(lane.get("species"), f"{context}.species")
        species = tuple(require_string(item, f"{context}.species") for item in raw_species)
        if not species or len(species) != len(set(species)):
            raise ExportValidationError(f"{context}.species must be a nonempty unique list")
        terms = normalized_terms(lane, context)
        if {item for item, _coefficient in terms} != set(species):
            raise ExportValidationError(f"{context}.terms and species must name the same set")
        require_exact_target(
            lane.get("reaction_target_hartree"), f"{context}.reaction_target_hartree"
        )
        require_optional_close(
            lane,
            "reaction_target_mha",
            1000.0 * PAPER_TARGET_HARTREE,
            context,
        )
        seed_policy_sha256 = require_sha256(
            lane.get("seed_policy_sha256"), f"{context}.seed_policy_sha256"
        )
        seed_expectation = normalized_seed_expectation(
            lane.get("seed_result_expectation"),
            species=species,
            context=f"{context}.seed_result_expectation",
        )
        if (
            require_int(
                lane.get("seed_total_t_gate_count"),
                f"{context}.seed_total_t_gate_count",
                minimum=0,
            )
            != seed_expectation["total_t_gate_count"]
        ):
            raise ExportValidationError(f"{context} seed T-count fields disagree")
        if (
            require_bool(lane.get("seed_validation_passes"), f"{context}.seed_validation_passes")
            is not True
        ):
            raise ExportValidationError(f"{context} preserved seed validation did not pass")
        spec = LaneSpec(
            lane_id,
            reaction_key,
            reaction,
            contract,
            species,
            terms,
            seed_policy_sha256,
            seed_expectation,
            campaign_id,
        )
        lanes[lane_id] = spec
        pair_set.add((reaction_key, contract))
        by_reaction.setdefault(reaction_key, []).append(spec)

    if set(by_reaction) != set(REACTION_KEYS):
        raise ExportValidationError("manifest reaction set differs from the fixed ten reactions")
    for reaction_key in REACTION_KEYS:
        specs = by_reaction[reaction_key]
        if {spec.contract for spec in specs} != set(CONTRACTS):
            raise ExportValidationError(f"{reaction_key} must have exactly both contract lanes")
        first, second = specs
        if (
            first.reaction != second.reaction
            or first.species != second.species
            or dict(first.terms) != dict(second.terms)
        ):
            raise ExportValidationError(
                f"{reaction_key} chemistry differs between equal-share and reaction-aware lanes"
            )
    return lanes


def _provenance_containers(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    containers: list[tuple[str, Mapping[str, Any]]] = [("manifest", manifest)]
    for key in ("autoresearch_provenance", "provenance", "runner_provenance", "autoresearch"):
        value = manifest.get(key)
        if isinstance(value, dict):
            containers.append((key, value))
    return containers


def _collect_setting(
    manifest: Mapping[str, Any], aliases: Sequence[str], expected: str, label: str
) -> str:
    found: list[tuple[str, str]] = []
    for container_name, container in _provenance_containers(manifest):
        for key in aliases:
            if key in container:
                found.append((f"{container_name}.{key}", require_string(container[key], label)))
    if not found:
        raise ExportValidationError(f"manifest does not record required paper provenance {label}")
    drift = {path: value for path, value in found if value != expected}
    if drift:
        raise ExportValidationError(
            f"paper provenance {label} changed: {drift}; expected {expected!r}"
        )
    return expected


def _collect_hash_provenance(value: object) -> dict[str, str]:
    found: dict[str, str] = {}

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                walk(node[key], (*path, str(key)))
            return
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*path, str(index)))
            return
        lowered = [item.lower() for item in path]
        in_hash_domain = bool(lowered) and any(
            token in lowered[0] for token in ("runner", "input", "source")
        )
        explicitly_a_hash = any(token in lowered[-1] for token in ("sha256", "hash"))
        looks_like_a_hash = isinstance(node, str) and SHA256_RE.fullmatch(node) is not None
        if not in_hash_domain or not (explicitly_a_hash or looks_like_a_hash):
            return
        label = ".".join(path)
        found[label] = require_sha256(node, f"manifest hash provenance {label}")

    walk(value, ())
    return found


def _safe_relative_path(value: object, context: str) -> Path:
    text = require_string(value, context)
    path = Path(text)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in text
        or "\x00" in text
        or path.as_posix() != text
        or any(part in {"", "."} for part in text.split("/"))
    ):
        raise ExportValidationError(f"{context} must be a safe relative path")
    return path


def _require_absolute_posix_path(value: object, context: str, *, allow_root: bool = False) -> str:
    text = require_string(value, context)
    if any(character in text for character in ("\n", "\r", "\x00")):
        raise ExportValidationError(f"{context} contains newline or NUL")
    path = PurePosixPath(text)
    components = text.split("/")[1:]
    if (
        not path.is_absolute()
        or any(component in {".", ".."} for component in components)
        or (text != "/" and any(not component for component in components))
        or path.as_posix() != text
        or (not allow_root and path == PurePosixPath("/"))
    ):
        raise ExportValidationError(f"{context} must be a canonical absolute POSIX path")
    return text


def validate_campaign_top_level(campaign_root: Path) -> None:
    """Require the complete runner-owned campaign boundary and no top-level noise."""
    require_exact_directory(
        campaign_root,
        context="campaign root",
        files=CAMPAIGN_TOP_FILES,
        directories=CAMPAIGN_TOP_DIRECTORIES,
    )
    require_exact_directory(
        campaign_root / "source",
        context="campaign source root",
        directories={"repo"},
    )
    require_exact_directory(
        campaign_root / "baselines",
        context="campaign baselines root",
        files={"hamiltonian_verification.json"},
    )


def validate_lane_structure(campaign_root: Path, lanes: Mapping[str, LaneSpec]) -> None:
    lanes_root = campaign_root / "lanes"
    require_exact_directory(
        lanes_root,
        context="campaign lanes root",
        directories=set(lanes),
    )
    for lane_id, lane in lanes.items():
        lane_root = lanes_root / lane_id
        require_exact_directory(
            lane_root,
            context=f"lane {lane_id} root",
            files={
                "initial_simple_pf.py",
                "last_result.json",
                "metadata.json",
                "simple_pf.py",
            },
            directories={"best"},
        )
        require_exact_directory(
            lane_root / "best",
            context=f"lane {lane_id} best root",
            files={"result.json", "simple_pf.py"},
        )
        if sha256_file(lane_root / "initial_simple_pf.py") != lane.seed_policy_sha256:
            raise ExportValidationError(
                f"lane {lane_id} initial policy differs from its fixed seed"
            )
        metadata = require_mapping(
            read_json(lane_root / "metadata.json"), f"lane {lane_id} metadata"
        )
        metadata_identity = {
            "lane_id": lane.lane_id,
            "reaction_key": lane.reaction_key,
            "reaction": lane.reaction,
            "contract": lane.contract,
            "species": list(lane.species),
            "terms": [
                {"species": species, "coefficient": coefficient}
                for species, coefficient in lane.terms
            ],
            "seed_policy_sha256": lane.seed_policy_sha256,
            "seed_result_expectation": lane.seed_result_expectation,
        }
        drift = {
            key: {"actual": metadata.get(key), "expected": expected}
            for key, expected in metadata_identity.items()
            if metadata.get(key) != expected
        }
        if drift:
            raise ExportValidationError(f"lane {lane_id} metadata differs from manifest: {drift}")


def validate_exact_tree_files(base: Path, expected_files: Iterable[str], context: str) -> None:
    """Require an exact recursive regular-file tree with no empty/unrecorded directories."""
    if base.is_symlink() or not base.is_dir():
        raise ExportValidationError(f"{context} must be a real directory: {base}")
    expected: set[str] = set()
    expected_directories: set[str] = set()
    for raw in expected_files:
        relative = _safe_relative_path(raw, f"{context} expected file")
        relative_name = relative.as_posix()
        if relative_name in expected:
            raise ExportValidationError(f"{context} repeats expected file {relative_name}")
        expected.add(relative_name)
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    symlinks: list[str] = []
    special: list[str] = []
    for path in base.rglob("*"):
        relative_name = path.relative_to(base).as_posix()
        if path.is_symlink():
            symlinks.append(relative_name)
        elif path.is_file():
            actual_files.add(relative_name)
        elif path.is_dir():
            actual_directories.add(relative_name)
        else:
            special.append(relative_name)
    if symlinks:
        raise ExportValidationError(f"{context} contains symlinks: {sorted(symlinks)}")
    if special:
        raise ExportValidationError(f"{context} contains special entries: {sorted(special)}")
    if actual_files != expected or actual_directories != expected_directories:
        raise ExportValidationError(
            f"{context} differs from its recorded file tree: "
            f"missing_files={sorted(expected - actual_files)}, "
            f"extra_files={sorted(actual_files - expected)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}, "
            f"extra_directories={sorted(actual_directories - expected_directories)}"
        )


def _validate_file_record(path: Path, record: object, context: str) -> None:
    payload = require_mapping(record, context)
    expected_hash = require_sha256(payload.get("sha256"), f"{context}.sha256")
    expected_size = require_int(payload.get("size_bytes"), f"{context}.size_bytes", minimum=0)
    if not path.is_file() or path.is_symlink():
        raise ExportValidationError(f"{context} points to a missing/non-regular snapshot: {path}")
    if path.stat().st_size != expected_size:
        raise ExportValidationError(f"{context} snapshot size changed: {path}")
    if sha256_file(path) != expected_hash:
        raise ExportValidationError(f"{context} snapshot SHA-256 changed: {path}")


def validate_snapshot_tree(base: Path, records: object, context: str) -> None:
    """Require an exact regular snapshot tree, pruning only real mutable runtime dirs."""
    declared_payload = require_mapping(records, context)
    if not declared_payload:
        raise ExportValidationError(f"{context} must not be empty")
    if not base.is_dir() or base.is_symlink():
        raise ExportValidationError(
            f"{context} snapshot root is missing, non-directory, or a symlink"
        )

    declared: set[str] = set()
    declared_directories: set[str] = set()
    for relative_text, record in declared_payload.items():
        relative = _safe_relative_path(relative_text, f"{context} key")
        if any(part in RUNTIME_SNAPSHOT_DIRS for part in relative.parts):
            raise ExportValidationError(f"{context} declares excluded runtime path {relative}")
        if "__pycache__" in relative.parts:
            raise ExportValidationError(f"{context} declares forbidden __pycache__ path {relative}")
        relative_name = relative.as_posix()
        if relative_name in declared:
            raise ExportValidationError(f"{context} has duplicate path {relative_name}")
        declared.add(relative_name)
        parent = relative.parent
        while parent != Path("."):
            declared_directories.add(parent.as_posix())
            parent = parent.parent
        _validate_file_record(base / relative, record, f"{context}.{relative_name}")

    actual: set[str] = set()
    actual_directories: set[str] = set()
    symlinks: list[str] = []
    special: list[str] = []

    def walk(directory: Path, relative_root: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = relative_root / entry.name
                relative_name = relative.as_posix()
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except FileNotFoundError as exc:
                    raise ExportValidationError(
                        f"{context} entry disappeared during validation: {relative_name}"
                    ) from exc
                if stat.S_ISLNK(mode):
                    symlinks.append(relative_name)
                elif stat.S_ISDIR(mode):
                    if entry.name in RUNTIME_SNAPSHOT_DIRS:
                        continue
                    actual_directories.add(relative_name)
                    walk(Path(entry.path), relative)
                elif stat.S_ISREG(mode):
                    actual.add(relative_name)
                else:
                    special.append(relative_name)

    walk(base, Path())
    if symlinks:
        raise ExportValidationError(f"{context} contains undeclared symlinks: {sorted(symlinks)}")
    if special:
        raise ExportValidationError(
            f"{context} contains special filesystem entries: {sorted(special)}"
        )
    if actual != declared or actual_directories != declared_directories:
        raise ExportValidationError(
            f"{context} tree differs from its manifest: "
            f"missing_files={sorted(declared - actual)}, "
            f"extra_files={sorted(actual - declared)}, "
            f"missing_directories={sorted(declared_directories - actual_directories)}, "
            f"extra_directories={sorted(actual_directories - declared_directories)}"
        )


def _read_checksum_file(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ExportValidationError(f"{path}:{line_number}: malformed checksum line")
        digest = require_sha256(parts[0].lower(), f"{path}:{line_number} digest")
        filename = parts[1].lstrip("*").strip().replace("\\", "/")
        if "/problem_cache/" in filename:
            filename = filename.rsplit("/problem_cache/", 1)[1]
        elif filename.startswith("problem_cache/"):
            filename = filename[len("problem_cache/") :]
        relative = _safe_relative_path(filename, f"{path}:{line_number} filename")
        if len(relative.parts) != 1 or relative.name in entries:
            raise ExportValidationError(f"{path}:{line_number}: unsafe or duplicate cache filename")
        entries[relative.name] = digest
    return entries


def validate_declared_campaign_files(campaign_root: Path, manifest: Mapping[str, Any]) -> None:
    """Verify every source, runner, input, and cache digest declared by the manifest."""
    for key, destination in (
        ("source_files", campaign_root / "source" / "repo"),
        ("runner_files", campaign_root / "runner"),
    ):
        validate_snapshot_tree(destination, manifest.get(key), f"manifest.json.{key}")

    inputs = require_mapping(manifest.get("inputs"), "manifest.json.inputs")
    input_locations = {
        "seed_manifest": Path("experiments/reactions10/inputs/seed_manifest.json"),
        "seed_validation": Path("experiments/reactions10/inputs/seed_validation.json"),
        "problem_cache_checksums": Path("experiments/reactions10/inputs/problem_cache.sha256"),
    }
    source_records = require_mapping(manifest["source_files"], "manifest.json.source_files")
    for name, relative in input_locations.items():
        record = require_mapping(inputs.get(name), f"manifest.json.inputs.{name}")
        record_path = _safe_relative_path(record.get("path"), f"manifest.json.inputs.{name}.path")
        if record_path.name != relative.name:
            raise ExportValidationError(f"manifest.json.inputs.{name}.path has the wrong filename")
        source_record = require_mapping(
            source_records.get(relative.as_posix()),
            f"manifest.json.source_files.{relative.as_posix()}",
        )
        input_hash = require_sha256(record.get("sha256"), f"manifest.json.inputs.{name}.sha256")
        if input_hash != PAPER_INPUT_HASHES[name]:
            raise ExportValidationError(
                f"manifest input {name} differs from the fixed paper input digest"
            )
        if input_hash != require_sha256(
            source_record.get("sha256"),
            f"manifest.json.source_files.{relative.as_posix()}.sha256",
        ) or require_int(
            record.get("size_bytes"), f"manifest.json.inputs.{name}.size_bytes", minimum=0
        ) != require_int(
            source_record.get("size_bytes"),
            f"manifest.json.source_files.{relative.as_posix()}.size_bytes",
            minimum=0,
        ):
            raise ExportValidationError(
                f"manifest input record {name} differs from source snapshot"
            )

    seed_copy = (
        campaign_root
        / "source"
        / "repo"
        / "experiments"
        / "reactions10"
        / "inputs"
        / "seed_manifest.json"
    )
    seed_manifest = require_mapping(read_json(seed_copy), "fixed seed manifest snapshot")
    seed_keys = require_list(seed_manifest.get("reaction_keys"), "seed reaction_keys")
    if seed_keys != list(REACTION_KEYS):
        raise ExportValidationError("fixed seed manifest reaction order/set changed")
    require_close(
        seed_manifest.get("reaction_target_mha"),
        1000.0 * PAPER_TARGET_HARTREE,
        "fixed seed manifest reaction_target_mha",
    )
    seed_reactions = require_mapping(seed_manifest.get("reactions"), "seed reactions")
    if set(seed_reactions) != set(REACTION_KEYS):
        raise ExportValidationError("fixed seed manifest reaction mapping changed")
    for index, raw_lane in enumerate(require_list(manifest.get("lanes"), "manifest lanes")):
        lane = require_mapping(raw_lane, f"manifest lane {index}")
        reaction_key = require_string(lane.get("reaction_key"), f"manifest lane {index} key")
        seed_reaction = require_mapping(
            seed_reactions.get(reaction_key), f"seed reaction {reaction_key}"
        )
        expected_species = tuple(
            require_string(value, f"seed reaction {reaction_key} species")
            for value in require_list(
                seed_reaction.get("species"), f"seed reaction {reaction_key} species"
            )
        )
        lane_species = tuple(
            require_string(value, f"manifest lane {index} species")
            for value in require_list(lane.get("species"), f"manifest lane {index} species")
        )
        if (
            lane.get("reaction") != seed_reaction.get("label")
            or lane_species != expected_species
            or dict(normalized_terms(lane, f"manifest lane {index}"))
            != dict(normalized_terms(seed_reaction, f"seed reaction {reaction_key}"))
        ):
            raise ExportValidationError(
                f"manifest lane {lane.get('lane_id')} chemistry differs from the fixed seed input"
            )

    validation_copy = (
        campaign_root
        / "source"
        / "repo"
        / "experiments"
        / "reactions10"
        / "inputs"
        / "seed_validation.json"
    )
    raw_validation = require_list(read_json(validation_copy), "fixed seed validation snapshot")
    validation_by_reaction: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_validation):
        row = require_mapping(raw_row, f"seed validation row {index}")
        reaction_key = require_string(
            row.get("reaction_key"), f"seed validation row {index}.reaction_key"
        )
        if reaction_key in validation_by_reaction:
            raise ExportValidationError(f"duplicate seed validation reaction {reaction_key}")
        validation_by_reaction[reaction_key] = row
    if set(validation_by_reaction) != set(REACTION_KEYS):
        raise ExportValidationError("fixed seed validation reaction set changed")
    for index, raw_lane in enumerate(require_list(manifest.get("lanes"), "manifest lanes")):
        lane = require_mapping(raw_lane, f"manifest lane {index}")
        reaction_key = require_string(lane.get("reaction_key"), f"manifest lane {index} key")
        contract = require_string(lane.get("contract"), f"manifest lane {index} contract")
        row = validation_by_reaction[reaction_key]
        policies = require_mapping(row.get("policies"), f"seed validation {reaction_key}.policies")
        policy = require_mapping(
            policies.get(contract), f"seed validation {reaction_key}.{contract}"
        )
        if (
            require_bool(
                policy.get("full_space_constraint_passes"),
                f"seed validation {reaction_key}.{contract}.full_space_constraint_passes",
            )
            is not True
        ):
            raise ExportValidationError(
                f"fixed seed validation failed for {reaction_key}/{contract}"
            )
        point = require_mapping(
            policy.get("full_space_point"),
            f"seed validation {reaction_key}.{contract}.full_space_point",
        )
        species = tuple(
            require_string(item, f"manifest lane {index} species")
            for item in require_list(lane.get("species"), f"manifest lane {index} species")
        )
        point_species = require_mapping(
            point.get("species"), f"seed validation {reaction_key}.{contract}.species"
        )
        fixed_expectation = normalized_seed_expectation(
            {
                "formula_order": row.get("formula_order"),
                "n_steps": row.get("n_steps"),
                "signed_reaction_error_hartree": point.get("signed_reaction_error_hartree"),
                "absolute_reaction_error_hartree": point.get("absolute_reaction_error_hartree"),
                "triangle_bound_hartree": point.get("triangle_bound_hartree"),
                "max_species_absolute_error_hartree": point.get(
                    "max_species_absolute_error_hartree"
                ),
                "equal_share_error_limit_hartree": point.get("equal_share_error_limit_hartree"),
                "total_t_gate_count": point.get("total_t_gate_count"),
                "species": {
                    name: {
                        "signed_energy_error_hartree": require_mapping(
                            point_species.get(name),
                            f"seed validation {reaction_key}.{contract}.species.{name}",
                        ).get("signed_energy_error_hartree"),
                        "t_gate_count": require_mapping(
                            point_species.get(name),
                            f"seed validation {reaction_key}.{contract}.species.{name}",
                        ).get("t_gate_count"),
                    }
                    for name in species
                },
            },
            species=species,
            context=f"fixed seed expectation {reaction_key}/{contract}",
        )
        manifest_expectation = normalized_seed_expectation(
            lane.get("seed_result_expectation"),
            species=species,
            context=f"manifest lane {index}.seed_result_expectation",
        )
        if manifest_expectation != fixed_expectation:
            raise ExportValidationError(
                f"manifest seed expectation differs from fixed validation for "
                f"{reaction_key}/{contract}"
            )
        if (
            require_int(
                lane.get("seed_total_t_gate_count"),
                f"manifest lane {index}.seed_total_t_gate_count",
                minimum=0,
            )
            != fixed_expectation["total_t_gate_count"]
        ):
            raise ExportValidationError(
                f"manifest seed T count differs from fixed validation for {reaction_key}/{contract}"
            )

    cache_files = require_mapping(
        inputs.get("problem_cache_files"), "manifest.json.inputs.problem_cache_files"
    )
    cache_count = require_int(
        inputs.get("problem_cache_file_count"),
        "manifest.json.inputs.problem_cache_file_count",
        minimum=0,
    )
    if cache_count != 42 or len(cache_files) != 42:
        raise ExportValidationError("paper campaign must declare exactly 42 problem-cache files")
    normalized_cache_files = {
        filename: require_sha256(digest, f"manifest.json.inputs.problem_cache_files.{filename}")
        for filename, digest in cache_files.items()
    }
    checksum_copy = (
        campaign_root
        / "source"
        / "repo"
        / "experiments"
        / "reactions10"
        / "inputs"
        / "problem_cache.sha256"
    )
    if _read_checksum_file(checksum_copy) != normalized_cache_files:
        raise ExportValidationError(
            "manifest problem-cache hashes differ from the fixed checksum-file snapshot"
        )
    require_exact_directory(
        campaign_root / "problem_cache",
        context="campaign problem-cache root",
        files=set(normalized_cache_files),
    )
    for filename, digest in cache_files.items():
        relative = _safe_relative_path(
            filename, f"manifest.json.inputs.problem_cache_files.{filename}"
        )
        if len(relative.parts) != 1:
            raise ExportValidationError("problem-cache manifest keys must be bare filenames")
        expected = normalized_cache_files[filename]
        path = campaign_root / "problem_cache" / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ExportValidationError(f"problem-cache snapshot changed: {filename}")


def validate_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    campaign_id = validate_campaign_identity(manifest)
    model = _collect_setting(manifest, ("model", "codex_model"), PAPER_MODEL, "model")
    reasoning = _collect_setting(
        manifest,
        ("reasoning_effort", "model_reasoning_effort"),
        PAPER_REASONING_EFFORT,
        "reasoning_effort",
    )
    cli = _collect_setting(
        manifest,
        ("codex_cli_version",),
        PAPER_CODEX_CLI_VERSION,
        "codex_cli_version",
    )
    package = _collect_setting(
        manifest,
        ("codex_package",),
        PAPER_CODEX_PACKAGE,
        "codex_package",
    )
    repository = require_mapping(manifest.get("repository"), "manifest.json.repository")
    if require_bool(repository.get("dirty"), "manifest.json.repository.dirty"):
        raise ExportValidationError("paper export requires a clean repository source snapshot")
    commit = require_string(repository.get("commit"), "manifest.json.repository.commit")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ExportValidationError("manifest.json.repository.commit is not a full Git object ID")
    branch = require_string(repository.get("branch"), "manifest.json.repository.branch")
    clean_status_sha256 = hashlib.sha256(b"").hexdigest()
    if (
        require_sha256(repository.get("status_sha256"), "manifest.json.repository.status_sha256")
        != clean_status_sha256
    ):
        raise ExportValidationError("clean repository provenance has a nonempty status digest")
    return {
        "campaign_id": campaign_id,
        "repository_branch": branch,
        "repository_commit": commit,
        "codex_cli_version": cli,
        "codex_package": package,
        "model": model,
        "reasoning_effort": reasoning,
        "declared_runner_input_source_hashes": _collect_hash_provenance(manifest),
    }


def _result_bool(result: Mapping[str, Any], key: str, expected: bool, context: str) -> None:
    actual = require_bool(result.get(key), f"{context}.{key}")
    if actual is not expected:
        raise ExportValidationError(f"{context}.{key}={actual}, recomputed value={expected}")


def validate_result(
    result: object,
    *,
    lane: LaneSpec,
    iteration: int,
    policy_path: Path,
    manifest_sha256: str,
) -> Candidate:
    context = f"iteration {iteration}/{lane.lane_id} result"
    payload = require_mapping(result, context)
    if payload.get("status") != "ok":
        raise ExportValidationError(f"{context}.status must be 'ok'")
    if lane.campaign_id is not None and payload.get("campaign_id") != lane.campaign_id:
        raise ExportValidationError(f"{context}.campaign_id mismatch")
    if (
        require_sha256(
            payload.get("campaign_manifest_sha256"),
            f"{context}.campaign_manifest_sha256",
        )
        != manifest_sha256
    ):
        raise ExportValidationError(f"{context}.campaign_manifest_sha256 mismatch")
    if payload.get("lane_id") != lane.lane_id:
        raise ExportValidationError(f"{context}.lane_id mismatch")
    if payload.get("reaction_key") != lane.reaction_key:
        raise ExportValidationError(f"{context}.reaction_key mismatch")
    if payload.get("reaction") != lane.reaction:
        raise ExportValidationError(f"{context}.reaction mismatch")
    if payload.get("contract") != lane.contract:
        raise ExportValidationError(f"{context}.contract mismatch")
    if require_int(payload.get("iteration"), f"{context}.iteration") != iteration:
        raise ExportValidationError(f"{context}.iteration mismatch")
    expected_candidate_relative = (
        Path("iterations") / f"iter_{iteration:04d}" / "candidates" / lane.lane_id / "simple_pf.py"
    )
    if _safe_relative_path(
        payload.get("candidate_relative_path"),
        f"{context}.candidate_relative_path",
    ) != expected_candidate_relative or _safe_relative_path(
        payload.get("problem_cache_relative_path"),
        f"{context}.problem_cache_relative_path",
    ) != Path("problem_cache"):
        raise ExportValidationError(f"{context} portable candidate/cache provenance mismatch")
    require_exact_target(
        payload.get("reaction_target_hartree"), f"{context}.reaction_target_hartree"
    )
    require_optional_close(
        payload,
        "reaction_target_mha",
        1000.0 * PAPER_TARGET_HARTREE,
        context,
    )

    if not policy_path.is_file() or policy_path.is_symlink():
        raise ExportValidationError(
            f"{context} candidate policy must be a regular file: {policy_path}"
        )
    policy_sha256 = require_sha256(payload.get("policy_sha256"), f"{context}.policy_sha256")
    actual_policy_sha256 = sha256_file(policy_path)
    if policy_sha256 != actual_policy_sha256:
        raise ExportValidationError(
            f"{context} policy hash mismatch: result={policy_sha256}, file={actual_policy_sha256}"
        )

    expected_terms = [
        {"coefficient": coefficient, "species": species} for species, coefficient in lane.terms
    ]
    fixed_fields: dict[str, object] = {
        "reaction_terms": expected_terms,
        "species_order": list(lane.species),
        "evaluation_mode": "full_hilbert_space_compiled_one_step_unitary_diagonalization",
        "ground_phase_matching": "maximum_ground_state_eigenvector_overlap",
        "compilation_backend": "pygridsynth",
        "t_count_aggregation": "sum_each_unique_species_once",
    }
    for field, expected in fixed_fields.items():
        if payload.get(field) != expected:
            raise ExportValidationError(f"{context}.{field} changed from fixed evaluator semantics")
    expected_source_provenance = {
        "archived_benchmark_sha256": ARCHIVED_REACTION_BENCHMARK_SHA256,
        "archived_evaluator_sha256": ARCHIVED_REACTION_EVALUATOR_SHA256,
    }
    if payload.get("source_provenance") != expected_source_provenance:
        raise ExportValidationError(f"{context}.source_provenance changed")

    raw_species = require_mapping(payload.get("species"), f"{context}.species")
    if set(raw_species) != set(lane.species):
        raise ExportValidationError(
            f"{context}.species set mismatch: expected {sorted(lane.species)}, got {sorted(raw_species)}"
        )
    signed_species_errors: dict[str, float] = {}
    absolute_species_errors: dict[str, float] = {}
    species_t_counts: dict[str, int] = {}
    exact_species_energies: dict[str, float] = {}
    pf_species_energies: dict[str, float] = {}
    species_scheduled_counts: dict[str, int] = {}
    for species in lane.species:
        species_context = f"{context}.species.{species}"
        item = require_mapping(raw_species[species], species_context)
        if item.get("species") != species:
            raise ExportValidationError(f"{species_context}.species mismatch")
        signed_error = require_finite(
            item.get("signed_energy_error_hartree"),
            f"{species_context}.signed_energy_error_hartree",
        )
        absolute_error = abs(signed_error)
        require_close(
            item.get("absolute_energy_error_hartree"),
            absolute_error,
            f"{species_context}.absolute_energy_error_hartree",
        )
        require_optional_close(
            item,
            "signed_energy_error_mha",
            1000.0 * signed_error,
            species_context,
        )
        require_optional_close(
            item,
            "absolute_energy_error_mha",
            1000.0 * absolute_error,
            species_context,
        )
        exact_energy = require_finite(
            item.get("exact_energy_hartree"), f"{species_context}.exact_energy_hartree"
        )
        pf_energy = require_finite(
            item.get("pf_energy_estimate_hartree"),
            f"{species_context}.pf_energy_estimate_hartree",
        )
        require_close(
            pf_energy - exact_energy,
            signed_error,
            f"{species_context} energy/error arithmetic",
        )
        formula_order = require_int(
            item.get("formula_order"), f"{species_context}.formula_order", minimum=1
        )
        n_steps = require_int(item.get("n_steps"), f"{species_context}.n_steps", minimum=1)
        if formula_order not in (1, 2):
            raise ExportValidationError(f"{species_context}.formula_order must be 1 or 2")
        method_config = require_mapping(
            item.get("method_config"), f"{species_context}.method_config"
        )
        if (
            require_int(
                method_config.get("formula_order"),
                f"{species_context}.method_config.formula_order",
            )
            != formula_order
            or require_int(method_config.get("n_steps"), f"{species_context}.method_config.n_steps")
            != n_steps
            or method_config.get("species") != species
        ):
            raise ExportValidationError(f"{species_context}.method_config drifted")
        zero_threshold = require_finite(
            item.get("zero_angle_threshold"), f"{species_context}.zero_angle_threshold"
        )
        method_threshold = require_finite(
            method_config.get("zero_angle_threshold"),
            f"{species_context}.method_config.zero_angle_threshold",
        )
        if zero_threshold < 0.0 or zero_threshold != method_threshold:
            raise ExportValidationError(f"{species_context} zero-angle threshold drifted")
        term_count = require_int(
            item.get("total_pauli_rotation_terms"),
            f"{species_context}.total_pauli_rotation_terms",
            minimum=1,
        )
        rotations_per_step = require_int(
            item.get("rotations_per_step"),
            f"{species_context}.rotations_per_step",
            minimum=1,
        )
        expected_rotations_per_step = term_count if formula_order == 1 else 2 * term_count - 1
        if rotations_per_step != expected_rotations_per_step:
            raise ExportValidationError(f"{species_context} PF rotation-count mismatch")
        scheduled = require_int(
            item.get("scheduled_rotation_count"),
            f"{species_context}.scheduled_rotation_count",
            minimum=0,
        )
        nonzero = require_int(
            item.get("nonzero_rotation_count"),
            f"{species_context}.nonzero_rotation_count",
            minimum=0,
        )
        skipped = require_int(
            item.get("skipped_rotation_count"),
            f"{species_context}.skipped_rotation_count",
            minimum=0,
        )
        one_nonzero = require_int(
            item.get("one_step_nonzero_rotation_count"),
            f"{species_context}.one_step_nonzero_rotation_count",
            minimum=0,
        )
        one_skipped = require_int(
            item.get("one_step_skipped_rotation_count"),
            f"{species_context}.one_step_skipped_rotation_count",
            minimum=0,
        )
        if (
            scheduled != rotations_per_step * n_steps
            or nonzero + skipped != scheduled
            or one_nonzero + one_skipped != rotations_per_step
            or nonzero != one_nonzero * n_steps
            or skipped != one_skipped * n_steps
        ):
            raise ExportValidationError(f"{species_context} scheduled rotation-count mismatch")
        one_step_t_count = require_int(
            item.get("one_step_t_gate_count"),
            f"{species_context}.one_step_t_gate_count",
            minimum=0,
        )
        t_count = require_int(
            item.get("t_gate_count"), f"{species_context}.t_gate_count", minimum=0
        )
        if t_count != one_step_t_count * n_steps:
            raise ExportValidationError(f"{species_context} T-count scaling mismatch")
        one_step_skipped_angle = require_finite(
            item.get("one_step_skipped_absolute_angle_sum"),
            f"{species_context}.one_step_skipped_absolute_angle_sum",
        )
        skipped_angle = require_finite(
            item.get("skipped_absolute_angle_sum"),
            f"{species_context}.skipped_absolute_angle_sum",
        )
        one_step_skip_bound = require_finite(
            item.get("one_step_skipped_rotation_operator_error_bound"),
            f"{species_context}.one_step_skipped_rotation_operator_error_bound",
        )
        skip_bound = require_finite(
            item.get("skipped_rotation_operator_error_bound"),
            f"{species_context}.skipped_rotation_operator_error_bound",
        )
        require_close(
            skipped_angle,
            one_step_skipped_angle * n_steps,
            f"{species_context} skipped angle accounting",
        )
        require_close(
            skip_bound,
            one_step_skip_bound * n_steps,
            f"{species_context} skipped error-bound accounting",
        )
        if (
            require_bool(
                item.get("stationary_step_policy_verified"),
                f"{species_context}.stationary_step_policy_verified",
            )
            is not True
        ):
            raise ExportValidationError(f"{species_context} lacks stationary-policy attestation")
        species_fixed = {
            "unitary_construction": (
                "compiled_one_step_product_formula_applied_and_diagonalized_in_full_hilbert_space"
            ),
            "phase_matching": "maximum_ground_state_eigenvector_overlap",
            "t_count_semantics": (
                "exact_count_of_T_characters_in_selected_PyGridSynth_gate_strings_times_n_steps"
            ),
        }
        for field, expected in species_fixed.items():
            if item.get(field) != expected:
                raise ExportValidationError(f"{species_context}.{field} changed")
        signed_species_errors[species] = signed_error
        absolute_species_errors[species] = absolute_error
        species_t_counts[species] = t_count
        exact_species_energies[species] = exact_energy
        pf_species_energies[species] = pf_energy
        species_scheduled_counts[species] = scheduled

    total_t_gate_count = sum(species_t_counts.values())
    if (
        require_int(payload.get("total_t_gate_count"), f"{context}.total_t_gate_count", minimum=0)
        != total_t_gate_count
    ):
        raise ExportValidationError(
            f"{context}.total_t_gate_count does not equal the unique-species sum "
            f"{total_t_gate_count}"
        )
    signed_reaction_error = sum(
        coefficient * signed_species_errors[species] for species, coefficient in lane.terms
    )
    absolute_reaction_error = abs(signed_reaction_error)
    triangle_bound = sum(
        abs(coefficient) * absolute_species_errors[species] for species, coefficient in lane.terms
    )
    stoichiometric_l1 = sum(abs(coefficient) for _species, coefficient in lane.terms)
    equal_share_limit = PAPER_TARGET_HARTREE / stoichiometric_l1
    max_species_error = max(absolute_species_errors.values())
    total_scheduled = sum(species_scheduled_counts.values())
    exact_reaction_energy = sum(
        coefficient * exact_species_energies[species] for species, coefficient in lane.terms
    )
    pf_reaction_energy = sum(
        coefficient * pf_species_energies[species] for species, coefficient in lane.terms
    )
    passes_equal_share = all(
        absolute_species_errors[species] <= equal_share_limit for species in lane.species
    )
    passes_reaction = absolute_reaction_error <= PAPER_TARGET_HARTREE
    passes_contract = passes_equal_share if lane.contract == "equal_share" else passes_reaction

    require_close(
        payload.get("signed_reaction_error_hartree"),
        signed_reaction_error,
        f"{context}.signed_reaction_error_hartree",
    )
    require_close(
        payload.get("absolute_reaction_error_hartree"),
        absolute_reaction_error,
        f"{context}.absolute_reaction_error_hartree",
    )
    require_close(
        payload.get("triangle_bound_hartree"),
        triangle_bound,
        f"{context}.triangle_bound_hartree",
    )
    require_close(
        payload.get("equal_share_error_limit_hartree"),
        equal_share_limit,
        f"{context}.equal_share_error_limit_hartree",
    )
    require_close(
        payload.get("max_species_absolute_error_hartree"),
        max_species_error,
        f"{context}.max_species_absolute_error_hartree",
    )
    require_close(
        payload.get("stoichiometric_abs_coefficient_sum"),
        stoichiometric_l1,
        f"{context}.stoichiometric_abs_coefficient_sum",
    )
    require_close(
        payload.get("exact_reaction_energy_hartree"),
        exact_reaction_energy,
        f"{context}.exact_reaction_energy_hartree",
    )
    require_close(
        payload.get("pf_reaction_energy_hartree"),
        pf_reaction_energy,
        f"{context}.pf_reaction_energy_hartree",
    )
    require_close(
        payload.get("triangle_inequality_slack_hartree"),
        triangle_bound - absolute_reaction_error,
        f"{context}.triangle_inequality_slack_hartree",
    )
    for key, value in (
        ("signed_reaction_error_mha", signed_reaction_error),
        ("absolute_reaction_error_mha", absolute_reaction_error),
        ("triangle_bound_mha", triangle_bound),
        ("equal_share_error_limit_mha", equal_share_limit),
        ("max_species_absolute_error_mha", max_species_error),
    ):
        require_optional_close(payload, key, 1000.0 * value, context)
    _result_bool(payload, "passes_equal_share_constraint", passes_equal_share, context)
    _result_bool(payload, "passes_reaction_constraint", passes_reaction, context)
    _result_bool(payload, "passes_contract", passes_contract, context)
    if triangle_bound + DERIVED_ABS_TOL < absolute_reaction_error:
        raise ExportValidationError(f"{context} violates the triangle inequality")
    if (
        require_bool(
            payload.get("triangle_inequality_holds"), f"{context}.triangle_inequality_holds"
        )
        is not True
    ):
        raise ExportValidationError(f"{context} lacks triangle-inequality attestation")
    if (
        require_int(
            payload.get("total_scheduled_rotation_count"),
            f"{context}.total_scheduled_rotation_count",
            minimum=0,
        )
        != total_scheduled
    ):
        raise ExportValidationError(f"{context}.total_scheduled_rotation_count is inconsistent")
    require_close(
        pf_reaction_energy - exact_reaction_energy,
        signed_reaction_error,
        f"{context} reaction-energy arithmetic",
    )
    violation_ratio = (
        max_species_error / equal_share_limit
        if lane.contract == "equal_share"
        else absolute_reaction_error / PAPER_TARGET_HARTREE
    )
    expected_score = (
        float(total_t_gate_count)
        if passes_contract
        else FAILURE_SCORE_BASE * (1.0 + max(1.0, violation_ratio)) ** 2 + float(total_t_gate_count)
    )
    score = require_finite(payload.get("score"), f"{context}.score")
    if not math.isclose(score, expected_score, rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ExportValidationError(f"{context}.score arithmetic mismatch")

    return Candidate(
        lane=lane,
        iteration=iteration,
        policy_sha256=policy_sha256,
        policy_path=policy_path,
        result=payload,
        total_t_gate_count=total_t_gate_count,
        signed_reaction_error_hartree=signed_reaction_error,
        absolute_reaction_error_hartree=absolute_reaction_error,
        triangle_bound_hartree=triangle_bound,
        equal_share_error_limit_hartree=equal_share_limit,
        max_species_absolute_error_hartree=max_species_error,
        passes_equal_share_constraint=passes_equal_share,
        passes_reaction_constraint=passes_reaction,
        passes_contract=passes_contract,
    )


def normalize_decisions(payload: object, iteration: int) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "decisions" in payload:
        if (
            "iteration" in payload
            and require_int(payload["iteration"], f"iteration {iteration} decisions.iteration")
            != iteration
        ):
            raise ExportValidationError(
                f"iteration {iteration} decisions wrapper has wrong iteration"
            )
        raw = require_list(payload["decisions"], f"iteration {iteration} decisions")
    elif isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = []
        for lane_id, item in payload.items():
            decision = dict(require_mapping(item, f"iteration {iteration} decision {lane_id}"))
            decision.setdefault("lane_id", lane_id)
            raw.append(decision)
    else:
        raise ExportValidationError(f"iteration {iteration} decisions.json has invalid shape")
    return [
        require_mapping(item, f"iteration {iteration} decisions[{index}]")
        for index, item in enumerate(raw)
    ]


def _record_hash(record: Mapping[str, Any], aliases: Sequence[str], context: str) -> str:
    values: list[tuple[str, str]] = []
    containers: list[tuple[str, Mapping[str, Any]]] = [("record", record)]
    for key in ("lineage", "policy", "provenance"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append((key, value))
    for container_name, container in containers:
        for alias in aliases:
            if alias in container:
                values.append(
                    (
                        f"{container_name}.{alias}",
                        require_sha256(container[alias], f"{context}.{container_name}.{alias}"),
                    )
                )
    if not values:
        raise ExportValidationError(f"{context} is missing {aliases[0]}")
    unique = {value for _name, value in values}
    if len(unique) != 1:
        raise ExportValidationError(f"{context} has conflicting lineage hashes: {values}")
    return values[0][1]


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_codex_jsonl_lifecycle(stdout_path: Path, context: str) -> dict[str, Any]:
    if not stdout_path.is_file() or stdout_path.is_symlink():
        raise ExportValidationError(f"{context} Codex JSONL stdout is missing or unsafe")
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stdout_path.read_bytes().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line.decode("utf-8"), object_pairs_hook=_duplicate_key_guard)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ExportValidationError(
                f"{context} Codex JSONL line {line_number} is invalid"
            ) from exc
        event = require_mapping(event, f"{context} Codex JSONL line {line_number}")
        require_string(event.get("type"), f"{context} Codex JSONL line {line_number}.type")
        events.append(event)
    starts = [
        (index, event) for index, event in enumerate(events) if event["type"] == "thread.started"
    ]
    terminals = [
        (index, event) for index, event in enumerate(events) if event["type"] == "turn.completed"
    ]
    failures = [event for event in events if event["type"] in {"turn.failed", "error"}]
    if len(starts) != 1 or len(terminals) != 1 or failures or starts[0][0] >= terminals[0][0]:
        raise ExportValidationError(
            f"{context} Codex JSONL does not prove one successful lifecycle"
        )
    start = starts[0][1]
    terminal = terminals[0][1]
    thread_id = require_string(start.get("thread_id"), f"{context} Codex thread ID")
    if len(thread_id) > 256:
        raise ExportValidationError(f"{context} Codex thread ID is too long")
    usage = require_mapping(terminal.get("usage"), f"{context} Codex usage")
    if not usage or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in usage.values()
    ):
        raise ExportValidationError(f"{context} Codex usage is malformed")
    return {
        "codex_thread_id": thread_id,
        "codex_start_event_sha256": _canonical_json_sha256(start),
        "codex_lifecycle_status": "turn_completed",
        "codex_terminal_event_sha256": _canonical_json_sha256(terminal),
        "codex_usage": usage,
        "codex_usage_sha256": _canonical_json_sha256(usage),
    }


def validate_mutation_record(
    record: object,
    *,
    lane: LaneSpec,
    iteration: int,
    expected_parent_iteration: int,
    expected_parent_sha256: str,
    expected_parent_policy_path: Path,
    candidate_sha256: str,
    mutation_root: Path,
    campaign_root: Path,
) -> None:
    context = f"iteration {iteration}/{lane.lane_id} mutation record"
    payload = require_mapping(record, context)
    expected_record_fields = {
        "schema_version",
        "lane_id",
        "iteration",
        "kind",
        "status",
        "mutation_outcome",
        "parent_iteration",
        "parent_policy_sha256",
        "prompt_sha256",
        "intent_sha256",
        "launch_command_sha256",
        "launch_nonce",
        "launch_intent_sha256",
        "launch_receipt_sha256",
        "exec_acknowledgement_sha256",
        "exec_acknowledgement_status",
        "exec_ack_protocol",
        "codex_thread_id",
        "codex_start_event_sha256",
        "codex_lifecycle_status",
        "codex_terminal_event_sha256",
        "codex_usage",
        "codex_usage_sha256",
        "archived_attempt_evidence",
        "archived_attempt_directory_paths",
        "process_sha256",
        "codex_stdout_sha256",
        "codex_stderr_sha256",
        "last_message_sha256",
        "work_file_records",
        "work_directory_paths",
        "work_policy_sha256",
        "candidate_policy_sha256",
        "codex_package",
        "codex_cli_version",
        "codex_version_output",
        "model",
        "reasoning_effort",
        "sandbox",
        "ephemeral",
        "ignored_user_config",
        "ignored_rules",
        "returncode",
        "timed_out",
        "changed",
        "valid",
        "validation_error",
        "unexpected_work_files",
        "recovered_after_interruption",
        "created_at",
        "completed_at",
    }
    if set(payload) != expected_record_fields:
        raise ExportValidationError(f"{context} has an unexpected schema")
    if payload.get("returncode") != 0 or payload.get("timed_out") is not False:
        raise ExportValidationError(
            f"{context} lacks successful process lifecycle completion evidence"
        )
    if (
        require_int(payload.get("schema_version"), f"{context}.schema_version") != 1
        or require_int(payload.get("iteration"), f"{context}.iteration") != iteration
        or require_int(payload.get("parent_iteration"), f"{context}.parent_iteration")
        != expected_parent_iteration
        or require_int(payload.get("returncode"), f"{context}.returncode") != 0
    ):
        raise ExportValidationError(f"{context} integer identity fields are invalid")
    for field in (
        "ephemeral",
        "ignored_user_config",
        "ignored_rules",
        "timed_out",
        "changed",
        "valid",
        "recovered_after_interruption",
    ):
        require_bool(payload.get(field), f"{context}.{field}")
    identity = {
        "schema_version": 1,
        "lane_id": lane.lane_id,
        "iteration": iteration,
        "kind": "autoresearch_proposal",
        "status": "candidate_ready",
        "parent_iteration": expected_parent_iteration,
        "parent_policy_sha256": expected_parent_sha256,
        "candidate_policy_sha256": candidate_sha256,
        "codex_package": PAPER_CODEX_PACKAGE,
        "codex_cli_version": PAPER_CODEX_CLI_VERSION,
        "model": PAPER_MODEL,
        "reasoning_effort": PAPER_REASONING_EFFORT,
        "sandbox": "workspace-write",
        "ephemeral": True,
        "ignored_user_config": True,
        "ignored_rules": True,
        "returncode": 0,
        "timed_out": False,
        "exec_acknowledgement_status": "codex_started",
        "exec_ack_protocol": EXEC_ACK_PROTOCOL,
        "codex_lifecycle_status": "turn_completed",
    }
    drift = {
        key: {"actual": payload.get(key), "expected": expected}
        for key, expected in identity.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ExportValidationError(f"{context} identity/provenance drift: {drift}")
    require_bool(
        payload.get("recovered_after_interruption"),
        f"{context}.recovered_after_interruption",
    )
    for field in ("created_at", "completed_at"):
        require_string(payload.get(field), f"{context}.{field}")
    version_output = require_string(
        payload.get("codex_version_output"), f"{context}.codex_version_output"
    )
    if re.findall(r"(?<![0-9.])([0-9]+\.[0-9]+\.[0-9]+)(?![0-9.])", version_output) != [
        PAPER_CODEX_CLI_VERSION
    ]:
        raise ExportValidationError(f"{context}.codex_version_output changed")

    prompt_path = mutation_root / "prompt.md"
    parent_snapshot_path = mutation_root / "parent_simple_pf.py"
    intent_path = mutation_root / "intent.json"
    launch_intent_path = mutation_root / "launch_intent.json"
    launch_receipt_path = mutation_root / "launch.json"
    process_path = mutation_root / "process.json"
    stdout_path = mutation_root / "codex.stdout"
    stderr_path = mutation_root / "codex.stderr"
    last_message_path = mutation_root / "last_message.txt"
    for path, label in (
        (prompt_path, "prompt"),
        (parent_snapshot_path, "parent snapshot"),
        (intent_path, "intent"),
        (launch_intent_path, "launch intent"),
        (launch_receipt_path, "launch receipt"),
        (process_path, "process receipt"),
        (stdout_path, "Codex stdout"),
        (stderr_path, "Codex stderr"),
        (last_message_path, "Codex last message"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ExportValidationError(f"{context} {label} is missing or non-regular")
    if last_message_path.stat().st_size <= 0:
        raise ExportValidationError(f"{context} Codex last message is empty")
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ExportValidationError(f"{context} prompt is not UTF-8") from exc
    prompt_sha256 = require_sha256(payload.get("prompt_sha256"), f"{context}.prompt_sha256")
    if sha256_file(prompt_path) != prompt_sha256 or sha256_file(parent_snapshot_path) != (
        expected_parent_sha256
    ):
        raise ExportValidationError(f"{context} prompt or parent snapshot hash mismatch")

    intent = require_mapping(read_json(intent_path), f"{context} intent")
    expected_parent_relative = (
        expected_parent_policy_path.resolve().relative_to(campaign_root.resolve()).as_posix()
    )
    expected_intent = {
        "schema_version": 1,
        "lane_id": lane.lane_id,
        "iteration": iteration,
        "parent_iteration": expected_parent_iteration,
        "parent_policy_path": expected_parent_relative,
        "parent_policy_sha256": expected_parent_sha256,
        "prompt_sha256": prompt_sha256,
        "codex_package": PAPER_CODEX_PACKAGE,
        "codex_cli_version": PAPER_CODEX_CLI_VERSION,
        "model": PAPER_MODEL,
        "reasoning_effort": PAPER_REASONING_EFFORT,
    }
    if set(intent) != {*expected_intent, "created_at"} or any(
        intent.get(key) != value for key, value in expected_intent.items()
    ):
        raise ExportValidationError(f"{context} intent schema/identity drift")
    if require_string(intent.get("created_at"), f"{context} intent.created_at") != payload.get(
        "created_at"
    ):
        raise ExportValidationError(f"{context} record creation time differs from intent")
    if require_sha256(payload.get("intent_sha256"), f"{context}.intent_sha256") != sha256_file(
        intent_path
    ):
        raise ExportValidationError(f"{context} intent file hash mismatch")

    launch_command_sha256 = require_sha256(
        payload.get("launch_command_sha256"), f"{context}.launch_command_sha256"
    )
    launch_nonce = require_sha256(payload.get("launch_nonce"), f"{context}.launch_nonce")
    launch_intent = require_mapping(read_json(launch_intent_path), f"{context} launch intent")
    expected_launch_intent = {
        "schema_version": 1,
        "lane_id": lane.lane_id,
        "iteration": iteration,
        "command_sha256": launch_command_sha256,
        "launch_nonce": launch_nonce,
        "exec_ack_protocol": EXEC_ACK_PROTOCOL,
        "required_start_event": "thread.started",
        "required_terminal_event": "turn.completed",
    }
    if set(launch_intent) != {*expected_launch_intent, "requested_at"} or any(
        launch_intent.get(key) != value for key, value in expected_launch_intent.items()
    ):
        raise ExportValidationError(f"{context} launch intent schema/identity drift")
    require_string(launch_intent.get("requested_at"), f"{context} launch intent.requested_at")
    if require_sha256(
        payload.get("launch_intent_sha256"), f"{context}.launch_intent_sha256"
    ) != sha256_file(launch_intent_path):
        raise ExportValidationError(f"{context} launch intent file hash mismatch")

    launch_receipt = require_mapping(read_json(launch_receipt_path), f"{context} launch receipt")
    launch_fields = {
        "schema_version",
        "pid",
        "pgid",
        "exec_pid",
        "work_dir",
        "command_sha256",
        "status",
        "exec_ack_protocol",
        "launch_nonce",
        "low_level_exec_status",
        "started_at_epoch",
        "exec_confirmed_at_epoch",
        "codex_thread_id",
        "codex_start_event_sha256",
        "codex_started_at_epoch",
    }
    if set(launch_receipt) != launch_fields:
        raise ExportValidationError(f"{context} launch receipt has an unexpected schema")
    if (
        require_int(launch_receipt.get("schema_version"), f"{context} launch.schema_version") != 1
        or launch_receipt.get("status") != "codex_started"
        or launch_receipt.get("low_level_exec_status") != "exec_confirmed"
        or launch_receipt.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or launch_receipt.get("command_sha256") != launch_command_sha256
        or launch_receipt.get("launch_nonce") != launch_nonce
    ):
        raise ExportValidationError(f"{context} does not prove a Codex startup")
    launch_pid = require_int(launch_receipt.get("pid"), f"{context} launch.pid", minimum=2)
    launch_pgid = require_int(launch_receipt.get("pgid"), f"{context} launch.pgid", minimum=2)
    launch_exec_pid = require_int(
        launch_receipt.get("exec_pid"), f"{context} launch.exec_pid", minimum=2
    )
    if launch_pid != launch_pgid or launch_exec_pid == launch_pid:
        raise ExportValidationError(f"{context} launch PID/PGID/exec PID mismatch")
    work_dir_text = _require_absolute_posix_path(
        launch_receipt.get("work_dir"), f"{context} launch.work_dir"
    )
    work_dir = PurePosixPath(work_dir_text)
    expected_work_suffix = (
        "iterations",
        f"iter_{iteration:04d}",
        "mutations",
        lane.lane_id,
        "work",
    )
    if tuple(work_dir.parts[-len(expected_work_suffix) :]) != expected_work_suffix:
        raise ExportValidationError(f"{context} launch work directory is malformed")
    timestamps = [
        require_finite(launch_receipt.get(field), f"{context} launch.{field}")
        for field in ("started_at_epoch", "exec_confirmed_at_epoch", "codex_started_at_epoch")
    ]
    if timestamps[0] <= 0.0 or timestamps != sorted(timestamps):
        raise ExportValidationError(f"{context} launch timestamps are invalid")
    launch_receipt_sha256 = sha256_file(launch_receipt_path)
    if any(
        require_sha256(payload.get(field), f"{context}.{field}") != launch_receipt_sha256
        for field in ("launch_receipt_sha256", "exec_acknowledgement_sha256")
    ):
        raise ExportValidationError(f"{context} launch receipt hash is not cross-linked")

    if require_sha256(payload.get("process_sha256"), f"{context}.process_sha256") != sha256_file(
        process_path
    ):
        raise ExportValidationError(f"{context} process receipt file hash mismatch")
    process = require_mapping(read_json(process_path), f"{context} process receipt")
    process_fields = {
        "command",
        "command_sha256",
        "returncode",
        "timed_out",
        "process_error",
        "lifecycle_error",
        "pid",
        "pgid",
        "exec_pid",
        "exec_acknowledgement_status",
        "exec_ack_protocol",
        "launch_nonce",
        "codex_thread_id",
        "codex_start_event_sha256",
        "codex_lifecycle_status",
        "codex_terminal_event_sha256",
        "codex_usage",
        "codex_usage_sha256",
        "codex_stdout_sha256",
        "codex_stderr_sha256",
        "last_message_sha256",
        "launch_receipt_sha256",
        "started_at",
        "completed_at",
    }
    if set(process) != process_fields:
        raise ExportValidationError(f"{context} process receipt has an unexpected schema")
    command_raw = require_list(process.get("command"), f"{context} process.command")
    last_message_text = str(work_dir.parent / "last_message.txt")
    expected_command = [
        "npx",
        "-y",
        PAPER_CODEX_PACKAGE,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-s",
        "workspace-write",
        "-C",
        work_dir_text,
        "-m",
        PAPER_MODEL,
        "-c",
        f'model_reasoning_effort="{PAPER_REASONING_EFFORT}"',
        "-o",
        last_message_text,
        "<prompt recorded in prompt.md>",
    ]
    if command_raw != expected_command or _canonical_json_sha256(process.get("codex_usage")) != (
        process.get("codex_usage_sha256")
    ):
        raise ExportValidationError(f"{context} process command or usage digest changed")
    if hashlib.sha256("\0".join([*command_raw[:-1], prompt_text]).encode("utf-8")).hexdigest() != (
        launch_command_sha256
    ):
        raise ExportValidationError(f"{context} launch command hash cannot be replayed")
    if (
        process.get("command_sha256") != launch_command_sha256
        or process.get("launch_receipt_sha256") != launch_receipt_sha256
        or require_int(process.get("pid"), f"{context} process.pid", minimum=2) != launch_pid
        or require_int(process.get("pgid"), f"{context} process.pgid", minimum=2) != launch_pgid
        or require_int(process.get("exec_pid"), f"{context} process.exec_pid", minimum=2)
        != launch_exec_pid
        or process.get("exec_acknowledgement_status") != "codex_started"
        or process.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or process.get("launch_nonce") != launch_nonce
        or require_int(process.get("returncode"), f"{context} process.returncode") != 0
        or require_bool(process.get("timed_out"), f"{context} process.timed_out") is not False
        or process.get("process_error") is not None
        or process.get("lifecycle_error") is not None
    ):
        raise ExportValidationError(f"{context} process/launch lifecycle cross-link mismatch")
    require_string(process.get("started_at"), f"{context} process.started_at")
    require_string(process.get("completed_at"), f"{context} process.completed_at")

    lifecycle = validate_codex_jsonl_lifecycle(stdout_path, context)
    for field, expected in lifecycle.items():
        if process.get(field) != expected or payload.get(field) != expected:
            raise ExportValidationError(f"{context} {field} differs from Codex JSONL")
    if launch_receipt.get("codex_thread_id") != lifecycle["codex_thread_id"] or (
        launch_receipt.get("codex_start_event_sha256") != lifecycle["codex_start_event_sha256"]
    ):
        raise ExportValidationError(f"{context} launch receipt differs from Codex JSONL")
    for field, path in (
        ("codex_stdout_sha256", stdout_path),
        ("codex_stderr_sha256", stderr_path),
        ("last_message_sha256", last_message_path),
    ):
        digest = sha256_file(path)
        if process.get(field) != digest or payload.get(field) != digest:
            raise ExportValidationError(f"{context} {field} evidence hash mismatch")

    validate_archived_attempt_evidence(mutation_root, record=payload, context=context)
    declared_work_files = validate_recorded_file_tree(
        mutation_root / "work",
        records=payload.get("work_file_records"),
        directory_paths=payload.get("work_directory_paths"),
        context=f"{context} work tree",
    )
    if "simple_pf.py" not in declared_work_files:
        raise ExportValidationError(f"{context} work tree has no policy")
    unexpected = _validated_directory_paths(
        payload.get("unexpected_work_files"), f"{context}.unexpected_work_files"
    )
    if set(unexpected) != declared_work_files - {"simple_pf.py"}:
        raise ExportValidationError(f"{context}.unexpected_work_files differs from work tree")
    work_policy_path = mutation_root / "work" / "simple_pf.py"
    work_policy_sha256 = sha256_file(work_policy_path)
    if (
        require_sha256(payload.get("work_policy_sha256"), f"{context}.work_policy_sha256")
        != work_policy_sha256
    ):
        raise ExportValidationError(f"{context} work policy hash mismatch")
    changed = work_policy_path.read_bytes() != parent_snapshot_path.read_bytes()
    if require_bool(payload.get("changed"), f"{context}.changed") is not changed:
        raise ExportValidationError(f"{context}.changed differs from work/parent bytes")
    valid = require_bool(payload.get("valid"), f"{context}.valid")
    validation_error = payload.get("validation_error")
    if (valid and validation_error is not None) or (
        not valid and (not isinstance(validation_error, str) or not validation_error)
    ):
        raise ExportValidationError(f"{context}.validation_error is inconsistent")
    work_directories = _validated_directory_paths(
        payload.get("work_directory_paths"), f"{context}.work_directory_paths"
    )
    use_mutation = changed and valid and not unexpected and not work_directories
    expected_outcome = "mutated" if use_mutation else "fallback_parent"
    if payload.get("mutation_outcome") != expected_outcome:
        raise ExportValidationError(f"{context}.mutation_outcome differs from work evidence")
    expected_candidate_sha256 = work_policy_sha256 if use_mutation else expected_parent_sha256
    if candidate_sha256 != expected_candidate_sha256:
        raise ExportValidationError(f"{context} candidate bytes do not follow mutation outcome")


def validate_baseline_record(
    record: object,
    *,
    lane: LaneSpec,
    candidate_sha256: str,
) -> None:
    context = f"iteration 0/{lane.lane_id} baseline record"
    payload = require_mapping(record, context)
    expected_fields = {
        "schema_version",
        "lane_id",
        "iteration",
        "kind",
        "status",
        "parent_iteration",
        "parent_policy_sha256",
        "prompt_sha256",
        "candidate_policy_sha256",
        "seed_policy_sha256",
        "codex_package",
        "codex_cli_version",
        "model",
        "reasoning_effort",
        "returncode",
        "timed_out",
        "changed",
        "valid",
        "validation_error",
        "created_at",
        "completed_at",
    }
    if set(payload) != expected_fields:
        raise ExportValidationError(f"{context} has an unexpected schema")
    if payload.get("lane_id") != lane.lane_id:
        raise ExportValidationError(f"{context}.lane_id mismatch")
    if require_int(payload.get("schema_version"), f"{context}.schema_version") != 1:
        raise ExportValidationError(f"{context}.schema_version must be 1")
    if require_int(payload.get("iteration"), f"{context}.iteration") != 0:
        raise ExportValidationError(f"{context}.iteration must be 0")
    if payload.get("kind") != "baseline" or payload.get("status") != "candidate_ready":
        raise ExportValidationError(f"{context} must be an immutable candidate-ready baseline")
    for key in ("parent_iteration", "parent_policy_sha256", "prompt_sha256"):
        if payload.get(key) is not None:
            raise ExportValidationError(f"{context}.{key} must be null")
    for key in ("codex_package", "codex_cli_version", "model", "reasoning_effort", "returncode"):
        if payload.get(key) is not None:
            raise ExportValidationError(f"{context}.{key} must be null")
    if (
        require_bool(payload.get("timed_out"), f"{context}.timed_out") is not False
        or require_bool(payload.get("changed"), f"{context}.changed") is not False
        or require_bool(payload.get("valid"), f"{context}.valid") is not True
        or payload.get("validation_error") is not None
    ):
        raise ExportValidationError(f"{context} baseline status fields are inconsistent")
    require_string(payload.get("created_at"), f"{context}.created_at")
    require_string(payload.get("completed_at"), f"{context}.completed_at")
    recorded_candidate = _record_hash(
        payload,
        ("candidate_policy_sha256", "policy_sha256"),
        context,
    )
    recorded_seed = require_sha256(
        payload.get("seed_policy_sha256"), f"{context}.seed_policy_sha256"
    )
    if (
        recorded_candidate != candidate_sha256
        or candidate_sha256 != lane.seed_policy_sha256
        or recorded_seed != lane.seed_policy_sha256
    ):
        raise ExportValidationError(
            f"{context} candidate, manifest seed, and baseline record hashes must agree"
        )


def validate_baseline_result(candidate: Candidate) -> None:
    context = f"iteration 0/{candidate.lane.lane_id} preserved seed replay"
    expectation = candidate.lane.seed_result_expectation
    if expectation is None:
        raise ExportValidationError(f"{context} has no fixed seed-result expectation")
    if candidate.total_t_gate_count != expectation["total_t_gate_count"]:
        raise ExportValidationError(f"{context} total T count drifted")
    for field in (
        "signed_reaction_error_hartree",
        "absolute_reaction_error_hartree",
        "triangle_bound_hartree",
        "max_species_absolute_error_hartree",
        "equal_share_error_limit_hartree",
    ):
        actual = require_finite(candidate.result.get(field), f"{context}.{field}")
        expected = float(expectation[field])
        if not math.isclose(actual, expected, rel_tol=SEED_REL_TOL, abs_tol=SEED_ABS_TOL):
            raise ExportValidationError(
                f"{context}.{field}={actual!r} does not reproduce fixed seed {expected!r}"
            )
    expected_species = require_mapping(expectation.get("species"), f"{context}.species")
    for species in candidate.lane.species:
        expected = require_mapping(expected_species.get(species), f"{context}.species.{species}")
        actual = require_mapping(
            candidate.result["species"].get(species), f"{context}.result.species.{species}"
        )
        if (
            require_int(
                actual.get("formula_order"),
                f"{context}.result.species.{species}.formula_order",
            )
            != expectation["formula_order"]
            or require_int(actual.get("n_steps"), f"{context}.result.species.{species}.n_steps")
            != expectation["n_steps"]
        ):
            raise ExportValidationError(f"{context} PF order/step drifted for {species}")
        if require_int(
            actual.get("t_gate_count"), f"{context}.result.species.{species}.t_gate_count"
        ) != require_int(expected.get("t_gate_count"), f"{context}.species.{species}.t_gate_count"):
            raise ExportValidationError(f"{context} species T count drifted for {species}")
        actual_error = require_finite(
            actual.get("signed_energy_error_hartree"),
            f"{context}.result.species.{species}.signed_energy_error_hartree",
        )
        expected_error = require_finite(
            expected.get("signed_energy_error_hartree"),
            f"{context}.species.{species}.signed_energy_error_hartree",
        )
        if not math.isclose(
            actual_error,
            expected_error,
            rel_tol=SEED_REL_TOL,
            abs_tol=SEED_ABS_TOL,
        ):
            raise ExportValidationError(f"{context} species error drifted for {species}")


def _optional_decision_hash(
    decision: Mapping[str, Any], key: str, expected: str, context: str
) -> None:
    if key in decision and require_sha256(decision[key], f"{context}.{key}") != expected:
        raise ExportValidationError(f"{context}.{key} does not match replay")


def validate_decision(
    decision: Mapping[str, Any],
    *,
    candidate: Candidate,
    incumbent: Candidate | None,
    expected_accept: bool,
) -> None:
    context = f"iteration {candidate.iteration}/{candidate.lane.lane_id} decision"
    if decision.get("lane_id") != candidate.lane.lane_id:
        raise ExportValidationError(f"{context}.lane_id mismatch")
    if (
        "iteration" in decision
        and require_int(decision["iteration"], f"{context}.iteration") != candidate.iteration
    ):
        raise ExportValidationError(f"{context}.iteration mismatch")
    if require_bool(decision.get("accepted"), f"{context}.accepted") is not expected_accept:
        raise ExportValidationError(f"{context}.accepted does not match strict replay")
    if candidate.iteration == 0:
        expected_reason = "initial_baseline"
    elif not candidate.passes_contract:
        expected_reason = "contract_failed"
    elif expected_accept:
        expected_reason = "strict_t_gate_improvement"
    else:
        expected_reason = "no_t_gate_improvement"
    for field in ("reason", "decision_reason"):
        if require_string(decision.get(field), f"{context}.{field}") != expected_reason:
            raise ExportValidationError(f"{context}.{field} does not match strict replay")
    if (
        require_bool(decision.get("passes_contract"), f"{context}.passes_contract")
        is not candidate.passes_contract
    ):
        raise ExportValidationError(f"{context}.passes_contract does not match recomputation")
    if (
        require_int(decision.get("candidate_t_gate_count"), f"{context}.candidate_t_gate_count")
        != candidate.total_t_gate_count
    ):
        raise ExportValidationError(f"{context}.candidate_t_gate_count does not match result")

    previous_t = None if incumbent is None else incumbent.total_t_gate_count
    previous_iteration = None if incumbent is None else incumbent.iteration
    previous_sha256 = None if incumbent is None else incumbent.policy_sha256
    if (
        require_optional_int(
            decision.get("previous_best_t_gate_count"), f"{context}.previous_best_t_gate_count"
        )
        != previous_t
    ):
        raise ExportValidationError(f"{context}.previous_best_t_gate_count does not match replay")
    if (
        "parent_iteration" in decision
        and require_optional_int(decision["parent_iteration"], f"{context}.parent_iteration")
        != previous_iteration
    ):
        raise ExportValidationError(f"{context}.parent_iteration does not match replay")
    if "parent_policy_sha256" in decision:
        actual_parent = decision["parent_policy_sha256"]
        if previous_sha256 is None:
            if actual_parent is not None:
                raise ExportValidationError(f"{context}.parent_policy_sha256 must be null")
        elif require_sha256(actual_parent, f"{context}.parent_policy_sha256") != previous_sha256:
            raise ExportValidationError(f"{context}.parent_policy_sha256 does not match replay")
    _optional_decision_hash(decision, "candidate_policy_sha256", candidate.policy_sha256, context)
    if "result_sha256" in decision:
        # The caller annotates this private value after reading the result file.
        result_sha256 = require_sha256(decision["result_sha256"], f"{context}.result_sha256")
        expected_result_sha256 = require_string(
            candidate.result["_validated_result_sha256"], f"{context} result digest"
        )
        if result_sha256 != expected_result_sha256:
            raise ExportValidationError(f"{context}.result_sha256 does not match the result file")

    selected = candidate if expected_accept else incumbent
    if selected is None:
        raise ExportValidationError(f"{context} has no incumbent after baseline replay")
    if require_int(decision.get("best_t_gate_count"), f"{context}.best_t_gate_count") != (
        selected.total_t_gate_count
    ):
        raise ExportValidationError(f"{context}.best_t_gate_count does not match replay")
    if (
        "best_iteration" in decision
        and require_int(decision["best_iteration"], f"{context}.best_iteration")
        != selected.iteration
    ):
        raise ExportValidationError(f"{context}.best_iteration does not match replay")
    _optional_decision_hash(decision, "best_policy_sha256", selected.policy_sha256, context)


def validate_iteration_directories(iterations_dir: Path) -> None:
    require_exact_directory(
        iterations_dir,
        context="campaign iterations root",
        directories={f"iter_{iteration:04d}" for iteration in PAPER_EVALUATION_ITERATIONS},
    )


def index_state_batches(state: object) -> dict[int, dict[str, dict[str, Any]]]:
    payload = require_mapping(state, "state.json")
    raw_batches = require_mapping(payload.get("batches"), "state.json.batches")
    indexed: dict[int, dict[str, dict[str, Any]]] = {
        iteration: {} for iteration in PAPER_EVALUATION_ITERATIONS
    }
    seen_submission_keys: set[str] = set()
    seen_job_names: set[str] = set()
    seen_job_ids: set[str] = set()
    for state_key, raw_batch in raw_batches.items():
        context = f"state.json.batches.{state_key}"
        batch = require_mapping(raw_batch, context)
        batch_id = require_string(batch.get("batch_id"), f"{context}.batch_id")
        if state_key != batch_id:
            raise ExportValidationError(f"{context}.batch_id differs from its state key")
        match = BATCH_ID_RE.fullmatch(batch_id)
        if match is None:
            raise ExportValidationError(f"{context} has an invalid runner batch ID")
        iteration = require_int(batch.get("iteration"), f"{context}.iteration")
        if iteration not in PAPER_EVALUATION_ITERATIONS or int(match.group(1)) != iteration:
            raise ExportValidationError(f"{context} has an invalid iteration identity")
        for attempt_index, raw_attempt in enumerate(
            require_list(batch.get("attempts"), f"{context}.attempts")
        ):
            attempt_context = f"{context}.attempts[{attempt_index}]"
            attempt = require_mapping(raw_attempt, attempt_context)
            submission_key = require_sha256(
                attempt.get("submission_key"), f"{attempt_context}.submission_key"
            )
            job_name = require_string(attempt.get("job_name"), f"{attempt_context}.job_name")
            job_id = require_string(attempt.get("job_id"), f"{attempt_context}.job_id")
            if re.fullmatch(r"[0-9]+", job_id) is None:
                raise ExportValidationError(f"{attempt_context}.job_id is not numeric")
            if (
                submission_key in seen_submission_keys
                or job_name in seen_job_names
                or job_id in seen_job_ids
            ):
                raise ExportValidationError(
                    f"{attempt_context} reuses a submission key, job name, or job receipt ID"
                )
            seen_submission_keys.add(submission_key)
            seen_job_names.add(job_name)
            seen_job_ids.add(job_id)
        indexed[iteration][batch_id] = batch
    missing = [iteration for iteration, batches in indexed.items() if not batches]
    if missing:
        raise ExportValidationError(
            f"fully collected campaign has no persisted batches for iterations {missing}"
        )
    return indexed


def _initial_batch_partition(lane_ids: Sequence[str], batch_size: int) -> list[list[str]]:
    batch_count = max(1, math.ceil(len(lane_ids) / batch_size))
    return [list(lane_ids[index::batch_count]) for index in range(batch_count)]


def _submission_identity(
    *,
    campaign_id: str,
    manifest_sha256: str,
    cluster_root: str,
    iteration: int,
    batch_id: str,
    attempt: int,
    lane_ids: Sequence[str],
    iteration_root: Path,
) -> tuple[str, str]:
    identity = {
        "campaign_id": campaign_id,
        "manifest_sha256": manifest_sha256,
        "cluster_root": cluster_root,
        "iteration": iteration,
        "batch_id": batch_id,
        "attempt": attempt,
        "candidates": [
            {
                "lane_id": lane_id,
                "policy_sha256": sha256_file(
                    iteration_root / "candidates" / lane_id / "simple_pf.py"
                ),
            }
            for lane_id in lane_ids
        ],
    }
    digest = hashlib.sha256(
        SUBMISSION_ID_DOMAIN
        + json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, f"rx_{digest[:32]}"


def validate_batch_structure(
    iteration_root: Path,
    *,
    iteration: int,
    batches: Mapping[str, Mapping[str, Any]],
    lane_ids: Sequence[str],
    campaign_id: str,
    manifest_sha256: str,
    cluster_root: str,
    cluster_python: str,
    batch_size: int,
) -> None:
    batches_root = iteration_root / "batches"
    expected_partition = _initial_batch_partition(lane_ids, batch_size)
    expected_batches = {
        f"iter{iteration:04d}_batch{index:02d}": group
        for index, group in enumerate(expected_partition)
    }
    if set(batches) != set(expected_batches):
        raise ExportValidationError(
            f"iteration {iteration} batch IDs differ from the deterministic partition"
        )
    require_exact_directory(
        batches_root,
        context=f"iteration {iteration} batches root",
        directories=set(batches),
    )
    for batch_id, raw_batch in batches.items():
        context = f"iteration {iteration} batch {batch_id}"
        batch = require_mapping(raw_batch, context)
        required_batch_fields = {
            "batch_id",
            "iteration",
            "campaign_id",
            "manifest_sha256",
            "lane_ids",
            "state",
            "retry_lane_ids",
            "retry_count",
            "attempts",
            "created_at",
            "updated_at",
            "collected_at",
        }
        allowed_batch_fields = required_batch_fields | {"retry_fingerprint"}
        if not required_batch_fields.issubset(batch) or set(batch) - allowed_batch_fields:
            raise ExportValidationError(f"{context} has an unexpected final schema")
        batch_lane_ids = [
            require_string(value, f"{context}.lane_ids")
            for value in require_list(batch.get("lane_ids"), f"{context}.lane_ids")
        ]
        retry_count = require_int(batch.get("retry_count"), f"{context}.retry_count", minimum=0)
        if (
            batch.get("batch_id") != batch_id
            or require_int(batch.get("iteration"), f"{context}.iteration") != iteration
            or batch.get("campaign_id") != campaign_id
            or batch.get("manifest_sha256") != manifest_sha256
            or batch_lane_ids != expected_batches[batch_id]
            or batch.get("state") != "collected"
            or require_list(batch.get("retry_lane_ids"), f"{context}.retry_lane_ids")
            or retry_count < 0
        ):
            raise ExportValidationError(f"{context} final identity/partition/state mismatch")
        require_string(batch.get("created_at"), f"{context}.created_at")
        require_string(batch.get("updated_at"), f"{context}.updated_at")
        require_string(batch.get("collected_at"), f"{context}.collected_at")
        if retry_count:
            require_sha256(batch.get("retry_fingerprint"), f"{context}.retry_fingerprint")
        elif "retry_fingerprint" in batch:
            raise ExportValidationError(f"{context} has retry fingerprint without retries")
        attempts = require_list(batch.get("attempts"), f"{context}.attempts")
        if not attempts:
            raise ExportValidationError(f"{context} has no durable submission attempt")
        if retry_count != len(attempts) - 1:
            raise ExportValidationError(f"{context}.retry_count does not match attempt history")
        attempts_by_name: dict[str, dict[str, Any]] = {}
        previous_retry_lane_ids: list[str] | None = None
        for index, raw_attempt in enumerate(attempts, 1):
            attempt = require_mapping(raw_attempt, f"{context} attempt")
            number = require_int(attempt.get("attempt"), f"{context} attempt number", minimum=1)
            if number != index:
                raise ExportValidationError(f"{context} attempts are not sequential")
            name = f"attempt_{number:02d}"
            if name in attempts_by_name:
                raise ExportValidationError(f"{context} repeats attempt {number}")
            attempts_by_name[name] = attempt
            attempt_context = f"{context} {name}"
            required_attempt_fields = {
                "attempt",
                "campaign_id",
                "manifest_sha256",
                "lane_ids",
                "submission_key",
                "job_name",
                "lane_ids_path",
                "state",
                "job_id",
                "created_at",
                "submitted_at",
                "execution",
            }
            is_final = index == len(attempts)
            if is_final:
                required_attempt_fields.add("collected_at")
            else:
                required_attempt_fields.add("retry_reason")
            if set(attempt) != required_attempt_fields:
                raise ExportValidationError(f"{attempt_context} has an unexpected schema")
            active_lane_ids = [
                require_string(value, f"{attempt_context}.lane_ids")
                for value in require_list(attempt.get("lane_ids"), f"{attempt_context}.lane_ids")
            ]
            if (
                not active_lane_ids
                or len(active_lane_ids) != len(set(active_lane_ids))
                or not set(active_lane_ids).issubset(set(batch_lane_ids))
                or attempt.get("campaign_id") != campaign_id
                or attempt.get("manifest_sha256") != manifest_sha256
            ):
                raise ExportValidationError(f"{attempt_context} lane/campaign identity mismatch")
            if index == 1 and active_lane_ids != batch_lane_ids:
                raise ExportValidationError(
                    f"{attempt_context} first attempt is not the full batch"
                )
            if previous_retry_lane_ids is not None and active_lane_ids != previous_retry_lane_ids:
                raise ExportValidationError(f"{attempt_context} does not match prior retry reason")
            expected_key, expected_job = _submission_identity(
                campaign_id=campaign_id,
                manifest_sha256=manifest_sha256,
                cluster_root=cluster_root,
                iteration=iteration,
                batch_id=batch_id,
                attempt=number,
                lane_ids=active_lane_ids,
                iteration_root=iteration_root,
            )
            if (
                attempt.get("submission_key") != expected_key
                or attempt.get("job_name") != expected_job
                or re.fullmatch(
                    r"[0-9]+", require_string(attempt.get("job_id"), f"{attempt_context}.job_id")
                )
                is None
                or attempt.get("state") != ("collected" if is_final else "retry_ready")
            ):
                raise ExportValidationError(
                    f"{attempt_context} submission/job/final state mismatch"
                )
            require_string(attempt.get("created_at"), f"{attempt_context}.created_at")
            require_string(attempt.get("submitted_at"), f"{attempt_context}.submitted_at")
            if is_final:
                require_string(attempt.get("collected_at"), f"{attempt_context}.collected_at")
                previous_retry_lane_ids = None
            else:
                retry_reason = require_mapping(
                    attempt.get("retry_reason"), f"{attempt_context}.retry_reason"
                )
                if not retry_reason or not all(
                    isinstance(value, str) and value for value in retry_reason.values()
                ):
                    raise ExportValidationError(f"{attempt_context}.retry_reason is invalid")
                previous_retry_lane_ids = [
                    lane_id for lane_id in batch_lane_ids if lane_id in retry_reason
                ]
                if set(previous_retry_lane_ids) != set(retry_reason):
                    raise ExportValidationError(f"{attempt_context}.retry_reason lanes are invalid")
            execution = require_mapping(attempt.get("execution"), f"{attempt_context}.execution")
            expected_execution_fields = {
                "sync_host",
                "submit_host",
                "account",
                "partition",
                "workers",
                "cpus_per_task",
                "memory",
                "time_limit",
                "cluster_root",
                "cluster_python",
            }
            if set(execution) != expected_execution_fields:
                raise ExportValidationError(f"{attempt_context}.execution has unexpected schema")
            for field in (
                "sync_host",
                "submit_host",
                "account",
                "partition",
                "memory",
                "time_limit",
            ):
                require_string(execution.get(field), f"{attempt_context}.execution.{field}")
            if (
                require_int(
                    execution.get("workers"), f"{attempt_context}.execution.workers", minimum=1
                )
                < 1
                or require_int(
                    execution.get("cpus_per_task"),
                    f"{attempt_context}.execution.cpus_per_task",
                    minimum=1,
                )
                < 1
                or execution.get("cluster_root") != cluster_root
                or execution.get("cluster_python") != cluster_python
            ):
                raise ExportValidationError(f"{attempt_context}.execution identity mismatch")
        batch_root = batches_root / batch_id
        require_exact_directory(
            batch_root,
            context=context,
            files={"batch.json"},
            directories=set(attempts_by_name),
        )
        if read_json(batch_root / "batch.json") != batch:
            raise ExportValidationError(f"{context} mirror differs from state.json")
        for attempt_name, attempt in attempts_by_name.items():
            attempt_context = f"{context} {attempt_name}"
            attempt_root = batch_root / attempt_name
            require_exact_directory(
                attempt_root,
                context=attempt_context,
                files={"lane_ids.txt"},
            )
            expected_relative = (
                Path("iterations")
                / f"iter_{iteration:04d}"
                / "batches"
                / batch_id
                / attempt_name
                / "lane_ids.txt"
            )
            recorded_relative = _safe_relative_path(
                attempt.get("lane_ids_path"), f"{attempt_context}.lane_ids_path"
            )
            if recorded_relative != expected_relative:
                raise ExportValidationError(f"{attempt_context}.lane_ids_path changed")
            expected_lane_ids = [
                require_string(value, f"{attempt_context}.lane_ids")
                for value in require_list(attempt.get("lane_ids"), f"{attempt_context}.lane_ids")
            ]
            if (attempt_root / "lane_ids.txt").read_text(encoding="utf-8").splitlines() != (
                expected_lane_ids
            ):
                raise ExportValidationError(f"{attempt_context} lane_ids.txt differs from state")


def validate_iteration_structure(
    iteration_root: Path,
    *,
    iteration: int,
    lane_ids: Sequence[str],
    batches: Mapping[str, Mapping[str, Any]],
    campaign_id: str,
    manifest_sha256: str,
    cluster_root: str,
    cluster_python: str,
    batch_size: int,
) -> None:
    lane_id_set = set(lane_ids)
    require_exact_directory(
        iteration_root,
        context=f"iteration {iteration} root",
        files=ITERATION_FILES,
        directories=ITERATION_DIRECTORIES,
    )
    require_exact_directory(
        iteration_root / "candidates",
        context=f"iteration {iteration} candidates root",
        directories=lane_id_set,
    )
    for lane_id in lane_ids:
        require_exact_directory(
            iteration_root / "candidates" / lane_id,
            context=f"iteration {iteration}/{lane_id} candidate root",
            files={"simple_pf.py"},
        )
    require_exact_directory(
        iteration_root / "results",
        context=f"iteration {iteration} results root",
        files={f"{lane_id}.json" for lane_id in lane_ids},
    )
    require_exact_directory(
        iteration_root / "mutations",
        context=f"iteration {iteration} mutations root",
        directories=lane_id_set,
    )
    validation = require_mapping(
        read_json(iteration_root / "validation.json"),
        f"iteration {iteration} validation report",
    )
    if (
        require_int(
            validation.get("schema_version"), f"iteration {iteration} validation.schema_version"
        )
        != 1
        or require_int(validation.get("iteration"), f"iteration {iteration} validation.iteration")
        != iteration
        or require_int(
            validation.get("validated_lane_count"),
            f"iteration {iteration} validation.validated_lane_count",
        )
        != PAPER_LANE_COUNT
        or require_mapping(
            validation.get("retryable_errors"),
            f"iteration {iteration} validation.retryable_errors",
        )
        or require_mapping(
            validation.get("fatal_errors"), f"iteration {iteration} validation.fatal_errors"
        )
    ):
        raise ExportValidationError(f"iteration {iteration} validation report is not complete")
    require_string(validation.get("validated_at"), f"iteration {iteration} validation.validated_at")
    validate_batch_structure(
        iteration_root,
        iteration=iteration,
        batches=batches,
        lane_ids=lane_ids,
        campaign_id=campaign_id,
        manifest_sha256=manifest_sha256,
        cluster_root=cluster_root,
        cluster_python=cluster_python,
        batch_size=batch_size,
    )


def _exact_tree_entries(base: Path, context: str) -> tuple[dict[str, Path], set[str]]:
    if not base.is_dir() or base.is_symlink():
        raise ExportValidationError(f"{context} must be a real directory: {base}")
    files: dict[str, Path] = {}
    directories: set[str] = set()

    def walk(directory: Path, relative_root: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = relative_root / entry.name
                relative_name = relative.as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ExportValidationError(f"{context} contains a symlink: {relative_name}")
                if stat.S_ISDIR(mode):
                    directories.add(relative_name)
                    walk(Path(entry.path), relative)
                elif stat.S_ISREG(mode):
                    files[relative_name] = Path(entry.path)
                else:
                    raise ExportValidationError(
                        f"{context} contains a special entry: {relative_name}"
                    )

    walk(base, Path())
    return files, directories


def _validated_directory_paths(value: object, context: str) -> list[str]:
    paths = [
        _safe_relative_path(raw, f"{context}[{index}]").as_posix()
        for index, raw in enumerate(require_list(value, context))
    ]
    if paths != sorted(set(paths)):
        raise ExportValidationError(f"{context} must be sorted and duplicate-free")
    return paths


def validate_recorded_file_tree(
    base: Path,
    *,
    records: object,
    directory_paths: object,
    context: str,
) -> set[str]:
    payload = require_mapping(records, f"{context} file records")
    declared_files: set[str] = set()
    for relative_text, raw_record in payload.items():
        relative = _safe_relative_path(relative_text, f"{context} file-record key")
        relative_name = relative.as_posix()
        record = require_mapping(raw_record, f"{context}.{relative_name}")
        if set(record) != {"sha256", "size_bytes"}:
            raise ExportValidationError(f"{context}.{relative_name} has an unexpected schema")
        _validate_file_record(base / relative, record, f"{context}.{relative_name}")
        declared_files.add(relative_name)
    declared_directories = set(
        _validated_directory_paths(directory_paths, f"{context} directory paths")
    )
    actual_files, actual_directories = _exact_tree_entries(base, context)
    if set(actual_files) != declared_files or actual_directories != declared_directories:
        raise ExportValidationError(
            f"{context} differs from its recorded file tree: "
            f"missing_files={sorted(declared_files - set(actual_files))}, "
            f"extra_files={sorted(set(actual_files) - declared_files)}, "
            f"missing_directories={sorted(declared_directories - actual_directories)}, "
            f"extra_directories={sorted(actual_directories - declared_directories)}"
        )
    return declared_files


def validate_archived_attempt_evidence(
    mutation_root: Path,
    *,
    record: Mapping[str, Any],
    context: str,
) -> bool:
    raw_evidence = require_list(
        record.get("archived_attempt_evidence"),
        f"{context}.archived_attempt_evidence",
    )
    archive_root = mutation_root / "archived_attempts"
    if not raw_evidence:
        if record.get("archived_attempt_directory_paths") != [] or archive_root.exists():
            raise ExportValidationError(f"{context} has unrecorded archived attempt state")
        return False

    records: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    nonce_names: set[str] = set()
    for index, raw_item in enumerate(raw_evidence):
        item_context = f"{context}.archived_attempt_evidence[{index}]"
        item = require_mapping(raw_item, item_context)
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ExportValidationError(f"{item_context} has an unexpected schema")
        full_relative = _safe_relative_path(item.get("path"), f"{item_context}.path")
        if (
            len(full_relative.parts) < 3
            or full_relative.parts[0] != "archived_attempts"
            or SHA256_RE.fullmatch(full_relative.parts[1]) is None
        ):
            raise ExportValidationError(f"{item_context}.path is not nonce-scoped archive evidence")
        archive_relative = Path(*full_relative.parts[1:])
        archive_name = archive_relative.as_posix()
        if archive_name in records:
            raise ExportValidationError(f"{item_context} repeats an archive file")
        records[archive_name] = {
            "sha256": require_sha256(item.get("sha256"), f"{item_context}.sha256"),
            "size_bytes": require_int(
                item.get("size_bytes"), f"{item_context}.size_bytes", minimum=0
            ),
        }
        ordered_paths.append(full_relative.as_posix())
        nonce_names.add(full_relative.parts[1])
    if ordered_paths != sorted(ordered_paths):
        raise ExportValidationError(f"{context}.archived_attempt_evidence is not sorted")

    prefixed_directories = _validated_directory_paths(
        record.get("archived_attempt_directory_paths"),
        f"{context}.archived_attempt_directory_paths",
    )
    archive_directories: list[str] = []
    directory_nonces: set[str] = set()
    for relative_text in prefixed_directories:
        relative = Path(relative_text)
        if (
            len(relative.parts) < 2
            or relative.parts[0] != "archived_attempts"
            or SHA256_RE.fullmatch(relative.parts[1]) is None
        ):
            raise ExportValidationError(f"{context} archived directory is not archive-scoped")
        directory_nonces.add(relative.parts[1])
        archive_directories.append(Path(*relative.parts[1:]).as_posix())
    if directory_nonces != nonce_names or any(
        nonce not in archive_directories for nonce in nonce_names
    ):
        raise ExportValidationError(f"{context} archived nonce directory closure is invalid")
    validate_recorded_file_tree(
        archive_root,
        records=records,
        directory_paths=archive_directories,
        context=f"{context} archived attempt tree",
    )
    for nonce in nonce_names:
        failure_path = archive_root / nonce / "failure.json"
        if failure_path.relative_to(archive_root).as_posix() not in records:
            raise ExportValidationError(f"{context} archived attempt {nonce} lacks failure.json")
        failure = require_mapping(read_json(failure_path), f"{context} archive {nonce} failure")
        if set(failure) != {"schema_version", "launch_nonce", "reason", "archived_at"}:
            raise ExportValidationError(f"{context} archive {nonce} failure schema changed")
        if (
            require_int(failure.get("schema_version"), f"{context} archive schema") != 1
            or failure.get("launch_nonce") != nonce
        ):
            raise ExportValidationError(f"{context} archive {nonce} identity mismatch")
        require_string(failure.get("reason"), f"{context} archive {nonce}.reason")
        require_string(failure.get("archived_at"), f"{context} archive {nonce}.archived_at")
    return True


def validate_mutation_directory_structure(
    mutation_root: Path,
    *,
    iteration: int,
    lane_id: str,
    record: Mapping[str, Any],
) -> None:
    context = f"iteration {iteration}/{lane_id} mutation root"
    if iteration == 0:
        require_exact_directory(mutation_root, context=context, files={"record.json"})
        return

    has_archives = validate_archived_attempt_evidence(mutation_root, record=record, context=context)
    required_files = {
        "codex.stderr",
        "codex.stdout",
        "intent.json",
        "last_message.txt",
        "launch.json",
        "launch_intent.json",
        "parent_simple_pf.py",
        "process.json",
        "prompt.md",
        "record.json",
    }
    require_exact_directory(
        mutation_root,
        context=context,
        files=required_files,
        directories={"work", *({"archived_attempts"} if has_archives else set())},
    )
    raw_unexpected = require_list(
        record.get("unexpected_work_files"), f"{context}.unexpected_work_files"
    )
    unexpected: list[str] = []
    for index, value in enumerate(raw_unexpected):
        relative = _safe_relative_path(value, f"{context}.unexpected_work_files[{index}]")
        name = relative.as_posix()
        if name == "simple_pf.py":
            raise ExportValidationError(f"{context} records its required policy as unexpected")
        unexpected.append(name)
    if unexpected != sorted(set(unexpected)):
        raise ExportValidationError(
            f"{context}.unexpected_work_files must be sorted and duplicate-free"
        )
    declared_work_files = validate_recorded_file_tree(
        mutation_root / "work",
        records=record.get("work_file_records"),
        directory_paths=record.get("work_directory_paths"),
        context=f"{context} work tree",
    )
    if "simple_pf.py" not in declared_work_files:
        raise ExportValidationError(f"{context} work tree has no simple_pf.py")
    if set(unexpected) != declared_work_files - {"simple_pf.py"}:
        raise ExportValidationError(f"{context}.unexpected_work_files differs from work records")
    if require_sha256(
        record.get("work_policy_sha256"), f"{context}.work_policy_sha256"
    ) != sha256_file(mutation_root / "work" / "simple_pf.py"):
        raise ExportValidationError(f"{context} work policy hash mismatch")


def validate_hamiltonian_verification(
    verification_path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    campaign_root: Path,
) -> None:
    context = "baselines/hamiltonian_verification.json"
    payload = require_mapping(read_json(verification_path), context)
    expected_top_fields = {
        "schema_version",
        "experiment",
        "diagonalization",
        "pauli_dense_check",
        "hamiltonian_count",
        "all_verified",
        "manifest_sha256",
        "verified_at",
        "rows",
    }
    if set(payload) != expected_top_fields:
        raise ExportValidationError(f"{context} has an unexpected schema")
    if (
        require_int(payload.get("schema_version"), f"{context}.schema_version") != 1
        or payload.get("experiment") != "reactions10_hamiltonian_cache_verification"
        or payload.get("diagonalization")
        != "numpy.linalg.eigh on fixed-electron Hamiltonian sectors"
        or payload.get("pauli_dense_check")
        != (
            "identity_coefficient plus cached evaluator Pauli labels/coefficients "
            "against the cached dense Hamiltonian"
        )
        or require_bool(payload.get("all_verified"), f"{context}.all_verified") is not True
        or require_int(payload.get("hamiltonian_count"), f"{context}.hamiltonian_count") != 21
        or require_sha256(payload.get("manifest_sha256"), f"{context}.manifest_sha256")
        != manifest_sha256
    ):
        raise ExportValidationError(f"{context} is incomplete or stale")
    require_string(payload.get("verified_at"), f"{context}.verified_at")

    inputs = require_mapping(manifest.get("inputs"), "manifest.json.inputs")
    cache_records = require_mapping(
        inputs.get("problem_cache_files"), "manifest.json.inputs.problem_cache_files"
    )
    expected_npz = {name for name in cache_records if name.endswith(".npz")}
    if len(expected_npz) != 21:
        raise ExportValidationError("manifest does not contain exactly 21 Hamiltonian NPZ files")
    rows = require_list(payload.get("rows"), f"{context}.rows")
    if len(rows) != 21:
        raise ExportValidationError(f"{context}.rows must contain exactly 21 entries")
    expected_row_fields = {
        "species",
        "cache_file",
        "cache_sha256",
        "metadata_sha256",
        "n_qubits",
        "active_electrons",
        "sector_dimension",
        "hermiticity_error",
        "pauli_dense_max_abs_residual_hartree",
        "pauli_dense_relative_frobenius_residual",
        "cached_ground_energy_hartree",
        "recomputed_ground_energy_hartree",
        "ground_energy_delta_hartree",
        "max_spectrum_delta_hartree",
        "cached_ground_vector_residual",
        "evaluator_ground_state_input_norm",
        "evaluator_ground_state_outside_sector_norm",
        "evaluator_ground_state_residual",
        "ground_eigenspace_dimension",
        "evaluator_recomputed_ground_eigenspace_overlap",
        "evaluator_cached_ground_eigenspace_overlap",
        "verified",
        "wall_seconds",
    }
    seen_cache_files: set[str] = set()
    seen_species: set[str] = set()
    species_order: list[str] = []
    for index, raw_row in enumerate(rows):
        row_context = f"{context}.rows[{index}]"
        row = require_mapping(raw_row, row_context)
        if set(row) != expected_row_fields:
            raise ExportValidationError(f"{row_context} has an unexpected schema")
        cache_file = _safe_relative_path(row.get("cache_file"), f"{row_context}.cache_file")
        if len(cache_file.parts) != 1 or cache_file.suffix != ".npz":
            raise ExportValidationError(f"{row_context}.cache_file is not a bare NPZ filename")
        cache_name = cache_file.name
        if cache_name in seen_cache_files or cache_name not in expected_npz:
            raise ExportValidationError(f"{row_context}.cache_file is duplicated or undeclared")
        metadata_name = cache_file.with_suffix(".json").name
        expected_cache_hash = require_sha256(
            cache_records.get(cache_name), f"manifest cache hash {cache_name}"
        )
        expected_metadata_hash = require_sha256(
            cache_records.get(metadata_name), f"manifest cache hash {metadata_name}"
        )
        cache_path = campaign_root / "problem_cache" / cache_name
        metadata_path = campaign_root / "problem_cache" / metadata_name
        if (
            not cache_path.is_file()
            or cache_path.is_symlink()
            or not metadata_path.is_file()
            or metadata_path.is_symlink()
            or require_sha256(row.get("cache_sha256"), f"{row_context}.cache_sha256")
            != expected_cache_hash
            or require_sha256(row.get("metadata_sha256"), f"{row_context}.metadata_sha256")
            != expected_metadata_hash
            or sha256_file(cache_path) != expected_cache_hash
            or sha256_file(metadata_path) != expected_metadata_hash
        ):
            raise ExportValidationError(f"{row_context} cache/metadata hash mismatch")
        metadata = require_mapping(read_json(metadata_path), f"cache metadata {metadata_name}")
        species = require_string(row.get("species"), f"{row_context}.species")
        if species != metadata.get("molecule") or species in seen_species:
            raise ExportValidationError(f"{row_context}.species is duplicated or mismatched")
        n_qubits = require_int(row.get("n_qubits"), f"{row_context}.n_qubits", minimum=1)
        active_electrons = require_int(
            row.get("active_electrons"), f"{row_context}.active_electrons", minimum=1
        )
        if (
            n_qubits > 10
            or n_qubits != require_int(metadata.get("n_qubits"), f"{metadata_name}.n_qubits")
            or active_electrons
            != require_int(metadata.get("active_electrons"), f"{metadata_name}.active_electrons")
        ):
            raise ExportValidationError(f"{row_context} cache identity fields mismatch")
        sector_dimension = require_int(
            row.get("sector_dimension"), f"{row_context}.sector_dimension", minimum=1
        )
        ground_dimension = require_int(
            row.get("ground_eigenspace_dimension"),
            f"{row_context}.ground_eigenspace_dimension",
            minimum=1,
        )
        if (
            active_electrons > n_qubits
            or sector_dimension != math.comb(n_qubits, active_electrons)
            or ground_dimension > sector_dimension
        ):
            raise ExportValidationError(f"{row_context} sector/eigenspace dimensions are invalid")
        for field, threshold in HAMILTONIAN_THRESHOLDS.items():
            value = require_finite(row.get(field), f"{row_context}.{field}")
            if value < 0.0 or value > threshold:
                raise ExportValidationError(f"{row_context}.{field} exceeds {threshold}")
        ground_delta = require_finite(
            row.get("ground_energy_delta_hartree"),
            f"{row_context}.ground_energy_delta_hartree",
        )
        if abs(ground_delta) > 1.0e-10:
            raise ExportValidationError(f"{row_context}.ground_energy_delta_hartree is excessive")
        cached_ground = require_finite(
            row.get("cached_ground_energy_hartree"),
            f"{row_context}.cached_ground_energy_hartree",
        )
        recomputed_ground = require_finite(
            row.get("recomputed_ground_energy_hartree"),
            f"{row_context}.recomputed_ground_energy_hartree",
        )
        require_close(
            recomputed_ground - cached_ground,
            ground_delta,
            f"{row_context} ground-energy arithmetic",
        )
        input_norm = require_finite(
            row.get("evaluator_ground_state_input_norm"),
            f"{row_context}.evaluator_ground_state_input_norm",
        )
        if abs(input_norm - 1.0) > 1.0e-10:
            raise ExportValidationError(f"{row_context} evaluator ground-state norm drift")
        for field in (
            "evaluator_recomputed_ground_eigenspace_overlap",
            "evaluator_cached_ground_eigenspace_overlap",
        ):
            overlap = require_finite(row.get(field), f"{row_context}.{field}")
            if overlap < 1.0 - 1.0e-8 or overlap > 1.0 + 1.0e-8:
                raise ExportValidationError(f"{row_context}.{field} is invalid")
        if require_bool(row.get("verified"), f"{row_context}.verified") is not True:
            raise ExportValidationError(f"{row_context} is not verified")
        if require_finite(row.get("wall_seconds"), f"{row_context}.wall_seconds") < 0.0:
            raise ExportValidationError(f"{row_context}.wall_seconds is negative")
        seen_cache_files.add(cache_name)
        seen_species.add(species)
        species_order.append(species)
    if seen_cache_files != expected_npz:
        raise ExportValidationError(f"{context} does not cover the fixed 21 Hamiltonians")
    if species_order != sorted(species_order):
        raise ExportValidationError(f"{context}.rows are not in deterministic species order")


def validate_state(
    state: object,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    lanes: Mapping[str, LaneSpec],
    incumbents: Mapping[str, Candidate],
    campaign_root: Path,
) -> None:
    payload = require_mapping(state, "state.json")
    expected_state_fields = {
        "schema_version",
        "experiment",
        "manifest_sha256",
        "campaign_id",
        "created_at",
        "updated_at",
        "current_collected_iteration",
        "next_iteration",
        "lanes",
        "batches",
        "last_validation_errors",
        "hamiltonian_verification_sha256",
    }
    if set(payload) != expected_state_fields:
        raise ExportValidationError("state.json has an unexpected final schema")
    if (
        require_int(payload.get("schema_version"), "state.json.schema_version") != 1
        or payload.get("experiment") != "reactions10_autoresearch"
    ):
        raise ExportValidationError("state.json schema/experiment mismatch")
    require_string(payload.get("created_at"), "state.json.created_at")
    require_string(payload.get("updated_at"), "state.json.updated_at")
    campaign_ids = {lane.campaign_id for lane in lanes.values()}
    if len(campaign_ids) != 1 or None in campaign_ids:
        raise ExportValidationError("manifest lanes do not share one fixed campaign identity")
    campaign_id = next(iter(campaign_ids))
    if payload.get("campaign_id") != campaign_id:
        raise ExportValidationError("state.json.campaign_id does not match manifest.json")
    if (
        require_sha256(payload.get("manifest_sha256"), "state.json.manifest_sha256")
        != manifest_sha256
    ):
        raise ExportValidationError("state.json.manifest_sha256 does not match manifest.json")
    if (
        require_int(
            payload.get("current_collected_iteration"),
            "state.json.current_collected_iteration",
        )
        != 50
        or payload.get("next_iteration") is not None
        or require_mapping(
            payload.get("last_validation_errors"), "state.json.last_validation_errors"
        )
    ):
        raise ExportValidationError("state.json is not in the exact completed state")

    verification_path = campaign_root / "baselines" / "hamiltonian_verification.json"
    validate_hamiltonian_verification(
        verification_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        campaign_root=campaign_root,
    )
    if require_sha256(
        payload.get("hamiltonian_verification_sha256"),
        "state.json.hamiltonian_verification_sha256",
    ) != sha256_file(verification_path):
        raise ExportValidationError("state.json does not bind the Hamiltonian verification report")

    state_lanes = require_mapping(payload.get("lanes"), "state.json.lanes")
    if set(state_lanes) != set(lanes):
        raise ExportValidationError("state.json lane set differs from manifest")
    final_decisions = normalize_decisions(
        read_json(
            campaign_root
            / "iterations"
            / f"iter_{PAPER_PROPOSAL_ITERATIONS:04d}"
            / "decisions.json"
        ),
        PAPER_PROPOSAL_ITERATIONS,
    )
    final_decisions_by_lane = {
        require_string(decision.get("lane_id"), "iteration 50 final decision.lane_id"): decision
        for decision in final_decisions
    }
    if len(final_decisions_by_lane) != PAPER_LANE_COUNT or set(final_decisions_by_lane) != set(
        lanes
    ):
        raise ExportValidationError("iteration 50 final decision lane set is invalid")

    for lane_id in sorted(lanes):
        context = f"state.json.lanes.{lane_id}"
        lane_state = require_mapping(state_lanes[lane_id], context)
        expected_lane_state_fields = {
            "last_collected_iteration",
            "proposal_count",
            "incumbent_iteration",
            "incumbent_policy_path",
            "incumbent_policy_sha256",
            "incumbent_t_gate_count",
            "incumbent_result_path",
            "status",
            "last_result_sha256",
        }
        if set(lane_state) != expected_lane_state_fields:
            raise ExportValidationError(f"{context} has an unexpected final schema")
        incumbent = incumbents[lane_id]
        if require_int(lane_state.get("incumbent_iteration"), f"{context}.incumbent_iteration") != (
            incumbent.iteration
        ):
            raise ExportValidationError(f"{context}.incumbent_iteration does not match replay")
        if require_int(
            lane_state.get("incumbent_t_gate_count"), f"{context}.incumbent_t_gate_count"
        ) != (incumbent.total_t_gate_count):
            raise ExportValidationError(f"{context}.incumbent_t_gate_count does not match replay")
        if (
            require_sha256(
                lane_state.get("incumbent_policy_sha256"), f"{context}.incumbent_policy_sha256"
            )
            != incumbent.policy_sha256
        ):
            raise ExportValidationError(f"{context} incumbent policy hash does not match replay")
        if (
            require_int(
                lane_state.get("last_collected_iteration"), f"{context}.last_collected_iteration"
            )
            != 50
        ):
            raise ExportValidationError(f"{context}.last_collected_iteration must be 50")
        if require_int(lane_state.get("proposal_count"), f"{context}.proposal_count") != (
            PAPER_PROPOSAL_ITERATIONS
        ):
            raise ExportValidationError(f"{context}.proposal_count must be 50")
        expected_status = (
            "accepted"
            if require_bool(
                final_decisions_by_lane[lane_id].get("accepted"),
                f"iteration 50/{lane_id} final decision.accepted",
            )
            else "rejected"
        )
        if lane_state.get("status") != expected_status:
            raise ExportValidationError(f"{context}.status does not match iteration 50")

        lane_root = campaign_root / "lanes" / lane_id
        for policy_path in (
            lane_root / "simple_pf.py",
            lane_root / "best" / "simple_pf.py",
        ):
            if sha256_file(policy_path) != incumbent.policy_sha256:
                raise ExportValidationError(f"{policy_path} does not match the replayed incumbent")

        expected_incumbent_result = (
            Path("iterations") / f"iter_{incumbent.iteration:04d}" / "results" / f"{lane_id}.json"
        )
        expected_incumbent_policy = (
            Path("iterations")
            / f"iter_{incumbent.iteration:04d}"
            / "candidates"
            / lane_id
            / "simple_pf.py"
        )
        if (
            _safe_relative_path(
                lane_state.get("incumbent_policy_path"), f"{context}.incumbent_policy_path"
            )
            != expected_incumbent_policy
        ):
            raise ExportValidationError(f"{context}.incumbent_policy_path does not match replay")
        recorded_incumbent_result = _safe_relative_path(
            lane_state.get("incumbent_result_path"), f"{context}.incumbent_result_path"
        )
        if recorded_incumbent_result != expected_incumbent_result:
            raise ExportValidationError(f"{context}.incumbent_result_path does not match replay")
        incumbent_result_hash = sha256_file(campaign_root / expected_incumbent_result)
        if sha256_file(lane_root / "best" / "result.json") != incumbent_result_hash:
            raise ExportValidationError(f"{context} best result differs from replayed incumbent")

        final_result = (
            campaign_root
            / "iterations"
            / f"iter_{PAPER_PROPOSAL_ITERATIONS:04d}"
            / "results"
            / f"{lane_id}.json"
        )
        final_result_hash = sha256_file(final_result)
        if (
            require_sha256(lane_state.get("last_result_sha256"), f"{context}.last_result_sha256")
            != final_result_hash
            or sha256_file(lane_root / "last_result.json") != final_result_hash
        ):
            raise ExportValidationError(f"{context} last result differs from iteration 50")


def validate_collection_transactions(
    campaign_root: Path,
    *,
    manifest_sha256: str,
    campaign_id: str,
) -> None:
    transactions_root = campaign_root / "transactions"
    expected_files = {
        f"collect_iter_{iteration:04d}.json" for iteration in PAPER_EVALUATION_ITERATIONS
    }
    require_exact_directory(
        transactions_root,
        context="campaign collection transactions",
        files=expected_files,
    )
    for iteration in PAPER_EVALUATION_ITERATIONS:
        context = f"collection transaction {iteration}"
        payload = require_mapping(
            read_json(transactions_root / f"collect_iter_{iteration:04d}.json"), context
        )
        expected_fields = {
            "schema_version",
            "phase",
            "iteration",
            "manifest_sha256",
            "created_at",
            "decisions",
            "desired_state",
            "replayed_at",
            "committed_at",
            "desired_state_sha256",
        }
        if set(payload) != expected_fields:
            raise ExportValidationError(f"{context} has an unexpected committed schema")
        if (
            require_int(payload.get("schema_version"), f"{context}.schema_version") != 1
            or payload.get("phase") != "committed"
            or require_int(payload.get("iteration"), f"{context}.iteration") != iteration
            or require_sha256(payload.get("manifest_sha256"), f"{context}.manifest_sha256")
            != manifest_sha256
        ):
            raise ExportValidationError(f"{context} is open, stale, or malformed")
        for field in ("created_at", "replayed_at", "committed_at"):
            require_string(payload.get(field), f"{context}.{field}")
        decisions = require_list(payload.get("decisions"), f"{context}.decisions")
        if len(decisions) != PAPER_LANE_COUNT:
            raise ExportValidationError(f"{context} does not contain exactly 20 decisions")
        desired_state = require_mapping(payload.get("desired_state"), f"{context}.desired_state")
        if (
            desired_state.get("campaign_id") != campaign_id
            or desired_state.get("manifest_sha256") != manifest_sha256
            or require_int(
                desired_state.get("current_collected_iteration"),
                f"{context}.desired_state.current_collected_iteration",
            )
            != iteration
            or desired_state.get("next_iteration")
            != (iteration + 1 if iteration < PAPER_PROPOSAL_ITERATIONS else None)
        ):
            raise ExportValidationError(f"{context}.desired_state completion markers mismatch")
        desired_hash = hashlib.sha256(
            json.dumps(desired_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            require_sha256(payload.get("desired_state_sha256"), f"{context}.desired_state_sha256")
            != desired_hash
        ):
            raise ExportValidationError(f"{context}.desired_state hash mismatch")
        iteration_decisions = normalize_decisions(
            read_json(campaign_root / "iterations" / f"iter_{iteration:04d}" / "decisions.json"),
            iteration,
        )
        if decisions != iteration_decisions:
            raise ExportValidationError(f"{context}.decisions differ from immutable iteration")
        if iteration == PAPER_PROPOSAL_ITERATIONS and desired_state != require_mapping(
            read_json(campaign_root / "state.json"), "state.json"
        ):
            raise ExportValidationError(
                f"{context}.desired_state differs from the committed final state.json"
            )


def load_and_replay(
    campaign_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, LaneSpec],
    list[Candidate],
    dict[str, Candidate],
    dict[str, Any],
]:
    validate_campaign_top_level(campaign_root)
    manifest_path = campaign_root / "manifest.json"
    state_path = campaign_root / "state.json"
    manifest = require_mapping(read_json(manifest_path), "manifest.json")
    manifest_hash = sha256_file(manifest_path)
    lanes = validate_manifest(manifest)
    validate_lane_structure(campaign_root, lanes)
    provenance = validate_provenance(manifest)
    validate_declared_campaign_files(campaign_root, manifest)
    state = read_json(state_path)
    batch_index = index_state_batches(state)
    iterations_dir = campaign_root / "iterations"
    validate_iteration_directories(iterations_dir)

    lane_ids = list(lanes)
    lane_id_set = set(lane_ids)
    campaign_id = require_sha256(manifest.get("campaign_id"), "manifest.json.campaign_id")
    execution = require_mapping(manifest.get("execution"), "manifest.json.execution")
    cluster_root = _require_absolute_posix_path(
        execution.get("cluster_root"), "manifest.json.execution.cluster_root"
    )
    cluster_python = _require_absolute_posix_path(
        execution.get("cluster_python"), "manifest.json.execution.cluster_python"
    )
    batch_size = require_int(
        execution.get("batch_size"), "manifest.json.execution.batch_size", minimum=1
    )
    validate_collection_transactions(
        campaign_root,
        manifest_sha256=manifest_hash,
        campaign_id=campaign_id,
    )
    incumbents: dict[str, Candidate] = {}
    candidates: list[Candidate] = []
    candidate_record_count = 0
    decision_record_count = 0
    mutation_record_count = 0

    for iteration in PAPER_EVALUATION_ITERATIONS:
        iter_dir = iterations_dir / f"iter_{iteration:04d}"
        candidates_dir = iter_dir / "candidates"
        results_dir = iter_dir / "results"
        validate_iteration_structure(
            iter_dir,
            iteration=iteration,
            lane_ids=lane_ids,
            batches=batch_index[iteration],
            campaign_id=campaign_id,
            manifest_sha256=manifest_hash,
            cluster_root=cluster_root,
            cluster_python=cluster_python,
            batch_size=batch_size,
        )

        decisions = normalize_decisions(read_json(iter_dir / "decisions.json"), iteration)
        if len(decisions) != PAPER_LANE_COUNT:
            raise ExportValidationError(
                f"iteration {iteration} must have exactly 20 decisions, got {len(decisions)}"
            )
        decisions_by_lane: dict[str, dict[str, Any]] = {}
        for decision in decisions:
            lane_id = require_string(
                decision.get("lane_id"), f"iteration {iteration} decision.lane_id"
            )
            if lane_id in decisions_by_lane:
                raise ExportValidationError(
                    f"iteration {iteration} duplicate decision for {lane_id}"
                )
            decisions_by_lane[lane_id] = decision
        if set(decisions_by_lane) != lane_id_set:
            raise ExportValidationError(f"iteration {iteration} decision lane set mismatch")

        mutation_records: dict[str, dict[str, Any]] = {}
        mutation_dir = iter_dir / "mutations"
        for lane_id in sorted(lane_ids):
            mutation_records[lane_id] = require_mapping(
                read_json(mutation_dir / lane_id / "record.json"),
                f"iteration {iteration}/{lane_id} mutation record",
            )
            validate_mutation_directory_structure(
                mutation_dir / lane_id,
                iteration=iteration,
                lane_id=lane_id,
                record=mutation_records[lane_id],
            )
        if iteration > 0:
            mutation_record_count += len(mutation_records)

        for lane_id in sorted(lane_ids):
            lane = lanes[lane_id]
            policy_path = candidates_dir / lane_id / "simple_pf.py"
            result_path = results_dir / f"{lane_id}.json"
            result = read_json(result_path)
            candidate = validate_result(
                result,
                lane=lane,
                iteration=iteration,
                policy_path=policy_path,
                manifest_sha256=manifest_hash,
            )
            # Private in-memory annotation only; never emitted.
            candidate.result["_validated_result_sha256"] = sha256_file(result_path)
            incumbent = incumbents.get(lane_id)
            if iteration == 0:
                if incumbent is not None or not candidate.passes_contract:
                    raise ExportValidationError(
                        f"iteration 0/{lane_id} baseline must be feasible and have no incumbent"
                    )
                validate_baseline_record(
                    mutation_records[lane_id],
                    lane=lane,
                    candidate_sha256=candidate.policy_sha256,
                )
                validate_baseline_result(candidate)
                expected_accept = True
            else:
                if incumbent is None:
                    raise ExportValidationError(f"iteration {iteration}/{lane_id} has no baseline")
                validate_mutation_record(
                    mutation_records[lane_id],
                    lane=lane,
                    iteration=iteration,
                    expected_parent_iteration=incumbent.iteration,
                    expected_parent_sha256=incumbent.policy_sha256,
                    expected_parent_policy_path=incumbent.policy_path,
                    candidate_sha256=candidate.policy_sha256,
                    mutation_root=mutation_dir / lane_id,
                    campaign_root=campaign_root,
                )
                expected_accept = (
                    candidate.passes_contract
                    and candidate.total_t_gate_count < incumbent.total_t_gate_count
                )
            validate_decision(
                decisions_by_lane[lane_id],
                candidate=candidate,
                incumbent=incumbent,
                expected_accept=expected_accept,
            )
            candidate.accepted = expected_accept
            if expected_accept:
                incumbents[lane_id] = candidate
            candidates.append(candidate)
            candidate_record_count += 1
            decision_record_count += 1

    if candidate_record_count != PAPER_CANDIDATE_COUNT:
        raise ExportValidationError(
            f"candidate record count={candidate_record_count}, expected {PAPER_CANDIDATE_COUNT}"
        )
    if decision_record_count != PAPER_CANDIDATE_COUNT:
        raise ExportValidationError(
            f"decision record count={decision_record_count}, expected {PAPER_CANDIDATE_COUNT}"
        )
    if mutation_record_count != PAPER_MUTATION_COUNT:
        raise ExportValidationError(
            f"mutation record count={mutation_record_count}, expected {PAPER_MUTATION_COUNT}"
        )
    validate_state(
        state,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        lanes=lanes,
        incumbents=incumbents,
        campaign_root=campaign_root,
    )
    for candidate in candidates:
        candidate.result.pop("_validated_result_sha256", None)
    return manifest, lanes, candidates, incumbents, provenance


def equal_share_selection_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.total_t_gate_count,
        candidate.max_species_absolute_error_hartree,
        candidate.triangle_bound_hartree,
        candidate.absolute_reaction_error_hartree,
        candidate.iteration,
        candidate.lane.lane_id,
        candidate.policy_sha256,
    )


def reaction_selection_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.total_t_gate_count,
        candidate.absolute_reaction_error_hartree,
        candidate.triangle_bound_hartree,
        candidate.max_species_absolute_error_hartree,
        candidate.iteration,
        candidate.lane.lane_id,
        candidate.policy_sha256,
    )


def select_pooled_candidates(
    candidates: Sequence[Candidate],
) -> dict[str, dict[str, Candidate]]:
    by_reaction: dict[str, list[Candidate]] = {key: [] for key in REACTION_KEYS}
    for candidate in candidates:
        by_reaction[candidate.lane.reaction_key].append(candidate)
    selected: dict[str, dict[str, Candidate]] = {}
    for reaction_key in REACTION_KEYS:
        pool = by_reaction[reaction_key]
        if len(pool) != len(CONTRACTS) * len(PAPER_EVALUATION_ITERATIONS):
            raise ExportValidationError(
                f"{reaction_key} pooled candidate count={len(pool)}, expected 102"
            )
        equal_share_pool = [item for item in pool if item.passes_equal_share_constraint]
        reaction_pool = [item for item in pool if item.passes_reaction_constraint]
        if not equal_share_pool or not reaction_pool:
            raise ExportValidationError(
                f"{reaction_key} has no feasible candidate for both selections"
            )
        equal_share = min(equal_share_pool, key=equal_share_selection_key)
        reaction_aware = min(reaction_pool, key=reaction_selection_key)
        if reaction_aware.total_t_gate_count > equal_share.total_t_gate_count:
            raise ExportValidationError(
                f"{reaction_key} reaction-feasible minimum is costlier than equal-share minimum"
            )
        selected[reaction_key] = {
            "equal_share": equal_share,
            "reaction_aware": reaction_aware,
        }
    return selected


def compact_species(candidate: Candidate) -> dict[str, Any]:
    return {
        species: {
            "absolute_energy_error_hartree": candidate.result["species"][species][
                "absolute_energy_error_hartree"
            ],
            "method_config": candidate.result["species"][species].get("method_config"),
            "signed_energy_error_hartree": candidate.result["species"][species][
                "signed_energy_error_hartree"
            ],
            "t_gate_count": candidate.result["species"][species]["t_gate_count"],
        }
        for species in candidate.lane.species
    }


def selection_payload(candidate: Candidate, policy_relative_path: str) -> dict[str, Any]:
    return {
        "absolute_reaction_error_hartree": candidate.absolute_reaction_error_hartree,
        "equal_share_error_limit_hartree": candidate.equal_share_error_limit_hartree,
        "max_species_absolute_error_hartree": candidate.max_species_absolute_error_hartree,
        "passes_equal_share_constraint": candidate.passes_equal_share_constraint,
        "passes_reaction_constraint": candidate.passes_reaction_constraint,
        "selected_policy_path": policy_relative_path,
        "selected_policy_sha256": candidate.policy_sha256,
        "selected_source_contract": candidate.lane.contract,
        "selected_source_iteration": candidate.iteration,
        "selected_source_lane": candidate.lane.lane_id,
        "signed_reaction_error_hartree": candidate.signed_reaction_error_hartree,
        "species": compact_species(candidate),
        "total_t_gate_count": candidate.total_t_gate_count,
        "triangle_bound_hartree": candidate.triangle_bound_hartree,
    }


def write_candidate_log(path: Path, candidates: Iterable[Candidate]) -> None:
    fieldnames = [
        "reaction_key",
        "source_contract",
        "source_lane",
        "iteration",
        "policy_sha256",
        "total_t_gate_count",
        "signed_reaction_error_hartree",
        "triangle_bound_hartree",
        "equal_share_error_limit_hartree",
        "max_species_absolute_error_hartree",
        "passes_equal_share_constraint",
        "passes_reaction_constraint",
        "passes_contract",
        "accepted",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda item: (item.iteration, item.lane.lane_id)):
            writer.writerow(
                {
                    "reaction_key": candidate.lane.reaction_key,
                    "source_contract": candidate.lane.contract,
                    "source_lane": candidate.lane.lane_id,
                    "iteration": candidate.iteration,
                    "policy_sha256": candidate.policy_sha256,
                    "total_t_gate_count": candidate.total_t_gate_count,
                    "signed_reaction_error_hartree": repr(candidate.signed_reaction_error_hartree),
                    "triangle_bound_hartree": repr(candidate.triangle_bound_hartree),
                    "equal_share_error_limit_hartree": repr(
                        candidate.equal_share_error_limit_hartree
                    ),
                    "max_species_absolute_error_hartree": repr(
                        candidate.max_species_absolute_error_hartree
                    ),
                    "passes_equal_share_constraint": str(
                        candidate.passes_equal_share_constraint
                    ).lower(),
                    "passes_reaction_constraint": str(candidate.passes_reaction_constraint).lower(),
                    "passes_contract": str(candidate.passes_contract).lower(),
                    "accepted": str(candidate.accepted).lower(),
                }
            )


def copy_selected_policies(
    output_root: Path,
    selected: Mapping[str, Mapping[str, Candidate]],
) -> dict[tuple[str, str], str]:
    paths: dict[tuple[str, str], str] = {}
    for reaction_key in REACTION_KEYS:
        for selection_contract in CONTRACTS:
            candidate = selected[reaction_key][selection_contract]
            relative = (
                Path("selected_policies") / reaction_key / selection_contract / "simple_pf.py"
            )
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate.policy_path, destination)
            if sha256_file(destination) != candidate.policy_sha256:
                raise ExportValidationError(f"copied selected policy hash changed: {relative}")
            paths[(reaction_key, selection_contract)] = relative.as_posix()
    return paths


def build_reaction_summary(
    selected: Mapping[str, Mapping[str, Candidate]],
    policy_paths: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for reaction_key in REACTION_KEYS:
        equal_share = selected[reaction_key]["equal_share"]
        reaction_aware = selected[reaction_key]["reaction_aware"]
        equal_t = equal_share.total_t_gate_count
        reaction_t = reaction_aware.total_t_gate_count
        rows.append(
            {
                "additional_t_gate_decrease_percent": 100.0 * (equal_t - reaction_t) / equal_t,
                "equal_share": selection_payload(
                    equal_share, policy_paths[(reaction_key, "equal_share")]
                ),
                "reaction": equal_share.lane.reaction,
                "reaction_aware": selection_payload(
                    reaction_aware, policy_paths[(reaction_key, "reaction_aware")]
                ),
                "reaction_key": reaction_key,
                "stoichiometric_terms": [
                    {"coefficient": coefficient, "species": species}
                    for species, coefficient in equal_share.lane.terms
                ],
            }
        )
    return {
        "chemical_accuracy_hartree": PAPER_TARGET_HARTREE,
        "reactions": rows,
        "schema_version": 1,
        "selection_rule": (
            "Pool candidates from both source lane contracts for each reaction; independently "
            "minimize T count among equal-share-feasible and reaction-feasible candidates. "
            "Ties use the relevant error metrics, iteration, source lane, and policy SHA-256."
        ),
    }


def build_summary(reaction_summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = require_list(reaction_summary["reactions"], "reaction summary rows")
    equal_total = sum(row["equal_share"]["total_t_gate_count"] for row in rows)
    reaction_total = sum(row["reaction_aware"]["total_t_gate_count"] for row in rows)
    gains = [float(row["additional_t_gate_decrease_percent"]) for row in rows]
    return {
        "arithmetic_mean_additional_t_gate_decrease_percent": statistics.fmean(gains),
        "candidate_record_count": PAPER_CANDIDATE_COUNT,
        "chemical_accuracy_hartree": PAPER_TARGET_HARTREE,
        "decision_record_count": PAPER_CANDIDATE_COUNT,
        "equal_share_total_t_gate_count": equal_total,
        "evaluation_iterations": list(PAPER_EVALUATION_ITERATIONS),
        "experiment": "reactions10_autoresearch",
        "lane_count": PAPER_LANE_COUNT,
        "mutation_record_count": PAPER_MUTATION_COUNT,
        "pooled_additional_t_gate_decrease_percent": (
            100.0 * (equal_total - reaction_total) / equal_total
        ),
        "proposal_iterations": PAPER_PROPOSAL_ITERATIONS,
        "reaction_aware_total_t_gate_count": reaction_total,
        "reaction_count": PAPER_REACTION_COUNT,
        "schema_version": 1,
        "status": "complete",
    }


def write_checksum_manifest(output_root: Path) -> None:
    manifest_path = output_root / "SHA256SUMS"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}"
        for path in sorted(item for item in output_root.rglob("*") if item.is_file())
        if path != manifest_path
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_no_campaign_root_leak(output_root: Path, campaign_root: Path) -> None:
    needles = {
        str(campaign_root).encode("utf-8"),
        str(campaign_root.resolve()).encode("utf-8"),
        f"file://{campaign_root}".encode("utf-8"),
        f"file://{campaign_root.resolve()}".encode("utf-8"),
    }
    needles.discard(b"")
    offenders: list[str] = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        if any(needle in data for needle in needles):
            offenders.append(path.relative_to(output_root).as_posix())
    if offenders:
        raise ExportValidationError(
            f"export leaks the absolute source campaign root in: {offenders}"
        )


def export_campaign(campaign_root: Path, output_dir: Path) -> None:
    campaign_root = campaign_root.expanduser().absolute()
    output_dir = output_dir.expanduser().absolute()
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        raise ExportValidationError(f"campaign root must be a real directory: {campaign_root}")
    campaign_root = campaign_root.resolve()
    if output_dir.exists():
        raise ExportValidationError(
            f"output path already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    manifest, _lanes, candidates, _incumbents, provenance = load_and_replay(campaign_root)
    selected = select_pooled_candidates(candidates)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        write_candidate_log(temporary / "candidate_log.csv", candidates)
        policy_paths = copy_selected_policies(temporary, selected)
        reaction_summary = build_reaction_summary(selected, policy_paths)
        write_json(temporary / "reaction_summary.json", reaction_summary)
        write_json(temporary / "summary.json", build_summary(reaction_summary))
        write_json(
            temporary / "provenance.json",
            {
                "campaign_id": manifest["campaign_id"],
                "campaign_manifest_sha256": sha256_file(campaign_root / "manifest.json"),
                "campaign_state_sha256": sha256_file(campaign_root / "state.json"),
                "experiment": manifest["experiment"],
                "paper_autoresearch": provenance,
                "schema_version": 1,
                "validation": {
                    "acceptance_rule": "baseline, then feasible and strictly lower T count",
                    "candidate_hashes_verified": PAPER_CANDIDATE_COUNT,
                    "decision_replay_records": PAPER_CANDIDATE_COUNT,
                    "mutation_lineage_records": PAPER_MUTATION_COUNT,
                    "scientific_quantities_recomputed": True,
                },
            },
        )
        (temporary / "FINAL_EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
        write_checksum_manifest(temporary)
        assert_no_campaign_root_leak(temporary, campaign_root)
        if output_dir.exists():
            raise ExportValidationError(
                f"output path appeared during export; refusing to overwrite: {output_dir}"
            )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate and atomically export a complete 50x20 reactions10 paper campaign."
        )
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_campaign(args.campaign_root, args.output_dir)


if __name__ == "__main__":
    main()
