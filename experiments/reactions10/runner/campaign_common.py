from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 1
REACTION_COUNT = 10
LANE_COUNT = 20
PROPOSAL_ITERATIONS = 50
FIRST_EVALUATION_ITERATION = 0
LAST_EVALUATION_ITERATION = PROPOSAL_ITERATIONS
EXPECTED_CACHE_FILES = 42
EXPECTED_CACHE_SPECIES = 21

HARTREE_PER_KCAL_MOL = 1.0 / 627.5094740631
CHEMICAL_ACCURACY_HARTREE = HARTREE_PER_KCAL_MOL
FAILURE_SCORE_BASE = 1.0e15

CODEX_PACKAGE = "@openai/codex@0.144.0"
CODEX_CLI_VERSION = "0.144.0"
CODEX_COMMAND = ("npx", "-y", CODEX_PACKAGE)
PAPER_MODEL = "gpt-5.5"
PAPER_REASONING_EFFORT = "high"
ARCHIVED_REACTION_BENCHMARK_SHA256 = (
    "09b713ebd784614ec774c73c22c0228b084d82bd9a3ea176589caf06918d965e"
)
ARCHIVED_REACTION_EVALUATOR_SHA256 = (
    "b81d8e936f71c0cbab76ba8f0b134b4ecbf2c610d95aaca04136b7aeb5309425"
)

SAFE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CAMPAIGN_ID_DOMAIN = b"hsopt-reactions10-campaign-v1\0"
# Python bytecode is executable input, not disposable runtime state.  Every
# runner entry point redirects bytecode into a new private prefix, so a
# __pycache__ below either immutable snapshot is always provenance drift.
RUNTIME_SNAPSHOT_DIRS = frozenset({".venv", ".pytest_cache", ".ruff_cache"})


def derive_campaign_id(preparation_nonce: str) -> str:
    if SHA256_RE.fullmatch(preparation_nonce) is None:
        raise ValueError("preparation nonce must be exactly 32 random bytes encoded as hex")
    return hashlib.sha256(CAMPAIGN_ID_DOMAIN + bytes.fromhex(preparation_nonce)).hexdigest()


def validate_absolute_posix_path(value: str, field: str, *, allow_root: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute POSIX path")
    if any(character in value for character in ("\n", "\r", "\x00")):
        raise ValueError(f"{field} cannot contain newline or NUL")
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute POSIX path")
    components = value.split("/")[1:]
    if any(component in {".", ".."} for component in components):
        raise ValueError(f"{field} cannot contain '.' or '..' components")
    if value != "/" and any(not component for component in components):
        raise ValueError(f"{field} cannot contain empty path components")
    if path.as_posix() != value:
        raise ValueError(f"{field} must be a canonical POSIX path")
    if not allow_root and path == PurePosixPath("/"):
        raise ValueError(f"{field} cannot be the filesystem root")
    return value


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def campaign_paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "state": root / "state.json",
        "lanes": root / "lanes",
        "iterations": root / "iterations",
        "logs": root / "logs",
        "source": root / "source",
        "problem_cache": root / "problem_cache",
        "runner": root / "runner",
        "receipts": root / "receipts",
        "transactions": root / "transactions",
        "locks": root / ".locks",
    }


def ensure_campaign_dirs(root: Path) -> dict[str, Path]:
    paths = campaign_paths(root)
    for key in (
        "lanes",
        "iterations",
        "logs",
        "source",
        "problem_cache",
        "runner",
        "receipts",
        "transactions",
        "locks",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    display = (
        path.name
        if relative_to is None
        else path.resolve().relative_to(relative_to.resolve()).as_posix()
    )
    return {
        "path": display,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def exact_flat_regular_file_names(root: Path, context: str) -> list[str]:
    """Return one directory's file names, rejecting links, dirs, and special entries."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{context} is missing, non-directory, or a symlink")
    names: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"{context} contains a non-regular entry: {entry.name}")
            names.append(entry.name)
    return sorted(names)


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)


def _snapshot_tree_entries(base: Path, context: str) -> tuple[set[str], set[str]]:
    """Return regular files/directories while pruning explicit mutable runtime trees."""

    if not base.is_dir() or base.is_symlink():
        raise RuntimeError(f"{context} snapshot root is missing, non-directory, or a symlink")
    files: set[str] = set()
    directories: set[str] = set()
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
                    raise RuntimeError(
                        f"{context} snapshot entry disappeared during verification: {relative_name}"
                    ) from exc
                if stat.S_ISLNK(mode):
                    symlinks.append(relative_name)
                elif stat.S_ISDIR(mode):
                    if entry.name in RUNTIME_SNAPSHOT_DIRS:
                        continue
                    directories.add(relative_name)
                    walk(Path(entry.path), relative)
                elif stat.S_ISREG(mode):
                    files.add(relative_name)
                else:
                    special.append(relative_name)

    walk(base, Path())
    if symlinks:
        raise RuntimeError(f"{context} snapshot contains symlinks: {sorted(symlinks)}")
    if special:
        raise RuntimeError(f"{context} snapshot contains special entries: {sorted(special)}")
    return files, directories


def verify_snapshot_records(root: Path, manifest: dict[str, Any]) -> None:
    for manifest_key, base in (
        ("source_files", root.resolve() / "source" / "repo"),
        ("runner_files", root.resolve() / "runner"),
    ):
        records = manifest.get(manifest_key)
        if not isinstance(records, dict) or not records:
            raise RuntimeError(f"campaign manifest has no {manifest_key}")
        declared_files: set[str] = set()
        declared_directories: set[str] = set()
        for relative, record in records.items():
            if not isinstance(relative, str):
                raise RuntimeError(f"unsafe snapshotted path in {manifest_key}: {relative!r}")
            relative_path = Path(relative)
            normalized = relative_path.as_posix()
            if (
                relative_path.is_absolute()
                or not relative_path.parts
                or normalized in {"", "."}
                or normalized != relative
                or ".." in relative_path.parts
                or any(part in RUNTIME_SNAPSHOT_DIRS for part in relative_path.parts)
                or normalized in declared_files
            ):
                raise RuntimeError(f"unsafe snapshotted path in {manifest_key}: {relative}")
            declared_files.add(normalized)
            for parent in relative_path.parents:
                if parent == Path("."):
                    break
                declared_directories.add(parent.as_posix())
            path = base / relative_path
            if (
                not path.is_file()
                or path.is_symlink()
                or not isinstance(record, dict)
                or record.get("sha256") != sha256_file(path)
                or record.get("size_bytes") != path.stat().st_size
            ):
                raise RuntimeError(f"snapshotted campaign source drift: {path}")
        actual_files, actual_directories = _snapshot_tree_entries(base, manifest_key)
        if actual_files != declared_files or actual_directories != declared_directories:
            raise RuntimeError(
                f"{manifest_key} snapshot tree drift: "
                f"missing_files={sorted(declared_files - actual_files)}, "
                f"undeclared_files={sorted(actual_files - declared_files)}, "
                f"missing_directories={sorted(declared_directories - actual_directories)}, "
                f"unexpected_directories={sorted(actual_directories - declared_directories)}"
            )


def verify_campaign_cache(root: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("inputs", {}).get("problem_cache_files")
    cache_root = root.resolve() / "problem_cache"
    if not isinstance(expected, dict) or len(expected) != EXPECTED_CACHE_FILES:
        raise RuntimeError("campaign manifest has no fixed 42-file cache checksum map")
    actual_names = exact_flat_regular_file_names(cache_root, "campaign problem cache")
    if set(actual_names) != set(expected):
        raise RuntimeError("campaign problem-cache file set drift")
    for name, digest in expected.items():
        path = cache_root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"campaign problem-cache checksum drift: {name}")


def ensure_campaign_runner_runtime(root: Path, script_path: Path, manifest: dict[str, Any]) -> None:
    expected_runner = root.resolve() / "runner"
    actual_script = script_path.resolve()
    expected_script = expected_runner / actual_script.name
    expected_common = expected_runner / "campaign_common.py"
    if (
        actual_script != expected_script.resolve()
        or Path(__file__).resolve() != expected_common.resolve()
    ):
        raise RuntimeError(
            "post-prepare commands must run from the immutable campaign_root/runner snapshot"
        )
    records = manifest.get("runner_files", {})
    for path in (expected_script, expected_common):
        record = records.get(path.relative_to(expected_runner).as_posix())
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(path):
            raise RuntimeError(
                f"active campaign runner file is not the snapshotted version: {path}"
            )


@contextlib.contextmanager
def campaign_lock(root: Path, name: str, *, blocking: bool = False) -> Iterator[None]:
    if SAFE_ID_RE.fullmatch(name) is None:
        raise ValueError(f"unsafe lock name {name!r}")
    lock_path = root.resolve() / ".locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            raise RuntimeError(f"campaign lock already held: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at={now_iso()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def open_collect_transactions(root: Path) -> list[Path]:
    transactions = campaign_paths(root)["transactions"]
    if not transactions.is_dir():
        return []
    result: list[Path] = []
    for path in sorted(transactions.glob("collect_iter_*.json")):
        try:
            phase = read_json(path).get("phase")
        except Exception:
            phase = "unreadable"
        if phase != "committed":
            result.append(path)
    return result


def require_campaign(
    root: Path, *, allow_open_collect_transaction: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = campaign_paths(root)
    if not paths["manifest"].is_file() or not paths["state"].is_file():
        raise FileNotFoundError(f"not a prepared reaction campaign: {root.resolve()}")
    manifest = read_json(paths["manifest"])
    state = read_json(paths["state"])
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("unsupported campaign manifest schema")
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("unsupported campaign state schema")
    actual_hash = manifest_sha256(paths["manifest"])
    if state.get("manifest_sha256") != actual_hash:
        raise RuntimeError("manifest.json changed after campaign preparation")
    preparation_nonce = manifest.get("preparation_nonce")
    campaign_id = manifest.get("campaign_id")
    if (
        not isinstance(preparation_nonce, str)
        or not isinstance(campaign_id, str)
        or derive_campaign_id(preparation_nonce) != campaign_id
        or state.get("campaign_id") != campaign_id
    ):
        raise RuntimeError("campaign preparation nonce/identity drift")
    if int(manifest.get("proposal_iterations", -1)) != PROPOSAL_ITERATIONS:
        raise RuntimeError("campaign does not use the fixed 50-proposal contract")
    if manifest.get("evaluation_iterations") != list(range(PROPOSAL_ITERATIONS + 1)):
        raise RuntimeError("campaign must evaluate exactly iterations 0 through 50")
    if (
        manifest.get("reaction_count") != REACTION_COUNT
        or manifest.get("lane_count") != LANE_COUNT
        or manifest.get("chemical_accuracy_hartree") != CHEMICAL_ACCURACY_HARTREE
        or manifest.get("autoresearch_provenance")
        != {
            "codex_package": CODEX_PACKAGE,
            "codex_cli_version": CODEX_CLI_VERSION,
            "model": PAPER_MODEL,
            "reasoning_effort": PAPER_REASONING_EFFORT,
        }
    ):
        raise RuntimeError("campaign fixed scientific/autoresearch contract drift")
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != LANE_COUNT:
        raise RuntimeError(f"campaign must contain exactly {LANE_COUNT} lanes")
    verify_snapshot_records(root, manifest)
    verify_campaign_cache(root, manifest)
    if not allow_open_collect_transaction:
        open_transactions = open_collect_transactions(root)
        if open_transactions:
            raise RuntimeError(
                "an interrupted collection transaction must be replayed first: "
                + ", ".join(str(path) for path in open_transactions)
            )
    return manifest, state


def lane_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in manifest["lanes"]:
        lane = dict(raw)
        lane_id = str(lane.get("lane_id", ""))
        if SAFE_ID_RE.fullmatch(lane_id) is None or lane_id in result:
            raise RuntimeError(f"invalid or duplicate lane id {lane_id!r}")
        result[lane_id] = lane
    return result


def expected_lane_ids(manifest: dict[str, Any]) -> list[str]:
    return [str(lane["lane_id"]) for lane in manifest["lanes"]]


def validate_iteration(iteration: int, *, mutation: bool = False) -> None:
    lower = 1 if mutation else FIRST_EVALUATION_ITERATION
    if not lower <= int(iteration) <= LAST_EVALUATION_ITERATION:
        kind = "mutation" if mutation else "evaluation"
        raise ValueError(f"{kind} iteration must be in [{lower}, {LAST_EVALUATION_ITERATION}]")


def iteration_dir(root: Path, iteration: int) -> Path:
    validate_iteration(iteration)
    return root.resolve() / "iterations" / f"iter_{iteration:04d}"


def candidate_path(root: Path, iteration: int, lane_id: str) -> Path:
    if SAFE_ID_RE.fullmatch(lane_id) is None:
        raise ValueError(f"unsafe lane id {lane_id!r}")
    return iteration_dir(root, iteration) / "candidates" / lane_id / "simple_pf.py"


def mutation_dir(root: Path, iteration: int, lane_id: str) -> Path:
    if SAFE_ID_RE.fullmatch(lane_id) is None:
        raise ValueError(f"unsafe lane id {lane_id!r}")
    return iteration_dir(root, iteration) / "mutations" / lane_id


def result_path(root: Path, iteration: int, lane_id: str) -> Path:
    if SAFE_ID_RE.fullmatch(lane_id) is None:
        raise ValueError(f"unsafe lane id {lane_id!r}")
    return iteration_dir(root, iteration) / "results" / f"{lane_id}.json"


def read_checksum_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0].lower()) is None:
            raise ValueError(f"{path}:{line_number}: expected '<sha256>  <path>'")
        source_name = parts[1].lstrip("*").strip().replace("\\", "/")
        if "/problem_cache/" in source_name:
            source_name = source_name.rsplit("/problem_cache/", 1)[1]
        elif source_name.startswith("problem_cache/"):
            source_name = source_name[len("problem_cache/") :]
        candidate = Path(source_name)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name in {"", ".", ".."}
        ):
            raise ValueError(f"{path}:{line_number}: unsafe cache path {source_name!r}")
        if candidate.name in entries:
            raise ValueError(f"{path}:{line_number}: duplicate cache file {candidate.name}")
        entries[candidate.name] = parts[0].lower()
    return entries


def verify_problem_cache(cache_root: Path, checksum_path: Path) -> dict[str, str]:
    cache_root = cache_root.resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"problem cache directory does not exist: {cache_root}")
    expected = read_checksum_manifest(checksum_path)
    if len(expected) != EXPECTED_CACHE_FILES:
        raise RuntimeError(
            f"fixed cache checksum list must contain {EXPECTED_CACHE_FILES} files, found {len(expected)}"
        )
    actual_files = exact_flat_regular_file_names(cache_root, "source problem cache")
    if set(actual_files) != set(expected):
        missing = sorted(set(expected) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected))
        raise RuntimeError(f"problem cache file set mismatch: missing={missing}, extra={extra}")
    json_stems = {Path(name).stem for name in expected if name.endswith(".json")}
    npz_stems = {Path(name).stem for name in expected if name.endswith(".npz")}
    if json_stems != npz_stems or len(json_stems) != EXPECTED_CACHE_SPECIES:
        raise RuntimeError(
            f"problem cache must contain JSON/NPZ pairs for {EXPECTED_CACHE_SPECIES} species"
        )
    mismatches = {
        name: {"expected": digest, "actual": sha256_file(cache_root / name)}
        for name, digest in expected.items()
        if sha256_file(cache_root / name) != digest
    }
    if mismatches:
        raise RuntimeError(f"problem cache checksum mismatch: {mismatches}")
    return expected


def copy_verified_cache(cache_root: Path, destination: Path, expected: dict[str, str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(expected):
        shutil.copy2(cache_root / name, destination / name)
    for name, digest in expected.items():
        if sha256_file(destination / name) != digest:
            raise RuntimeError(f"copied cache checksum mismatch for {name}")


def snapshot_files(
    *, source_root: Path, destination_root: Path, relative_paths: Sequence[Path]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe snapshot path {relative}")
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required snapshot source missing: {source}")
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records[relative.as_posix()] = {
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
    return records


def snapshot_tree(
    *, source_root: Path, destination_root: Path, excluded_names: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    excluded = excluded_names or set()
    relative_paths = [
        path.relative_to(source_root)
        for path in source_root.rglob("*")
        if path.is_file()
        and not any(part in excluded for part in path.relative_to(source_root).parts)
    ]
    return snapshot_files(
        source_root=source_root,
        destination_root=destination_root,
        relative_paths=relative_paths,
    )


def load_policy(path: Path):
    module_name = f"reaction_policy_{sha256_file(path)[:16]}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load policy {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def validate_policy(path: Path, species: Sequence[str]) -> tuple[bool, str | None]:
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        module = load_policy(path)
        if not callable(getattr(module, "set_active_species", None)):
            raise ValueError("policy must define set_active_species(species)")
        if not callable(getattr(module, "get_method_config", None)):
            raise ValueError("policy must define get_method_config()")
        if not callable(getattr(module, "choose_synthesis_epsilon", None)):
            raise ValueError("policy must define choose_synthesis_epsilon(...)")
        sample = {
            "label": "IXYZ",
            "coefficient": 0.03125,
            "support_size": 3,
            "x_count": 1,
            "y_count": 1,
            "z_count": 1,
        }
        for species_key in species:
            module.set_active_species(str(species_key))
            config = module.get_method_config()
            formula_order = int(getattr(config, "formula_order"))
            n_steps = int(getattr(config, "n_steps"))
            threshold = float(getattr(config, "zero_angle_threshold"))
            config_species = str(getattr(config, "species"))
            config_name = str(getattr(config, "name"))
            synthesis_budget = float(getattr(config, "synthesis_error_budget_hartree"))
            if config_species != species_key or not config_name:
                raise ValueError(f"{species_key}: config species/name identity mismatch")
            if formula_order not in (1, 2):
                raise ValueError(f"{species_key}: formula_order must be 1 or 2")
            if n_steps <= 0:
                raise ValueError(f"{species_key}: n_steps must be positive")
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    f"{species_key}: zero_angle_threshold must be finite and nonnegative"
                )
            if not math.isfinite(synthesis_budget) or synthesis_budget <= 0.0:
                raise ValueError(
                    f"{species_key}: synthesis_error_budget_hartree must be finite and positive"
                )
            epsilon = float(module.choose_synthesis_epsilon(sample, 1.0e-4, 0, config, 12))
            if not math.isfinite(epsilon) or epsilon <= 0.0:
                raise ValueError(f"{species_key}: invalid synthesis epsilon {epsilon}")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def config_dict(config: object) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(getattr(config, "__dict__", {}))


def load_benchmark(root: Path):
    source_repo = root.resolve() / "source" / "repo"
    script = source_repo / "experiments" / "reactions10" / "benchmark.py"
    if not script.is_file():
        raise FileNotFoundError(f"snapshotted reaction benchmark is missing: {script}")
    source_text = str(source_repo)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module_name = f"reaction_benchmark_{sha256_file(script)[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load reaction benchmark {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "evaluate_policy", None)):
        raise AttributeError("reaction benchmark must expose evaluate_policy(...)")
    return module


def evaluate_candidate(
    root: Path,
    lane: dict[str, Any],
    iteration: int,
    *,
    campaign_id: str,
    campaign_manifest_sha256: str,
) -> dict[str, Any]:
    lane_id = str(lane["lane_id"])
    policy_path = candidate_path(root, iteration, lane_id)
    if not policy_path.is_file():
        raise FileNotFoundError(f"immutable candidate is missing: {policy_path}")
    if exact_flat_regular_file_names(policy_path.parent, "immutable candidate directory") != [
        "simple_pf.py"
    ]:
        raise RuntimeError(f"immutable candidate directory is not exact for {lane_id}")
    record_path = mutation_dir(root, iteration, str(lane["lane_id"])) / "record.json"
    record = read_json(record_path)
    policy_hash = sha256_file(policy_path)
    if record.get("candidate_policy_sha256") != policy_hash:
        raise RuntimeError(f"candidate hash differs from mutation record for {lane['lane_id']}")
    benchmark = load_benchmark(root)
    result = benchmark.evaluate_policy(
        policy_path=policy_path,
        reaction_key=str(lane["reaction_key"]),
        contract=str(lane["contract"]),
        cache_root=root.resolve() / "problem_cache",
        iteration=int(iteration),
        lane_id=str(lane["lane_id"]),
    )
    if not isinstance(result, dict):
        raise TypeError("evaluate_policy must return a dict")
    result = dict(result)
    result["campaign_id"] = campaign_id
    result["campaign_manifest_sha256"] = campaign_manifest_sha256
    result["candidate_relative_path"] = relative_to_campaign(root, policy_path)
    result["problem_cache_relative_path"] = "problem_cache"
    result["evaluated_at"] = now_iso()
    validate_result(
        result,
        lane=lane,
        iteration=iteration,
        candidate_hash=policy_hash,
        campaign_id=campaign_id,
        campaign_manifest_sha256=campaign_manifest_sha256,
    )
    return result


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-12)


def validate_result(
    result: dict[str, Any],
    *,
    lane: dict[str, Any],
    iteration: int,
    candidate_hash: str,
    campaign_id: str,
    campaign_manifest_sha256: str,
) -> None:
    lane_id = str(lane["lane_id"])
    if result.get("status") != "ok":
        raise ValueError(f"{lane_id}: evaluator status is {result.get('status')!r}")
    identity = {
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "lane_id": lane_id,
        "reaction_key": str(lane["reaction_key"]),
        "contract": str(lane["contract"]),
        "iteration": int(iteration),
        "policy_sha256": candidate_hash,
        "candidate_relative_path": (
            f"iterations/iter_{iteration:04d}/candidates/{lane_id}/simple_pf.py"
        ),
        "problem_cache_relative_path": "problem_cache",
    }
    for field, expected in identity.items():
        if result.get(field) != expected:
            raise ValueError(
                f"{lane_id}: {field} mismatch: expected {expected!r}, got {result.get(field)!r}"
            )
    expected_terms = [
        {"coefficient": float(item["coefficient"]), "species": str(item["species"])}
        for item in lane["terms"]
    ]
    expected_species = [str(item) for item in lane["species"]]
    fixed_fields = {
        "reaction": str(lane["reaction"]),
        "reaction_terms": expected_terms,
        "species_order": expected_species,
        "evaluation_mode": "full_hilbert_space_compiled_one_step_unitary_diagonalization",
        "ground_phase_matching": "maximum_ground_state_eigenvector_overlap",
        "compilation_backend": "pygridsynth",
        "t_count_aggregation": "sum_each_unique_species_once",
    }
    for field, expected in fixed_fields.items():
        if result.get(field) != expected:
            raise ValueError(f"{lane_id}: fixed evaluator field mismatch for {field}")
    if result.get("source_provenance") != {
        "archived_benchmark_sha256": ARCHIVED_REACTION_BENCHMARK_SHA256,
        "archived_evaluator_sha256": ARCHIVED_REACTION_EVALUATOR_SHA256,
    }:
        raise ValueError(f"{lane_id}: fixed evaluator source provenance mismatch")
    species_results = result.get("species")
    if not isinstance(species_results, dict) or set(species_results) != set(expected_species):
        raise ValueError(f"{lane_id}: result species set does not match lane")
    signed_errors: dict[str, float] = {}
    absolute_errors: dict[str, float] = {}
    species_t_counts: dict[str, int] = {}
    exact_energies: dict[str, float] = {}
    pf_energies: dict[str, float] = {}
    species_scheduled_counts: dict[str, int] = {}
    for species in expected_species:
        row = species_results[species]
        if not isinstance(row, dict):
            raise ValueError(f"{lane_id}: species result for {species} is not an object")
        signed = _finite_number(row.get("signed_energy_error_hartree"), f"{species}.signed_error")
        absolute = _finite_number(
            row.get("absolute_energy_error_hartree"), f"{species}.absolute_error"
        )
        if not _close(absolute, abs(signed)):
            raise ValueError(f"{lane_id}: inconsistent signed/absolute error for {species}")
        exact_energy = _finite_number(
            row.get("exact_energy_hartree"), f"{species}.exact_energy_hartree"
        )
        pf_energy = _finite_number(
            row.get("pf_energy_estimate_hartree"), f"{species}.pf_energy_estimate_hartree"
        )
        if not _close(pf_energy - exact_energy, signed):
            raise ValueError(f"{lane_id}: species energy/error arithmetic mismatch for {species}")
        if row.get("species") != species:
            raise ValueError(f"{lane_id}: species identity mismatch for {species}")
        formula_order = _nonnegative_integer(row.get("formula_order"), f"{species}.formula_order")
        n_steps = _nonnegative_integer(row.get("n_steps"), f"{species}.n_steps")
        if formula_order not in (1, 2) or n_steps <= 0:
            raise ValueError(f"{lane_id}: invalid formula order/step count for {species}")
        method_config = row.get("method_config")
        if not isinstance(method_config, dict):
            raise ValueError(f"{lane_id}: missing method_config for {species}")
        if (
            int(method_config.get("formula_order", -1)) != formula_order
            or int(method_config.get("n_steps", -1)) != n_steps
            or method_config.get("species") != species
        ):
            raise ValueError(f"{lane_id}: method config drift for {species}")
        zero_threshold = _finite_number(
            row.get("zero_angle_threshold"), f"{species}.zero_angle_threshold"
        )
        method_threshold = _finite_number(
            method_config.get("zero_angle_threshold"),
            f"{species}.method_config.zero_angle_threshold",
        )
        if zero_threshold < 0.0 or zero_threshold != method_threshold:
            raise ValueError(f"{lane_id}: zero-angle threshold drift for {species}")
        term_count = _nonnegative_integer(
            row.get("total_pauli_rotation_terms"), f"{species}.total_pauli_rotation_terms"
        )
        rotations_per_step = _nonnegative_integer(
            row.get("rotations_per_step"), f"{species}.rotations_per_step"
        )
        expected_rotations_per_step = term_count if formula_order == 1 else 2 * term_count - 1
        if rotations_per_step != expected_rotations_per_step:
            raise ValueError(f"{lane_id}: PF rotation-count mismatch for {species}")
        scheduled = _nonnegative_integer(
            row.get("scheduled_rotation_count"), f"{species}.scheduled_rotation_count"
        )
        nonzero = _nonnegative_integer(
            row.get("nonzero_rotation_count"), f"{species}.nonzero_rotation_count"
        )
        skipped = _nonnegative_integer(
            row.get("skipped_rotation_count"), f"{species}.skipped_rotation_count"
        )
        one_nonzero = _nonnegative_integer(
            row.get("one_step_nonzero_rotation_count"),
            f"{species}.one_step_nonzero_rotation_count",
        )
        one_skipped = _nonnegative_integer(
            row.get("one_step_skipped_rotation_count"),
            f"{species}.one_step_skipped_rotation_count",
        )
        if (
            scheduled != rotations_per_step * n_steps
            or nonzero + skipped != scheduled
            or one_nonzero + one_skipped != rotations_per_step
            or nonzero != one_nonzero * n_steps
            or skipped != one_skipped * n_steps
        ):
            raise ValueError(f"{lane_id}: scheduled/nonzero/skipped count mismatch for {species}")
        one_step_t = _nonnegative_integer(
            row.get("one_step_t_gate_count"), f"{species}.one_step_t_gate_count"
        )
        t_count_raw = _nonnegative_integer(row.get("t_gate_count"), f"{species}.t_gate_count")
        if t_count_raw != one_step_t * n_steps:
            raise ValueError(f"{lane_id}: T-count scaling mismatch for {species}")
        one_step_skipped_angle = _finite_number(
            row.get("one_step_skipped_absolute_angle_sum"),
            f"{species}.one_step_skipped_absolute_angle_sum",
        )
        skipped_angle = _finite_number(
            row.get("skipped_absolute_angle_sum"), f"{species}.skipped_absolute_angle_sum"
        )
        one_step_skip_bound = _finite_number(
            row.get("one_step_skipped_rotation_operator_error_bound"),
            f"{species}.one_step_skipped_rotation_operator_error_bound",
        )
        skip_bound = _finite_number(
            row.get("skipped_rotation_operator_error_bound"),
            f"{species}.skipped_rotation_operator_error_bound",
        )
        if not _close(skipped_angle, one_step_skipped_angle * n_steps) or not _close(
            skip_bound, one_step_skip_bound * n_steps
        ):
            raise ValueError(f"{lane_id}: skipped-rotation accounting mismatch for {species}")
        if (
            row.get("stationary_step_policy_verified") is not True
            or row.get("unitary_construction")
            != "compiled_one_step_product_formula_applied_and_diagonalized_in_full_hilbert_space"
            or row.get("phase_matching") != "maximum_ground_state_eigenvector_overlap"
            or row.get("t_count_semantics")
            != "exact_count_of_T_characters_in_selected_PyGridSynth_gate_strings_times_n_steps"
        ):
            raise ValueError(f"{lane_id}: fixed species evaluator semantics drift for {species}")
        signed_errors[species] = signed
        absolute_errors[species] = absolute
        species_t_counts[species] = t_count_raw
        exact_energies[species] = exact_energy
        pf_energies[species] = pf_energy
        species_scheduled_counts[species] = scheduled
    terms = [(float(item["coefficient"]), str(item["species"])) for item in lane["terms"]]
    if {species for _coefficient, species in terms} != set(expected_species):
        raise ValueError(f"{lane_id}: lane stoichiometry does not cover its species exactly")
    target = float(lane["reaction_target_hartree"])
    coefficient_norm = sum(abs(coefficient) for coefficient, _species in terms)
    equal_limit = target / coefficient_norm
    signed_reaction_error = sum(
        coefficient * signed_errors[species] for coefficient, species in terms
    )
    triangle_bound = sum(
        abs(coefficient) * absolute_errors[species] for coefficient, species in terms
    )
    max_species_error = max(absolute_errors.values())
    total_t_count = sum(species_t_counts.values())
    total_scheduled = sum(species_scheduled_counts.values())
    exact_reaction_energy = sum(
        coefficient * exact_energies[species] for coefficient, species in terms
    )
    pf_reaction_energy = sum(coefficient * pf_energies[species] for coefficient, species in terms)
    expected_numbers = {
        "reaction_target_hartree": target,
        "reaction_target_mha": 1000.0 * target,
        "stoichiometric_abs_coefficient_sum": coefficient_norm,
        "equal_share_error_limit_hartree": equal_limit,
        "equal_share_error_limit_mha": 1000.0 * equal_limit,
        "exact_reaction_energy_hartree": exact_reaction_energy,
        "pf_reaction_energy_hartree": pf_reaction_energy,
        "signed_reaction_error_hartree": signed_reaction_error,
        "signed_reaction_error_mha": 1000.0 * signed_reaction_error,
        "absolute_reaction_error_hartree": abs(signed_reaction_error),
        "absolute_reaction_error_mha": 1000.0 * abs(signed_reaction_error),
        "triangle_bound_hartree": triangle_bound,
        "triangle_bound_mha": 1000.0 * triangle_bound,
        "triangle_inequality_slack_hartree": triangle_bound - abs(signed_reaction_error),
        "max_species_absolute_error_hartree": max_species_error,
        "max_species_absolute_error_mha": 1000.0 * max_species_error,
    }
    for field, expected in expected_numbers.items():
        actual = _finite_number(result.get(field), field)
        if not _close(actual, expected):
            raise ValueError(f"{lane_id}: arithmetic mismatch for {field}")
    if abs(signed_reaction_error) > triangle_bound + 1.0e-12:
        raise ValueError(f"{lane_id}: triangle inequality is violated by result arithmetic")
    if result.get("triangle_inequality_holds") is not True:
        raise ValueError(f"{lane_id}: evaluator did not attest the triangle inequality")
    if result.get("total_t_gate_count") != total_t_count:
        raise ValueError(f"{lane_id}: total_t_gate_count is not the unique-species sum")
    if result.get("total_scheduled_rotation_count") != total_scheduled:
        raise ValueError(f"{lane_id}: total scheduled rotations are inconsistent")
    if not _close(pf_reaction_energy - exact_reaction_energy, signed_reaction_error):
        raise ValueError(f"{lane_id}: reaction-energy arithmetic mismatch")
    passes_equal_share = all(error <= equal_limit for error in absolute_errors.values())
    passes_reaction = abs(signed_reaction_error) <= target
    passes_contract = passes_equal_share if lane["contract"] == "equal_share" else passes_reaction
    flags = {
        "passes_equal_share_constraint": passes_equal_share,
        "passes_reaction_constraint": passes_reaction,
        "passes_contract": passes_contract,
    }
    for field, expected in flags.items():
        if result.get(field) is not expected:
            raise ValueError(f"{lane_id}: inconsistent {field}")
    violation_ratio = (
        max_species_error / equal_limit
        if lane["contract"] == "equal_share"
        else abs(signed_reaction_error) / target
    )
    expected_score = (
        float(total_t_count)
        if passes_contract
        else FAILURE_SCORE_BASE * (1.0 + max(1.0, violation_ratio)) ** 2 + float(total_t_count)
    )
    score = _finite_number(result.get("score"), "score")
    if not math.isclose(score, expected_score, rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(f"{lane_id}: score arithmetic mismatch")


def result_validation_error(
    path: Path,
    *,
    lane: dict[str, Any],
    iteration: int,
    candidate_hash: str,
    campaign_id: str,
    campaign_manifest_sha256: str,
) -> str | None:
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise TypeError("result JSON is not an object")
        validate_result(
            payload,
            lane=lane,
            iteration=iteration,
            candidate_hash=candidate_hash,
            campaign_id=campaign_id,
            campaign_manifest_sha256=campaign_manifest_sha256,
        )
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def relative_to_campaign(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(campaign_paths(root)["state"], state)
