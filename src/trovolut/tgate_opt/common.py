from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PFS_ROOT_RELATIVE = Path("examples/ft_protocols/pfs")
PHASE_OBJECTIVE = "ground_state_phase_estimation"


def _find_repo_root(start: Path) -> Path:
    override = os.environ.get("TROVOLUT_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / PFS_ROOT_RELATIVE / "model_registry.py").exists()
        ):
            return candidate
    raise RuntimeError(
        "could not locate trovolut repo root; set TROVOLUT_REPO_ROOT or run inside the repository"
    )


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
DEFAULT_FAMILY = "pf1"
DEFAULT_MOLECULE = "h2"


@dataclass(frozen=True)
class Lane:
    family: str
    molecule: str
    display_name: str
    root_dir: Path
    lane_dir: Path
    editable_script: Path
    snapshots_dir: Path
    results_path: Path
    instruction_path: Path | None
    max_parallel: int = 1
    target_iteration: int = 100
    supported: bool = True
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.family}:{self.molecule}"


def repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def _registry_lanes(repo_root: Path, family: str, relative_path: str) -> list[Lane]:
    root = repo_root / relative_path
    registry = root / "model_registry.py"
    evaluator = root / "evaluate.py"
    queued = root / "queued_track_iteration.py"
    if not registry.exists():
        return []
    module_name = f"_trovolut_{family.replace('-', '_')}_registry"
    module = _load_module(registry, module_name)
    lanes: list[Lane] = []
    for molecule in getattr(module, "ACTIVE_MOLECULES", ()):
        spec = module.get_spec(molecule)
        supported = evaluator.exists() and queued.exists() and Path(spec.editable_script).exists()
        lanes.append(
            Lane(
                family=family,
                molecule=molecule,
                display_name=str(spec.display_name),
                root_dir=root,
                lane_dir=Path(spec.lane_dir),
                editable_script=Path(spec.editable_script),
                snapshots_dir=Path(spec.snapshots_dir),
                results_path=Path(spec.results_path),
                instruction_path=Path(spec.instruction_path),
                max_parallel=int(getattr(spec, "max_parallel", 1)),
                target_iteration=int(getattr(spec, "target_iteration", 100)),
                supported=supported,
                notes="" if supported else "Scaffold is present but evaluator/live simple_pf.py is missing.",
            )
        )
    return lanes


def discover_lanes(repo_root: Path = REPO_ROOT) -> list[Lane]:
    lanes: list[Lane] = []
    lanes.extend(_registry_lanes(repo_root, "pf1", str(PFS_ROOT_RELATIVE)))
    return lanes


def _candidate_target_paths(target: str | Path, repo_root: Path) -> list[Path]:
    raw = Path(target)
    if raw.is_absolute():
        return [raw.resolve()]

    candidates = [
        (Path.cwd() / raw).resolve(),
        (repo_root / raw).resolve(),
    ]
    if raw.parts and raw.parts[0] == "pfs":
        candidates.append((repo_root / "examples/ft_protocols" / raw).resolve())

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def lane_for_path(target: str | Path, repo_root: Path = REPO_ROOT, require_supported: bool = True) -> Lane:
    paths = _candidate_target_paths(target, repo_root)

    matches: list[Lane] = []
    for lane in discover_lanes(repo_root):
        lane_dir = lane.lane_dir.resolve()
        for path in paths:
            if path == lane_dir or path.is_relative_to(lane_dir):
                matches.append(lane)
                break
    if not matches:
        checked = ", ".join(str(path) for path in paths)
        raise SystemExit(f"no T-gate PF lane found for path: {target}; checked: {checked}")
    lane = sorted(matches, key=lambda item: len(str(item.lane_dir)), reverse=True)[0]
    if require_supported and not lane.supported:
        raise SystemExit(f"lane {lane.key} is not runnable yet: {lane.notes}")
    return lane


def resolve_lane(
    family: str | None = None,
    molecule: str | None = None,
    repo_root: Path = REPO_ROOT,
    require_supported: bool = True,
    target: str | Path | None = None,
) -> Lane:
    if target is not None:
        return lane_for_path(target, repo_root=repo_root, require_supported=require_supported)
    family = family or DEFAULT_FAMILY
    molecule = molecule or DEFAULT_MOLECULE
    matches = [lane for lane in discover_lanes(repo_root) if lane.family == family and lane.molecule == molecule]
    if not matches:
        available = ", ".join(lane.key for lane in discover_lanes(repo_root)) or "none"
        raise SystemExit(f"unknown lane {family}:{molecule}; available lanes: {available}")
    lane = matches[0]
    if require_supported and not lane.supported:
        raise SystemExit(f"lane {lane.key} is not runnable yet: {lane.notes}")
    return lane


def next_iteration(lane: Lane) -> int:
    if not lane.snapshots_dir.exists():
        return 0
    values: list[int] = []
    for child in lane.snapshots_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            values.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(values, default=-1) + 1


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def _phase_accuracy_row(row: dict[str, str]) -> bool:
    objective = (row.get("objective") or "").strip()
    if objective and objective != PHASE_OBJECTIVE:
        return False
    if "passes_accuracy_constraint" not in row:
        return False
    return _truthy(row.get("passes_accuracy_constraint"))


def best_row(lane: Lane) -> dict[str, str] | None:
    if not lane.results_path.exists():
        return None
    best: dict[str, str] | None = None
    best_score: float | None = None
    with lane.results_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            if not _phase_accuracy_row(row):
                continue
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if best_score is None or score < best_score:
                best = row
                best_score = score
    return best


def baseline_result(lane: Lane) -> dict[str, Any] | None:
    path = lane.snapshots_dir / "iter_0000" / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def summarize_lane(lane: Lane) -> dict[str, Any]:
    baseline = baseline_result(lane)
    best = best_row(lane)
    baseline_t = baseline.get("t_gate_count") if baseline else None
    best_t: int | None = None
    if best and best.get("t_gate_count") not in {None, ""}:
        try:
            best_t = int(float(best["t_gate_count"]))
        except ValueError:
            best_t = None
    reduction_pct = None
    if isinstance(baseline_t, int) and best_t is not None and baseline_t > 0:
        reduction_pct = 100.0 * (baseline_t - best_t) / baseline_t
    return {
        "key": lane.key,
        "family": lane.family,
        "molecule": lane.molecule,
        "display_name": lane.display_name,
        "supported": lane.supported,
        "notes": lane.notes,
        "editable_script": repo_relative(lane.editable_script),
        "last_archived_iteration": next_iteration(lane) - 1,
        "target_iteration": lane.target_iteration,
        "baseline_t_gate_count": baseline_t,
        "best_iteration": int(best["iteration"]) if best and best.get("iteration") else None,
        "best_score": float(best["score"]) if best and best.get("score") else None,
        "best_t_gate_count": best_t,
        "t_gate_reduction_pct": reduction_pct,
    }


def evaluate_command(lane: Lane, script: Path | None = None) -> list[str]:
    script = script or lane.editable_script
    return [
        "uv",
        "run",
        "python",
        str(lane.root_dir / "evaluate.py"),
        "--molecule",
        lane.molecule,
        "--script",
        str(script),
    ]


def track_command(lane: Lane, description: str, script: Path | None = None) -> list[str]:
    script = script or lane.editable_script
    return [
        "uv",
        "run",
        "python",
        str(lane.root_dir / "queued_track_iteration.py"),
        "--molecule",
        lane.molecule,
        "--script",
        str(script),
        "--max-parallel",
        str(lane.max_parallel),
        "--description",
        description,
    ]


def restore_command(lane: Lane, script: Path | None = None) -> list[str]:
    script = script or lane.editable_script
    return [
        "uv",
        "run",
        "python",
        str(lane.root_dir / "restore_best_iteration.py"),
        "--molecule",
        lane.molecule,
        "--script",
        str(script),
    ]


def shell_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)
