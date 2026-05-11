from __future__ import annotations

import importlib.util
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PFS_ROOT_RELATIVE = Path("examples/pfs")
DEFAULT_FAMILY = "pf"
DEFAULT_MOLECULE = "h2"
DEFAULT_EVALUATOR = "energy"
FAMILY_ALIASES = {
    "pf": "pf",
    "pf1": "pf",
    "pfs": "pf",
    "product_formula": "pf",
}


def _find_repo_root(start: Path) -> Path:
    override = os.environ.get("HSOPT_REPO_ROOT") or os.environ.get("TROVOLUT_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (start, *start.parents):
        pfs_root = candidate / PFS_ROOT_RELATIVE
        if (
            (candidate / "pyproject.toml").exists()
            and (pfs_root / "chemistry.py").exists()
            and (pfs_root / "evaluate_energy.py").exists()
            and (pfs_root / "evaluate_fidelity.py").exists()
        ):
            return candidate
    raise RuntimeError(
        "could not locate hsopt repo root; set HSOPT_REPO_ROOT or run inside the repository"
    )


REPO_ROOT = _find_repo_root(Path(__file__).resolve())


@dataclass(frozen=True)
class Lane:
    family: str
    molecule: str
    display_name: str
    root_dir: Path
    lane_dir: Path
    editable_script: Path
    evaluator_wrapper: Path
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


def _pfs_lanes(repo_root: Path, family: str, relative_path: Path) -> list[Lane]:
    root = repo_root / relative_path
    chemistry = root / "chemistry.py"
    if not chemistry.exists():
        return []
    module = _load_module(chemistry, f"_hsopt_{family}_chemistry")
    lanes: list[Lane] = []
    for molecule in getattr(module, "ACTIVE_MOLECULES", ()):
        spec = module.get_spec(molecule)
        lane_dir = Path(spec.lane_dir)
        editable_script = Path(spec.editable_script)
        evaluator_wrapper = lane_dir / "evaluate.py"
        supported = (
            editable_script.exists()
            and evaluator_wrapper.exists()
            and (root / "evaluate_energy.py").exists()
            and (root / "evaluate_fidelity.py").exists()
        )
        lanes.append(
            Lane(
                family=family,
                molecule=molecule,
                display_name=str(spec.display_name),
                root_dir=root,
                lane_dir=lane_dir,
                editable_script=editable_script,
                evaluator_wrapper=evaluator_wrapper,
                target_iteration=int(getattr(spec, "target_iteration", 100)),
                supported=supported,
                notes="" if supported else "Missing simple_pf.py or one of the two evaluator entry points.",
            )
        )
    return lanes


def discover_lanes(repo_root: Path = REPO_ROOT) -> list[Lane]:
    return _pfs_lanes(repo_root, DEFAULT_FAMILY, PFS_ROOT_RELATIVE)


def normalize_family(family: str | None) -> str:
    value = (family or DEFAULT_FAMILY).strip().lower()
    return FAMILY_ALIASES.get(value, value)


def _candidate_target_paths(target: str | Path, repo_root: Path) -> list[Path]:
    raw = Path(target)
    if raw.is_absolute():
        return [raw.resolve()]

    candidates = [
        (Path.cwd() / raw).resolve(),
        (repo_root / raw).resolve(),
    ]
    if raw.parts and raw.parts[0] == "pfs":
        candidates.append((repo_root / "examples" / raw).resolve())
    if len(raw.parts) >= 3 and raw.parts[:3] == ("examples", "ft_protocols", "pfs"):
        candidates.append((repo_root / "examples" / "pfs" / Path(*raw.parts[3:])).resolve())

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
    family = normalize_family(family)
    molecule = molecule or DEFAULT_MOLECULE
    matches = [lane for lane in discover_lanes(repo_root) if lane.family == family and lane.molecule == molecule]
    if not matches:
        available = ", ".join(lane.key for lane in discover_lanes(repo_root)) or "none"
        raise SystemExit(f"unknown lane {family}:{molecule}; available lanes: {available}")
    lane = matches[0]
    if require_supported and not lane.supported:
        raise SystemExit(f"lane {lane.key} is not runnable yet: {lane.notes}")
    return lane


def normalize_evaluator(evaluator: str | None) -> str:
    value = (evaluator or DEFAULT_EVALUATOR).strip().lower()
    if value in {"energy", "phase", "qpe"}:
        return "energy"
    if value in {"fidelity", "state"}:
        return "fidelity"
    raise SystemExit(f"unknown evaluator {evaluator!r}; use 'energy' or 'fidelity'")


def evaluator_path(lane: Lane, evaluator: str | None = None) -> Path:
    selected = normalize_evaluator(evaluator)
    return lane.root_dir / f"evaluate_{selected}.py"


def next_iteration(_lane: Lane) -> int:
    return 0


def summarize_lane(lane: Lane) -> dict[str, Any]:
    return {
        "key": lane.key,
        "family": lane.family,
        "molecule": lane.molecule,
        "display_name": lane.display_name,
        "supported": lane.supported,
        "notes": lane.notes,
        "editable_script": repo_relative(lane.editable_script),
        "evaluator_wrapper": repo_relative(lane.evaluator_wrapper),
        "energy_evaluator": repo_relative(lane.root_dir / "evaluate_energy.py"),
        "fidelity_evaluator": repo_relative(lane.root_dir / "evaluate_fidelity.py"),
        "target_iteration": lane.target_iteration,
    }


def evaluate_command(
    lane: Lane,
    script: Path | None = None,
    evaluator: str | None = None,
    epsilon: float | None = None,
) -> list[str]:
    selected = normalize_evaluator(evaluator)
    script = script or lane.editable_script
    command = [
        "uv",
        "run",
        "python",
        str(evaluator_path(lane, selected)),
        "--molecule",
        lane.molecule,
        "--script",
        str(script),
    ]
    if selected == "fidelity" and epsilon is not None:
        command.extend(["--epsilon", str(epsilon)])
    return command


def track_command(
    lane: Lane,
    _description: str,
    script: Path | None = None,
    evaluator: str | None = None,
    epsilon: float | None = None,
) -> list[str]:
    return evaluate_command(lane, script=script, evaluator=evaluator, epsilon=epsilon)


def restore_command(_lane: Lane, _script: Path | None = None) -> list[str]:
    return [
        "python",
        "-c",
        "print('{\"status\":\"noop\",\"reason\":\"simplified pfs layout has no in-tree snapshots\"}')",
    ]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
