"""Shared, deterministic helpers for the paper-figure entry points."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from hsopt.view.reporting import verify_strict_export


CHEMICAL_ACCURACY_HARTREE = 1.0 / 627.5094740631
CHEMICAL_ACCURACY_MHA = 1000.0 * CHEMICAL_ACCURACY_HARTREE

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK_GREEN = "#3E7E00"
LIGHT_GREEN = "#C7E596"
GRAPHITE = "#4D4D4D"
LIGHT_GRAPHITE = "#A0A0A0"
PALE_GRAPHITE = "#D6D6D6"
RED = "#C00000"


def apply_paper_style(*, compact: bool = False) -> None:
    """Apply the manuscript's LaTeX-rendered Matplotlib style."""

    base = 15 if compact else 18
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": base,
            "axes.labelsize": base + 2,
            "axes.titlesize": base + 2,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 2,
            "figure.titlesize": base + 4,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def strict_export(path: Path, family: str) -> tuple[Path, dict[str, Any]]:
    """Verify a completed export and return its resolved path and manifest."""

    path = path.expanduser().resolve()
    verify_strict_export(path)
    manifest = read_json(path / "export_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("family") != family:
        raise ValueError(f"{path} is not a complete {family} export")
    return path, manifest


def as_float(value: object, *, label: str = "value") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def as_int(value: object, *, label: str = "value") -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_rows(
    rows: Iterable[Mapping[str, str]],
    *fields: str,
) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    return dict(grouped)


def output_file(output_dir: Path, filename: str) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def external_output_dir(output_dir: Path, *read_roots: Path) -> Path:
    """Refuse to place generated files inside immutable input or export trees."""

    output_dir = output_dir.expanduser().resolve()
    for root in read_roots:
        root = root.expanduser().resolve()
        if root.is_file():
            root = root.parent
        if output_dir == root or output_dir.is_relative_to(root):
            raise ValueError(f"output directory must be outside immutable input tree: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def deterministic_sample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size != y.size:
        raise ValueError("paired arrays have different lengths")
    if x.size <= maximum:
        return x, y
    indices = np.random.default_rng(seed).choice(x.size, size=maximum, replace=False)
    indices.sort()
    return x[indices], y[indices]


def molecule_label(name: str) -> str:
    """Return a compact LaTeX label for a simple molecular formula."""

    charge = ""
    if name.endswith("+") or name.endswith("-"):
        charge = rf"^{{{name[-1]}}}"
        name = name[:-1]
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", name)
    if not tokens or "".join(element + count for element, count in tokens) != name:
        return name.replace("_", r"\_")
    body = "".join(
        rf"\mathrm{{{element}}}" + (rf"_{{{count}}}" if count else "") for element, count in tokens
    )
    return f"${body}{charge}$"


REACTION_LABELS = {
    "hcn_hnc": r"$\mathrm{HCN}\!\to\!\mathrm{HNC}$",
    "hco_hoc_cation": r"$\mathrm{HCO}^{+}\!\to\!\mathrm{HOC}^{+}$",
    "hno_noh": r"$\mathrm{HNO}\!\to\!\mathrm{NOH}$",
    "h2co_hcoh": r"$\mathrm{H}_{2}\mathrm{CO}\!\to\!\mathrm{HCOH}$",
    "acetylene_vinylidene": r"$\mathrm{HCCH}\!\to\!\mathrm{H}_{2}\mathrm{CC}$",
    "h2_cl2_hcl": r"$\mathrm{H}_{2}+\mathrm{Cl}_{2}\!\to\!2\mathrm{HCl}$",
    "n2_h2_nh3": r"$\mathrm{N}_{2}+3\mathrm{H}_{2}\!\to\!2\mathrm{NH}_{3}$",
    "co_h2o_co2_h2": (
        r"$\mathrm{CO}+\mathrm{H}_{2}\mathrm{O}\!\to\!"
        r"\mathrm{CO}_{2}+\mathrm{H}_{2}$"
    ),
    "h2_f2_hf": r"$\mathrm{H}_{2}+\mathrm{F}_{2}\!\to\!2\mathrm{HF}$",
    "co_h2_ch4_h2o": (
        r"$\mathrm{CO}+3\mathrm{H}_{2}\!\to\!"
        r"\mathrm{CH}_{4}+\mathrm{H}_{2}\mathrm{O}$"
    ),
}

REACTION_ORDER = tuple(REACTION_LABELS)
