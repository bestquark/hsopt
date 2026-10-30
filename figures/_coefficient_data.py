"""Pinned HamLib coefficient inputs for the Figure 1 illustration inset."""

from dataclasses import dataclass
import hashlib
import json
import matplotlib as mpl
import numpy as np
from _paths import ROOT

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK_GREEN = "#3E7E00"
ZERO_THRESHOLD = 1e-15


@dataclass(frozen=True)
class HamLibHamiltonian:
    label: str
    collection: str
    archive: str
    dataset: str
    nqubits: int
    terms: int


INSTANCES = (
    HamLibHamiltonian(
        label=r"$\mathrm{CoH}$",
        collection="transition_metals",
        archive="CoH.zip",
        dataset="ham_JW30",
        nqubits=30,
        terms=56832,
    ),
    HamLibHamiltonian(
        label=r"$\mathrm{O}_3$",
        collection="standard",
        archive="O3.zip",
        dataset="ham_JW22",
        nqubits=22,
        terms=5466,
    ),
    HamLibHamiltonian(
        label=r"$\mathrm{Na}_2$",
        collection="standard",
        archive="Na2.zip",
        dataset="ham_JW20",
        nqubits=20,
        terms=2951,
    ),
    HamLibHamiltonian(
        label=r"$\mathrm{NaLi}$",
        collection="standard",
        archive="NaLi.zip",
        dataset="ham_JW-18",
        nqubits=18,
        terms=3720,
    ),
)


def apply_mlp_style() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "figure.titlesize": 24,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.frameon": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def coefficient_magnitudes(instance):
    path = ROOT / "experiments/abs_protocol/results/hamlib-diagnostics/schematic_coefficients.npz"
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        key = instance.archive.removesuffix(".zip")
        values = archive[key]
        row = metadata["instances"][key]
        assert row["dataset"] == instance.dataset
        assert (
            hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            == row["array_sha256"]
        )
        return values
