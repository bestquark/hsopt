"""
Generic hardware abstraction for trovolut.

Provides backend-agnostic base classes for describing hardware units and
composable clusters.  The core layer only stores a ``name`` and a free-form
``specs`` dictionary -- backend subclasses (GPU, CPUSpec, QPUSpec, ...)
add domain-specific typed accessors.

Example (raw ``BaseHardware``)::

    cpu = BaseHardware(
        "EPYC-9654",
        specs={"memory_bytes": 768 * 1024**3, "cores": 96, "fp64_flops": 5.4e12},
    )

    qpu = BaseHardware(
        "IBM-Eagle",
        specs={"n_qubits": 127, "t1_us": 100, "gate_fidelity": 0.999},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# BaseHardware
# ============================================================

@dataclass(frozen=True)
class BaseHardware:
    """Single hardware unit -- generic enough for GPUs, CPUs, QPUs, etc.

    Only ``name`` and ``specs`` live at the base level.  Domain-specific
    fields (``memory_bytes``, ``clock_hz``, ``n_qubits``, ...) belong in
    backend subclasses that expose them as typed ``@property`` accessors
    reading from the ``specs`` dict.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"A100-80GB"``, ``"IBM-Eagle"``).
    specs : dict[str, float]
        Free-form key-value specifications.  Subclasses add typed
        ``@property`` accessors that delegate to this dict.
    """

    name: str
    specs: dict[str, float] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------

    def spec(self, key: str, default: float = 0.0) -> float:
        """Lookup a spec by key (e.g. ``'fp32_flops'``, ``'n_qubits'``)."""
        return self.specs.get(key, default)

    # -- composability (returns generic HardwareCluster) -------------------

    def __mul__(self, n: int) -> "HardwareCluster":
        if not isinstance(n, int):
            return NotImplemented
        return HardwareCluster(unit=self, count=n)

    def __rmul__(self, n: int) -> "HardwareCluster":
        return self.__mul__(n)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


# ============================================================
# HardwareCluster
# ============================================================

class HardwareCluster:
    """Composable collection of :class:`BaseHardware` (or subclass) units.

    Supports arithmetic composition::

        cluster = 4 * gpus.A100               # homogeneous
        cluster = 2 * gpus.A100 + 2 * gpus.H100  # heterogeneous

    And aggregation queries::

        cluster.total_spec("memory_bytes")
        cluster.total_spec("fp16_flops")
        cluster.max_spec("memory_bandwidth")
    """

    def __init__(self, unit: BaseHardware, count: int = 1) -> None:
        self.units: list[tuple[BaseHardware, int]] = [(unit, count)]

    # -- scalar properties ------------------------------------------------

    @property
    def total_memory_bytes(self) -> int:
        """Total memory across all units (reads ``specs["memory_bytes"]``)."""
        return int(self.total_spec("memory_bytes"))

    @property
    def total_units(self) -> int:
        """Total number of hardware units."""
        return sum(c for _, c in self.units)

    # -- spec aggregation -------------------------------------------------

    def total_spec(self, key: str) -> float:
        """Sum a spec across all units (e.g. total ``fp32_flops``)."""
        return sum(u.spec(key) * c for u, c in self.units)

    def max_spec(self, key: str) -> float:
        """Max of a spec across unit types."""
        values = [u.spec(key) for u, _ in self.units if u.spec(key) != 0.0]
        return max(values) if values else 0.0

    # -- arithmetic -------------------------------------------------------

    def __mul__(self, n: int) -> "HardwareCluster":
        if not isinstance(n, int):
            return NotImplemented
        new = HardwareCluster.__new__(HardwareCluster)
        new.units = [(u, c * n) for u, c in self.units]
        return new

    def __rmul__(self, n: int) -> "HardwareCluster":
        return self.__mul__(n)

    def __add__(self, other: "HardwareCluster | BaseHardware") -> "HardwareCluster":
        if isinstance(other, BaseHardware):
            other = HardwareCluster(other, 1)
        if not isinstance(other, HardwareCluster):
            return NotImplemented
        new = HardwareCluster.__new__(HardwareCluster)
        new.units = list(self.units) + list(other.units)
        return new

    def __radd__(self, other: Any) -> "HardwareCluster":
        if isinstance(other, (BaseHardware, HardwareCluster)):
            if isinstance(other, BaseHardware):
                other = HardwareCluster(other, 1)
            return other.__add__(self)
        return NotImplemented

    # -- display ----------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for u, c in self.units:
            parts.append(f"{c}x {u.name}")
        return f"HardwareCluster({', '.join(parts)})"


# ============================================================
# Exports
# ============================================================

__all__ = [
    "BaseHardware",
    "HardwareCluster",
]
