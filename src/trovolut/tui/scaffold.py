"""Pure template-generation logic for evaluator and workflow scaffolds.

Every public function returns a string (no I/O).  Screens handle file
writing and user notifications.
"""

from __future__ import annotations

from typing import Any

# ── Mapping tables ────────────────────────────────────────────────────

RUNNER_MAP: dict[str, str] = {
    "cpu": "CPURunner",
    "gpu": "LegateRunner",
    "gpu-mp-emulator": "CPURunner",
    "qpu": "QPURunner",
}

BACKEND_CONSTRAINTS: dict[str, list[str]] = {
    "cpu": ["Time", "Error"],
    "gpu": ["Time", "Error", "LegateGPUSeconds", "LegateFramebuffer", "LegateMakespan"],
    "gpu-mp-emulator": ["Time", "Error", "MemoryCost", "FlopsCost", "SpaceTimeCost", "MatMulCount"],
    "qpu": ["Time", "Error"],
}

CONSTRAINT_IMPORTS: dict[str, str] = {
    "Time": "trovolut.core.base_constraints",
    "Error": "trovolut.core.base_constraints",
    "LegateGPUSeconds": "trovolut.backends.gpu.constraints",
    "LegateFramebuffer": "trovolut.backends.gpu.constraints",
    "LegateMakespan": "trovolut.backends.gpu.constraints",
    "MemoryCost": "trovolut.backends.gpu_mp_emulator.constraints",
    "FlopsCost": "trovolut.backends.gpu_mp_emulator.constraints",
    "SpaceTimeCost": "trovolut.backends.gpu_mp_emulator.constraints",
    "MatMulCount": "trovolut.backends.gpu_mp_emulator.constraints",
}

CONSTRAINT_PARAMS: dict[str, list[dict[str, Any]]] = {
    "Time": [
        {"name": "max_seconds", "type": "float", "default": 30.0},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "Error": [
        {"name": "max_error", "type": "float", "default": 1.0},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "inverse"},
    ],
    "LegateGPUSeconds": [
        {"name": "max_gpu_seconds", "type": "float", "default": 10.0},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "LegateFramebuffer": [
        {"name": "max_mb", "type": "float", "default": 1000.0},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "LegateMakespan": [
        {"name": "max_seconds", "type": "float", "default": 30.0},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "MemoryCost": [
        {"name": "max_memory_bytes", "type": "float", "default": 104857600.0},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "FlopsCost": [
        {"name": "max_flops", "type": "float", "default": 1e12},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "SpaceTimeCost": [
        {"name": "max_product", "type": "float", "default": 1e15},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
    "MatMulCount": [
        {"name": "max_count", "type": "int", "default": 100},
        {"name": "weight", "type": "float", "default": 1.0},
        {"name": "hard", "type": "bool", "default": False},
        {"name": "score_fn", "type": "str", "default": "linear"},
    ],
}

EVOLUTION_METHODS: dict[str, list[str]] = {
    "cpu": ["TrotterEvolution", "ExactEvolution", "QDriftEvolution"],
    "gpu-mp-emulator": [
        "MPTrotterEvolution",
        "MPQDriftEvolution",
        "MPTaylorEvolution",
        "MPPadeEvolution",
    ],
    "gpu": ["TrotterEvolution", "ExactEvolution", "QDriftEvolution"],
    "qpu": ["TrotterEvolution", "QDriftEvolution"],
}


# ── Evaluator scaffold ────────────────────────────────────────────────


def generate_evaluator_scaffold(
    backend: str,
    constraints_config: list[dict[str, Any]],
    run_fn_name: str = "run_dynamics",
    n_trials: int = 10,
    timeout: int = 30,
    gpus: int = 1,
    cpus: int = 4,
    fbmem_mb: int = 256,
) -> str:
    """Return the text of an OpenEvolve-compatible evaluator file.

    Parameters
    ----------
    backend:
        One of ``cpu``, ``gpu``, ``gpu-mp-emulator``, ``qpu``.
    constraints_config:
        List of dicts, each ``{"type": "Time", "params": {"max_seconds": 30.0, ...}}``.
    run_fn_name:
        Name of the entry-point function in the initial program.
    n_trials / timeout:
        Runner parameters.
    gpus / cpus / fbmem_mb:
        Legate runner parameters (only used when *backend* == ``"gpu"``).
    """
    runner_cls = RUNNER_MAP[backend]
    lines: list[str] = []

    # ── imports ────────────────────────────────────────────
    lines.append("import numpy as np")
    lines.append("import scipy")
    lines.append("import trovolut as trv")
    lines.append(f"trv.set_backend({backend!r})")
    lines.append(f"from trovolut.filemaker.runners import {runner_cls}")
    lines.append("from trovolut.core.base_constraints import Group")

    # Collect constraint imports (deduplicate by module)
    constraint_types_used: list[str] = [c["type"] for c in constraints_config]
    imports_by_module: dict[str, list[str]] = {}
    for ct in constraint_types_used:
        mod = CONSTRAINT_IMPORTS[ct]
        imports_by_module.setdefault(mod, [])
        if ct not in imports_by_module[mod]:
            imports_by_module[mod].append(ct)
    for mod, names in sorted(imports_by_module.items()):
        lines.append(f"from {mod} import {', '.join(sorted(names))}")

    lines.append("")
    lines.append("")

    # ── helper functions ──────────────────────────────────
    lines.append("# --- Helper functions (fill in your problem-specific logic) ---")
    lines.append("")
    lines.append("_cached_hams = None")
    lines.append("")
    lines.append("")
    lines.append("def get_hamiltonian():")
    lines.append('    """Build Hamiltonian terms for your system."""')
    lines.append("    # TODO: Construct your Hamiltonian here.")
    lines.append("    #   Example: trv.qm.ElectronicHamiltonian.from_xyz(...).to_qubits()")
    lines.append("    raise NotImplementedError")
    lines.append("")
    lines.append("")
    lines.append("def get_test_cases():")
    lines.append('    """Generate a random test case (hams, psi0, t)."""')
    lines.append("    global _cached_hams")
    lines.append("    if _cached_hams is None:")
    lines.append("        _cached_hams = get_hamiltonian()")
    lines.append("    dim = _cached_hams[0].shape[0]")
    lines.append("    psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)")
    lines.append("    psi0 /= np.linalg.norm(psi0)")
    lines.append("    t = 1.0")
    lines.append("    return _cached_hams, psi0, t")
    lines.append("")
    lines.append("")
    lines.append("def exact_fn(hams, t, psi0):")
    lines.append('    """Exact evolution via matrix exponential."""')
    lines.append("    return scipy.linalg.expm(-1j * sum(hams) * t) @ psi0")
    lines.append("")
    lines.append("")

    # ── runner ────────────────────────────────────────────
    lines.append("# --- Runner and constraints ---")
    if runner_cls == "LegateRunner":
        lines.append(
            f"runner = {runner_cls}(gpus={gpus}, cpus={cpus}, fbmem_mb={fbmem_mb})"
        )
    elif runner_cls == "QPURunner":
        lines.append(f"runner = {runner_cls}()")
    else:
        lines.append(f"runner = {runner_cls}()")
    lines.append("")

    # ── constraints ───────────────────────────────────────
    constraint_exprs: list[str] = []
    for c in constraints_config:
        ct = c["type"]
        params = c.get("params", {})
        parts = [f"{k}={v!r}" for k, v in params.items()]
        constraint_exprs.append(f"{ct}({', '.join(parts)})")

    if constraint_exprs:
        inner = ", ".join(constraint_exprs)
        lines.append(f"constraints = Group([{inner}])")
    else:
        lines.append("constraints = Group([])")
    lines.append("")
    lines.append("")

    # ── evaluate() ────────────────────────────────────────
    lines.append("def evaluate(program_path):")
    lines.append('    """OpenEvolve-compatible evaluator entry point."""')
    lines.append("    return runner.evaluate(")
    lines.append("        program_path,")
    lines.append("        test_generator=get_test_cases,")
    lines.append("        exact_fn=exact_fn,")
    lines.append("        constraints=constraints,")
    lines.append(f"        n_trials={n_trials},")
    lines.append(f"        timeout={timeout},")
    lines.append(f"        run_fn_name={run_fn_name!r},")
    lines.append("    )")
    lines.append("")
    lines.append("")
    lines.append("evaluator = evaluate")
    lines.append("")
    lines.append("")

    # ── __main__ ──────────────────────────────────────────
    lines.append('if __name__ == "__main__":')
    lines.append("    import sys, argparse")
    lines.append("    if len(sys.argv) > 1:")
    lines.append("        parser = argparse.ArgumentParser()")
    lines.append('        parser.add_argument("--program_path", type=str, required=True)')
    lines.append("        args = parser.parse_args()")
    lines.append("        result = evaluate(args.program_path)")
    lines.append('        print(f"combined_score: {result[\'combined_score\']}")')
    lines.append('        print(f"std_score: {result[\'std_score\']}")')
    lines.append("")

    return "\n".join(lines)


# ── Workflow scaffold ─────────────────────────────────────────────────

_EVOLUTION_IMPORT_MAP: dict[str, str] = {
    # cpu
    "TrotterEvolution": "trovolut.backends.cpu.dynamics",
    "ExactEvolution": "trovolut.backends.cpu.dynamics",
    "QDriftEvolution": "trovolut.backends.cpu.dynamics",
    # gpu-mp-emulator
    "MPTrotterEvolution": "trovolut.backends.gpu_mp_emulator.dynamics",
    "MPQDriftEvolution": "trovolut.backends.gpu_mp_emulator.dynamics",
    "MPTaylorEvolution": "trovolut.backends.gpu_mp_emulator.dynamics",
    "MPPadeEvolution": "trovolut.backends.gpu_mp_emulator.dynamics",
}

_IS_MP_METHOD: set[str] = {
    "MPTrotterEvolution",
    "MPQDriftEvolution",
    "MPTaylorEvolution",
    "MPPadeEvolution",
}


def generate_workflow_scaffold(
    backend: str,
    evolution_method: str,
    trotter_steps: int = 50,
    order: int = 2,
    include_api_reference: bool = True,
    include_evo_markers: bool = True,
) -> str:
    """Return the text of an initial-program (workflow) file.

    Parameters
    ----------
    backend:
        Backend name.
    evolution_method:
        Class name, e.g. ``"MPTrotterEvolution"``.
    trotter_steps:
        Number of product-formula steps.
    order:
        Suzuki-Trotter order (1, 2, 4, …).
    include_api_reference:
        Embed class docstrings as comments for LLM context.
    include_evo_markers:
        Wrap the tunable block with ``# EVOLVE-BLOCK-START/END`` markers.
    """
    is_mp = evolution_method in _IS_MP_METHOD
    lines: list[str] = []

    # ── imports ────────────────────────────────────────────
    lines.append("import scipy")
    lines.append("import numpy as np")
    lines.append("import trovolut as trv")
    lines.append(f"trv.set_backend({backend!r})")
    lines.append("")

    # ── optional API reference block ──────────────────────
    if include_api_reference:
        lines.append("# " + "=" * 70)
        lines.append("# API Reference (auto-generated for LLM context)")
        lines.append("# You may use or modify calls to any of these in the EVOLVE-BLOCK.")
        lines.append("# " + "=" * 70)
        if is_mp:
            lines.append("#")
            lines.append("# Key classes for mixed-precision evolution:")
            lines.append("#   trv.dynamics.SuzukiTrotter()     — 2nd-order product formula")
            lines.append("#   trv.dynamics.LieTrotter()         — 1st-order product formula")
            lines.append("#   trv.dynamics.PrecisionConfig(compute, storage, accumulate)")
            lines.append("#   trv.dynamics.Precision.FP64 / FP32 / FP16 / BF16 / FP8_E4M3 / …")
            lines.append(f"#   trv.dynamics.{evolution_method}(trotter_steps, formula, precision_schedule)")
        else:
            lines.append("#")
            lines.append("# Key classes for dynamics:")
            lines.append(f"#   trv.dynamics.{evolution_method}")
        lines.append("")

    # ── run_dynamics ──────────────────────────────────────
    lines.append("def run_dynamics(hams, t, psi0):")
    if is_mp:
        lines.append(f'    """Mixed-precision evolution ({evolution_method}, order {order}, {trotter_steps} steps)."""')
    else:
        lines.append(f'    """Time evolution ({evolution_method}, order {order}, {trotter_steps} steps)."""')
    lines.append("    import numpy as np")
    lines.append("")

    # Tunable block
    if include_evo_markers:
        lines.append("    # EVOLVE-BLOCK-START")

    if is_mp:
        if order == 1:
            lines.append("    formula = trv.dynamics.LieTrotter()")
        elif order == 2:
            lines.append("    formula = trv.dynamics.SuzukiTrotter()")
        else:
            lines.append(f"    formula = trv.dynamics.suzuki({order})")
        lines.append("    precision_schedule = trv.dynamics.PrecisionConfig(")
        lines.append("        compute=trv.dynamics.Precision.FP32,")
        lines.append("        storage=trv.dynamics.Precision.FP32,")
        lines.append("        accumulate=trv.dynamics.Precision.FP64,")
        lines.append("    )")
        lines.append(f"    n_steps = {trotter_steps}")

        if include_evo_markers:
            lines.append("    # EVOLVE-BLOCK-END")

        lines.append(f"    evolution = trv.dynamics.{evolution_method}(")
        lines.append("        trotter_steps=n_steps,")
        lines.append("        formula=formula,")
        lines.append("        precision_schedule=precision_schedule,")
        lines.append("    )")
        lines.append("    return evolution.evolve(hams, psi0, t)")
    else:
        lines.append(f"    n_steps = {trotter_steps}")
        lines.append(f"    order = {order}")

        if include_evo_markers:
            lines.append("    # EVOLVE-BLOCK-END")

        if evolution_method == "ExactEvolution":
            lines.append(f"    evolution = trv.dynamics.{evolution_method}()")
        else:
            lines.append(f"    # TODO: Instantiate {evolution_method} with your parameters")
            lines.append(f"    evolution = trv.dynamics.{evolution_method}()")
        lines.append("    return evolution.evolve(hams, psi0, t)")

    lines.append("")

    return "\n".join(lines)
