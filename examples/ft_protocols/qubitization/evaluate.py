from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protocol_common.problems import (  # noqa: E402
    load_protocol_problem,
    molecule_choices,
    score_from_accuracy_proxy,
)


def load_method_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("qubitization_method", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load method file: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for symbol in ("get_method_config", "estimate_qubitization_qpe"):
        if not hasattr(module, symbol):
            raise AttributeError(f"method file is missing required symbol {symbol!r}")
    return module


def evaluate(molecule: str, script_path: Path) -> Dict[str, object]:
    start = time.perf_counter()
    module = load_method_module(script_path)
    problem = load_protocol_problem(molecule)
    estimate = module.estimate_qubitization_qpe(problem)

    allowed = float(problem["chemical_accuracy_hartree"])
    actual = estimate.get("accuracy_proxy_hartree")
    t_gate_count = int(estimate["t_gate_count"])
    passes = actual is not None and float(actual) <= allowed

    return {
        "experiment": f"qubitization_qpe_{molecule}",
        "protocol": "qubitization_qpe",
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "qualtran_qubitization_qpe_resource_proxy",
        "score": score_from_accuracy_proxy(t_gate_count, allowed, actual),
        "passes_accuracy_constraint": passes,
        "accuracy_metric_name": "accuracy_proxy_hartree",
        "accuracy_metric_value": actual,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": allowed,
        "t_gate_count": t_gate_count,
        "logical_qubits": int(estimate["logical_qubits"]),
        "qualtran_count_status": estimate["qualtran_count_status"],
        "resource_estimate": estimate,
        "cas_label": problem["cas_label"],
        "basis": problem["basis"],
        "localization": problem["localization"],
        "evaluation_wall_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one Qualtran qubitization-QPE resource policy.")
    parser.add_argument("--molecule", required=True, choices=molecule_choices())
    parser.add_argument("--script", required=True, help="Path to the editable simple_qubitization.py file.")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

