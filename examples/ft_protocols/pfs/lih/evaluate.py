from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(os.environ.get("TROVOLUT_BENCHMARK_ROOT", Path(__file__).resolve().parent)).resolve()
FT_PROTOCOL_ROOT = BENCHMARK_ROOT.parents[1]
if str(FT_PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(FT_PROTOCOL_ROOT))


def evaluate_selected(source: Path):
    evaluator = os.environ.get("TROVOLUT_PFS_EVALUATOR", "energy").strip().lower()
    if evaluator in {"energy", "phase", "qpe"}:
        from pfs.evaluate_energy import evaluate

        return evaluate("lih", source)
    if evaluator in {"fidelity", "state"}:
        from pfs.evaluate_fidelity import evaluate

        epsilon = os.environ.get("TROVOLUT_PFS_FIDELITY_EPSILON")
        return evaluate("lih", source, epsilon=float(epsilon) if epsilon else None)
    raise SystemExit(f"unknown TROVOLUT_PFS_EVALUATOR={evaluator!r}; use 'energy' or 'fidelity'")


def main() -> int:
    source = Path(os.environ.get("TROVOLUT_SOURCE_FILE", BENCHMARK_ROOT / "simple_pf.py")).resolve()
    print(json.dumps(evaluate_selected(source), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
