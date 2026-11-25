"""Generate compact loose/tight PF2 policies from sealed calibration records.

Only the ProductFormula class is mutated in an external candidate.py. Repetitions
set r and therefore dt = target_time/r; build_step selects order and angles;
synthesis_epsilon may vary by term and angle. Uniform precision is a baseline
starting point, not an evaluator requirement. Calibration/provenance live in
hsopt.campaign.calibration; existing baseline code is emitted unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Mapping

from hsopt.campaign import calibration


def __getattr__(name: str) -> Any:
    """Preserve the calibration API used by existing campaign commands."""
    return getattr(calibration, name)


class ProductFormula:
    parameters = {}

    def __init__(self, problem):
        self.n_steps, self.epsilon = self.parameters[str(problem["name"])]

    def repetitions(self):
        return self.n_steps

    def build_step(self, terms, step_time):
        ordered = sorted(terms, key=lambda term: term["label"])
        half = [(term, 0.5 * float(term["coefficient"]) * step_time) for term in ordered[:-1]]
        center = [(ordered[-1], float(ordered[-1]["coefficient"]) * step_time)]
        return half + center + list(reversed(half))

    def synthesis_epsilon(self, term, angle):
        _ = (term, angle)
        return self.epsilon


def product_formula(
    problem: Mapping[str, Any], parameters: Mapping[str, Mapping[str, Any]]
) -> ProductFormula:
    """Bind sealed per-species r/epsilon rows and return the small PF object."""
    table = {
        name: (int(row["n_steps"]), float(row["uniform_epsilon"]))
        for name, row in parameters.items()
    }
    bound = type("ProductFormula", (ProductFormula,), {"parameters": table})
    return bound(problem)


def _policy_source() -> str:
    """Emit this same class without dependencies on the host calibration code."""
    source = Path(__file__).read_text(encoding="utf-8")
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "ProductFormula"
    )
    return ast.get_source_segment(source, node) + "\n"


def format_parameters(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["{"]
    for name, row in sorted(rows.items()):
        lines.append(
            f"        {json.dumps(name)}: "
            f"({int(row['n_steps'])}, {float(row['uniform_epsilon'])!r}),"
        )
    lines.append("    }")
    return "\n".join(lines)


def render(
    source: Path,
    *,
    baseline: str = "loose",
    calibration_path: Path | None = None,
    calibration_root: Path | None = None,
) -> str:
    rows, _calibration_paths = calibration.load_baselines(
        source,
        baseline=baseline,
        calibration_path=calibration_path,
        calibration_root=calibration_root,
    )
    policy_source = _policy_source().replace(
        "parameters = {}", "parameters = " + format_parameters(rows), 1
    )
    calibration_lines = "\n".join(
        f"Calibration SHA-256: {rows[name]['calibration_sha256']}" for name in sorted(rows)
    )
    contract = calibration.paper_contract(baseline)
    variant_line = "" if baseline == "loose" else f"Baseline variant: {baseline}\n"
    derivation_label = "bound" if baseline == "loose" else "beta"
    return f'''"""Generated iteration-0 product formula; mutate the class, not the benchmark.

{variant_line}Baseline: {contract["display_name"]}, adapted to 1 kcal/mol.
{calibration_lines}
Full {derivation_label}/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with CUDA-Q.
"""


{policy_source}'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=calibration.BASELINE_VARIANTS, default="loose")
    calibrations = parser.add_mutually_exclusive_group(required=True)
    calibrations.add_argument("--calibration", type=Path)
    calibrations.add_argument("--calibration-root", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    expected = render(
        args.source.resolve(),
        baseline=args.baseline,
        calibration_path=args.calibration.resolve() if args.calibration else None,
        calibration_root=args.calibration_root.resolve() if args.calibration_root else None,
    )
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"generated baseline is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
