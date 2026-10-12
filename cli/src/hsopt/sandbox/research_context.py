"""Small, immutable scientific context for one isolated mutation proposal."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

MUTATION_PROTOCOL_VERSION = "hamiltonian_context_history_v8_fixed_hf_qpe_reference"

POLICY_CONTRACT = {
    "repetitions": "Return a positive integer; step_time = target_time / repetitions().",
    "build_step": (
        "Called ONCE per species. Return a nonempty list/tuple of (trusted term or Pauli label, "
        "finite real angle) pairs. This ONE schedule is repeated unchanged. A self.step_index "
        "counter does not alternate steps. Put all stages of a higher-order or alternating "
        "composition inside the returned schedule, with the appropriate substep angles."
    ),
    "angle_convention": "Each angle alpha implements exp(+i alpha P); compiled Rz angle is -2*alpha.",
    "custom_angles": (
        "Angles are editable independently for every scheduled Pauli term. They need not equal "
        "coefficient*step_time. You may use label-specific weights, support/coefficient-dependent "
        "corrections, negative substeps, grouping, and reordered or repeated terms. Return only "
        "labels already present in the immutable Hamiltonian. Changing an angle never changes "
        "the trusted Hamiltonian or exact target."
    ),
    "synthesis_epsilon": (
        "Called for each scheduled (term, angle); return a finite tolerance strictly between "
        "0 and 1. This controls synthesis precision, independently of the physical angle. "
        "It may vary by Pauli label, coefficient, support, angle, and species."
    ),
    "term_fields": ["label", "coefficient", "support_size", "x_count", "y_count", "z_count"],
    "species": "__init__(problem) is called separately per species; use problem['name'] for species policies.",
    "schedule_limit": "At most 16 * number_of_Hamiltonian_terms rotations in one step.",
    "python": (
        "One ProductFormula class, literal top-level constants, optional import math. "
        "No other imports, top-level helper functions, generators, private attributes such as "
        "self._cache, getattr, setattr, set, or external execution. Use ordinary public attributes."
    ),
    "available_builtins": [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "object",
        "range",
        "reversed",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    ],
    "metadata_boundary": (
        "Hamiltonian summaries are host-derived context, not additional problem keys. "
        "Read the original JSON for actual keys. In particular nonidentity_one_norm is "
        "a summary field; compute sum([abs(t['coefficient']) for t in problem['terms']]) "
        "if needed. Terms are dictionaries, never objects with coefficient attributes."
    ),
    "objective": (
        "Minimize confidence-adjusted T95 for the fixed HF-accessible energy observable. "
        "Simulation error compares compiled and exact evolution from the SAME HF determinant "
        "using the same lowest-accessible-peak rule. Ground-state preparation bias is a separate "
        "diagnostic, not the acceptance error. The HF state and reference remain immutable. "
        "Once feasible, only a feasible strict T95 improvement is accepted. Before feasibility, "
        "lower exact contract error is progress. The test is not full-unitary fidelity."
    ),
}


def policy_signature(path: Path) -> str:
    """Ignore formatting/comments/docstrings when spotting an already evaluated proposal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_summary(source: Path, kind: str) -> dict[str, Any]:
    """Summarize concrete coefficients, with raw JSON still available for detailed inspection."""
    payload = _read(source)
    if kind == "absolute":
        sources = [(source, "problem.json")]
        reaction = None
    else:
        sources = list(
            dict.fromkeys(
                (source.parent / row["problem"], row["problem"]) for row in payload["terms"]
            )
        )
        reaction = {key: payload[key] for key in ("name", "terms") if key in payload}
    summaries = []
    for path, sandbox_path in sources:
        problem = _read(path)
        terms = problem["terms"]
        ordered = sorted(terms, key=lambda term: -abs(float(term["coefficient"])))
        summaries.append(
            {
                "source": sandbox_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **{
                    key: problem[key]
                    for key in (
                        "name",
                        "n_qubits",
                        "electrons",
                        "target_time",
                        "identity_coefficient",
                    )
                },
                "term_count": len(terms),
                "nonidentity_one_norm": sum(abs(float(t["coefficient"])) for t in terms),
                "support_histogram": dict(
                    Counter(str(sum(c != "I" for c in t["label"])) for t in terms)
                ),
                "pauli_families": dict(
                    Counter(
                        "diagonal" if set(t["label"]) <= {"I", "Z"} else "off_diagonal"
                        for t in terms
                    )
                ),
                "largest_terms": [
                    {"label": t["label"], "coefficient": t["coefficient"]} for t in ordered[:12]
                ],
                "smallest_terms": [
                    {"label": t["label"], "coefficient": t["coefficient"]} for t in ordered[-4:]
                ],
            }
        )
    return {"kind": kind, "reaction": reaction, "hamiltonians": summaries}


def lane_history(lane: Path, run_dir: Path) -> dict[str, Any]:
    """Use only this lane's scored proposals and recent failed attempts, never sibling runs."""
    if not (lane / "manifest.json").is_file():
        return {"recent_proposals": [], "recent_failures": []}
    proposals = []
    paths = sorted((lane / "iterations").glob("[0-9][0-9][0-9][0-9]"))
    for path in paths:
        if path.resolve() == run_dir.resolve() or not (path / "result.json").is_file():
            continue
        result = _read(path / "result.json")
        decision = _read(path / "decision.json")
        diff = ""
        before, after = path / "candidate.before.py", path / "candidate.py"
        if before.is_file() and after.is_file():
            diff = "\n".join(
                difflib.unified_diff(
                    before.read_text().splitlines(), after.read_text().splitlines(), n=1
                )
            )[:1800]
        proposals.append(
            {
                "iteration": int(path.name),
                "accepted": decision.get("accepted"),
                "acceptance_reason": decision.get("acceptance_reason"),
                "candidate_sha256": hashlib.sha256(after.read_bytes()).hexdigest()
                if after.is_file()
                else None,
                "policy": result.get("policy"),
                "score": result.get("score"),
                "passes_chemical_accuracy": result.get("passes_chemical_accuracy"),
                "contract_error_mha": result.get(
                    "contract_error_mha", result.get("absolute_energy_error_mha")
                ),
                "error": str(result.get("error", ""))[:400],
                "change": diff,
            }
        )
    failure_paths = sorted(
        (lane / "mutation_failures").glob("*/mutation_result.json"), key=lambda p: p.stat().st_mtime
    )[-3:]
    failures = []
    for path in failure_paths:
        failure = _read(path)
        result = _read(path.parent / "result.json")
        failures.append(
            {
                "attempt": path.parent.name,
                "status": failure.get("status"),
                "error": str(failure.get("error") or result.get("error") or "interrupted")[:600],
            }
        )
    return {
        "recent_proposals": proposals[-8:],
        "recent_failures": failures,
        "instruction": "Learn from rejections and errors; do not repeat an already evaluated policy. This history belongs only to your lane.",
    }


def duplicate_policy(candidate: Path, incumbent: Path) -> str | None:
    signature = policy_signature(candidate)
    if signature == policy_signature(incumbent):
        return "current incumbent (including formatting-only edits)"
    lane = incumbent.parent
    for result_path in sorted((lane / "iterations").glob("*/result.json")):
        previous = result_path.parent / "candidate.py"
        if previous.resolve() == candidate.resolve() or not previous.is_file():
            continue
        if _read(result_path).get("status") != "ok":
            continue
        try:
            if signature == policy_signature(previous):
                return f"scored iteration {result_path.parent.name}"
        except SyntaxError:
            continue
    return None
