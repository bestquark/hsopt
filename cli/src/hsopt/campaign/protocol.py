"""Shared-PF trajectories: one proposal, a fixed suite, and portable case requests.

The existing evaluator, cached HF references and reaction contracts remain fixed.
Machines, GPU batching and agent concurrency belong to the external runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from ..evaluation import compiled_unitary as judge
from ..sandbox.openshell import mutate
from ..sandbox.mutation_retry import (
    clear_mutation_retry,
    mutation_readiness,
    record_mutation_failure,
)
from . import calibration
from .interface import _lane_lock, evaluation_request, run_evaluation
from .loop import _archive_mutation_attempt
from .queue import _enqueue_lock, _find_task, _snapshot_reference, _snapshot_source, enqueue

SCHEMA = "hsopt-shared-protocol-v1"
FAMILIES = {"abs_protocol": "absolute", "diff_protocol": "difference"}
OBJECTIVE = "geometric_mean_baseline_relative_t95"
ROOT = Path(__file__).resolve().parents[4]


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid suite input path: {relative}")
    return path


def suite_at(run: Path, *, verify_inputs: bool = True) -> dict[str, Any]:
    manifest = read(run / "manifest.json")
    if sha(run / "suite.json") != manifest["suite_sha256"]:
        raise ValueError("sealed suite changed")
    if manifest["protocol_evaluator_sha256"] != sha(Path(__file__)):
        raise ValueError("shared-protocol acceptance code changed; use its pinned revision")
    suite = read(run / "suite.json")
    if verify_inputs:
        for name, digest in manifest["input_sha256"].items():
            if sha(within(run, name)) != digest:
                raise ValueError(f"sealed input changed: {name}")
    return suite


def bind_policy(candidate: Path, parameters: dict[str, Any], output: Path) -> None:
    """Fill only the reserved baseline table; the shared algorithm stays identical."""
    tree = judge.validate_policy_source(candidate)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    slots = [
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "baseline_parameters" for t in n.targets)
    ]
    if len(slots) != 1 or ast.literal_eval(slots[0].value) != {}:
        raise ValueError("preserve baseline_parameters = {}; the host binds sealed calibration")
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Attribute)
            and n.attr == "baseline_parameters"
            and isinstance(n.ctx, ast.Store)
        ):
            raise ValueError("baseline_parameters is read-only")
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
            if any(
                isinstance(a, ast.Attribute) and a.attr == "baseline_parameters"
                for a in ast.walk(n.value)
            ):
                raise ValueError("baseline_parameters is read-only")
    table = {
        name: (
            int(row["n_steps"]),
            float(row["uniform_epsilon"]),
            float(row["reference_trotter_number_beta"]),
        )
        for name, row in parameters.items()
    }
    slots[0].value = ast.parse(repr(table), mode="eval").body
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    judge.validate_policy_source(output)


def _validate_reference(kind: str, source: Path, reference: Path) -> None:
    payload = read(reference)
    if kind == "absolute":
        judge._reference_energy(judge.load_problem(source), payload)
        return
    if (
        payload.get("observable") != "exact_unitary_signed_energy_difference"
        or payload.get("evaluation_contract_version") not in judge.EXACT_REFERENCE_CONTRACT_VERSIONS
        or payload.get("reference_target") != judge.REFERENCE_TARGET
        or payload.get("reaction_sha256") != sha(source)
    ):
        raise ValueError("reaction needs a matching, certified HF-accessible reference")
    for item in read(source)["terms"]:
        judge._reference_energy(
            judge.load_problem(source.parent / item["problem"]), payload["species"][item["species"]]
        )


def prepare(
    selection: Path,
    run: Path,
    *,
    family: str,
    baseline: str,
    iterations: int,
    replicate: str,
    error_contract: str | None = None,
) -> dict[str, Any]:
    """Reuse beta calibrations and cached references; never recalibrate/diagonalize."""
    if family not in FAMILIES or baseline not in calibration.BASELINE_VARIANTS or iterations < 1:
        raise ValueError("invalid family, baseline or mutation budget")
    kind = FAMILIES[family]
    if (
        kind == "difference"
        and error_contract not in judge.DIFFERENCE_ERROR_CONTRACTS
        or kind == "absolute"
        and error_contract is not None
    ):
        raise ValueError("select exactly one reaction contract for diff_protocol only")
    selected = read(selection.resolve())["cases"]
    if not selected or len({row["id"] for row in selected}) != len(selected):
        raise ValueError("select a nonempty suite with unique case IDs")
    if run.exists() and any(run.iterdir()):
        raise ValueError("shared run directory must be empty")
    run.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in selected:
        case_id = str(row["id"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
            raise ValueError("case IDs must be simple names")
        base = selection.resolve().parent
        source, reference = (base / row["source"]).resolve(), (base / row["reference"]).resolve()
        _validate_reference(kind, source, reference)
        parameters, _ = calibration.load_baselines(
            source,
            baseline=baseline,
            calibration_path=(base / row["calibration"]).resolve()
            if row.get("calibration")
            else None,
            calibration_root=(base / row["calibration_root"]).resolve()
            if row.get("calibration_root")
            else None,
        )
        case = run / "inputs" / case_id
        case.mkdir(parents=True)
        source_name = _snapshot_source(kind, source, case)
        _snapshot_reference(kind, source, reference, case)
        cases.append(
            {
                "id": case_id,
                "source": f"inputs/{case_id}/{source_name}",
                "reference": f"inputs/{case_id}/reference.json",
                "parameters": parameters,
            }
        )
    suite = {
        "schema": SCHEMA,
        "family": family,
        "kind": kind,
        "baseline": baseline,
        "objective": OBJECTIVE,
        "difference_error_contract": error_contract,
        "cases": cases,
    }
    write(run / "suite.json", suite)
    hashes = {str(p.relative_to(run)): sha(p) for p in (run / "inputs").rglob("*.json")}
    for p in (run / "inputs").rglob("*.json"):
        p.chmod(0o444)
    (run / "suite.json").chmod(0o444)
    write(
        run / "manifest.json",
        {
            "schema": SCHEMA,
            "family": family,
            "replicate": replicate,
            "iterations_target": iterations,
            "baseline": baseline,
            "difference_error_contract": error_contract,
            "suite_sha256": sha(run / "suite.json"),
            "input_sha256": hashes,
            "evaluation_contract_version": judge.EVALUATION_CONTRACT_VERSION,
            "protocol_evaluator_sha256": sha(Path(__file__)),
        },
    )
    shutil.copyfile(ROOT / "experiments" / family / "initial_pf.py", run / "initial_pf.py")
    (run / "initial_pf.py").chmod(0o444)
    shutil.copyfile(run / "initial_pf.py", run / "incumbent.py")
    write(
        run / "state.json", {"iterations_completed": 0, "pending_round": None, "initialized": False}
    )
    with _lane_lock(run):
        return _enqueue_round(run, 0, run / "initial_pf.py")


def _enqueue_round(run: Path, iteration: int, candidate: Path) -> dict[str, Any]:
    suite = suite_at(run)
    folder = run / "iterations" / f"{iteration:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "candidate.py"
    if target.resolve() != candidate.resolve():
        shutil.copyfile(candidate, target)
    requests = []
    for case in suite["cases"]:
        bound = folder / "bound_policies" / f"{case['id']}.py"
        bind_policy(target, case["parameters"], bound)
        queued = enqueue(
            run / "queue",
            kind=suite["kind"],
            source=within(run, case["source"]),
            policy=bound,
            reference=within(run, case["reference"]),
            difference_error_contract=suite["difference_error_contract"],
        )
        task = Path(queued["path"])
        request = evaluation_request(task)
        write(task / "evaluation_request.json", request)
        requests.append({"case_id": case["id"], "request": request})
    bundle = {
        "schema": SCHEMA,
        "suite_sha256": sha(run / "suite.json"),
        "iteration": iteration,
        "shared_candidate_sha256": sha(target),
        "requests": requests,
    }
    write(folder / "requests.json", bundle)
    state = read(run / "state.json")
    state["pending_round"] = iteration
    state["pending_request_sha256"] = sha(folder / "requests.json")
    write(run / "state.json", state)
    return {
        "status": "awaiting_evaluation",
        "iteration": iteration,
        "case_count": len(requests),
        "requests": str(folder / "requests.json"),
    }


def _pending(run: Path) -> tuple[Path, dict[str, Any]]:
    # Whole-suite inputs are checked at round creation and acceptance. Case
    # receipts check their own task snapshot; avoid N full-suite scans per round.
    suite = suite_at(run, verify_inputs=False)
    state = read(run / "state.json")
    if state["pending_round"] is None:
        raise ValueError("no pending shared round")
    folder = run / "iterations" / f"{state['pending_round']:04d}"
    if sha(folder / "requests.json") != state["pending_request_sha256"]:
        raise ValueError("shared request bundle changed")
    bundle = read(folder / "requests.json")
    case_ids = [row["case_id"] for row in bundle["requests"]]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != {
        row["id"] for row in suite["cases"]
    }:
        raise ValueError("shared request bundle must include every suite case exactly once")
    if bundle["suite_sha256"] != sha(run / "suite.json") or bundle[
        "shared_candidate_sha256"
    ] != sha(folder / "candidate.py"):
        raise ValueError("shared round snapshot changed")
    return folder, bundle


def _check_receipt(
    run: Path, bundle: dict[str, Any], case_id: str, receipt: dict[str, Any]
) -> Path:
    expected = next(row["request"] for row in bundle["requests"] if row["case_id"] == case_id)
    found = _find_task(run / "queue", expected["task_id"])
    if found is None:
        raise ValueError("case task disappeared")
    task = found[1]
    actual = evaluation_request(task)
    for key in ("schema", "task_id", "request_sha256", "evaluator_sha256"):
        if receipt.get(key) != expected[key] or actual[key] != expected[key]:
            raise ValueError("foreign or stale case receipt")
    result = receipt["result"]
    if (
        result.get("task_id") != expected["task_id"]
        or result.get("evaluation_contract_version") != judge.EVALUATION_CONTRACT_VERSION
    ):
        raise ValueError("case result has wrong task or evaluator contract")
    if result.get("execution_backend") != "cupy-lanczos":
        raise ValueError("shared case receipt must come from GPU Lanczos")
    contract = read(run / "suite.json")["difference_error_contract"]
    if result.get("difference_error_contract") != contract:
        raise ValueError("reaction contract mismatch")
    return task


def record_receipt(run: Path, case_id: str, receipt: dict[str, Any]) -> None:
    """Portable, idempotent delivery; no acceptance until every case returns."""
    with _lane_lock(run), _enqueue_lock(run / "queue"):
        folder, bundle = _pending(run)
        task = _check_receipt(run, bundle, case_id, receipt)
        path = folder / "receipts" / f"{case_id}.json"
        if path.exists() and read(path) != receipt:
            raise ValueError("conflicting case receipt")
        task_receipt = task / "evaluation_receipt.json"
        if task_receipt.exists() and read(task_receipt) != receipt:
            raise ValueError("conflicting task receipt")
        write(task_receipt, receipt)
        write(task / "result.json", receipt["result"])
        if task.parent.name == "pending":
            phase = "completed" if receipt["result"]["status"] == "ok" else "failed"
            task.rename(run / "queue" / phase / task.name)
        write(path, receipt)


def evaluate_pending(
    run: Path,
    *,
    case_ids: list[str] | None = None,
    backend: str = "cupy-lanczos",
    accelerator: str = "gpu",
) -> dict[str, Any]:
    """Worker helper; a deployment may split the requests across any GPU workers."""
    if backend != "cupy-lanczos":
        raise ValueError(
            "shared-protocol evaluation requires GPU Lanczos; no full-diagonalization fallback"
        )
    folder, bundle = _pending(run)
    known = {row["case_id"] for row in bundle["requests"]}
    selected = known if case_ids is None else set(case_ids)
    if not selected <= known:
        raise ValueError("unknown suite case")
    for row in bundle["requests"]:
        if row["case_id"] not in selected:
            continue
        path = folder / "receipts" / f"{row['case_id']}.json"
        if path.exists():
            _check_receipt(run, bundle, row["case_id"], read(path))
            continue
        request = row["request"]
        task = _find_task(run / "queue", request["task_id"])[1]
        cached = task / "evaluation_receipt.json"
        receipt = (
            read(cached)
            if cached.exists()
            else run_evaluation(request, task_dir=task, backend=backend, accelerator=accelerator)
        )
        record_receipt(run, row["case_id"], receipt)
    return {"status": "evaluated", "case_count": len(selected), "iteration": bundle["iteration"]}


def aggregate(results: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    """Equal-weight geometric mean, with case-wise feasibility and error feedback."""
    if not results or results.keys() != baselines.keys():
        raise ValueError("incomplete suite results")
    metrics, ratios, errors = [], [], []
    for case_id in sorted(results):
        row, baseline = results[case_id], baselines[case_id]
        denominator = baseline.get("confidence_adjusted_t_count")
        if (
            not isinstance(denominator, (int, float))
            or not math.isfinite(denominator)
            or denominator <= 0
        ):
            raise ValueError(f"{case_id}: baseline needs a certified positive T95")
        cost = row.get("confidence_adjusted_t_count")
        valid = (
            row.get("status") == "ok"
            and isinstance(cost, (int, float))
            and math.isfinite(cost)
            and cost >= 0
        )
        error = row.get("contract_error_hartree", row.get("absolute_energy_error_hartree"))
        valid = valid and isinstance(error, (int, float)) and math.isfinite(error) and error >= 0
        passed = bool(
            valid
            and row.get("passes_chemical_accuracy")
            and error < judge.CHEMICAL_ACCURACY_HARTREE
        )
        ratio = cost / denominator if valid else None
        if valid:
            ratios.append(ratio)
            errors.append(error / judge.CHEMICAL_ACCURACY_HARTREE)
        metrics.append(
            {
                "case_id": case_id,
                "t95": cost if valid else None,
                "relative_t95": ratio,
                "error_hartree": error if valid else None,
                "passes_chemical_accuracy": passed,
                "evaluation_error": row.get("error") if not valid else None,
            }
        )
    failed = [row["case_id"] for row in metrics if not row["passes_chemical_accuracy"]]
    score = (
        (
            0.0
            if min(ratios) == 0
            else math.exp(math.fsum(math.log(r) for r in ratios) / len(ratios))
        )
        if len(ratios) == len(results)
        else None
    )
    return {
        "status": "ok" if score is not None else "error",
        "protocol_objective": OBJECTIVE,
        "score": score,
        "mean_relative_t95": math.fsum(ratios) / len(ratios) if score is not None else None,
        "passes_chemical_accuracy": not failed,
        "failed_case_ids": failed,
        "case_metrics": metrics,
        "worst_normalized_error": max(errors) if score is not None else None,
        "case_count": len(results),
    }


def accept(candidate: dict[str, Any], incumbent: dict[str, Any]) -> tuple[bool, str]:
    if candidate["status"] != "ok":
        return False, "evaluation_failed"
    if not incumbent["passes_chemical_accuracy"]:
        return False, "baseline_accuracy_must_be_resolved"
    if not candidate["passes_chemical_accuracy"]:
        return False, "a_suite_case_failed_accuracy"
    improved = candidate["score"] < incumbent["score"] or (
        candidate["score"] == incumbent["score"] == 0
        and candidate["mean_relative_t95"] < incumbent["mean_relative_t95"]
    )
    return improved, "aggregate_t95_improved" if improved else "no_aggregate_improvement"


def advance(run: Path, *, propose: bool = True, timeout_seconds: int = 900) -> dict[str, Any]:
    """Reconcile the whole suite, then make at most one shared OpenShell proposal."""
    with _lane_lock(run):
        state = read(run / "state.json")
        if state["pending_round"] is not None:
            folder, bundle = _pending(run)
            missing = [
                r["case_id"]
                for r in bundle["requests"]
                if not (folder / "receipts" / f"{r['case_id']}.json").is_file()
            ]
            if missing:
                return {
                    "status": "awaiting_evaluation",
                    "missing_case_ids": missing,
                    "requests": str(folder / "requests.json"),
                }
            suite_at(run)
            for row in bundle["requests"]:
                _check_receipt(
                    run,
                    bundle,
                    row["case_id"],
                    read(folder / "receipts" / f"{row['case_id']}.json"),
                )
            results = {
                r["case_id"]: read(folder / "receipts" / f"{r['case_id']}.json")["result"]
                for r in bundle["requests"]
            }
            baseline = (
                results
                if bundle["iteration"] == 0
                else read(run / "iterations/0000/case_results.json")
            )
            summary = aggregate(results, baseline)
            write(folder / "case_results.json", results)
            write(folder / "result.json", summary)
            if bundle["iteration"] == 0:
                if summary["status"] != "ok":
                    raise ValueError(
                        "iteration 0 has uncertified case results; resolve before mutating"
                    )
                if not summary["passes_chemical_accuracy"]:
                    write(
                        folder / "decision.json",
                        {
                            "accepted": False,
                            "acceptance_reason": "baseline_accuracy_failed",
                            "failed_case_ids": summary["failed_case_ids"],
                        },
                    )
                    return {
                        "status": "needs_attention",
                        "reason": "iteration_0_accuracy_failed",
                        "failed_case_ids": summary["failed_case_ids"],
                    }
                accepted, reason = True, "initial_shared_baseline"
                state["initialized"] = True
            else:
                accepted, reason = accept(summary, read(run / "incumbent_result.json"))
                state["iterations_completed"] = bundle["iteration"]
            write(
                folder / "decision.json",
                {
                    "accepted": accepted,
                    "acceptance_reason": reason,
                    "shared_candidate_sha256": bundle["shared_candidate_sha256"],
                    "suite_sha256": bundle["suite_sha256"],
                },
            )
            if accepted:
                shutil.copyfile(folder / "candidate.py", run / "incumbent.py")
                write(run / "incumbent_result.json", summary)
            state["pending_round"] = None
            state["pending_request_sha256"] = None
            write(run / "state.json", state)
        manifest = read(run / "manifest.json")
        if state["iterations_completed"] >= manifest["iterations_target"]:
            return {"status": "complete", **state}
        if not propose:
            return {"status": "ready_to_mutate", **state}
        readiness = mutation_readiness(run)
        if readiness is not None:
            return readiness
        iteration = state["iterations_completed"] + 1
        folder = run / "iterations" / f"{iteration:04d}"
        proposed = False
        if folder.exists() and any(folder.iterdir()):
            outcome = folder / "mutation_result.json"
            validation = folder / "protocol_validation.json"
            if (
                outcome.exists()
                and read(outcome).get("status") == "mutated"
                and (not validation.exists() or read(validation)["status"] != "failed")
            ):
                # Recover an interruption after the successful proposal without
                # invoking another agent or discarding any case requests.
                proposed = True
            else:
                _archive_mutation_attempt(run, folder)
        try:
            if not proposed:
                mutate(
                    run / "incumbent.py",
                    run / "incumbent_result.json",
                    folder,
                    kind="protocol",
                    source=run / "suite.json",
                    timeout_seconds=timeout_seconds,
                )
            bind_policy(
                folder / "candidate.py",
                suite_at(run)["cases"][0]["parameters"],
                folder / "binding_check.py",
            )
            (folder / "binding_check.py").unlink()
            write(folder / "protocol_validation.json", {"status": "passed"})
        except Exception as exc:
            if folder.exists():
                write(folder / "protocol_validation.json", {"status": "failed", "error": str(exc)})
            record_mutation_failure(run, exc, iteration=iteration)
            raise
        clear_mutation_retry(run)
        return _enqueue_round(run, iteration, folder / "candidate.py")


def main(*, family: str, worker: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if worker:
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--case-id", action="append")
        parser.add_argument("--accelerator", default="gpu")
        args = parser.parse_args()
        if read(args.run_dir / "manifest.json")["family"] != family:
            parser.error("wrong protocol family")
        result = evaluate_pending(args.run_dir, case_ids=args.case_id, accelerator=args.accelerator)
    else:
        sub = parser.add_subparsers(dest="action", required=True)
        prepare_parser = sub.add_parser("prepare")
        prepare_parser.add_argument("--selection", required=True, type=Path)
        prepare_parser.add_argument("--run-dir", required=True, type=Path)
        prepare_parser.add_argument(
            "--baseline", required=True, choices=calibration.BASELINE_VARIANTS
        )
        prepare_parser.add_argument("--iterations", required=True, type=int)
        prepare_parser.add_argument("--replicate", required=True)
        if family == "diff_protocol":
            prepare_parser.add_argument(
                "--error-contract", required=True, choices=judge.DIFFERENCE_ERROR_CONTRACTS
            )
        advance_parser = sub.add_parser("advance")
        advance_parser.add_argument("--run-dir", required=True, type=Path)
        advance_parser.add_argument("--no-mutate", action="store_true")
        advance_parser.add_argument("--timeout-seconds", type=int, default=900)
        deliver = sub.add_parser("deliver")
        deliver.add_argument("--run-dir", required=True, type=Path)
        deliver.add_argument("--case-id", required=True)
        deliver.add_argument("--receipt", required=True, type=Path)
        args = parser.parse_args()
        if args.action == "prepare":
            result = prepare(
                args.selection,
                args.run_dir,
                family=family,
                baseline=args.baseline,
                iterations=args.iterations,
                replicate=args.replicate,
                error_contract=getattr(args, "error_contract", None),
            )
        else:
            if read(args.run_dir / "manifest.json")["family"] != family:
                parser.error("wrong protocol family")
            if args.action == "deliver":
                record_receipt(args.run_dir, args.case_id, read(args.receipt))
                result = {"status": "recorded"}
            else:
                result = advance(
                    args.run_dir, propose=not args.no_mutate, timeout_seconds=args.timeout_seconds
                )
    print(json.dumps(result, sort_keys=True))
    return 0
