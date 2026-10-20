"""Three isolated ELOTE searches sharing scientific inputs and evaluation budgets.

GPU placement is delegated to the ordinary portable request queue. PySR's child
process never imports CUDA-Q, and never sees held-out Hamiltonians or results.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..campaign import generalization as g
from ..campaign import protocol as p
from ..campaign.interface import _lane_lock
from ..sandbox.openshell import mutate
from .tolerance import FEATURES, expression_source, tolerance_policy, training_features

METHODS = ("agent_only", "pysr_only", "agent_pysr")
SCHEMA = "hsopt-search-comparison-v1"


class BudgetExhausted(RuntimeError):
    pass


class NumericalEvaluationError(RuntimeError):
    """An unresolved evaluator is infrastructure feedback, never a bad-rule score."""


def signature(source):
    import hashlib

    return hashlib.sha256(
        ast.dump(ast.parse(source), include_attributes=False).encode()
    ).hexdigest()


def prepare_comparison(
    selection,
    root,
    *,
    seeds,
    train_size,
    test_size,
    budget,
    baseline,
    policy_profile="fermionic_nested",
    energy_target="ground_branch",
):
    if budget < 1 or not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("select a positive evaluation budget and distinct nonnegative seeds")
    if root.exists() and any(root.iterdir()):
        raise ValueError("comparison directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "methods": list(METHODS),
        "seeds": seeds,
        "evaluation_budget_per_method_seed": budget,
        "budget_unit": "distinct policy sources evaluated on the training suite; all hybrid probes count",
        "baseline_evaluations_excluded_from_search_budget": True,
        "baseline": baseline,
        "policy_profile": policy_profile,
        "energy_target": energy_target,
        "train_size": train_size,
        "test_size": test_size,
        "objective": p.OBJECTIVE,
        "evaluator": "cupy-lanczos",
        "pysr_search_space": "tolerance arithmetic; fixed supplied splitting and repetitions",
        "agent_search_space": "complete permitted ProductFormula",
        "comparison_scope": "matched data, start, judge and evaluation cap; different expressivity",
        "test_use": "freeze final training incumbent; evaluate once; no selection or feedback",
        "search_code_sha256": p.sha(Path(__file__)),
        "tolerance_code_sha256": p.sha(Path(__file__).with_name("tolerance.py")),
        "worker_code_sha256": p.sha(Path(__file__).with_name("pysr_worker.py")),
        "runs": [],
    }
    for seed in seeds:
        split = root / "splits" / f"seed-{seed:02d}"
        g.split(selection, split, seed=seed, train_size=train_size, test_size=test_size)
        for method in METHODS:
            run = root / method / f"seed-{seed:02d}"
            g.prepare_training(
                split,
                run,
                iterations=budget,
                replicate=f"seed-{seed:02d}",
                baseline=baseline,
                energy_target=energy_target,
                policy_profile=policy_profile,
            )
            meta = {
                "schema": SCHEMA,
                "method": method,
                "seed": seed,
                "comparison_root": str(root.resolve()),
                "split": str(split.resolve()),
                "evaluation_budget": budget,
                "source_sha256": manifest["search_code_sha256"],
            }
            p.write(run / "search.json", meta)
            (run / "search.json").chmod(0o444)
            manifest["runs"].append(str(run.relative_to(root)))
    p.write(root / "comparison.json", manifest)
    (root / "comparison.json").chmod(0o444)
    return manifest


class TrainingOracle:
    def __init__(self, run: Path, *, worker="external", timeout=86400):
        self.run = run.resolve()
        self.meta = p.read(self.run / "search.json")
        if self.meta["source_sha256"] != p.sha(Path(__file__)):
            raise ValueError("search controller changed; use the comparison's pinned source")
        comparison = p.read(Path(self.meta["comparison_root"]) / "comparison.json")
        for filename, key in [
            ("tolerance.py", "tolerance_code_sha256"),
            ("pysr_worker.py", "worker_code_sha256"),
        ]:
            if p.sha(Path(__file__).with_name(filename)) != comparison[key]:
                raise ValueError("symbolic search implementation changed")
        p.suite_at(self.run)
        if worker not in {"external", "local"}:
            raise ValueError("worker must be external or local GPU")
        self.worker, self.timeout = worker, timeout
        self.lock = threading.RLock()

    @property
    def remaining(self):
        return (
            self.meta["evaluation_budget"] - p.read(self.run / "state.json")["iterations_completed"]
        )

    def _collect(self):
        state = p.read(self.run / "state.json")
        if state.get("pending_round") is None:
            return
        if self.worker == "local":
            p.evaluate_pending(self.run)
        deadline = time.monotonic() + self.timeout
        while True:
            result = p.advance(self.run, propose=False)
            if result["status"] != "awaiting_evaluation":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU results pending; resume this run without resubmitting")
            time.sleep(1)
        if result["status"] == "needs_attention":
            raise NumericalEvaluationError(f"baseline needs attention: {result}")
        current = p.read(self.run / "state.json")
        if not current["initialized"]:
            raise NumericalEvaluationError("baseline is not certified and accurate")
        iteration = current["iterations_completed"]
        summary = p.read(self.run / "iterations" / f"{iteration:04d}" / "result.json")
        if summary["status"] != "ok":
            raise NumericalEvaluationError(
                "uncertified evaluation; repair/retry before search resumes"
            )

    def initialize(self):
        with self.lock:
            self._collect()
            self._check_numerical_hold()
            if not p.read(self.run / "state.json")["initialized"]:
                raise NumericalEvaluationError("evaluate iteration 0 before searching")

    def _check_numerical_hold(self):
        for path in (self.run / "iterations").glob("*/result.json"):
            if p.read(path)["status"] != "ok":
                raise NumericalEvaluationError(f"unresolved evaluator result: {path}")

    def evaluate(self, source: str, *, origin: dict):
        with self.lock:
            self._collect()
            self._check_numerical_hold()
            wanted = signature(source)
            # Exact-source cache, never cached test results or an approximate surrogate.
            for path in sorted((self.run / "iterations").glob("*/result.json")):
                candidate = path.parent / "candidate.py"
                if signature(candidate.read_text()) == wanted:
                    return self._reply(path.parent, cached=True)
            with _lane_lock(self.run):
                state = p.read(self.run / "state.json")
                if state.get("pending_round") is not None:
                    raise RuntimeError("another proposal is pending")
                if self.remaining <= 0:
                    raise BudgetExhausted("training evaluation budget exhausted")
                iteration = state["iterations_completed"] + 1
                folder = self.run / "iterations" / f"{iteration:04d}"
                folder.mkdir(exist_ok=True)
                candidate = folder / "candidate.py"
                candidate.write_text(source)
                # Same validator, binding and literal compilation as ordinary autoresearch.
                p.judge.validate_policy_source(candidate)
                p.bind_policy(
                    candidate,
                    p.suite_at(self.run)["cases"][0]["parameters"],
                    folder / "binding_check.py",
                )
                (folder / "binding_check.py").unlink()
                p.write(folder / "search_origin.json", origin)
                p._enqueue_round(self.run, iteration, candidate)
            self._collect()
            return self._reply(folder, cached=False)

    def _reply(self, folder, *, cached):
        result = p.read(folder / "result.json")
        if result["status"] != "ok":
            raise NumericalEvaluationError("an uncertified result cannot score a symbolic rule")
        feasible = result["passes_chemical_accuracy"]
        # Physical inaccuracy is a valid search rejection; solver errors raise instead.
        loss = result["score"] if feasible else 1e6 + len(result["failed_case_ids"])
        return {
            "loss": loss,
            "feasible": feasible,
            "cached": cached,
            "iteration": int(folder.name),
            "result": result,
            "candidate_sha256": p.sha(folder / "candidate.py"),
            "candidate_path": str(folder / "candidate.py"),
        }


class SymbolicSearch:
    def __init__(
        self, oracle, *, python=sys.executable, max_evals=20, seconds=180, maxsize=12, max_calls=100
    ):
        self.oracle, self.python = oracle, python
        self.max_evals, self.seconds, self.maxsize, self.max_calls = (
            max_evals,
            seconds,
            maxsize,
            max_calls,
        )
        self.lock = threading.Lock()

    def search(self, request):
        with self.lock:
            if self.oracle.meta["method"] not in {"pysr_only", "agent_pysr"}:
                raise ValueError("PySR is disabled for the agent-only control")
            self.oracle.initialize()
            if self.oracle.remaining <= 0:
                return {"status": "budget_exhausted"}
            root = self.oracle.run / "symbolic_searches"
            root.mkdir(exist_ok=True)
            existing = list(root.glob("call-*"))
            if len(existing) >= self.max_calls:
                return {"status": "tool_call_limit"}
            folder = root / f"call-{len(existing) + 1:04d}"
            folder.mkdir()
            base = request.get("base_policy") or (self.oracle.run / "initial_pf.py").read_text()
            if self.oracle.meta["method"] == "pysr_only" and signature(base) != signature(
                (self.oracle.run / "initial_pf.py").read_text()
            ):
                raise ValueError("pure PySR must keep its initial splitting and repetitions")
            base_path = folder / "base.py"
            base_path.write_text(base)
            p.judge.validate_policy_source(base_path)
            expression_source(request.get("guess", "0.0"))
            limits = {
                "max_evals": min(
                    int(request.get("max_evals", self.max_evals)),
                    self.max_evals,
                    self.oracle.remaining,
                ),
                "seconds": min(float(request.get("seconds", self.seconds)), self.seconds),
                "maxsize": min(int(request.get("maxsize", self.maxsize)), self.maxsize),
            }
            if limits["max_evals"] < 1 or limits["seconds"] <= 0 or limits["maxsize"] < 3:
                raise ValueError("symbolic search limits must be positive")
            operators = request.get("operators", ["+", "-", "*", "/", "abs", "sqrt"])
            if not operators or set(operators) - {"+", "-", "*", "/", "abs", "sqrt"}:
                raise ValueError("unsupported symbolic-search operators")
            if not set(operators) & {"+", "-", "*", "/"}:
                raise ValueError("at least one binary operator is required")
            features = training_features(self.oracle.run, base_path, folder / "bindings")
            p.write(
                folder / "request.json",
                {
                    **request,
                    **limits,
                    "operators": operators,
                    "base_sha256": p.sha(base_path),
                    "training_suite_sha256": p.sha(self.oracle.run / "suite.json"),
                },
            )
            config = {
                **limits,
                "features": features,
                "feature_names": FEATURES,
                "operators": operators,
                "guesses": [request.get("guess", "0.0")],
                "max_evals": 5 * limits["max_evals"],  # callbacks include duplicates/invalid trees
                "seed": self.oracle.meta["seed"] + len(existing),
                "output_directory": str(folder / "pysr"),
            }
            before = self.oracle.remaining
            best, done, versions = None, {}, {}
            start = time.monotonic()
            with (folder / "worker.stderr").open("w") as log:
                child = subprocess.Popen(
                    [self.python, "-u", str(Path(__file__).with_name("pysr_worker.py"))],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                )
                try:
                    child.stdin.write(json.dumps(config) + "\n")
                    child.stdin.flush()
                    for line in child.stdout:
                        with (folder / "worker.stdout").open("a") as stdout_log:
                            stdout_log.write(line)
                        if line.startswith("HSOPT_FITNESS "):
                            message = json.loads(line.removeprefix("HSOPT_FITNESS "))
                            if before - self.oracle.remaining >= limits["max_evals"]:
                                done = {"status": "evaluation_budget_reached"}
                                break
                            if (
                                before > self.oracle.remaining
                                and time.monotonic() - start >= limits["seconds"]
                            ):
                                # Finish the in-flight physical evaluation, then stop
                                # even during PySR's initial population scoring.
                                done = {"status": "time_budget_reached"}
                                break
                            expr = message["expression"]
                            source = tolerance_policy(base, expr)
                            reply = self.oracle.evaluate(
                                source,
                                origin={
                                    "method": self.oracle.meta["method"],
                                    "tool": "hsopt_pysr",
                                    "call": folder.name,
                                    "expression": expr,
                                    "base_sha256": p.sha(base_path),
                                },
                            )
                            with (folder / "fitness.jsonl").open("a") as stream:
                                stream.write(json.dumps({"expression": expr, **reply}) + "\n")
                            if reply["feasible"] and (best is None or reply["loss"] < best["loss"]):
                                best = {**reply, "expression": expr, "source": source}
                            child.stdin.write(json.dumps({"loss": reply["loss"]}) + "\n")
                            child.stdin.flush()
                        elif line.startswith("HSOPT_READY "):
                            versions = json.loads(line.removeprefix("HSOPT_READY "))
                        elif line.startswith("HSOPT_DONE "):
                            done = {
                                "status": "complete",
                                **json.loads(line.removeprefix("HSOPT_DONE ")),
                            }
                        elif line.startswith("HSOPT_ERROR "):
                            raise RuntimeError(
                                json.loads(line.removeprefix("HSOPT_ERROR "))["error"]
                            )
                    if not done:
                        raise RuntimeError("PySR worker exited without a completion receipt")
                except Exception as exc:
                    p.write(folder / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
                    raise
                finally:
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=30)
                    try:
                        child.stdin.close()
                    except BrokenPipeError:
                        pass
                    child.stdout.close()
            result = {
                **done,
                **versions,
                "wall_seconds": time.monotonic() - start,
                "new_suite_evaluations": before - self.oracle.remaining,
                "remaining_evaluations": self.oracle.remaining,
                "best": best,
                "test_feedback_used": False,
            }
            p.write(folder / "result.json", result)
            return result


def tool_server(search, host="127.0.0.1", port=0, *, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(403)
                return
            if self.path != "/search":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 256000:
                self.send_error(413)
                return
            try:
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("search request must be an object")
                payload, status = search.search(request), 200
            except Exception as exc:
                payload, status = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}, 400
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@contextmanager
def agent_capability(method, endpoint=None, token=None):
    # Never accidentally give the agent-only control a tool inherited from the shell.
    keys = ("HSOPT_PYSR_URL", "HSOPT_PYSR_TOKEN")
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.pop("HSOPT_PYSR_URL", None)
        os.environ.pop("HSOPT_PYSR_TOKEN", None)
        if method == "agent_pysr":
            if not endpoint or not token:
                raise ValueError("hybrid needs a PySR service URL reachable from OpenShell")
            os.environ["HSOPT_PYSR_URL"] = endpoint
            os.environ["HSOPT_PYSR_TOKEN"] = token
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_search(oracle, symbolic, *, tool_url=None, listen="127.0.0.1", port=0, agent_timeout=1800):
    method = oracle.meta["method"]
    if method == "agent_pysr" and (not tool_url or not port):
        raise ValueError("hybrid needs --tool-url reachable from OpenShell and an explicit --port")
    token = secrets.token_urlsafe(32)
    server = tool_server(symbolic, listen, port, token=token) if method == "agent_pysr" else None
    started = time.monotonic()
    with (oracle.run / "controller_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "start",
                    "at": datetime.now(UTC).isoformat(),
                    "method": method,
                    "worker": oracle.worker,
                    "pysr_evaluations_per_call": symbolic.max_evals,
                    "agent_timeout_seconds": agent_timeout,
                }
            )
            + "\n"
        )
    try:
        oracle.initialize()
        while oracle.remaining > 0:
            before = oracle.remaining
            if method == "pysr_only":
                guess = "0.0"
                for decision in reversed(
                    sorted((oracle.run / "iterations").glob("*/decision.json"))
                ):
                    if p.read(decision).get("accepted"):
                        origin = decision.parent / "search_origin.json"
                        if origin.exists():
                            guess = p.read(origin).get("expression", "0.0")
                        break
                symbolic.search({"max_evals": symbolic.max_evals, "guess": guess})
            else:
                turns = oracle.run / "agent_turns"
                turns.mkdir(exist_ok=True)
                folder = turns / f"{len(list(turns.iterdir())) + 1:04d}"
                try:
                    with agent_capability(method, tool_url, token):
                        mutate(
                            oracle.run / "incumbent.py",
                            oracle.run / "incumbent_result.json",
                            folder,
                            kind="protocol",
                            source=oracle.run / "suite.json",
                            timeout_seconds=agent_timeout,
                        )
                except Exception as exc:
                    # A tool can finish the entire budget before the agent writes
                    # a final file. Certified proposals already own the trajectory;
                    # a missing/duplicate final edit must not discard that science.
                    if method != "agent_pysr" or oracle.remaining == before:
                        raise
                    oracle.initialize()  # A numerical hold still blocks completion.
                    p.write(
                        folder / "controller_progress.json",
                        {
                            "reason": "certified_inner_search_proposals_preserved",
                            "agent_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue
                try:
                    reply = oracle.evaluate(
                        (folder / "candidate.py").read_text(),
                        origin={"method": method, "tool": "opencode", "agent_turn": folder.name},
                    )
                    p.write(folder / "search_evaluation.json", reply)
                except BudgetExhausted:
                    # Inner probes can spend the final evaluation; their certified
                    # improvements already went through ordinary acceptance.
                    break
            if oracle.remaining == before:
                raise RuntimeError(
                    "search made no new proposal; resume explicitly with a new hypothesis"
                )
        split = Path(oracle.meta["split"])
        test = oracle.run / "held_out"
        g.prepare_test(split, oracle.run, test)
        while True:
            state = p.read(test / "state.json")
            if state.get("pending_round") is not None:
                if oracle.worker == "local":
                    p.evaluate_pending(test)
                elif g.advance_test(test)["status"] == "awaiting_evaluation":
                    return {"status": "test_pending", "run": str(test)}
            result = g.advance_test(test)
            if result["status"] == "needs_attention":
                raise NumericalEvaluationError(f"held-out baseline needs attention: {result}")
            if result["status"] == "complete":
                return {"status": "complete", "report": str(test / "generalization_report.json")}
    finally:
        with (oracle.run / "controller_events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "exit",
                        "at": datetime.now(UTC).isoformat(),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                + "\n"
            )
        if server:
            server.shutdown()
            server.server_close()


def comparison_report(root):
    manifest = p.read(root / "comparison.json")
    rows = []
    for relative in manifest["runs"]:
        run = root / relative
        state = p.read(run / "state.json")
        results = [p.read(path) for path in (run / "iterations").glob("*/result.json")]
        traces = list((run / "agent_turns").glob("*/agent_audit.json"))
        tokens = Counter()
        token_complete = True
        model_requests = 0
        for path in traces:
            metrics = p.read(path)["opencode"]
            tokens.update(metrics.get("tokens", {}))
            token_complete &= bool(metrics.get("token_usage_complete", False))
            model_requests += metrics.get("inference_steps", 0)
        raw = [p.read(path) for path in (run / "iterations").glob("*/case_results.json")]
        events = run / "controller_events.jsonl"
        controller_seconds = (
            sum(
                json.loads(line).get("elapsed_seconds", 0)
                for line in events.read_text().splitlines()
            )
            if events.exists()
            else None
        )
        test = run / "held_out/generalization_report.json"
        rows.append(
            {
                "run": relative,
                "evaluations_completed": state["iterations_completed"],
                "evaluation_budget": manifest["evaluation_budget_per_method_seed"],
                "molecular_or_reaction_receipts": sum(r["case_count"] for r in results),
                "agent_turns": len(list((run / "agent_turns").glob("*/mutation_protocol.json"))),
                "symbolic_search_calls": len(list((run / "symbolic_searches").glob("call-*"))),
                "controller_elapsed_seconds": controller_seconds,
                "evaluation_seconds_sum": sum(
                    float(case.get("evaluation_wall_seconds") or 0)
                    for cases in raw
                    for case in cases.values()
                ),
                "evaluation_timing_complete": all(
                    isinstance(case.get("evaluation_wall_seconds"), (int, float))
                    for cases in raw
                    for case in cases.values()
                ),
                "tokens": dict(tokens),
                "token_usage_complete": token_complete,
                "inference_steps": model_requests,
                "agent_metrics_files": [str(p.relative_to(root)) for p in traces],
                "training": p.read(run / "incumbent_result.json")
                if (run / "incumbent_result.json").exists()
                else None,
                "held_out": p.read(test)["result"] if test.exists() else None,
            }
        )
    report = {
        "schema": SCHEMA,
        "at": datetime.now(UTC).isoformat(),
        "comparison_scope": manifest["comparison_scope"],
        "runs": rows,
    }
    p.write(root / "comparison_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cmd = sub.add_parser("prepare")
    cmd.add_argument("--selection", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--seed", type=int, action="append", required=True)
    cmd.add_argument("--train-size", type=int, required=True)
    cmd.add_argument("--test-size", type=int, required=True)
    cmd.add_argument("--budget", type=int, required=True)
    cmd.add_argument("--baseline", choices=["tight", "loose"], required=True)
    cmd.add_argument(
        "--policy-profile", choices=["pauli", "fermionic_nested"], default="fermionic_nested"
    )
    cmd.add_argument(
        "--energy-target", choices=["ground_branch", "hf_accessible"], default="ground_branch"
    )
    cmd = sub.add_parser("run")
    cmd.add_argument("--run-dir", type=Path, required=True)
    cmd.add_argument("--worker", choices=["external", "local"], default="external")
    cmd.add_argument("--pysr-python", default=sys.executable)
    cmd.add_argument("--pysr-evals-per-call", type=int, default=20)
    cmd.add_argument("--pysr-seconds", type=float, default=180)
    cmd.add_argument("--tool-url")
    cmd.add_argument("--listen", default="127.0.0.1")
    cmd.add_argument("--port", type=int, default=0)
    cmd.add_argument("--agent-timeout", type=int, default=1800)
    cmd = sub.add_parser("report")
    cmd.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_comparison(
            args.selection.resolve(),
            args.output.resolve(),
            seeds=args.seed,
            train_size=args.train_size,
            test_size=args.test_size,
            budget=args.budget,
            baseline=args.baseline,
            policy_profile=args.policy_profile,
            energy_target=args.energy_target,
        )
    elif args.action == "report":
        result = comparison_report(args.root.resolve())
    else:
        oracle = TrainingOracle(args.run_dir, worker=args.worker)
        symbolic = SymbolicSearch(
            oracle,
            python=args.pysr_python,
            max_evals=args.pysr_evals_per_call,
            seconds=args.pysr_seconds,
        )
        result = run_search(
            oracle,
            symbolic,
            tool_url=args.tool_url,
            listen=args.listen,
            port=args.port,
            agent_timeout=args.agent_timeout,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
