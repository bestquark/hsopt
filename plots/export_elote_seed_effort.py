#!/usr/bin/env python3
"""Read-only campaign audit: freeze completed rounds, decisions and agent effort.

Run with --ssh HOST to read remote files without installing the exporter there.
Use --snapshot JSON to replay an existing frozen audit without live SSH reads.
Only the requested local output directory is written. Raw Hamiltonians, case
receipts and agent messages are never exported. Input/output counts are kept
separate; cached and reasoning fields are reported separately, never added to
output. OpenCode step-finish IDs deduplicate resumed or copied event streams.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess


TOKEN_SEMANTICS = (
    "Input and output are separate OpenCode step_finish receipt fields, summed "
    "across model calls including repeated input context. Reasoning and cache "
    "fields are reported separately, never added to output. Preparation and "
    "editing are already concatenated in opencode.jsonl; preparation_result.json "
    "and agent_audit.json are not separately summed. Stable "
    "(sessionID,messageID,part.id) keys deduplicate repeated receipts globally. "
    "No missing usage is imputed as zero. Fields ending _mean_per_complete_round "
    "use only rounds with complete usage for every identified attempt. The LaTeX "
    "table reports these means in separate input and output columns. Fields "
    "ending _observed_per_completed_round_lower_bound divide "
    "observed totals by all completed rounds, including partially logged rounds. "
    "Printed means use ordinary rounding with M/B suffixes."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def collect(campaign):
    campaign = Path(campaign)
    sources, rounds, attempts, summary = [], [], [], []
    cutoff = {}
    for seed in sorted(campaign.glob("seed-*")):
        completed = sorted(int(p.parent.name) for p in (seed / "iterations").glob("[0-9][0-9][0-9][0-9]/decision.json") if int(p.parent.name) > 0)
        assert completed == list(range(1, max(completed, default=0) + 1)), seed
        cutoff[seed.name] = len(completed)
    captured = datetime.now(timezone.utc).isoformat()
    seen_events = set()

    def load(path):
        data = path.read_bytes()
        sources.append({"path": str(path.relative_to(campaign)), "bytes": len(data), "sha256": digest(data)})
        return json.loads(data)

    def token_attempt(seed, n, path, archived):
        metrics_path = path / "opencode_metrics.json"
        metrics = load(metrics_path) if metrics_path.exists() else None
        raw_path = path / "opencode.jsonl"
        if not raw_path.exists():
            raw_path = path / "opencode.jsonl.gz"
        record = {"seed": seed.name, "round": n, "attempt": str(path.relative_to(seed)), "archived_retry": archived,
                  "token_source": None, "complete": False, "input": None, "output": None, "reasoning": None,
                  "cache_read": None, "cache_write": None, "finished_steps": 0, "started_steps": 0,
                  "duplicate_step_receipts_excluded": 0}
        if raw_path.exists():
            raw_digest = hashlib.sha256()
            uncompressed_bytes = 0
            totals = Counter()
            finished = started = duplicate = malformed = 0
            session_ids = set()
            with (gzip.open(raw_path, "rb") if raw_path.suffix == ".gz" else raw_path.open("rb")) as stream:
                for raw_line in stream:
                    raw_digest.update(raw_line)
                    uncompressed_bytes += len(raw_line)
                    try:
                        event = json.loads(raw_line)
                    except (ValueError, UnicodeDecodeError):
                        malformed += 1
                        continue
                    if event.get("type") == "step_start":
                        started += 1
                    if event.get("type") != "step_finish":
                        continue
                    finished += 1
                    part = event.get("part") or {}
                    identity = (event.get("sessionID"), part.get("messageID"), part.get("id"))
                    if all(identity):
                        if identity in seen_events:
                            duplicate += 1
                            continue
                        seen_events.add(identity)
                    else:
                        raise ValueError(f"Step-finish receipt has no stable ID: {raw_path}")
                    session_ids.add(event.get("sessionID"))
                    tokens = part.get("tokens") or {}
                    if not isinstance(tokens.get("input"), (int, float)) or not isinstance(tokens.get("output"), (int, float)):
                        raise ValueError(f"Step-finish receipt lacks input/output: {raw_path}")
                    for name in ("input", "output", "reasoning"):
                        totals[name] += tokens.get(name, 0)
                    for name in ("read", "write"):
                        totals["cache_" + name] += (tokens.get("cache") or {}).get(name, 0)
            sources.append({"path": str(raw_path.relative_to(campaign)), "bytes_uncompressed": uncompressed_bytes,
                            "bytes_on_disk": raw_path.stat().st_size,
                            "sha256_uncompressed": raw_digest.hexdigest()})
            record.update(token_source="deduplicated_raw_step_finish", finished_steps=finished, started_steps=started,
                          duplicate_step_receipts_excluded=duplicate, malformed_lines=malformed, sessions=sorted(session_ids),
                          complete=finished > 0 and started == finished and not malformed and (metrics is None or metrics.get("token_usage_complete") is True))
            if finished:
                record.update({name: totals[name] for name in ("input", "output", "reasoning", "cache_read", "cache_write")})
            if metrics and not duplicate:
                for name, value in totals.items():
                    assert value == metrics.get("tokens", {}).get(name, 0), (raw_path, name, value, metrics.get("tokens"))
        elif metrics and metrics.get("raw_log_present") and metrics.get("tokens"):
            raise ValueError(
                f"Raw inference log required to deduplicate token receipts: {path}; "
                "metrics-only totals cannot establish an observed lower bound."
            )
        return record

    for seed_name, maximum in cutoff.items():
        seed = campaign / seed_name
        seed_rounds, seed_attempts = [], []
        for n in range(1, maximum + 1):
            folder = seed / "iterations" / f"{n:04d}"
            decision = load(folder / "decision.json")
            result = load(folder / "result.json")
            reason = decision.get("acceptance_reason")
            errors = [str(c["evaluation_error"]) for c in result.get("case_metrics", []) if c.get("evaluation_error")]
            numerical = [error for error in errors if re.search(r"lanczos|uncertif|inconclusive|converg|residual", error, re.I)]
            execution = [error for error in errors if error not in numerical]
            if decision.get("accepted") is True:
                category = "accepted"
            elif reason == "no_aggregate_improvement":
                category = "nonimproving"
            elif reason in ("a_suite_case_failed_accuracy", "certified_qpe_energy_contract_violation"):
                category = "accuracy_rejection"
            elif reason == "evaluation_failed":
                category = (
                    "mixed_evaluation_failure" if numerical and execution
                    else "numerical_uncertified" if numerical
                    else "execution_failure" if execution
                    else "unknown_evaluation_failure"
                )
            else:
                category = "other_not_retained"
            retry_paths = sorted((seed / "mutation_failures").glob(f"{n:04d}-attempt-*"))
            records = [token_attempt(seed, n, path, True) for path in retry_paths]
            records.append(token_attempt(seed, n, folder, False))
            seed_attempts.extend(records)
            complete = all(row["complete"] for row in records)
            row = {"seed": seed_name, "round": n, "accepted": decision.get("accepted") is True,
                   "acceptance_reason": reason, "outcome_category": category,
                   "numerical_error_cases": len(numerical), "execution_error_cases": len(execution),
                   "error_signatures": sorted(set(error.split("[")[0][:200] for error in errors)),
                   "archived_retries": len(retry_paths), "token_attempts": len(records), "token_usage_complete": complete,
                   "score": result.get("score"), "candidate_sha256": decision.get("shared_candidate_sha256")}
            for name in ("input", "output", "reasoning", "cache_read", "cache_write"):
                observed = [r[name] for r in records if r[name] is not None]
                row[name + "_observed"] = sum(observed) if observed else None
            seed_rounds.append(row)
        count = Counter(row["outcome_category"] for row in seed_rounds)
        covered = [row for row in seed_rounds if row["token_usage_complete"]]
        entry = {"seed": seed_name, "completed_rounds": maximum, "accepted": count["accepted"],
                 "not_retained": maximum - count["accepted"], "accepted_percent": 100 * count["accepted"] / maximum if maximum else None,
                 "not_retained_percent": 100 * (maximum - count["accepted"]) / maximum if maximum else None,
                 "outcomes": dict(count), "decision_reasons": dict(Counter(row["acceptance_reason"] for row in seed_rounds)),
                 "uncertified_decisions": count["numerical_uncertified"] + count["mixed_evaluation_failure"],
                 "rounds_with_numerical_error": sum(row["numerical_error_cases"] > 0 for row in seed_rounds),
                 "rounds_with_execution_error": sum(row["execution_error_cases"] > 0 for row in seed_rounds),
                 "archived_retry_attempts": sum(row["archived_retries"] for row in seed_rounds),
                 "token_attempts": len(seed_attempts), "token_complete_attempts": sum(row["complete"] for row in seed_attempts),
                 "token_complete_rounds": len(covered), "token_coverage_percent": 100 * len(covered) / maximum if maximum else None,
                 "duplicate_step_receipts_excluded": sum(row["duplicate_step_receipts_excluded"] for row in seed_attempts)}
        for name in ("input", "output", "reasoning", "cache_read", "cache_write"):
            observed = [row[name] for row in seed_attempts if row[name] is not None]
            entry[name + "_observed_total"] = sum(observed) if observed else None
            entry[name + "_observed_per_completed_round_lower_bound"] = sum(observed) / maximum if observed and maximum else None
            values = [row[name + "_observed"] for row in covered if row[name + "_observed"] is not None]
            entry[name + "_mean_per_complete_round"] = statistics.mean(values) if values else None
        summary.append(entry)
        rounds.extend(seed_rounds)
        attempts.extend(seed_attempts)
    return {"schema": "elote-seed-effort-v1", "campaign": str(campaign), "captured_utc": captured,
            "completed_round_cutoffs": cutoff, "seeds": summary, "rounds": rounds, "attempts": attempts,
            "sources": sources,
            "scope": "Round 0 excluded. Every completed decision in each frozen 1..N prefix included. Archived mutation_failures/<round>-attempt-* retries included only for these completed rounds; pending-round work excluded.",
            "token_semantics": TOKEN_SEMANTICS,
            "outcome_semantics": "Not retained means any decision accepted=false. Accuracy rejections, infeasible code/execution errors, and unresolved numerical certificates are separate subtypes; uncertified does not establish physical inaccuracy."}


def mean_tokens(value):
    """Format a fully logged-round token mean with an M/B suffix."""
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError("Token means must be finite and nonnegative")
    scale, suffix = (Decimal("1000000000"), "B") if value >= 1000000000 else (Decimal("1000000"), "M")
    amount = (value / scale).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return rf"{amount:.4f}\,\mathrm{{{suffix}}}"


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "seed_effort.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in (("seeds", result["seeds"]), ("rounds", result["rounds"]), ("attempts", result["attempts"]), ("provenance", result["sources"])):
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    table = []
    mapping = result.get("paper_run_label_to_original_seed")
    labels = {seed: int(label) for label, seed in mapping.items()} if mapping else {}
    if mapping and (
        len(labels) != len(mapping)
        or set(labels) != {row["seed"] for row in result["seeds"]}
        or any(label <= 0 for label in labels.values())
    ):
        raise ValueError("Paper labels must map one-to-one onto the reported original runs")
    for row in result["seeds"]:
        a, b = row["input_mean_per_complete_round"], row["output_mean_per_complete_round"]
        input_tokens = rf"${mean_tokens(a)}$" if a is not None else "---"
        output_tokens = rf"${mean_tokens(b)}$" if b is not None else "---"
        label = labels.get(row["seed"], int(row["seed"][-2:]))
        table.append(f"{label} & {row['completed_rounds']} & {row['accepted_percent']:.1f} & {row['not_retained_percent']:.1f} & {row['uncertified_decisions']} & {input_tokens} & {output_tokens} " + r"\\")
    (output / "table_body.tex").write_text("\n".join(table) + "\n")
    manifest = {"exporter_sha256": digest(Path(__file__).read_bytes()), "files": {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign")
    source.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ssh")
    args = parser.parse_args()
    if args.snapshot:
        if args.ssh:
            parser.error("--ssh is only valid with --campaign")
        snapshot_bytes = args.snapshot.read_bytes()
        result = json.loads(snapshot_bytes)
        if result.get("schema") != "elote-seed-effort-v1":
            parser.error("--snapshot requires an elote-seed-effort-v1 audit")
        result["token_semantics"] = TOKEN_SEMANTICS
        result["replayed_from"] = {"path": str(args.snapshot.resolve()), "sha256": digest(snapshot_bytes)}
    elif args.ssh:
        command = "python3 - --campaign " + shlex.quote(args.campaign) + " --output -"
        remote = subprocess.run(["ssh", args.ssh, command], input=Path(__file__).read_text(), capture_output=True, text=True, check=True)
        result = json.loads(remote.stdout)
    else:
        result = collect(args.campaign)
    if args.output == "-":
        print(json.dumps(result))
    else:
        write_outputs(result, Path(args.output))
        print(json.dumps({"captured_utc": result["captured_utc"], "seeds": result["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
