"""Render deterministic, final-only PDF reports from strict normalized exports.

The renderer deliberately does not know how to read a campaign database or a
mutation workspace.  Its input is the portable output of one of the strict
paper exporters, including its completion marker and checksum manifest.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HAMLIB_LANES = 82
HAMLIB_RUNS = 5
HAMLIB_ITERATIONS = 50
REACTIONS = 10
REACTION_CONTRACTS = 2
REACTION_ITERATIONS = 50
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ReportValidationError(RuntimeError):
    """Raised when an export cannot prove that final reports are complete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ReportValidationError(f"missing or unsafe normalized table: {path.name}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_strict_export(export_dir: Path) -> None:
    export_dir = export_dir.expanduser().resolve()
    marker = export_dir / "FINAL_EXPORT_COMPLETE"
    checksum_path = export_dir / "SHA256SUMS"
    if (
        not marker.is_file()
        or marker.is_symlink()
        or marker.read_text(encoding="utf-8") != "complete\n"
    ):
        raise ReportValidationError("strict export completion marker is missing or invalid")
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ReportValidationError("strict export checksum manifest is missing or unsafe")
    declared: dict[str, str] = {}
    for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", line)
        if match is None:
            raise ReportValidationError(f"invalid SHA256SUMS line {number}")
        digest, relative_text = match.groups()
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in declared:
            raise ReportValidationError(f"unsafe or duplicate checksum path: {relative_text}")
        path = export_dir / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ReportValidationError(f"checksum mismatch: {relative_text}")
        declared[relative_text] = digest
    actual = {
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(declared) != actual:
        raise ReportValidationError("SHA256SUMS does not cover the exact strict-export file set")


_ASCII_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "→": "->",
        "←": "<-",
        "×": "x",
        "≤": "<=",
        "≥": ">=",
        "Δ": "Delta",
        "δ": "delta",
        "ε": "epsilon",
        "π": "pi",
        "µ": "u",
        "·": "*",
        "“": '"',
        "”": '"',
        "’": "'",
    }
)
_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: object) -> str:
    text = " ".join(str(value if value is not None else "").split())
    text = text.translate(_ASCII_TRANSLATION)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "replace").decode("ascii")
    return "".join(_LATEX_ESCAPE.get(character, character) for character in text)


def _float(value: object, *, default: float = math.nan) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(f"expected a numeric report value, got {value!r}") from exc


def _integer(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(f"expected an integer report value, got {value!r}") from exc


def _number(value: object, digits: int = 6) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return "--"
    if number == 0:
        return "0"
    if abs(number) < 10 ** (-(digits - 1)) or abs(number) >= 10**6:
        return f"{number:.{max(1, digits - 1)}e}"
    return f"{number:.{digits}g}"


def _yes(value: object) -> str:
    return "yes" if str(value).lower() in {"1", "true", "yes"} else "no"


def _pf(order: object) -> str:
    return f"PF{_integer(order)}" if value_present(order) else "--"


def value_present(value: object) -> bool:
    return value not in (None, "", "None", "null")


def _qpe_cost(row: Mapping[str, object], prefix: str = "") -> str:
    cost = row.get(f"{prefix}total_t_gates_qpe")
    raw = row.get(f"{prefix}per_attempt_t_gates_qpe")
    repetitions = row.get(f"{prefix}qpe_repetitions")
    if value_present(raw) and value_present(repetitions):
        return f"{cost} ({raw} x {repetitions})"
    return str(cost or "--")


def _reaction_qpe_cost(row: Mapping[str, object]) -> str:
    cost = row.get("total_t_gate_count")
    raw = row.get("per_attempt_t_gate_count")
    if value_present(raw):
        return f"{cost} (raw {raw})"
    return str(cost or "--")


def _order(value: object) -> str:
    if not value_present(value):
        return "--"
    return "custom" if str(value).lower() == "custom" else str(value)


def _schedule(row: Mapping[str, object], prefix: str = "") -> str:
    strategy = str(row.get(f"{prefix}schedule_strategy") or "stationary_trotter")
    if strategy == "custom":
        return "custom"
    if strategy == "qdrift":
        samples = row.get(f"{prefix}qdrift_samples")
        return f"qdrift; samples={samples}" if value_present(samples) else "qdrift"
    steps = row.get(f"{prefix}n_steps")
    label = strategy.replace("_", " ")
    return f"{label}; r={steps}" if value_present(steps) else label


def _assert_energy(row: Mapping[str, str], prefix: str = "") -> float:
    exact = _float(row.get(f"{prefix}ground_state_energy_hartree"))
    estimate = _float(row.get(f"{prefix}ground_state_energy_estimate_hartree"))
    stored = _float(row.get(f"{prefix}ground_state_energy_error_hartree"))
    computed = abs(estimate - exact)
    if not math.isclose(computed, abs(stored), rel_tol=1e-9, abs_tol=2e-12):
        raise ReportValidationError("HamLib exact-energy error arithmetic is inconsistent")
    return computed


def _document(title: str, body: str) -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,landscape,margin=0.45in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs,longtable,array,microtype}}
\usepackage[table]{{xcolor}}
\definecolor{{rowgray}}{{gray}}{{0.94}}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.16}}
\setlength{{\LTpre}}{{4pt}}
\setlength{{\LTpost}}{{4pt}}
\pagestyle{{plain}}
\pdfinfoomitdate=1
\pdftrailerid{{}}
\begin{{document}}
\begin{{center}}\Large\bfseries {latex_escape(title)}\end{{center}}
{body}
\end{{document}}
"""


def _longtable(columns: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    header_line = " & ".join(rf"\textbf{{{latex_escape(item)}}}" for item in header) + r" \\"
    rendered: list[str] = [
        r"{\scriptsize",
        rf"\begin{{longtable}}{{{columns}}}",
        r"\toprule",
        header_line,
    ]
    rendered.extend(
        [r"\midrule", r"\endfirsthead", r"\toprule", header_line, r"\midrule", r"\endhead"]
    )
    for index, row in enumerate(rows):
        if index % 2:
            rendered.append(r"\rowcolor{rowgray}")
        rendered.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    rendered.extend([r"\bottomrule", r"\end{longtable}", "}"])
    return "\n".join(rendered)


def _hamlib_summary(lane_id: str, rows: Sequence[Mapping[str, str]]) -> str:
    first = rows[0]
    table_rows: list[list[object]] = []
    for row in sorted(rows, key=lambda item: item["run"]):
        ar = f"{row['accepted']}/{row['rejected']}"
        baseline_error = 1000.0 * _assert_energy(row, "baseline_")
        final_error = 1000.0 * _assert_energy(row, "best_")
        common = [row["run"], ar]
        table_rows.append(
            common
            + [
                "Initial",
                f"{_pf(row['baseline_formula_order'])}; {_order(row['baseline_term_order_strategy'])}",
                _schedule(row, "baseline_"),
                f"eps={_number(row['baseline_uniform_epsilon'], 4)}; zero={_number(row['baseline_zero_angle_threshold'], 3)}",
                _qpe_cost(row, "baseline_"),
                _number(baseline_error, 5),
                "--",
            ]
        )
        table_rows.append(
            common
            + [
                f"Final incumbent (accepted Iter. {row['best_iteration']})",
                f"{_pf(row['best_formula_order'])}; {_order(row['best_term_order_strategy'])}",
                _schedule(row, "best_"),
                f"eps={_number(row['best_uniform_epsilon'], 4)}; zero={_number(row['best_zero_angle_threshold'], 3)}",
                _qpe_cost(row, "best_"),
                _number(final_error, 5),
                f"{_number(row['reduction_percent'], 5)}%",
            ]
        )
    identity = (
        f"Lane {lane_id}; molecule {first['molecule']}; {first['nqubits']} qubits; "
        f"{first['electrons']} electrons; {first['terms']} Pauli terms. "
        "A/R denotes accepted/rejected mutations. Energy errors are measured against the exact "
        "ground-state energy in the fixed evaluator. T95 is the 95%-confidence "
        "cost; parentheses show per-attempt T count x QPE attempts."
    )
    body = (
        latex_escape(identity)
        + "\n"
        + _longtable(
            r"llp{4.0cm}p{3.4cm}p{3.3cm}p{3.8cm}rrr",
            (
                "Run",
                "A/R",
                "Protocol",
                "Product formula / order",
                "Schedule / repetitions",
                "Epsilon / zero threshold",
                "T95 (U x M x R)",
                "|Delta E| (mHa)",
                "Reduction",
            ),
            table_rows,
        )
    )
    return _document(f"HamLib82 summary: {lane_id}", body)


def _hamlib_extended(
    lane_id: str,
    summaries: Sequence[Mapping[str, str]],
    candidates: Sequence[Mapping[str, str]],
) -> str:
    baseline_rows: list[list[object]] = []
    for row in sorted(summaries, key=lambda item: item["run"]):
        baseline_rows.append(
            [
                row["run"],
                _pf(row["baseline_formula_order"]),
                _schedule(row, "baseline_"),
                _order(row["baseline_term_order_strategy"]),
                _qpe_cost(row, "baseline_"),
                _number(row["baseline_ground_state_energy_estimate_hartree"], 9),
                _number(1000.0 * _assert_energy(row, "baseline_"), 6),
            ]
        )
    mutation_rows: list[list[object]] = []
    for row in sorted(candidates, key=lambda item: (item["run"], _integer(item["iteration"]))):
        has_energy = all(
            value_present(row.get(key))
            for key in (
                "ground_state_energy_hartree",
                "ground_state_energy_estimate_hartree",
                "ground_state_energy_error_hartree",
            )
        )
        error = _assert_energy(row) if has_energy else math.nan
        mutation_rows.append(
            [
                row["run"],
                row["iteration"],
                "accepted" if _yes(row["accepted"]) == "yes" else "rejected",
                row.get("mutation_summary") or "--",
                f"{_pf(row['formula_order'])}; {_order(row['term_order_strategy'])}; {_schedule(row)}",
                _qpe_cost(row),
                _number(row["ground_state_energy_estimate_hartree"], 8),
                _number(1000.0 * error, 5),
                _yes(row["passes_accuracy"]),
                row.get("decision_reason") or "--",
            ]
        )
    exact = summaries[0]["baseline_ground_state_energy_hartree"]
    intro = (
        f"Lane {lane_id}. Exact ground-state energy: {_number(exact, 10)} Ha. "
        "Each row records the attempted mutation and the fixed evaluator outcome; rejected rows "
        "do not replace the incumbent."
    )
    body = (
        latex_escape(intro)
        + "\n\\textbf{Initial product formula}\n"
        + _longtable(
            r"llp{4.4cm}p{3.2cm}rrr",
            (
                "Run",
                "PF",
                "Schedule / repetitions",
                "Term order",
                "T95 (U x M x R)",
                "Estimated E (Ha)",
                "|Delta E| (mHa)",
            ),
            baseline_rows,
        )
    )
    body += "\n\\textbf{Mutation log}\n" + _longtable(
        r"lllp{7.0cm}p{4.0cm}rrrll",
        (
            "Run",
            "Iter.",
            "Decision",
            "Mutation summary",
            "Attempted PF settings",
            "T95 (U x M x R)",
            "Estimated E",
            "|Delta E| mHa",
            "Pass",
            "Reason",
        ),
        mutation_rows,
    )
    return _document(f"HamLib82 extended log: {lane_id}", body)


def _species_signature(raw: object) -> str:
    try:
        species = json.loads(str(raw)) if not isinstance(raw, dict) else raw
    except json.JSONDecodeError as exc:
        raise ReportValidationError("invalid species_json in reaction export") from exc
    parts = []
    for name, item in sorted(species.items()):
        schedule = item.get("schedule_strategy") or (item.get("method_config") or {}).get(
            "schedule_strategy", "stationary_trotter"
        )
        order = item.get("term_order_strategy") or (item.get("method_config") or {}).get(
            "term_order_strategy", "input"
        )
        repetitions = item.get("n_steps")
        if schedule == "custom":
            schedule_text = "custom"
        else:
            schedule_text = f"{str(schedule).replace('_', ' ')} r={repetitions}"
        qpe = ""
        if value_present(item.get("confidence_adjusted_t_gate_count")):
            qpe = (
                f"/T95={item['confidence_adjusted_t_gate_count']}"
                f"(raw={item.get('per_attempt_t_gate_count')},"
                f"R={item.get('qpe_repetitions')},"
                f"p={_number(item.get('qpe_peak_probability_from_hf'), 4)})"
            )
        parts.append(f"{name}:PF{item.get('formula_order')}/{_order(order)}/{schedule_text}{qpe}")
    return "; ".join(parts)


def _reaction_incumbents(
    rows: Sequence[Mapping[str, str]],
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    ordered = sorted(rows, key=lambda item: _integer(item["iteration"]))
    if not ordered or _integer(ordered[0]["iteration"]) != 0:
        raise ReportValidationError("reaction lane has no iteration-0 baseline")
    incumbent = ordered[0]
    for row in ordered[1:]:
        if _yes(row["accepted"]) == "yes":
            incumbent = row
    return ordered[0], incumbent


def _reaction_error(row: Mapping[str, str]) -> float:
    signed = _float(row["pf_reaction_energy_hartree"]) - _float(
        row["exact_reaction_energy_hartree"]
    )
    if not math.isclose(
        signed,
        _float(row["signed_reaction_error_hartree"]),
        rel_tol=1e-9,
        abs_tol=2e-12,
    ) or not math.isclose(
        abs(signed),
        _float(row["absolute_reaction_error_hartree"]),
        rel_tol=1e-9,
        abs_tol=2e-12,
    ):
        raise ReportValidationError("reaction exact-energy error arithmetic is inconsistent")
    return signed


def _reaction_summary(
    reaction_key: str,
    rows: Sequence[Mapping[str, str]],
    pooled: Mapping[str, Any],
) -> str:
    by_contract: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_contract[row["source_contract"]].append(row)
    table_rows: list[list[object]] = []
    for contract in sorted(by_contract):
        baseline, final = _reaction_incumbents(by_contract[contract])
        proposals = [item for item in by_contract[contract] if _integer(item["iteration"]) > 0]
        accepted = sum(_yes(item["accepted"]) == "yes" for item in proposals)
        ar = f"{accepted}/{len(proposals) - accepted}"
        reduction = (
            100.0
            * (_integer(baseline["total_t_gate_count"]) - _integer(final["total_t_gate_count"]))
            / _integer(baseline["total_t_gate_count"])
        )
        for label, item, reduction_text in (
            ("Initial", baseline, "--"),
            (
                f"Final incumbent (accepted Iter. {item_iteration(final)})",
                final,
                f"{_number(reduction, 5)}%",
            ),
        ):
            reaction_error = abs(_reaction_error(item))
            table_rows.append(
                [
                    contract,
                    ar,
                    label,
                    _species_signature(item["species_json"]),
                    _reaction_qpe_cost(item),
                    _number(1000.0 * reaction_error, 5),
                    _number(1000.0 * _float(item["max_species_absolute_error_hartree"]), 5),
                    reduction_text,
                ]
            )
    pooled_rows = []
    for selection in ("equal_share", "reaction_aware"):
        item = pooled[selection]
        pooled_rows.append(
            [
                selection,
                item["selected_source_contract"],
                item["selected_source_iteration"],
                _reaction_qpe_cost(item),
                _number(1000.0 * _float(item["absolute_reaction_error_hartree"]), 5),
                _number(1000.0 * _float(item["max_species_absolute_error_hartree"]), 5),
            ]
        )
    body = latex_escape(
        f"Reaction {pooled['reaction']}. A/R denotes accepted/rejected mutations. "
        "Lane incumbents are shown separately from the pooled paper selections."
    )
    body += "\n" + _longtable(
        r"llp{3.8cm}p{7.3cm}rrrr",
        (
            "Contract",
            "A/R",
            "Protocol",
            "PF parameters by species",
            "T95 (sum U x M x R)",
            "|dE reaction| (mHa)",
            "max |dE species| (mHa)",
            "Reduction",
        ),
        table_rows,
    )
    body += "\n\\textbf{Pooled paper selections}\n" + _longtable(
        r"llllrr",
        (
            "Selection",
            "Source contract",
            "Iteration",
            "T95 (sum U x M x R)",
            "|dE reaction| (mHa)",
            "max |dE species| (mHa)",
        ),
        pooled_rows,
    )
    return _document(f"Reactions10 summary: {reaction_key}", body)


def item_iteration(row: Mapping[str, str]) -> str:
    return str(_integer(row["iteration"]))


def _reaction_extended(
    reaction_key: str, rows: Sequence[Mapping[str, str]], pooled: Mapping[str, Any]
) -> str:
    baseline_rows: list[list[object]] = []
    mutation_rows: list[list[object]] = []
    for row in sorted(
        rows, key=lambda item: (item["source_contract"], _integer(item["iteration"]))
    ):
        target = baseline_rows if _integer(row["iteration"]) == 0 else mutation_rows
        if target is baseline_rows:
            signed = _reaction_error(row)
            target.append(
                [
                    row["source_contract"],
                    _species_signature(row["species_json"]),
                    _reaction_qpe_cost(row),
                    _number(row["exact_reaction_energy_hartree"], 9),
                    _number(row["pf_reaction_energy_hartree"], 9),
                    _number(1000.0 * abs(signed), 5),
                ]
            )
        else:
            has_energy = all(
                value_present(row.get(key))
                for key in (
                    "pf_reaction_energy_hartree",
                    "exact_reaction_energy_hartree",
                    "signed_reaction_error_hartree",
                    "absolute_reaction_error_hartree",
                )
            )
            signed = _reaction_error(row) if has_energy else math.nan
            target.append(
                [
                    row["source_contract"],
                    row["iteration"],
                    "accepted" if _yes(row["accepted"]) == "yes" else "rejected",
                    row.get("mutation_summary") or "--",
                    _species_signature(row["species_json"]),
                    _reaction_qpe_cost(row),
                    _number(1000.0 * signed, 5),
                    _number(1000.0 * _float(row["max_species_absolute_error_hartree"]), 5),
                    _yes(row["passes_contract"]),
                    row.get("decision_reason") or "--",
                ]
            )
    body = latex_escape(
        f"Reaction {pooled['reaction']}. The exact reaction energy and every mutation error are "
        "recomputed from the normalized per-species evaluator results."
    )
    body += "\n\\textbf{Initial product formulas}\n" + _longtable(
        r"lp{11cm}rrrr",
        (
            "Contract",
            "PF parameters by species",
            "T95 (sum U x M x R)",
            "Exact E_rxn",
            "PF E_rxn",
            "|Delta E_rxn| mHa",
        ),
        baseline_rows,
    )
    body += "\n\\textbf{Mutation log}\n" + _longtable(
        r"lllp{5.5cm}p{5.5cm}rrrll",
        (
            "Contract",
            "Iter.",
            "Decision",
            "Mutation summary",
            "Attempted PF settings",
            "T95 (sum U x M x R)",
            "signed dE (mHa)",
            "max species error (mHa)",
            "Pass",
            "Reason",
        ),
        mutation_rows,
    )
    return _document(f"Reactions10 extended log: {reaction_key}", body)


def _compile_pdf(tex: str, destination: Path, latexmk: str) -> None:
    build = Path(tempfile.mkdtemp(prefix="hsopt-latex-"))
    try:
        source = build / "report.tex"
        source.write_text(tex, encoding="utf-8")
        environment = os.environ.copy()
        environment.update({"LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": "315532800"})
        process = subprocess.run(
            [
                latexmk,
                "-pdf",
                "-halt-on-error",
                "-file-line-error",
                "-interaction=nonstopmode",
                "-no-shell-escape",
                source.name,
            ],
            cwd=build,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        pdf = build / "report.pdf"
        if process.returncode or not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            tail = "\n".join(process.stdout.splitlines()[-80:])
            raise ReportValidationError(f"LaTeX compilation failed:\n{tail}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, destination)
    finally:
        shutil.rmtree(build, ignore_errors=True)


def _hamlib_reportlab_content(
    lane_id: str,
    summaries: Sequence[Mapping[str, str]],
    candidates: Sequence[Mapping[str, str]],
    *,
    extended: bool,
) -> tuple[str, list[tuple[str, list[str]]]]:
    first = summaries[0]
    intro = (
        f"Lane {lane_id}; molecule {first['molecule']}; {first['nqubits']} qubits; "
        f"{first['electrons']} electrons; {first['terms']} Pauli terms."
    )
    baseline_lines = []
    for row in sorted(summaries, key=lambda item: item["run"]):
        baseline_error = 1000.0 * _assert_energy(row, "baseline_")
        best_error = 1000.0 * _assert_energy(row, "best_")
        baseline_lines.append(
            f"{row['run']} | initial | T95={_qpe_cost(row, 'baseline_')} | "
            f"|dE|={_number(baseline_error, 6)} mHa | "
            f"{_pf(row['baseline_formula_order'])}, n_steps={row['baseline_n_steps']}, "
            f"order={_order(row['baseline_term_order_strategy'])}"
        )
        baseline_lines.append(
            f"{row['run']} | final after accepted iter {row['best_iteration']} | "
            f"T95={_qpe_cost(row, 'best_')} | |dE|={_number(best_error, 6)} mHa | "
            f"reduction={_number(row['reduction_percent'], 6)}% | "
            f"accepted/rejected={row['accepted']}/{row['rejected']}"
        )
    if not extended:
        return intro, [("Initial and final incumbents", baseline_lines)]

    mutation_lines = []
    for row in sorted(candidates, key=lambda item: (item["run"], _integer(item["iteration"]))):
        has_energy = all(
            value_present(row.get(key))
            for key in (
                "ground_state_energy_hartree",
                "ground_state_energy_estimate_hartree",
                "ground_state_energy_error_hartree",
            )
        )
        error_text = f"{_number(1000.0 * _assert_energy(row), 6)} mHa" if has_energy else "--"
        mutation_lines.append(
            f"{row['run']} iter {row['iteration']} | "
            f"{'accepted' if _yes(row['accepted']) == 'yes' else 'rejected'} | "
            f"T95={_qpe_cost(row)} | |dE|={error_text} | "
            f"pass={_yes(row.get('passes_accuracy'))} | "
            f"{row.get('mutation_summary') or '--'} | "
            f"reason={row.get('decision_reason') or '--'} | "
            f"agent tools={row.get('agent_tool_calls') or '--'}, "
            f"web={row.get('agent_web_searches') or '0'}, "
            f"tokens={row.get('agent_total_tokens') or '--'}"
        )
    return intro, [
        ("Initial product formulas", baseline_lines),
        ("Mutation log", mutation_lines),
    ]


def _reaction_reportlab_content(
    reaction_key: str,
    rows: Sequence[Mapping[str, str]],
    pooled: Mapping[str, Any],
    *,
    extended: bool,
) -> tuple[str, list[tuple[str, list[str]]]]:
    intro = f"Reaction {pooled['reaction']} ({reaction_key})."
    by_contract: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_contract[row["source_contract"]].append(row)
    summary_lines = []
    for contract in sorted(by_contract):
        baseline, final = _reaction_incumbents(by_contract[contract])
        proposals = [item for item in by_contract[contract] if _integer(item["iteration"]) > 0]
        accepted = sum(_yes(item["accepted"]) == "yes" for item in proposals)
        reduction = (
            100.0
            * (_integer(baseline["total_t_gate_count"]) - _integer(final["total_t_gate_count"]))
            / _integer(baseline["total_t_gate_count"])
        )
        summary_lines.append(
            f"{contract} | initial | T95={_reaction_qpe_cost(baseline)} | "
            f"|dE reaction|={_number(1000.0 * abs(_reaction_error(baseline)), 6)} mHa | "
            f"max |dE species|="
            f"{_number(1000.0 * _float(baseline['max_species_absolute_error_hartree']), 6)} mHa"
        )
        summary_lines.append(
            f"{contract} | final after accepted iter {item_iteration(final)} | "
            f"T95={_reaction_qpe_cost(final)} | "
            f"|dE reaction|={_number(1000.0 * abs(_reaction_error(final)), 6)} mHa | "
            f"reduction={_number(reduction, 6)}% | "
            f"accepted/rejected={accepted}/{len(proposals) - accepted}"
        )
    pooled_lines = []
    for selection in ("equal_share", "reaction_aware"):
        item = pooled[selection]
        pooled_lines.append(
            f"{selection} <- {item['selected_source_contract']} iter "
            f"{item['selected_source_iteration']} | T95={_reaction_qpe_cost(item)} | "
            f"|dE reaction|="
            f"{_number(1000.0 * _float(item['absolute_reaction_error_hartree']), 6)} mHa"
        )
    if not extended:
        return intro, [
            ("Initial and final incumbents", summary_lines),
            ("Pooled selections", pooled_lines),
        ]

    mutation_lines = []
    for row in sorted(
        rows, key=lambda item: (item["source_contract"], _integer(item["iteration"]))
    ):
        has_energy = all(
            value_present(row.get(key))
            for key in (
                "pf_reaction_energy_hartree",
                "exact_reaction_energy_hartree",
                "signed_reaction_error_hartree",
                "absolute_reaction_error_hartree",
            )
        )
        signed = _reaction_error(row) if has_energy else math.nan
        mutation_lines.append(
            f"{row['source_contract']} iter {row['iteration']} | "
            f"{'accepted' if _yes(row['accepted']) == 'yes' else 'rejected'} | "
            f"T95={_reaction_qpe_cost(row)} | "
            f"signed dE={_number(1000.0 * signed, 6)} mHa | "
            f"pass={_yes(row.get('passes_contract'))} | "
            f"{row.get('mutation_summary') or '--'} | "
            f"reason={row.get('decision_reason') or '--'} | "
            f"agent tools={row.get('agent_tool_calls') or '--'}, "
            f"web={row.get('agent_web_searches') or '0'}, "
            f"tokens={row.get('agent_total_tokens') or '--'}"
        )
    return intro, [
        ("Initial and final incumbents", summary_lines),
        ("Mutation log", mutation_lines),
        ("Pooled selections", pooled_lines),
    ]


def _compile_reportlab_pdf(
    title: str,
    intro: str,
    sections: Sequence[tuple[str, Sequence[str]]],
    destination: Path,
) -> None:
    """Render a portable, deterministic PDF without a system TeX installation."""

    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    class InvariantCanvas(Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["invariant"] = 1
            super().__init__(*args, **kwargs)

    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HSOptTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "HSOptHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        spaceBefore=8,
        spaceAfter=4,
    )
    line_style = ParagraphStyle(
        "HSOptLine",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=6.5,
        leading=8.2,
        spaceAfter=2,
    )
    intro_style = ParagraphStyle(
        "HSOptIntro",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        spaceAfter=6,
    )
    document = SimpleDocTemplate(
        str(destination),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        title=title,
        author="HSOpt",
    )
    story: list[Any] = [
        Paragraph(html.escape(title), title_style),
        Paragraph(html.escape(intro), intro_style),
    ]
    for heading, lines in sections:
        story.extend(
            [
                Spacer(1, 2),
                Paragraph(html.escape(heading), heading_style),
            ]
        )
        story.extend(Paragraph(html.escape(str(line)), line_style) for line in lines)
    document.build(story, canvasmaker=InvariantCanvas)
    if not destination.read_bytes().startswith(b"%PDF-"):
        raise ReportValidationError(f"ReportLab did not create a PDF: {destination}")


def _export_shape(export_dir: Path, family: str) -> dict[str, Any]:
    """Read dynamic exact-campaign dimensions, with legacy paper defaults."""

    path = export_dir / "export_manifest.json"
    if not path.is_file():
        return {
            "family": family,
            "instance_count": HAMLIB_LANES if family == "abs_energy" else REACTIONS,
            "lane_count": (
                HAMLIB_LANES * HAMLIB_RUNS
                if family == "abs_energy"
                else REACTIONS * REACTION_CONTRACTS
            ),
            "iterations": (HAMLIB_ITERATIONS if family == "abs_energy" else REACTION_ITERATIONS),
            "replicates": (
                [f"run_{index:02d}" for index in range(1, HAMLIB_RUNS + 1)]
                if family == "abs_energy"
                else ["equal_share", "reaction_aware"]
            ),
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("family") != family:
        raise ReportValidationError("exact-campaign export manifest is incomplete or mismatched")
    required = ("instance_count", "lane_count", "iterations", "replicates")
    if any(key not in manifest for key in required):
        raise ReportValidationError("exact-campaign export manifest lacks dimensions")
    return manifest


def _validate_hamlib(export_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    candidates = _read_csv(export_dir / "candidate_log.csv")
    summaries = _read_csv(export_dir / "lane_run_summary.csv")
    shape = _export_shape(export_dir, "abs_energy")
    expected_candidates = int(shape["lane_count"]) * int(shape["iterations"])
    if len(candidates) != expected_candidates:
        raise ReportValidationError(
            f"HamLib candidate_log.csv has {len(candidates)} rows, expected {expected_candidates}"
        )
    if len(summaries) != int(shape["lane_count"]):
        raise ReportValidationError(
            f"HamLib lane_run_summary.csv has {len(summaries)} rows, expected {shape['lane_count']}"
        )
    keys = {(row["run"], row["lane_id"], _integer(row["iteration"])) for row in candidates}
    if len(keys) != len(candidates):
        raise ReportValidationError("HamLib candidate log has duplicate run/lane/iteration rows")
    expected_iterations = set(range(1, int(shape["iterations"]) + 1))
    expected_runs = set(map(str, shape["replicates"]))
    for lane in {row["lane_id"] for row in summaries}:
        lane_rows = [row for row in candidates if row["lane_id"] == lane]
        summary_runs = {row["run"] for row in summaries if row["lane_id"] == lane}
        if summary_runs != expected_runs:
            raise ReportValidationError(f"{lane} lacks the declared replicate set")
        for run in summary_runs:
            iterations = {_integer(row["iteration"]) for row in lane_rows if row["run"] == run}
            if iterations != expected_iterations:
                raise ReportValidationError(f"{lane}/{run} lacks contiguous proposal iterations")
    return candidates, summaries


def _validate_reactions(
    export_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Mapping[str, Any]]]:
    candidates = _read_csv(export_dir / "candidate_log.csv")
    summary = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
    reaction_summary = json.loads(
        (export_dir / "reaction_summary.json").read_text(encoding="utf-8")
    )
    shape = _export_shape(export_dir, "diff_energy")
    expected_candidates = int(shape["lane_count"]) * (int(shape["iterations"]) + 1)
    if (
        summary.get("status") != "complete"
        or int(summary.get("candidate_record_count", -1)) != expected_candidates
    ):
        raise ReportValidationError(
            "reaction summary does not attest the complete declared campaign"
        )
    if len(candidates) != expected_candidates:
        raise ReportValidationError(
            f"reaction candidate_log.csv has {len(candidates)} rows, expected {expected_candidates}"
        )
    keys = {
        (row["reaction_key"], row["source_contract"], _integer(row["iteration"]))
        for row in candidates
    }
    if len(keys) != len(candidates):
        raise ReportValidationError("reaction candidate log has duplicate contract iterations")
    pooled = {row["reaction_key"]: row for row in reaction_summary["reactions"]}
    if len(pooled) != int(shape["instance_count"]):
        raise ReportValidationError(
            "reaction_summary.json does not contain the declared reaction set"
        )
    expected_iterations = set(range(int(shape["iterations"]) + 1))
    for reaction_key in pooled:
        for contract in map(str, shape["replicates"]):
            iterations = {
                _integer(row["iteration"])
                for row in candidates
                if row["reaction_key"] == reaction_key and row["source_contract"] == contract
            }
            if iterations != expected_iterations:
                raise ReportValidationError(
                    f"{reaction_key}/{contract} lacks contiguous evaluation iterations"
                )
    return candidates, pooled


def _write_report_metadata(
    root: Path,
    family: str,
    pdfs: Sequence[Path],
    export_dir: Path,
    *,
    engine: str,
) -> None:
    manifest = {
        "family": family,
        "schema_version": 1,
        "source": "strict normalized final export",
        "render_engine": engine,
        "source_checksum_manifest_sha256": sha256_file(export_dir / "SHA256SUMS"),
        "source_completion_marker_sha256": sha256_file(export_dir / "FINAL_EXPORT_COMPLETE"),
        "pdf_count": len(pdfs),
        "pdfs": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(pdfs)
        ],
    }
    (root / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum = root / "SHA256SUMS"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_reports(
    family: str,
    export_dir: Path,
    output_dir: Path,
    *,
    latexmk: str = "latexmk",
    engine: str = "auto",
) -> None:
    export_dir = export_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().absolute()
    verify_strict_export(export_dir)
    if output_dir == export_dir or export_dir in output_dir.parents:
        raise ReportValidationError("report output must not modify the strict source export")
    if output_dir.exists():
        raise ReportValidationError(f"refusing to overwrite report output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    pdfs: list[Path] = []
    try:
        if engine not in {"auto", "latex", "reportlab"}:
            raise ReportValidationError(f"unsupported report engine: {engine}")
        latex_path = shutil.which(latexmk)
        if latex_path is None and Path(latexmk).is_file():
            latex_path = latexmk
        selected_engine = (
            ("latex" if latex_path is not None else "reportlab") if engine == "auto" else engine
        )
        if selected_engine == "latex" and latex_path is None:
            raise ReportValidationError(f"LaTeX engine is unavailable: {latexmk}")
        if family == "abs_energy":
            candidates, summaries = _validate_hamlib(export_dir)
            shape = _export_shape(export_dir, family)
            lane_ids = sorted({row["lane_id"] for row in summaries})
            if len(lane_ids) != int(shape["instance_count"]):
                raise ReportValidationError(
                    "HamLib export does not contain the declared Hamiltonian set"
                )
            for lane_id in lane_ids:
                if SAFE_ID.fullmatch(lane_id) is None:
                    raise ReportValidationError(f"unsafe HamLib lane ID: {lane_id}")
                directory = temporary / "hamiltonians" / lane_id
                lane_summaries = [row for row in summaries if row["lane_id"] == lane_id]
                lane_candidates = [row for row in candidates if row["lane_id"] == lane_id]
                for name, title, tex, content in (
                    (
                        "summary.pdf",
                        f"HamLib82 summary: {lane_id}",
                        _hamlib_summary(lane_id, lane_summaries),
                        _hamlib_reportlab_content(
                            lane_id,
                            lane_summaries,
                            lane_candidates,
                            extended=False,
                        ),
                    ),
                    (
                        "extended_log.pdf",
                        f"HamLib82 extended log: {lane_id}",
                        _hamlib_extended(lane_id, lane_summaries, lane_candidates),
                        _hamlib_reportlab_content(
                            lane_id,
                            lane_summaries,
                            lane_candidates,
                            extended=True,
                        ),
                    ),
                ):
                    destination = directory / name
                    if selected_engine == "latex":
                        _compile_pdf(tex, destination, str(latex_path))
                    else:
                        _compile_reportlab_pdf(title, *content, destination)
                    pdfs.append(destination)
        elif family == "diff_energy":
            candidates, pooled = _validate_reactions(export_dir)
            shape = _export_shape(export_dir, family)
            for reaction_key in sorted(pooled):
                if SAFE_ID.fullmatch(reaction_key) is None:
                    raise ReportValidationError(f"unsafe reaction key: {reaction_key}")
                directory = temporary / "reactions" / reaction_key
                rows = [row for row in candidates if row["reaction_key"] == reaction_key]
                for name, title, tex, content in (
                    (
                        "summary.pdf",
                        f"Reaction-energy summary: {reaction_key}",
                        _reaction_summary(reaction_key, rows, pooled[reaction_key]),
                        _reaction_reportlab_content(
                            reaction_key,
                            rows,
                            pooled[reaction_key],
                            extended=False,
                        ),
                    ),
                    (
                        "extended_log.pdf",
                        f"Reaction-energy extended log: {reaction_key}",
                        _reaction_extended(reaction_key, rows, pooled[reaction_key]),
                        _reaction_reportlab_content(
                            reaction_key,
                            rows,
                            pooled[reaction_key],
                            extended=True,
                        ),
                    ),
                ):
                    destination = directory / name
                    if selected_engine == "latex":
                        _compile_pdf(tex, destination, str(latex_path))
                    else:
                        _compile_reportlab_pdf(title, *content, destination)
                    pdfs.append(destination)
        else:
            raise ReportValidationError(f"unsupported report family: {family}")
        expected = 2 * int(shape["instance_count"])
        if len(pdfs) != expected:
            raise ReportValidationError(f"built {len(pdfs)} PDFs, expected {expected}")
        _write_report_metadata(
            temporary,
            family,
            pdfs,
            export_dir,
            engine=selected_engine,
        )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build completion-gated PDF reports from a strict paper export."
    )
    parser.add_argument("--family", choices=("abs_energy", "diff_energy"), required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latexmk", default="latexmk")
    parser.add_argument("--engine", choices=("auto", "latex", "reportlab"), default="auto")
    args = parser.parse_args()
    generate_reports(
        args.family,
        args.export_dir,
        args.output_dir,
        latexmk=args.latexmk,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
