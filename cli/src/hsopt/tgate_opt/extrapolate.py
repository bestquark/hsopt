from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .common import REPO_ROOT


BASELINE_METADATA = REPO_ROOT / "baseline/reactions.json"
DEFAULT_FEATURES = (
    "log_active_orbitals",
    "log_total_pauli_rotation_terms",
)
DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "experiments" / "_runs" / "remote_sleepy_runs_20260506"
DEFAULT_FEMOCO_FEATURES = {
    "molecule": "femoco",
    "display_name": "FeMoco CAS(54e,54o)",
    "active_electrons": 54,
    "active_orbitals": 54,
    "n_qubits": 108,
    "target_time": 1.0,
}


@dataclass(frozen=True)
class GainRecord:
    molecule: str
    baseline_t_count: float
    optimized_t_count: float
    source: str
    metadata: Mapping[str, Any]
    evaluator: str | None = None

    @property
    def gain(self) -> float:
        return self.baseline_t_count / self.optimized_t_count

    @property
    def log_gain(self) -> float:
        return math.log(self.gain)


@dataclass(frozen=True)
class BayesianLinearFit:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    alpha: float
    beta_scale: float
    degrees_of_freedom: float
    n_observations: int
    residual_log_scale: float


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _alias_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in ("", None):
            return row[alias]
    return None


def _load_metadata(molecule: str) -> dict[str, Any]:
    if BASELINE_METADATA.exists():
        registry = json.loads(BASELINE_METADATA.read_text())
        metadata = registry["historical_metadata"]["species"].get(molecule)
        if metadata is not None:
            return {"molecule": molecule, "display_name": molecule, **metadata}
    return {"molecule": molecule, "display_name": molecule}


def _with_estimated_pauli_terms(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    if "total_pauli_rotation_terms" not in result:
        active_orbitals = _positive_float(result.get("active_orbitals"))
        if active_orbitals is not None:
            result["total_pauli_rotation_terms"] = int(round(active_orbitals**4))
    if "n_qubits" not in result:
        active_orbitals = _positive_float(result.get("active_orbitals"))
        if active_orbitals is not None:
            result["n_qubits"] = int(round(2.0 * active_orbitals))
    if "active_orbitals" not in result:
        n_qubits = _positive_float(result.get("n_qubits"))
        if n_qubits is not None:
            result["active_orbitals"] = n_qubits / 2.0
    return result


def _record_metadata(molecule: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = _load_metadata(molecule)
    if extra:
        for key, value in extra.items():
            if value not in ("", None):
                metadata[key] = value
    return _with_estimated_pauli_terms(metadata)


def _parse_json_payload(path: Path) -> Any:
    return json.loads(path.read_text())


def load_gain_table(path: Path) -> list[GainRecord]:
    """Load explicit baseline/optimized T-count records from CSV or JSON.

    Expected columns/keys include `molecule`, `baseline_t_count`, and
    `optimized_t_count`. Common aliases such as `baseline_t`, `start_t_count`,
    `best_t_count`, and `final_t_count` are accepted.
    """

    path = path.resolve()
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows: Iterable[Mapping[str, Any]] = list(csv.DictReader(handle))
    else:
        payload = _parse_json_payload(path)
        if isinstance(payload, Mapping):
            rows = payload.get("records", payload.get("data", []))
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a list of gain records")

    records: list[GainRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path} record {index} is not an object")
        molecule = str(_alias_value(row, ("molecule", "name", "key")) or "").strip()
        if not molecule:
            raise ValueError(f"{path} record {index} is missing molecule")
        baseline = _positive_float(
            _alias_value(
                row, ("baseline_t_count", "baseline_t", "start_t_count", "reference_t_count")
            )
        )
        optimized = _positive_float(
            _alias_value(row, ("optimized_t_count", "optimized_t", "best_t_count", "final_t_count"))
        )
        if baseline is None or optimized is None:
            raise ValueError(
                f"{path} record {index} must have positive baseline and optimized T counts"
            )
        metadata = _record_metadata(molecule, row)
        records.append(
            GainRecord(
                molecule=molecule,
                baseline_t_count=baseline,
                optimized_t_count=optimized,
                source=f"{path}:{index}",
                metadata=metadata,
                evaluator=str(row.get("evaluator")) if row.get("evaluator") else None,
            )
        )
    return records


def _events_from_file(path: Path) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"could not parse {path}:{line_number}: {exc}") from exc
            if isinstance(event, Mapping):
                events.append(event)
    return events


def _valid_campaign_results(events_path: Path) -> list[Mapping[str, Any]]:
    valid: list[Mapping[str, Any]] = []
    for event in _events_from_file(events_path):
        if event.get("type") != "checkpoint_evaluated":
            continue
        result = event.get("result")
        if not isinstance(result, Mapping):
            continue
        passes_accuracy = _boolish(result.get("passes_accuracy_constraint"))
        if passes_accuracy is False:
            continue
        t_count = _positive_float(result.get("t_gate_count"))
        if t_count is None:
            t_count = _positive_float(result.get("state_diagnostic_t_gate_count"))
        if t_count is None:
            t_count = _positive_float(result.get("score"))
        if t_count is None:
            continue
        item = dict(result)
        item["_t_count_for_gain"] = t_count
        item["_event_iteration"] = event.get("iteration")
        valid.append(item)
    return valid


def load_campaign_gain_records(root: Path) -> list[GainRecord]:
    """Build one gain record per campaign `events.jsonl`.

    Each record compares the first valid checkpoint in that campaign against the
    best valid checkpoint seen later in the same file. This measures campaign
    gain, not an evaluator-framework change.
    """

    root = root.resolve()
    events_files = sorted(root.rglob("events.jsonl")) if root.exists() else []
    records: list[GainRecord] = []
    for events_path in events_files:
        valid = _valid_campaign_results(events_path)
        if not valid:
            continue
        first = valid[0]
        best = min(valid, key=lambda item: float(item["_t_count_for_gain"]))
        baseline = float(first["_t_count_for_gain"])
        optimized = float(best["_t_count_for_gain"])
        if optimized <= 0.0 or baseline <= 0.0:
            continue
        molecule = str(best.get("molecule") or first.get("molecule") or events_path.parents[2].name)
        metadata = _record_metadata(molecule, first | best)
        records.append(
            GainRecord(
                molecule=molecule,
                baseline_t_count=baseline,
                optimized_t_count=optimized,
                source=str(
                    events_path.relative_to(REPO_ROOT)
                    if events_path.is_relative_to(REPO_ROOT)
                    else events_path
                ),
                metadata=metadata,
                evaluator=str(first.get("objective")) if first.get("objective") else None,
            )
        )
    return records


def collect_gain_records(
    *,
    tables: Sequence[Path] = (),
    campaign_roots: Sequence[Path] = (),
) -> list[GainRecord]:
    records: list[GainRecord] = []
    for table in tables:
        records.extend(load_gain_table(table))
    for root in campaign_roots:
        records.extend(load_campaign_gain_records(root))
    return records


def feature_value(metadata: Mapping[str, Any], name: str) -> float:
    metadata = _with_estimated_pauli_terms(metadata)
    if name.startswith("log_"):
        base = name[4:]
        value = _positive_float(metadata.get(base))
        if value is None:
            raise KeyError(f"feature {name!r} requires positive metadata field {base!r}")
        return math.log(value)
    value = _positive_float(metadata.get(name))
    if value is None:
        raise KeyError(f"feature {name!r} requires positive metadata field {name!r}")
    return value


def design_matrix(records: Sequence[GainRecord], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            [feature_value(record.metadata, feature) for feature in feature_names]
            for record in records
        ],
        dtype=np.float64,
    )


def _standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = (
        np.std(features, axis=0, ddof=1) if features.shape[0] > 1 else np.ones(features.shape[1])
    )
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return (features - mean) / scale, mean, scale


def fit_bayesian_log_gain(
    records: Sequence[GainRecord],
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    *,
    prior_scale: float = 2.0,
    prior_alpha: float = 2.0,
    prior_beta: float = 0.5,
) -> BayesianLinearFit:
    if not records:
        raise ValueError("at least one gain record is required")
    if prior_scale <= 0.0 or prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("prior_scale, prior_alpha, and prior_beta must be positive")

    raw_features = design_matrix(records, feature_names)
    standardized, mean, scale = _standardize_features(raw_features)
    x = np.column_stack([np.ones(len(records), dtype=np.float64), standardized])
    y = np.asarray([record.log_gain for record in records], dtype=np.float64)

    beta0 = np.zeros(x.shape[1], dtype=np.float64)
    prior_cov_inv = np.eye(x.shape[1], dtype=np.float64) / float(prior_scale**2)
    posterior_precision = prior_cov_inv + x.T @ x
    posterior_cov = np.linalg.inv(posterior_precision)
    posterior_beta = posterior_cov @ (prior_cov_inv @ beta0 + x.T @ y)
    alpha = float(prior_alpha + 0.5 * len(records))
    beta_scale = float(
        prior_beta
        + 0.5
        * (
            y.T @ y
            + beta0.T @ prior_cov_inv @ beta0
            - posterior_beta.T @ posterior_precision @ posterior_beta
        )
    )
    beta_scale = max(beta_scale, np.finfo(float).tiny)
    residual_log_scale = math.sqrt(beta_scale / alpha)
    return BayesianLinearFit(
        feature_names=tuple(feature_names),
        feature_mean=mean,
        feature_scale=scale,
        beta=posterior_beta,
        covariance=posterior_cov,
        alpha=alpha,
        beta_scale=beta_scale,
        degrees_of_freedom=2.0 * alpha,
        n_observations=len(records),
        residual_log_scale=residual_log_scale,
    )


def _target_design(fit: BayesianLinearFit, target_metadata: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(
        [feature_value(target_metadata, feature) for feature in fit.feature_names], dtype=np.float64
    )
    standardized = (raw - fit.feature_mean) / fit.feature_scale
    return np.concatenate([[1.0], standardized])


def predict_log_gain_interval(
    fit: BayesianLinearFit,
    target_metadata: Mapping[str, Any],
    *,
    interval: float = 0.9,
    include_observation_noise: bool = True,
) -> dict[str, float]:
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must be between 0 and 1")
    x = _target_design(fit, target_metadata)
    loc = float(x @ fit.beta)
    variance_multiplier = 1.0 if include_observation_noise else 0.0
    scale = math.sqrt(
        fit.beta_scale / fit.alpha * (variance_multiplier + float(x.T @ fit.covariance @ x))
    )
    tail = (1.0 - interval) / 2.0
    return {
        "lower": float(student_t.ppf(tail, fit.degrees_of_freedom, loc=loc, scale=scale)),
        "median": loc,
        "upper": float(student_t.ppf(1.0 - tail, fit.degrees_of_freedom, loc=loc, scale=scale)),
        "scale": scale,
        "df": fit.degrees_of_freedom,
    }


def target_feature_distance(
    fit: BayesianLinearFit, target_metadata: Mapping[str, Any]
) -> dict[str, float]:
    x = _target_design(fit, target_metadata)[1:]
    return {
        "max_abs_standardized_feature": float(np.max(np.abs(x))) if x.size else 0.0,
        "euclidean_standardized_feature": float(np.linalg.norm(x)),
    }


def summarize_training_records(records: Sequence[GainRecord]) -> dict[str, Any]:
    gains = np.asarray([record.gain for record in records], dtype=np.float64)
    return {
        "count": len(records),
        "molecules": sorted({record.molecule for record in records}),
        "gain_factor_min": float(np.min(gains)),
        "gain_factor_median": float(np.median(gains)),
        "gain_factor_max": float(np.max(gains)),
    }


def extrapolate_gain(
    records: Sequence[GainRecord],
    target_metadata: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    interval: float = 0.9,
    target_baseline_t_count: float | None = None,
    prior_scale: float = 2.0,
    prior_alpha: float = 2.0,
    prior_beta: float = 0.5,
) -> dict[str, Any]:
    fit = fit_bayesian_log_gain(
        records,
        feature_names,
        prior_scale=prior_scale,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    target = _with_estimated_pauli_terms(target_metadata)
    log_interval = predict_log_gain_interval(fit, target, interval=interval)
    gain = {key: math.exp(log_interval[key]) for key in ("lower", "median", "upper")}
    actionable_gain = {key: max(1.0, value) for key, value in gain.items()}
    distance = target_feature_distance(fit, target)
    result: dict[str, Any] = {
        "model": "bayesian_linear_regression_log_gain",
        "feature_names": list(feature_names),
        "interval_probability": interval,
        "target": target,
        "training": summarize_training_records(records),
        "log_gain": log_interval,
        "gain_factor": gain,
        "actionable_gain_factor": actionable_gain,
        "empirical_gain_factor_range": {
            "worst_observed": min(record.gain for record in records),
            "best_observed": max(record.gain for record in records),
        },
        "target_feature_distance": distance,
        "records": [
            {
                "molecule": record.molecule,
                "baseline_t_count": record.baseline_t_count,
                "optimized_t_count": record.optimized_t_count,
                "gain_factor": record.gain,
                "source": record.source,
                "evaluator": record.evaluator,
            }
            for record in records
        ],
    }
    if distance["max_abs_standardized_feature"] > 3.0:
        result["extrapolation_warning"] = (
            "Target features are outside the small-molecule training range; "
            "treat the posterior interval as a sensitivity analysis, not a calibrated guarantee."
        )
    if target_baseline_t_count is not None:
        baseline = _positive_float(target_baseline_t_count)
        if baseline is None:
            raise ValueError("target_baseline_t_count must be positive")
        result["target_baseline_t_count"] = baseline
        result["target_optimized_t_count"] = {
            "lower_best_case": baseline / actionable_gain["upper"],
            "median": baseline / actionable_gain["median"],
            "upper_worst_case": baseline / actionable_gain["lower"],
        }
        result["target_t_gate_savings"] = {
            "lower_worst_case": baseline - baseline / actionable_gain["lower"],
            "median": baseline - baseline / actionable_gain["median"],
            "upper_best_case": baseline - baseline / actionable_gain["upper"],
        }
    return result


def _load_target_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _target_from_args(args: argparse.Namespace) -> dict[str, Any]:
    target = dict(DEFAULT_FEMOCO_FEATURES)
    if args.target_json:
        target.update(_load_target_json(Path(args.target_json)))
    overrides = {
        "active_electrons": args.active_electrons,
        "active_orbitals": args.active_orbitals,
        "n_qubits": args.n_qubits,
        "total_pauli_rotation_terms": args.total_pauli_rotation_terms,
        "centered_hamiltonian_norm": args.centered_hamiltonian_norm,
    }
    for key, value in overrides.items():
        if value is not None:
            target[key] = value
    return _with_estimated_pauli_terms(target)


def _format_scientific(value: float) -> str:
    return f"{value:.6e}"


def _svg_polyline(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _nice_log_ticks(min_value: float, max_value: float) -> list[float]:
    lower = math.floor(math.log10(min_value))
    upper = math.ceil(math.log10(max_value))
    ticks: list[float] = []
    for exponent in range(lower, upper + 1):
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * 10.0**exponent
            if min_value <= value <= max_value:
                ticks.append(value)
    return ticks


def _format_gain_tick(value: float) -> str:
    if 0.1 <= value < 100.0:
        return f"{value:g}x"
    return f"{value:.0e}x"


def write_uncertainty_svg(
    path: Path,
    records: Sequence[GainRecord],
    target_metadata: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    interval: float = 0.9,
    prior_scale: float = 2.0,
    prior_alpha: float = 2.0,
    prior_beta: float = 0.5,
) -> Path:
    fit = fit_bayesian_log_gain(
        records,
        feature_names,
        prior_scale=prior_scale,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    target = _with_estimated_pauli_terms(target_metadata)
    target_orbitals = _positive_float(target.get("active_orbitals")) or 54.0
    target_terms = _positive_float(target.get("total_pauli_rotation_terms")) or target_orbitals**4

    observed_x = [feature_value(record.metadata, "active_orbitals") for record in records]
    observed_y = [record.gain for record in records]
    x_min = max(1.0, min(observed_x) * 0.8)
    x_max = max(target_orbitals * 1.08, max(observed_x) * 1.15)
    xs = np.geomspace(x_min, x_max, 120)
    path_rows: list[dict[str, float]] = []
    for active_orbitals in xs:
        pauli_terms = target_terms * (float(active_orbitals) / target_orbitals) ** 4
        metadata = dict(target)
        metadata["active_orbitals"] = float(active_orbitals)
        metadata["n_qubits"] = 2.0 * float(active_orbitals)
        metadata["total_pauli_rotation_terms"] = max(1.0, pauli_terms)
        predicted = predict_log_gain_interval(fit, metadata, interval=interval)
        path_rows.append(
            {
                "x": float(active_orbitals),
                "lower": math.exp(predicted["lower"]),
                "median": math.exp(predicted["median"]),
                "upper": math.exp(predicted["upper"]),
            }
        )

    target_prediction = predict_log_gain_interval(fit, target, interval=interval)
    target_gain = math.exp(target_prediction["median"])
    y_values = observed_y + [1.0, target_gain]
    for row in path_rows:
        y_values.extend([row["lower"], row["median"], row["upper"]])
    y_min = max(min(y_values) * 0.75, 1e-6)
    y_max = max(y_values) * 1.35

    width = 960
    height = 560
    left = 92
    right = 34
    top = 48
    bottom = 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)
    log_y_min = math.log10(y_min)
    log_y_max = math.log10(y_max)

    def px(value: float) -> float:
        return left + (math.log10(value) - log_x_min) / (log_x_max - log_x_min) * plot_width

    def py(value: float) -> float:
        return top + (log_y_max - math.log10(value)) / (log_y_max - log_y_min) * plot_height

    band_points = [(px(row["x"]), py(row["upper"])) for row in path_rows]
    band_points.extend((px(row["x"]), py(row["lower"])) for row in reversed(path_rows))
    median_points = [(px(row["x"]), py(row["median"])) for row in path_rows]

    x_ticks = sorted(
        {
            tick
            for tick in (2, 3, 4, 5, 10, 20, 50, round(target_orbitals))
            if x_min <= float(tick) <= x_max
        }
    )
    y_ticks = _nice_log_ticks(y_min, y_max)
    interval_percent = int(round(interval * 100.0))
    molecule_label = html.escape(str(target.get("display_name", target.get("molecule", "FeMoco"))))

    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#202124}",
        ".small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:700}",
        ".grid{stroke:#d9dde3;stroke-width:1}.axis{stroke:#2f3338;stroke-width:1.4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="28">FeMoco T-gate gain extrapolation uncertainty</text>',
        f'<text class="small" x="{left}" y="46">Bayesian log-gain regression; shaded band is {interval_percent}% posterior prediction interval</text>',
    ]

    for tick in x_ticks:
        x = px(float(tick))
        svg.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>'
        )
        svg.append(
            f'<text class="small" text-anchor="middle" x="{x:.2f}" y="{top + plot_height + 22}">{tick:g}</text>'
        )
    for tick in y_ticks:
        y = py(tick)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text class="small" text-anchor="end" x="{left - 10}" y="{y + 4:.2f}">{_format_gain_tick(tick)}</text>'
        )

    one_y = py(1.0)
    svg.append(
        f'<line x1="{left}" y1="{one_y:.2f}" x2="{left + plot_width}" y2="{one_y:.2f}" '
        'stroke="#68707a" stroke-width="1.4" stroke-dasharray="5 5"/>'
    )
    svg.append(
        f'<text class="small" x="{left + plot_width - 92}" y="{one_y - 8:.2f}" fill="#68707a">no improvement</text>'
    )

    svg.append(f'<polygon points="{_svg_polyline(band_points)}" fill="#8ec5ff" opacity="0.38"/>')
    svg.append(
        f'<polyline points="{_svg_polyline(median_points)}" fill="none" stroke="#1358bf" stroke-width="3"/>'
    )

    for record, x_value, y_value in zip(records, observed_x, observed_y):
        cx = px(x_value)
        cy = py(y_value)
        title = html.escape(
            f"{record.molecule}: {record.gain:.3g}x gain; "
            f"{record.baseline_t_count:.0f} -> {record.optimized_t_count:.0f} T gates"
        )
        svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.2" fill="#202124"><title>{title}</title></circle>'
        )

    target_x = px(target_orbitals)
    target_y = py(target_gain)
    svg.append(
        f'<line x1="{target_x:.2f}" y1="{top}" x2="{target_x:.2f}" y2="{top + plot_height}" '
        'stroke="#c62828" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    svg.append(
        f'<circle cx="{target_x:.2f}" cy="{target_y:.2f}" r="6" fill="#c62828">'
        f"<title>{molecule_label}: median raw gain {target_gain:.3g}x</title></circle>"
    )
    svg.append(
        f'<text class="small" text-anchor="end" x="{target_x - 8:.2f}" y="{top + 18}">{molecule_label}</text>'
    )

    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text class="label" text-anchor="middle" x="{left + plot_width / 2:.2f}" y="{height - 24}">active spatial orbitals (log scale)</text>',
            f'<text class="label" transform="translate(24 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">gain factor: baseline / optimized T count (log scale)</text>',
            f'<rect x="{left + 18}" y="{top + 16}" width="198" height="70" fill="#fff" stroke="#d1d5db"/>',
            f'<rect x="{left + 32}" y="{top + 31}" width="28" height="11" fill="#8ec5ff" opacity="0.38"/>',
            f'<text class="small" x="{left + 68}" y="{top + 42}">{interval_percent}% prediction interval</text>',
            f'<line x1="{left + 32}" y1="{top + 58}" x2="{left + 60}" y2="{top + 58}" stroke="#1358bf" stroke-width="3"/>',
            f'<text class="small" x="{left + 68}" y="{top + 62}">posterior median</text>',
            f'<circle cx="{left + 46}" cy="{top + 76}" r="4.2" fill="#202124"/>',
            f'<text class="small" x="{left + 68}" y="{top + 80}">observed campaign gain</text>',
            "</svg>",
        ]
    )

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")
    return path


def _print_text_summary(result: Mapping[str, Any]) -> None:
    interval = 100.0 * float(result["interval_probability"])
    gain = result["gain_factor"]
    actionable_gain = result["actionable_gain_factor"]
    training = result["training"]
    print("FeMoco T-gate gain extrapolation")
    print(f"model: {result['model']}")
    print(f"features: {', '.join(result['feature_names'])}")
    print(f"training records: {training['count']} across {', '.join(training['molecules'])}")
    print(
        "observed gain factor range: "
        f"{training['gain_factor_min']:.3g}x to {training['gain_factor_max']:.3g}x "
        f"(median {training['gain_factor_median']:.3g}x)"
    )
    print(
        f"posterior gain factor ({interval:.0f}% interval): "
        f"{gain['median']:.3g}x "
        f"[{gain['lower']:.3g}x, {gain['upper']:.3g}x]"
    )
    if gain != actionable_gain:
        print(
            "actionable gain factor with no-change fallback: "
            f"{actionable_gain['median']:.3g}x "
            f"[{actionable_gain['lower']:.3g}x, {actionable_gain['upper']:.3g}x]"
        )
    if "target_baseline_t_count" in result:
        optimized = result["target_optimized_t_count"]
        savings = result["target_t_gate_savings"]
        print(f"target baseline T count: {_format_scientific(result['target_baseline_t_count'])}")
        print(
            f"target optimized T count ({interval:.0f}% interval): "
            f"{_format_scientific(optimized['median'])} "
            f"[{_format_scientific(optimized['lower_best_case'])}, "
            f"{_format_scientific(optimized['upper_worst_case'])}]"
        )
        print(
            f"target T-gate savings ({interval:.0f}% interval): "
            f"{_format_scientific(savings['median'])} "
            f"[{_format_scientific(savings['lower_worst_case'])}, "
            f"{_format_scientific(savings['upper_best_case'])}]"
        )
    distance = result["target_feature_distance"]
    print(
        "extrapolation distance: "
        f"max |z|={distance['max_abs_standardized_feature']:.2f}, "
        f"euclidean={distance['euclidean_standardized_feature']:.2f}"
    )
    if result.get("extrapolation_warning"):
        print(f"warning: {result['extrapolation_warning']}")


def build_extrapolate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--gain-table",
        action="append",
        default=[],
        help="CSV/JSON file with molecule, baseline_t_count, and optimized_t_count records.",
    )
    parser.add_argument(
        "--campaign-root",
        action="append",
        default=[],
        help="Autoresearch campaign root containing events.jsonl files.",
    )
    parser.add_argument(
        "--use-default-campaign-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Include {DEFAULT_CAMPAIGN_ROOT.relative_to(REPO_ROOT)} when it exists.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        dest="features",
        help=f"Feature to use in the log-gain model. Defaults to {', '.join(DEFAULT_FEATURES)}.",
    )
    parser.add_argument(
        "--interval", type=float, default=0.9, help="Posterior interval probability."
    )
    parser.add_argument(
        "--target-baseline-t-count",
        type=float,
        help="Optional FeMoco baseline T count to convert gain factors into absolute optimized counts.",
    )
    parser.add_argument("--target-json", help="JSON object with target molecule feature overrides.")
    parser.add_argument("--active-electrons", type=int, help="Target active electron count.")
    parser.add_argument("--active-orbitals", type=int, help="Target spatial active orbital count.")
    parser.add_argument("--n-qubits", type=int, help="Target spin-orbital qubit count.")
    parser.add_argument(
        "--total-pauli-rotation-terms", type=int, help="Target Pauli rotation term count."
    )
    parser.add_argument(
        "--centered-hamiltonian-norm", type=float, help="Target centered Hamiltonian norm."
    )
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--prior-alpha", type=float, default=2.0)
    parser.add_argument("--prior-beta", type=float, default=0.5)
    parser.add_argument(
        "--plot",
        help="Optional SVG path for a posterior uncertainty plot over active-orbital scale.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def run_extrapolate_cli(args: argparse.Namespace) -> int:
    campaign_roots = [Path(path) for path in args.campaign_root]
    if args.use_default_campaign_root and DEFAULT_CAMPAIGN_ROOT.exists():
        campaign_roots.insert(0, DEFAULT_CAMPAIGN_ROOT)
    records = collect_gain_records(
        tables=[Path(path) for path in args.gain_table],
        campaign_roots=campaign_roots,
    )
    if not records:
        raise SystemExit(
            "no gain records found; pass --gain-table or --campaign-root with valid campaign events"
        )
    target = _target_from_args(args)
    result = extrapolate_gain(
        records,
        target,
        feature_names=tuple(args.features or DEFAULT_FEATURES),
        interval=args.interval,
        target_baseline_t_count=args.target_baseline_t_count,
        prior_scale=args.prior_scale,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
    )
    if args.plot:
        plot_path = write_uncertainty_svg(
            Path(args.plot),
            records,
            target,
            feature_names=tuple(args.features or DEFAULT_FEATURES),
            interval=args.interval,
            prior_scale=args.prior_scale,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
        )
        result["plot_path"] = str(plot_path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text_summary(result)
        if args.plot:
            print(f"plot: {result['plot_path']}")
    return 0
