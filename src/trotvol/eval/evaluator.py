# evaluator.py
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Any, List, Sequence
from dataclasses import dataclass

from .metrics import Metric
from .runner import TrialRunner


@dataclass
class EvalConfig:
    trials: int = 10
    n: int | None = None
    m: int | None = None
    t: float | None = None
    seeds: Sequence[int] | None = None

    # fallback distributions if n/m/t not given
    def sample_problem_params(self, rng: np.random.Generator):
        n = self.n
        m = self.m
        t = self.t
        if n is None:
            # Beta(2,5) scaled [10,200]
            n = int(10 + (200 - 10) * rng.beta(2, 5))
        if m is None:
            m = int(2 + (5 - 2) * rng.beta(2, 5))
        if t is None:
            t = float(0.01 + (1.0 - 0.01) * rng.beta(2, 5))
        return n, m, t

    def make_seeds(self):
        if self.seeds is not None:
            return list(self.seeds)
        return [1337 + i for i in range(self.trials)]


@dataclass
class CombinerWeights:
    """Weights for each metric scalar. All should be >=0.
    We'll subtract them because lower metric.scalar is 'better'."""

    w_err: float = 0.5
    w_mem: float = 0.5
    w_time: float = 0.1
    w_normdev: float = 0.0  # optionally penalize non-unitarity


@dataclass
class EvaluationSummary:
    combined_score: float
    per_metric: Dict[str, Dict[str, float]]
    trials: int


class Evaluator:
    """
    High-level evaluator that:
    - runs multiple trials of a candidate program via TrialRunner
    - aggregates arbitrary Metric objects
    - produces a combined score
    """

    def __init__(
        self,
        runner: TrialRunner,
        metrics: List[Metric],
        combiner: CombinerWeights,
        mem_cap_mb: float,
    ):
        self.runner = runner
        self.metrics = metrics
        self.combiner = combiner
        self.mem_cap_mb = mem_cap_mb

    def evaluate(self, program_path: str, cfg: EvalConfig) -> Dict[str, Any]:
        # basic input validation
        if (not isinstance(program_path, str)) or (not os.path.exists(program_path)):
            # hard fail -> huge penalties
            return {
                "combined_score": -1e9,
                "error": "program_path_invalid",
                "trials": 0,
            }

        rng = np.random.default_rng(12345)
        n, m, t = cfg.sample_problem_params(rng)
        seeds = cfg.make_seeds()

        # collect per-metric raw per-trial scalars
        metric_vals: Dict[str, List[float]] = {mt.name: [] for mt in self.metrics}

        trial_payloads: List[Dict[str, Any]] = []
        for sd in seeds:
            payload = self.runner.run_single_trial(program_path, n, m, t, sd)
            trial_payloads.append(payload)

            for mt in self.metrics:
                try:
                    v = mt.accumulate_single(payload)
                except Exception:
                    v = 1e9
                metric_vals[mt.name].append(v)

        # aggregate
        per_metric_summary: Dict[str, Dict[str, float]] = {}
        metric_scalars: Dict[str, float] = {}
        for mt in self.metrics:
            agg = mt.aggregate(metric_vals[mt.name])
            per_metric_summary[mt.name] = dict(agg.summary)
            metric_scalars[mt.name] = float(agg.scalar)

        # build combined score.
        # Convention: higher is better for final "combined_score".
        # We'll negate the costs.
        w = self.combiner
        score = (
            -w.w_err * metric_scalars.get("err", 0.0)
            - w.w_mem * metric_scalars.get("mem", 0.0)
            - w.w_time * metric_scalars.get("time_ms", 0.0)
            - w.w_normdev * metric_scalars.get("norm_dev", 0.0)
        )

        # also expose human-readable aliases
        out = {
            "combined_score": float(score),
            "trials": len(seeds),
        }
        # flatten summaries into top-level convenience keys
        for k, v in per_metric_summary.items():
            for sk, sv in v.items():
                out[f"{k}_{sk}"] = float(sv)

        # keep raw per-metric dict in case caller wants structured info
        out["per_metric"] = per_metric_summary
        return out
