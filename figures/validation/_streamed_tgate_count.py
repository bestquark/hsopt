"""Bounded-memory literal CUDA-Q counts for the canonical Pauli PF2 schedule.

Only scalar rotation keys and tiny single-qubit compiler outputs are retained.
This module never constructs a molecular unitary, statevector or energy. The
caller must first verify that both frozen policies use canonical lexicographic
PF2 and no Clifford-frame or fermionic grouping extensions.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from _hamlib_scaling import iter_terms
from hsopt.evaluation import compiled_unitary as ev


COMPILER_CACHE_MAXSIZE = 65536


def streamed_cudaq_pair(
    text,
    n_qubits,
    center_label,
    baseline,
    policy,
    target_time,
    workers=2,
    chunk_size=2048,
    *,
    term_stride=1,
    term_offset=0,
    term_start=0,
    term_stop=None,
):
    """Count actual synthesized gates with bounded chunks and an LRU cache.

    Identical (rounded angle, tolerance) keys share compilation within a chunk.
    The bounded cache may evict old keys, so ``unique_compilation_count`` counts
    cache misses (including recompilation after eviction), not globally unique
    rotations. No cache proportional to the complete Hamiltonian is created.

    A nondefault ``term_stride`` selects global nonidentity-term indices modulo
    that stride. The caller must keep the FULL Hamiltonian's normalization,
    allocation, policy instances and center label fixed, then sum every offset
    exactly once. Partial results explicitly identify their incomplete scope.
    A contiguous ``[term_start, term_stop)`` range is also supported, separately
    from modulo selection, so adjacent equal compiler keys stay together.
    """
    started = time.perf_counter()
    for value, name in ((n_qubits, "n_qubits"), (workers, "workers"), (chunk_size, "chunk_size")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(term_stride) is not int or term_stride < 1:
        raise ValueError("term_stride must be a positive integer")
    if type(term_offset) is not int or not 0 <= term_offset < term_stride:
        raise ValueError("require 0 <= term_offset < term_stride")
    if type(term_start) is not int or term_start < 0:
        raise ValueError("term_start must be a nonnegative integer")
    if term_stop is not None and (type(term_stop) is not int or term_stop <= term_start):
        raise ValueError("term_stop must be an integer greater than term_start")
    ranged = term_start != 0 or term_stop is not None
    if ranged and term_stride != 1:
        raise ValueError("choose contiguous ranges or modulo partitions, not both")
    target_time = float(target_time)
    if not math.isfinite(target_time) or target_time <= 0:
        raise ValueError("target_time must be finite and positive")
    if (
        not isinstance(center_label, str)
        or len(center_label) != n_qubits
        or set(center_label) - set("IXYZ")
        or set(center_label) == {"I"}
    ):
        raise ValueError("center_label must be a nonidentity Pauli label")
    formulas = (baseline, policy)
    repetitions = [formula.repetitions() for formula in formulas]
    if any(type(value) is not int or value < 1 for value in repetitions):
        raise ValueError("ProductFormula.repetitions() must return a positive integer")
    step_times = [target_time / steps for steps in repetitions]

    # Bypass the production compiler's unbounded cache without modifying it.
    # Test doubles and other noncached callables are supported as well.
    underlying = getattr(ev._compile_rz, "__wrapped__", ev._compile_rz)
    compiler = lru_cache(maxsize=COMPILER_CACHE_MAXSIZE)(underlying)
    ev._cudaq_synth()  # Initialize native bindings before entering worker threads.
    totals = [0, 0]
    minima, maxima = [1.0, 1.0], [0.0, 0.0]
    pending = []
    terms_seen = center_occurrences = 0
    selected_terms = selected_center_occurrences = 0
    max_label = None
    synthesis_seconds = 0.0
    compilation_count = 0

    def flush(pool):
        nonlocal synthesis_seconds, compilation_count
        if not pending:
            return
        keys = list(dict.fromkeys(item[2] for item in pending))
        synthesis_started = time.perf_counter()
        if pool is None:
            values = [compiler(*key) for key in keys]
        else:
            values = list(pool.map(lambda key: compiler(*key), keys))
        synthesis_seconds += time.perf_counter() - synthesis_started
        counts = {}
        for key, result in zip(keys, values, strict=True):
            value = result["t_count"]
            if type(value) is not int or value < 0:
                raise ValueError("CUDA-Q returned an invalid integer T count")
            counts[key] = value
        for index, multiplicity, key in pending:
            totals[index] += multiplicity * counts[key]
        compilation_count = compiler.cache_info().misses
        pending.clear()

    pool = (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hsopt-count-synthesis")
        if workers > 1
        else None
    )
    try:
        for term in iter_terms(text, n_qubits):
            # Match elote.transformed_terms for the required empty frame.
            term = {**term, "original_label": term["label"], "frame_sign": 1}
            terms_seen += 1
            label = term["label"]
            max_label = label if max_label is None or label > max_label else max_label
            multiplicity = 1 if label == center_label else 2
            center_occurrences += label == center_label
            if (terms_seen - 1) % term_stride != term_offset:
                continue
            if terms_seen - 1 < term_start or (
                term_stop is not None and terms_seen - 1 >= term_stop
            ):
                continue
            selected_terms += 1
            selected_center_occurrences += label == center_label
            for index, (formula, step_time) in enumerate(zip(formulas, step_times, strict=True)):
                # Match canonical build_step's floating-point operation order:
                # (.5 * coefficient) * (t/r), not (coefficient * t / r) / 2.
                angle = (
                    term["coefficient"] * step_time
                    if multiplicity == 1
                    else 0.5 * term["coefficient"] * step_time
                )
                if not math.isfinite(angle):
                    raise ValueError("nonfinite scheduled rotation angle")
                epsilon = float(formula.synthesis_epsilon(dict(term), angle))
                if not math.isfinite(epsilon) or not 0 < epsilon < 1:
                    raise ValueError(
                        "ProductFormula.synthesis_epsilon() must return a value in (0, 1)"
                    )
                key = (float(round(-2.0 * angle, 15)), epsilon)
                minima[index] = min(minima[index], epsilon)
                maxima[index] = max(maxima[index], epsilon)
                pending.append((index, multiplicity, key))
            if selected_terms % chunk_size == 0:
                flush(pool)
        flush(pool)
        if not terms_seen or center_occurrences != 1 or max_label != center_label:
            raise ValueError("center_label must occur once and be the lexicographic maximum")
        if not selected_terms:
            raise ValueError("term partition contains no nonidentity terms")
        if term_stop is not None and term_stop > terms_seen:
            raise ValueError("term_stop exceeds the global nonidentity term count")
        result = {
            "baseline_t_count": totals[0] * repetitions[0],
            "policy_t_count": totals[1] * repetitions[1],
            "baseline_repetitions": repetitions[0],
            "policy_repetitions": repetitions[1],
            "baseline_epsilon_min": minima[0],
            "baseline_epsilon_max": maxima[0],
            "policy_epsilon_min": minima[1],
            "policy_epsilon_max": maxima[1],
            "rotations_per_step": 2 * terms_seen - 1,
            "nonidentity_terms": terms_seen,
            "count_method": "cudaq",
            "exact_compiled_count": True,
            "analytic_estimate": False,
            "energy_evaluated": False,
            "accuracy_certified": False,
            "count_scope": "one_complete_evolution_excluding_QPE_and_state_preparation",
            "unique_compilation_count": compilation_count,
            "compilation_count_definition": "cache_misses_including_recompilation_after_eviction",
            "compiler_cache_maxsize": COMPILER_CACHE_MAXSIZE,
            "compiler_cache_peak_entries": compiler.cache_info().currsize,
            "worker_threads": workers,
            "chunk_size": chunk_size,
            "synthesis_seconds": synthesis_seconds,
            "synthesis_options": dict(ev.SYNTHESIS_OPTIONS),
            "wall_seconds": time.perf_counter() - started,
        }
        if term_stride != 1 or ranged:
            result.update(
                partial_count=True,
                count_scope="partition_of_one_complete_evolution_excluding_QPE_and_state_preparation",
                selected_terms=selected_terms,
                selected_center_occurrences=selected_center_occurrences,
                selected_rotations_per_step=2 * selected_terms - selected_center_occurrences,
                term_partition={
                    "scheme": "global_nonidentity_index_mod_stride",
                    "stride": term_stride,
                    "offset": term_offset,
                }
                if not ranged
                else {
                    "scheme": "global_nonidentity_index_range",
                    "start": term_start,
                    "stop": term_stop if term_stop is not None else terms_seen,
                },
            )
        return result
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        compiler.cache_clear()
