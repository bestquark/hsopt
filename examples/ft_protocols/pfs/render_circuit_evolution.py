from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np

import evaluate
from model_registry import get_spec


EPSILON_PALETTE = [
    "#0f4c81",
    "#2f76a6",
    "#5da3c6",
    "#91c7da",
    "#f2b35d",
    "#dd7f43",
    "#c95a38",
    "#8c3d2e",
]


class CaptureDone(RuntimeError):
    pass


class IdentityAction:
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.complex128)

    def __matmul__(self, state: np.ndarray) -> np.ndarray:
        return self(state)


@dataclass(frozen=True)
class SnapshotRecord:
    iteration: int
    score: float
    status: str
    passes_constraint: bool
    fidelity: float
    fidelity_floor: float
    t_gate_count: int
    description: str
    snapshot_dir: Path


def snapshot_dir(molecule: str, iteration: int) -> Path:
    return get_spec(molecule).snapshots_dir / f"iter_{iteration:04d}"


def results_rows(molecule: str) -> List[SnapshotRecord]:
    spec = get_spec(molecule)
    if not spec.results_path.exists():
        return []
    rows: List[SnapshotRecord] = []
    with spec.results_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                rows.append(
                    SnapshotRecord(
                        iteration=int(row["iteration"]),
                        score=float(row["score"]),
                        status=str(row["status"]),
                        passes_constraint=str(
                            row.get("passes_accuracy_constraint", row.get("passes_fidelity_constraint", ""))
                        ).strip().lower()
                        in {"true", "1", "yes"},
                        fidelity=float(row["fidelity"]),
                        fidelity_floor=float(row["fidelity_floor"]),
                        t_gate_count=int(float(row["t_gate_count"])),
                        description=str(row.get("description", "")),
                        snapshot_dir=Path(str(row["snapshot_dir"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda item: item.iteration)
    return rows


def baseline_t_count(molecule: str) -> int | None:
    rows = results_rows(molecule)
    if not rows:
        return None
    return rows[0].t_gate_count


def best_t_count_through(molecule: str, iteration: int) -> int | None:
    best: int | None = None
    for row in results_rows(molecule):
        if row.iteration > iteration or not row.passes_constraint:
            continue
        if best is None or row.t_gate_count < best:
            best = row.t_gate_count
    return best


def compilation_key(label: str, alpha: float, epsilon: float) -> Tuple[str, str, str]:
    return (str(label), f"{float(alpha):.15e}", f"{float(epsilon):.15e}")


def unique_compilation_lookup(result_payload: Dict[str, object]) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for item in result_payload.get("unique_compilations", []):
        try:
            key = compilation_key(
                str(item["label"]),
                float(item["alpha"]),
                float(item["epsilon"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        lookup[key] = item
    return lookup


def first_step_operations(molecule: str, script_path: Path, result_payload: Dict[str, object]) -> Dict[str, object]:
    module = evaluate.load_method_module(script_path)
    if not hasattr(module, "choose_synthesis_epsilon"):
        raise RuntimeError(f"{script_path} is missing choose_synthesis_epsilon; cannot reconstruct circuit colors")

    choose_fn = module.choose_synthesis_epsilon
    config = module.get_method_config()
    problem = evaluate.build_problem(molecule)
    compile_lookup = unique_compilation_lookup(result_payload)
    identity_action = IdentityAction()
    operations: List[Dict[str, object]] = []

    def compile_stub(label: str, alpha: float, epsilon: float) -> Dict[str, object]:
        key = compilation_key(label, round(float(alpha), 15), float(epsilon))
        cached = compile_lookup.get(key, {})
        gate_count = int(cached.get("gate_count", 0))
        t_gate_count = int(cached.get("t_gate_count", 0))
        operator_error = float(cached.get("operator_error", 0.0))
        return {
            "cache_key": "|".join(key),
            "label": str(label),
            "alpha": float(round(float(alpha), 15)),
            "theta": float(round(-2.0 * float(alpha), 15)),
            "epsilon": float(epsilon),
            "gate_string": str(cached.get("gate_string", "")),
            "gate_count": gate_count,
            "t_gate_count": t_gate_count,
            "operator_error": operator_error,
            "unitary": identity_action,
            "state_action": identity_action,
        }

    def choose_wrapper(term: Dict[str, object], alpha: float, step_index: int, cfg: object) -> float:
        if step_index > 0:
            raise CaptureDone()
        epsilon = float(choose_fn(term, alpha, step_index, cfg))
        key = compilation_key(str(term["label"]), round(float(alpha), 15), epsilon)
        cached = compile_lookup.get(key, {})
        support_qubits = [idx for idx, pauli in enumerate(str(term["label"])) if pauli != "I"]
        operations.append(
            {
                "order": len(operations) + 1,
                "step_index": int(step_index),
                "label": str(term["label"]),
                "coefficient": float(term["coefficient"]),
                "alpha": float(round(float(alpha), 15)),
                "epsilon": epsilon,
                "support_qubits": support_qubits,
                "support_min": min(support_qubits),
                "support_max": max(support_qubits),
                "support_size": len(support_qubits),
                "t_gate_count": int(cached.get("t_gate_count", 0)),
                "gate_count": int(cached.get("gate_count", 0)),
                "operator_error": float(cached.get("operator_error", 0.0)),
            }
        )
        return epsilon

    module.choose_synthesis_epsilon = choose_wrapper
    problem["compile_pauli_rotation"] = compile_stub
    try:
        module.run_product_formula(problem, exact_angles=False)
    except CaptureDone:
        pass
    finally:
        module.choose_synthesis_epsilon = choose_fn

    if not operations:
        raise RuntimeError(f"no first-step operations were captured for {script_path}")

    return {
        "operations": operations,
        "n_qubits": int(problem["n_qubits"]),
        "target_time": float(problem["target_time"]),
        "display_name": str(problem["display_name"]),
        "method_config": result_payload.get("method_config", {}),
    }


def color_map(epsilon_values: List[float]) -> Dict[float, str]:
    unique = sorted(set(float(value) for value in epsilon_values))
    return {value: EPSILON_PALETTE[idx % len(EPSILON_PALETTE)] for idx, value in enumerate(unique)}


def write_operation_table(operations: List[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "order",
                "label",
                "coefficient",
                "alpha",
                "epsilon",
                "t_gate_count",
                "gate_count",
                "operator_error",
                "support_qubits",
            ]
        )
        for item in operations:
            writer.writerow(
                [
                    item["order"],
                    item["label"],
                    item["coefficient"],
                    item["alpha"],
                    item["epsilon"],
                    item["t_gate_count"],
                    item["gate_count"],
                    item["operator_error"],
                    ",".join(str(value) for value in item["support_qubits"]),
                ]
            )


def render_snapshot_figure(
    molecule: str,
    iteration: int,
    result_payload: Dict[str, object],
    capture_payload: Dict[str, object],
    png_path: Path,
    svg_path: Path,
) -> None:
    operations = capture_payload["operations"]
    n_qubits = int(capture_payload["n_qubits"])
    baseline_t = baseline_t_count(molecule)
    best_t = best_t_count_through(molecule, iteration)
    reduction = 0.0
    if baseline_t and best_t is not None and baseline_t > 0:
        reduction = 100.0 * (baseline_t - best_t) / baseline_t

    cmap = color_map([float(item["epsilon"]) for item in operations])
    method_config = capture_payload.get("method_config", {})
    n_steps = method_config.get("n_steps")
    target_time = float(capture_payload.get("target_time", result_payload.get("target_time", 0.0)))
    display_name = str(result_payload.get("display_name", capture_payload["display_name"]))

    fig = plt.figure(figsize=(17.8, 7.9), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[6.7, 2.3],
        left=0.045,
        right=0.985,
        top=0.93,
        bottom=0.18,
        wspace=0.08,
    )

    ax_circuit = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_circuit.set_facecolor("white")
    ax_side.set_facecolor("white")

    y_positions = list(range(n_qubits))

    def style_circuit_axis(ax: plt.Axes, op_count: int) -> None:
        ax.set_xlim(0.5, op_count + 0.5)
        ax.set_ylim(-0.8, n_qubits - 0.2)
        ax.invert_yaxis()
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"q{idx}" for idx in range(n_qubits)], fontsize=12)
        ax.tick_params(axis="x", labelsize=11, colors="#444444")
        ax.tick_params(axis="y", labelsize=12, colors="#333333")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c6c6c6")
        ax.spines["bottom"].set_color("#c6c6c6")
        for y in y_positions:
            ax.axhline(y, color="#d9d9d9", linewidth=0.85, zorder=0)
        ax.set_title("PF step 1 circuit", loc="left", fontsize=17, color="#111111", pad=10)

    style_circuit_axis(ax_circuit, len(operations))

    def draw_ops(ax: plt.Axes, ops: List[Dict[str, object]], width: float) -> None:
        for idx, item in enumerate(ops, start=1):
            x = idx
            y0 = float(item["support_min"]) - 0.34
            height = float(item["support_max"] - item["support_min"]) + 0.68
            color = cmap[float(item["epsilon"])]
            rect = patches.FancyBboxPatch(
                (x - width / 2.0, y0),
                width,
                height,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                linewidth=0.35,
                edgecolor="#263238",
                facecolor=color,
                alpha=0.95,
            )
            ax.add_patch(rect)
            for qubit in item["support_qubits"]:
                ax.add_patch(
                    patches.Circle(
                        (x, float(qubit)),
                        radius=min(0.12, width * 0.23),
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.92,
                    )
                )
    box_width = 0.82 if len(operations) <= 48 else 0.74 if len(operations) <= 120 else 0.66
    draw_ops(ax_circuit, operations, width=box_width)

    ax_circuit.set_xticks([])
    label_transform = mtransforms.blended_transform_factory(ax_circuit.transData, ax_circuit.transAxes)
    op_count = len(operations)
    if op_count <= 20:
        label_font_size = 8.2
    elif op_count <= 48:
        label_font_size = 7.2
    elif op_count <= 120:
        label_font_size = 6.0
    elif op_count <= 280:
        label_font_size = 5.0
    elif op_count <= 700:
        label_font_size = 4.2
    else:
        label_font_size = 3.6
    for idx, item in enumerate(operations, start=1):
        ax_circuit.text(
            idx,
            -0.035,
            str(item["label"]),
            transform=label_transform,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="top",
            fontsize=label_font_size,
            color="#5a5a5a",
            clip_on=False,
        )

    ax_side.axis("off")
    ax_side.text(0.04, 0.96, display_name, ha="left", va="top", fontsize=20, color="#111111")
    ax_side.text(0.04, 0.905, f"Iteration {iteration:04d}", ha="left", va="top", fontsize=16, color="#333333")

    side_lines = [
        f"Repeats: {int(n_steps):,}" if n_steps is not None else "Repeats: n/a",
        f"Target time: {target_time:g}",
        f"T gates: {int(result_payload['t_gate_count']):,}",
        f"Fidelity: {float(result_payload['fidelity']):.12f}",
        f"Floor: {float(result_payload['chemical_accuracy_fidelity_floor']):.12f}",
        f"Reduction: {reduction:.2f}%",
        f"Ops / step: {len(operations):,}",
        f"Precisions: {len(cmap)}",
    ]
    ax_side.text(
        0.04,
        0.82,
        "\n".join(side_lines),
        ha="left",
        va="top",
        fontsize=13,
        color="#222222",
        linespacing=1.4,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#d2d2d2"},
    )

    ax_side.text(0.04, 0.34, "Gate precision", ha="left", va="top", fontsize=15, color="#111111")
    legend_y = 0.30
    for epsilon in sorted(cmap):
        ax_side.add_patch(
            patches.FancyBboxPatch(
                (0.04, legend_y - 0.018),
                0.065,
                0.04,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                linewidth=0.6,
                edgecolor="#263238",
                facecolor=cmap[epsilon],
                transform=ax_side.transAxes,
            )
        )
        ax_side.text(
            0.125,
            legend_y,
            f"{epsilon:.1e}",
            ha="left",
            va="center",
            fontsize=12,
            color="#333333",
            transform=ax_side.transAxes,
        )
        legend_y -= 0.055

    fig.savefig(svg_path, dpi=180, facecolor="white")
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.close(fig)


def render_snapshot_artifacts(molecule: str, iteration: int) -> Dict[str, str]:
    snap_dir = snapshot_dir(molecule, iteration)
    result_path = snap_dir / "result.json"
    script_path = snap_dir / "simple_pf.py"
    if not result_path.exists() or not script_path.exists():
        raise FileNotFoundError(f"missing snapshot inputs in {snap_dir}")

    result_payload = json.loads(result_path.read_text())
    capture_payload = first_step_operations(molecule, script_path, result_payload)

    png_path = snap_dir / "precision_circuit.png"
    svg_path = snap_dir / "precision_circuit.svg"
    tsv_path = snap_dir / "precision_circuit.tsv"

    render_snapshot_figure(molecule, iteration, result_payload, capture_payload, png_path, svg_path)
    write_operation_table(capture_payload["operations"], tsv_path)

    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "tsv": str(tsv_path),
    }


def available_iterations(molecule: str) -> List[int]:
    iterations: List[int] = []
    for row in results_rows(molecule):
        if (row.snapshot_dir / "result.json").exists() and (row.snapshot_dir / "simple_pf.py").exists():
            iterations.append(row.iteration)
    return iterations


def build_gif(molecule: str, duration_ms: int = 450) -> Path:
    spec = get_spec(molecule)
    frame_paths: List[Path] = []
    for iteration in available_iterations(molecule):
        snap_dir = snapshot_dir(molecule, iteration)
        frame_path = snap_dir / "precision_circuit.png"
        if not frame_path.exists():
            render_snapshot_artifacts(molecule, iteration)
        frame_paths.append(frame_path)
    if not frame_paths:
        raise RuntimeError(f"no rendered circuit frames available for {molecule}")

    output_path = spec.lane_dir / "circuit_evolution.gif"
    raw_frames = [np.asarray(imageio.imread(frame_path)) for frame_path in frame_paths]
    normalized_frames: List[np.ndarray] = []
    max_height = 0
    max_width = 0
    for frame in raw_frames:
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
            rgb = frame[:, :, :3].astype(np.float32)
            frame = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        elif frame.ndim == 3 and frame.shape[2] > 3:
            frame = frame[:, :, :3]
        max_height = max(max_height, int(frame.shape[0]))
        max_width = max(max_width, int(frame.shape[1]))
        normalized_frames.append(frame)

    frames: List[np.ndarray] = []
    for frame in normalized_frames:
        canvas = np.full((max_height, max_width, 3), 255, dtype=np.uint8)
        height, width = frame.shape[:2]
        y0 = (max_height - height) // 2
        x0 = (max_width - width) // 2
        canvas[y0 : y0 + height, x0 : x0 + width] = frame[:, :, :3]
        frames.append(canvas)
    imageio.mimsave(output_path, frames, duration=max(duration_ms, 20) / 1000.0, loop=0)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render per-iteration precision-colored circuits and build an evolution GIF.")
    parser.add_argument("--molecule", required=True, choices=tuple(sorted(get_spec.__globals__["MOLECULE_SPECS"])))
    parser.add_argument("--iteration", type=int, help="Render one archived iteration.")
    parser.add_argument("--all", action="store_true", help="Render all archived iterations with results.")
    parser.add_argument("--gif", action="store_true", help="Build or rebuild the lane GIF after rendering.")
    args = parser.parse_args()

    if args.iteration is None and not args.all:
        parser.error("pass --iteration N or --all")

    rendered: List[Dict[str, str]] = []
    if args.iteration is not None:
        rendered.append(render_snapshot_artifacts(args.molecule, args.iteration))
    if args.all:
        for iteration in available_iterations(args.molecule):
            rendered.append(render_snapshot_artifacts(args.molecule, iteration))
    for item in rendered:
        print(json.dumps(item))
    if args.gif:
        print(json.dumps({"gif": str(build_gif(args.molecule))}))


if __name__ == "__main__":
    main()
