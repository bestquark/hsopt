# analyze_pareto.py
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "code_evolution"
TEMPLATE = "openevolve_output_{lam}/best/best_program_info.json"

# 10 lambdas from 0.00 to 1.00 inclusive
LAMBDAS = [f"{x:.1f}" for x in np.linspace(0.0, 1.0, 6)]


def read_point(lam: str):
    p = BASE / TEMPLATE.format(lam=lam)
    if not p.exists():
        print(f"[skip] missing file: {p}", file=sys.stderr)
        return None
    try:
        data = json.loads(p.read_text())
        m = data.get("metrics", {})
        acc = m.get("acc")
        mem = m.get("mem_norm")
        if acc is None or mem is None:
            print(f"[skip] metrics incomplete in {p}", file=sys.stderr)
            return None
        return {
            "lambda": lam,
            "acc": float(acc),
            "mem_norm": float(mem),
            "path": str(p),
        }
    except Exception as e:
        print(f"[skip] failed reading {p}: {e}", file=sys.stderr)
        return None


def pareto_front(points):
    """Pareto frontier for maximizing accuracy and minimizing memory."""
    idxs = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if (
                b["acc"] >= a["acc"]
                and b["mem_norm"] <= a["mem_norm"]
                and (b["acc"] > a["acc"] or b["mem_norm"] < a["mem_norm"])
            ):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    idxs.sort(key=lambda k: (points[k]["mem_norm"], -points[k]["acc"]))
    return idxs


def main():
    points = [read_point(lam) for lam in LAMBDAS]
    points = [p for p in points if p is not None]

    if not points:
        print("No valid points found.", file=sys.stderr)
        sys.exit(1)

    # Write CSV
    out_csv = PROJECT_ROOT / "visualization/pareto_points.csv"
    with out_csv.open("w") as f:
        f.write("lambda,acc,mem_norm,path\n")
        for p in points:
            f.write(f"{p['lambda']},{p['acc']:.8f},{p['mem_norm']:.8f},{p['path']}\n")
    print(f"wrote {out_csv}")

    # Compute Pareto frontier
    frontier_idx = pareto_front(points)
    frontier = [points[i] for i in frontier_idx]

    # Make plot
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=140)
    ax.scatter(
        [p["mem_norm"] for p in points],
        [p["acc"] for p in points],
        s=30,
        alpha=0.7,
        label="runs",
    )

    if frontier:
        xs = [p["mem_norm"] for p in frontier]
        ys = [p["acc"] for p in frontier]
        ax.plot(xs, ys, linewidth=2, label="Pareto frontier")
        ax.scatter(xs, ys, s=50, zorder=3)

    for p in points:
        ax.annotate(
            p["lambda"],
            (p["mem_norm"], p["acc"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Memory usage (mem_norm)")
    ax.set_ylabel("Accuracy (acc)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_png = PROJECT_ROOT / "visualization/pareto_plot.png"
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"wrote {out_png}")

    print("\nPareto frontier:")
    for p in frontier:
        print(f"  λ={p['lambda']}: acc={p['acc']:.4f}, mem_norm={p['mem_norm']:.6f}")


if __name__ == "__main__":
    main()
