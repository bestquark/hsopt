# analyze_pareto.py
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# set latex font for plots
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

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
        err = m.get("err")
        mem = m.get("mem_norm")
        if err is None or mem is None:
            print(f"[skip] metrics incomplete in {p}", file=sys.stderr)
            return None
        return {
            "lambda": lam,
            "err": float(err),
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
                b["err"] >= a["err"]
                and b["mem_norm"] <= a["mem_norm"]
                and (b["err"] > a["err"] or b["mem_norm"] < a["mem_norm"])
            ):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    idxs.sort(key=lambda k: (points[k]["mem_norm"], -points[k]["err"]))
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
        f.write("lambda,err,mem_norm,path\n")
        for p in points:
            f.write(f"{p['lambda']},{p['err']:.8f},{p['mem_norm']:.8f},{p['path']}\n")
    print(f"wrote {out_csv}")

    # Make plot
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=140)

    xs = [p["mem_norm"] for p in points]
    ys = [p["err"] for p in points]
    cs = [float(p["lambda"]) for p in points]  # numeric λ for colormap
    yerrs = [p.get("err", 0) / np.sqrt(max(1, p.get("trials", 1))) for p in points]

    # light gray error bars behind the points
    for x, y, ye in zip(xs, ys, yerrs):
        ax.errorbar(
            x,
            y,
            yerr=ye,
            fmt="none",
            ecolor="0.1",
            elinewidth=1,
            alpha=0.6,
            zorder=1,
            capsize=2,
        )

    # colored scatter by λ
    sc = ax.scatter(
        xs, ys, c=cs, cmap="viridis", s=40, alpha=0.9, zorder=2, label="runs"
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("$\\lambda$")

    # Pareto frontier
    # if frontier:
    #     fx = [p["mem_norm"] for p in frontier]
    #     fy = [p["err"] for p in frontier]
    #     ax.plot(fx, fy, linewidth=2, label="Pareto frontier", zorder=3)
    #     ax.scatter(fx, fy, s=50, zorder=4)

    # annotate each point with its λ
    # for p in points:
    #     ax.annotate(p["lambda"], (p["mem_norm"], p["err"]), fontsize=8,
    #                 xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Memory usage (normalized)")
    ax.set_ylabel("Trotter error")
    ax.grid(True, alpha=0.25)
    # ax.legend()

    out_png = PROJECT_ROOT / "visualization/pareto_plot.png"
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"wrote {out_png}")

    # print("\nPareto frontier:")
    # for p in frontier:
    #     print(f"  λ={p['lambda']}: err={p['err']:.4f}, mem_norm={p['mem_norm']:.6f}")


if __name__ == "__main__":
    main()
