"""Rebuild only the figures used by the current manuscript from frozen inputs.

No Overleaf checkout, agent, scheduler, GPU, or fresh scientific evaluation is
needed. Released PDFs stay in figures/rendered; regenerated copies go to the
chosen output directory, which defaults to the ignored figures/generated.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from _paths import MANIFEST, ROOT, data_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/generated")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    for name in manifest["data_files"]:
        data_path(name)
    env = dict(os.environ, HSOPT_FIGURE_OUTPUT=str(output), MPLBACKEND="Agg")
    scripts = ROOT / "figures"
    for name in (
        "plot_autoresearch_histograms.py",
        "assemble_elote_figure2.py",
        "plot_elote_search_and_energy.py",
        "plot_reaction_elote_delta_radial.py",
        "plot_reaction_elote_delta_energy_bars.py",
        "plot_hamlib_molecular_dataset_active_spaces.py",
        "plot_baseline_initialization_vs_qubits.py",
    ):
        subprocess.run([sys.executable, str(scripts / name)], env=env, check=True, cwd=ROOT)
    banner = output / ".build/banner"
    shutil.copytree(scripts / "sources/hsopt_autoresearch_loop", banner, dirs_exist_ok=True)
    shutil.copyfile(
        output / ".build/coefficient_histograms_2x2.pdf", banner / "coefficient_histograms_2x2.pdf"
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "banner.tex"],
        cwd=banner,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    shutil.copyfile(banner / "banner.pdf", output / "hsopt_autoresearch_loop.pdf")
    expected = set(manifest["figures"])
    assert {p.name for p in output.glob("*.pdf")} == expected
    report = {
        "schema": "hsopt-manuscript-render-v1",
        "paper_source_commit": manifest["paper_source_commit"],
        "outputs": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in sorted(expected)
        },
        "processing": "Frozen measured data; no new simulation or policy selection",
    }
    (output / "render-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
