# Figure 1 source artwork

`banner.tex` is the editable LaTeX/TikZ assembly for the approved Figure 1.
It contains the circuit, vertical energy-error schematic, candidate product-formula
snippets, typography, and flow arrows. Only the robot is clipped from the original
`generated-banner.png`; the original prompts remain as provenance.

The four molecular images were rendered with 3Dmol.js from explicit atoms and
bonds. Their schematic geometries illustrate the species, not the benchmark
coordinates. Run `npm install`, `npx playwright install chromium`, then
`npm run molecules` in this directory to regenerate them. An existing Chromium
can be supplied using `CHROMIUM_PATH`. Rebuilding the figure itself requires only
the supplied PNGs and LaTeX, with no browser or npm installation.

The coefficient inset is native PGF, so the figure is typeset in a single LaTeX
run without nested-PDF font conflicts. It uses the unchanged pinned HamLib arrays in
`results/paper/hamlib-diagnostics/schematic_coefficients.npz`.
The energy-error points, circuit and candidate code are schematic rather than
recorded evaluations. Candidate acceptance additionally requires a lower objective,
as described in the manuscript.

From the HSOpt root, `python figures/build.py` rebuilds the figures from their
frozen inputs. For a standalone rebuild of this assembly, run
`pdflatex -interaction=nonstopmode -halt-on-error banner.tex` in a temporary copy
of this directory. The released figure is
`figures/rendered/hsopt_autoresearch_loop.pdf`.

The 3:1 layout uses larger internal type and a horizontal error label. Molecular
renders have 2560 x 1440 pixels and camera margins to prevent clipping. Fonts,
histograms, circuits, labels and arrows remain vector graphics. The original robot
retains its source raster resolution.
