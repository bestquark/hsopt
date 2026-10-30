# Figure 1 source artwork

The approved ImageGen illustration is generated-banner.png. Its original prompts
are retained as artwork provenance; the figure build never invokes an image model.
The numerical coefficient inset uses the pinned HamLib arrays in
experiments/abs_protocol/results/hamlib-diagnostics/schematic_coefficients.npz.

From the HSOpt root, run python figures/build.py to rebuild the inset and original
banner.tex assembly. The renderer writes a temporary copy under the selected
output directory and preserves these source assets. The published result is
figures/rendered/hsopt_autoresearch_loop.pdf.
