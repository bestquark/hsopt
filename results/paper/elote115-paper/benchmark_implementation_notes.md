# Benchmark implementation notes

These notes retain implementation details omitted from the compact manuscript
appendices. They describe the reported experiments and do not change the
scientific protocols or their evaluated results.

## Software and evaluation

The five molecular searches and five reaction searches used OpenCode 2.0.26
in OpenShell 0.1.3, with Nemotron 3.5 Lightning generating code proposals.
CUDA-Q synthesized rotations in Matsumoto–Amano normal form. GPU Lanczos
applied the compiled circuit to vectors without constructing its full matrix.
The paper's Methods and Results identify these tools, their roles, and the
evaluation hardware.

For the 18–22-qubit tests, the eigenpair-residual tolerance was 1e-9 Ha
and the energy-stability tolerance was 1e-10 Ha. Independent starts were
required to agree, and numerically unresolved results were distinguished
from demonstrated chemical-accuracy failures.

Each cost includes every rotation in every Trotter repetition, even when the
compiler reuses an already synthesized sequence. Compilation reuse does not
reduce the number of gates executed by the full evolution.

## Grouped tight baseline

The training baseline was constructed from Hermitian one- and two-body
fermionic blocks. Layers contained blocks with disjoint spin-orbital support.
Commutation of the Pauli strings within each layer was checked after
Jordan–Wigner mapping. These layers entered the symmetric forward-and-reverse
sequence of the nested second-order formula. Consecutive matching Pauli
rotations were combined, and each group was compiled into individual rotations.

For each training Hamiltonian, the grouped formula was calibrated against its
exact ground-state energy at a 0.1 mHa Trotter-error target to determine the
tight beta parameter. A model fitted to all 115 training calibrations predicted
the testing parameters. That model was fixed before testing, and testing
energies were used only to check accuracy. The campaign used normalized
evolution time t = 1/lambda. The separate bound-based diagnostic shown in
Figure 6 used t = 1 Ha^-1.

## Interpreting rotation counts

Aggregation preserves the Hamiltonian, although changing rotation order can
change the finite-step approximation and its energy error. Rotation counts
are different from T-gate counts because some nonzero-angle rotations are
Clifford operations. The manuscript reports compiled T counts and QPE-inclusive
T95 costs separately. The frozen policy diagnostics in `elote_understanding/`
contain the rotation-level checks and exact policy sources.

The baseline and selected molecular ELOTE contained the same Pauli strings.
Summed signed angles per string agreed to within 5.6e-17 radians over one
step. No nonzero coefficients were removed using a magnitude threshold.
