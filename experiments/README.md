# Paper Experiments

This tree contains portable experiment definitions and completion-gated paper
artifacts. It does not contain live campaign workspaces.

- `hamlib121/` defines five independent autoresearch runs over the fixed
  121-Hamiltonian benchmark.
- `reactions10/` is reserved for the ten-reaction error-allocation comparison.

Campaign databases, mutation directories, scheduler logs, telemetry, agent
transcripts, virtual environments, and intermediate checkpoints stay in
external storage. Only a reusable runner and compact, validated final results
belong here.
