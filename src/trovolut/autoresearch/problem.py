"""Problem definition for the autoresearch loop.

A Problem specifies what to optimize: the target script, constraints,
budget, and how to parse metrics. It generates the configuration needed
by OpenEvolve or the agent-based optimization loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class Problem:
    """Describes an autoresearch optimization problem.

    Parameters
    ----------
    target : str or Path
        Path to the Python script to optimize.
    constraints : Group, list of Constraint, or None
        Constraints that define success criteria.
    budget_seconds : float
        Per-experiment timeout in seconds.
    evolvable_regions : str
        ``"whole_file"`` or ``"markers"`` (for EVO_START/EVO_END blocks).
    metric_parser : str
        ``"json"`` or ``"keyvalue"`` — how to parse metrics from stdout.
    runner : str or None
        Runner type: ``"generic"``, ``"pytorch"``, ``"cpu"``, ``"gpu"``.
        If None, auto-detected from target content.
    description : str
        Human-readable description of the optimization goal.
    output_dir : str
        Directory for autoresearch output (checkpoints, history, etc.).
    """

    target: Union[str, Path] = ""
    constraints: Optional[object] = None  # Group or list of Constraint
    budget_seconds: float = 300.0
    evolvable_regions: str = "whole_file"
    metric_parser: str = "json"
    runner: Optional[str] = None
    description: str = ""
    output_dir: str = "autoresearch_output"

    def __post_init__(self):
        self.target = Path(self.target) if self.target else Path()

    def validate(self) -> None:
        """Validate the problem definition.

        Raises
        ------
        ValueError
            If the problem is misconfigured.
        FileNotFoundError
            If the target script does not exist.
        """
        if not self.target or not Path(self.target).exists():
            raise FileNotFoundError(f"Target script not found: {self.target}")

        if self.evolvable_regions not in ("whole_file", "markers"):
            raise ValueError(
                f"evolvable_regions must be 'whole_file' or 'markers', "
                f"got {self.evolvable_regions!r}"
            )

        if self.metric_parser not in ("json", "keyvalue"):
            raise ValueError(
                f"metric_parser must be 'json' or 'keyvalue', "
                f"got {self.metric_parser!r}"
            )

        if self.budget_seconds <= 0:
            raise ValueError(f"budget_seconds must be positive, got {self.budget_seconds}")

    def _detect_runner(self) -> str:
        """Auto-detect the appropriate runner from target content."""
        if self.runner is not None:
            return self.runner

        try:
            content = Path(self.target).read_text()
        except Exception:
            return "generic"

        if "import torch" in content or "from torch" in content:
            return "pytorch"
        if "import cudaq" in content or "from cudaq" in content:
            return "gpu"
        if "import trovolut" in content or "from trovolut" in content:
            return "cpu"
        return "generic"

    def to_program_md(self) -> str:
        """Generate agent instructions (program.md) for LLM-based optimization.

        Returns a markdown string describing the optimization problem,
        constraints, and instructions for the LLM agent.
        """
        lines = [
            "# Autoresearch Optimization Problem",
            "",
        ]

        if self.description:
            lines.append(f"## Goal\n{self.description}\n")

        lines.append(f"## Target Script\n`{self.target}`\n")

        if self.evolvable_regions == "markers":
            lines.append(
                "## Evolvable Regions\n"
                "Only modify code between `# EVOLVE-BLOCK-START` and "
                "`# EVOLVE-BLOCK-END` markers.\n"
            )
        else:
            lines.append("## Evolvable Regions\nThe entire file can be modified.\n")

        # Constraints
        if self.constraints is not None:
            if hasattr(self.constraints, "generate_llm_instructions"):
                lines.append("## Constraints")
                lines.append(self.constraints.generate_llm_instructions())
                lines.append("")
            elif isinstance(self.constraints, (list, tuple)):
                lines.append("## Constraints")
                for c in self.constraints:
                    if hasattr(c, "to_prompt"):
                        lines.append(f"- {c.to_prompt()}")
                lines.append("")

        lines.extend([
            "## Metrics",
            f"Metrics are parsed from stdout using `{self.metric_parser}` format.",
            f"Per-experiment timeout: {self.budget_seconds}s",
            "",
            "## Output Format",
        ])

        if self.metric_parser == "json":
            lines.append(
                'Print a JSON object to stdout: `{"time": 1.5, "error": 0.01}`'
            )
        else:
            lines.append(
                "Print key-value pairs to stdout:\n```\ntime: 1.5\nerror: 0.01\n```"
            )

        return "\n".join(lines)

    def to_openevolve_config(self):
        """Generate an OpenEvolve Config object for this problem.

        Returns an ``openevolve.config.Config`` with settings derived
        from this Problem definition.
        """
        try:
            from openevolve.config import Config
        except ImportError:
            raise ImportError(
                "OpenEvolve is required for autoresearch. "
                "Install with: pip install openevolve"
            )

        config = Config()
        config.timeout = int(self.budget_seconds)

        return config


__all__ = [
    "Problem",
]
