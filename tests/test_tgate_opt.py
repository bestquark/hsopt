from __future__ import annotations

from pathlib import Path

import pytest

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes
from hsopt.tgate_opt.t_gepa_cli import build_parser as build_gepa_parser


def test_clean_exact_branch_has_no_legacy_pfs_lanes() -> None:
    assert discover_lanes() == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["dataset"],
        ["evaluate-prompt", "--prompt-file", "/tmp/seed.md"],
        ["optimize"],
    ],
)
def test_hsgepa_write_commands_require_explicit_destinations(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        build_gepa_parser().parse_args(arguments)


def test_hsgepa_accepts_caller_supplied_external_paths(tmp_path: Path) -> None:
    dataset = build_gepa_parser().parse_args(
        ["dataset", "--output", str(tmp_path / "dataset.json")]
    )
    assert dataset.output == str(tmp_path / "dataset.json")

    evaluation = build_gepa_parser().parse_args(
        [
            "evaluate-prompt",
            "--prompt-file",
            str(tmp_path / "seed.md"),
            "--run-root",
            str(tmp_path / "manual"),
        ]
    )
    assert evaluation.run_root == str(tmp_path / "manual")

    optimization = build_gepa_parser().parse_args(
        ["optimize", "--run-root", str(tmp_path / "optimization")]
    )
    assert optimization.run_root == str(tmp_path / "optimization")


def test_install_docs_cover_runtime_and_agent_skill_separately() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "hsopt" / "SKILL.md").read_text(encoding="utf-8")
    readme_words = " ".join(readme.split())
    skill_words = " ".join(skill.split())

    assert "git clone https://github.com/bestquark/hsopt.git" in readme
    assert "cloning the Python runtime does not automatically install it" in readme_words
    assert "## Agent skill" in readme
    assert "installed agent skill and the benchmark runtime as separate pieces" in skill_words
    assert "Direct evaluator runs" in skill_words and "`uv sync`" in skill_words
