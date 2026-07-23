from __future__ import annotations

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes


def test_clean_exact_branch_has_no_legacy_pfs_lanes() -> None:
    assert discover_lanes() == []


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
