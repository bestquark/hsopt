from __future__ import annotations

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes


def test_clean_exact_branch_has_no_legacy_pfs_lanes() -> None:
    assert discover_lanes() == []


def test_runtime_is_bundled_inside_the_agent_skill() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "hsopt" / "SKILL.md").read_text(encoding="utf-8")
    runtime = REPO_ROOT / "skills" / "hsopt" / "scripts" / "hsopt"
    readme_words = " ".join(readme.split())
    skill_words = " ".join(skill.split())

    assert runtime.is_dir()
    assert (runtime / "compiled_unitary.py").is_file()
    assert not (REPO_ROOT / "src").exists()
    assert "git clone https://github.com/bestquark/hsopt.git" in readme
    assert "skills/hsopt/scripts/hsopt/" in readme_words
    assert "## Agent skill" in readme
    assert "single source bundle" in skill_words
    assert "Direct evaluator runs" in skill_words and "`uv sync`" in skill_words
