from __future__ import annotations

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes


def test_clean_exact_branch_has_no_legacy_pfs_lanes() -> None:
    assert discover_lanes() == []


def test_agent_skill_is_separate_from_runtime_package() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "hsopt" / "SKILL.md").read_text(encoding="utf-8")
    runtime = REPO_ROOT / "src" / "hsopt"
    readme_words = " ".join(readme.split())
    skill_words = " ".join(skill.split())

    assert runtime.is_dir()
    assert (runtime / "compiled_unitary.py").is_file()
    assert not (REPO_ROOT / "skills" / "hsopt" / "scripts").exists()
    assert "git clone https://github.com/bestquark/hsopt.git" in readme
    assert "src/hsopt/" in readme_words
    assert "## Agent skill" in readme
    assert "repository runtime is ordinary package code" in skill_words
    assert len(skill.splitlines()) < 130
