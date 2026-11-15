from pathlib import Path

import resume_agent.cli as cli
from resume_agent.app import build_app
from resume_agent.chat import ResumeChat
from resume_agent.profile import load_profile
from tests.test_matching import PromptRoutedLLM

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_match_prints_a_markdown_summary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "LLM", lambda: PromptRoutedLLM())
    cli.main(["--profile", str(EXAMPLES / "profile"), "match", str(EXAMPLES / "job.txt")])
    assert "Overall fit: 82/100" in capsys.readouterr().out


def test_match_can_print_raw_json_with_the_judge(monkeypatch, capsys):
    monkeypatch.setattr(cli, "LLM", lambda: PromptRoutedLLM())
    cli.main(["--profile", str(EXAMPLES / "profile"), "match", str(EXAMPLES / "job.txt"), "--json", "--judge"])
    assert '"quality_score": 8' in capsys.readouterr().out


def test_the_example_profile_loads_and_the_app_builds():
    profile = load_profile(EXAMPLES / "profile")
    assert profile.name == "Alex Rivera" and "Spark" in profile.resume
    assert build_app(profile, ResumeChat(profile)) is not None
