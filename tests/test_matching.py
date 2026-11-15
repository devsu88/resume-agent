import json

import pytest

from resume_agent.llm import LLMError
from resume_agent.matching import run_match
from resume_agent.render import render_match

COMPARISON = {
    "matching_summary": {"total_jd_skills": 4, "total_matches": 3, "match_percentage": 75},
    "matched_skills": [{"jd_skill": "Python", "candidate_skill": "Python", "match_strength": "strong"}],
    "unmatched_jd_skills": [{"skill": "Kubernetes", "criticality": "nice-to-have"}],
    "assessment_insights": {"strengths": ["Python"], "gaps": ["Kubernetes"], "overall_fit_score": "82", "recommendation": "Interview."},
}
JUDGE = {"quality_score": 8, "overall_verdict": "reliable", "suggested_improvements": ["Check dbt"]}

MARKERS = {
    "RESUME SKILLS EXTRACTION PROMPT": "extract",
    "Skills Validation Specialist": "validate",
    "Skills Taxonomy Classifier": "classify",
    "HR Talent Acquisition analyst": "compare",
    "SKILL MATCHING JUDGE PROMPT": "judge",
}


class PromptRoutedLLM:
    """Answers according to which pipeline prompt it receives (the two branches run in parallel)."""

    def __init__(self, comparison=None, judge=None, fence=False, failing_step=None):
        self.comparison, self.judge, self.fence, self.failing_step = comparison or COMPARISON, judge or JUDGE, fence, failing_step
        self.steps = []

    def complete(self, prompt, system=None):
        step = next(name for marker, name in MARKERS.items() if marker in prompt)
        self.steps.append(step)
        if step == self.failing_step:
            raise LLMError("model down")
        payload = {"compare": self.comparison, "judge": self.judge}.get(step, {"skills": []})
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return f"```json\n{text}\n```" if self.fence else text


def test_the_comparison_is_returned_without_calling_the_judge():
    llm = PromptRoutedLLM()
    result = run_match("resume", "job", llm)
    assert result.comparison == COMPARISON and result.judge is None
    assert "judge" not in llm.steps


def test_with_the_judge_both_results_come_back():
    llm = PromptRoutedLLM()
    result = run_match("resume", "job", llm, judge=True)
    assert result.judge == JUDGE
    assert llm.steps[-2:] == ["compare", "judge"]


def test_the_resume_and_the_job_are_both_processed_before_the_comparison():
    llm = PromptRoutedLLM()
    run_match("resume", "job", llm)
    assert llm.steps.count("classify") == 2
    assert sorted(llm.steps[:-1]) == ["classify", "classify", "extract", "validate"]
    assert llm.steps[-1] == "compare"


def test_answers_wrapped_in_a_markdown_fence_are_parsed():
    assert run_match("resume", "job", PromptRoutedLLM(fence=True)).comparison == COMPARISON


def test_an_answer_that_is_not_json_is_an_error():
    with pytest.raises(LLMError, match="valid JSON"):
        run_match("resume", "job", PromptRoutedLLM(comparison="sorry, I cannot do that"))


def test_a_failing_model_call_stops_the_run():
    with pytest.raises(LLMError, match="model down"):
        run_match("resume", "job", PromptRoutedLLM(failing_step="validate"))


def test_the_markdown_summary_shows_score_gaps_and_judge_verdict():
    text = render_match(run_match("resume", "job", PromptRoutedLLM(), judge=True))
    assert "Overall fit: 82/100" in text and "3 of 4 required skills" in text
    assert "| Python | Python | strong |" in text and "**Kubernetes** (nice-to-have)" in text
    assert "Quality score:** 8" in text and "reliable" in text and "Check dbt" in text


def test_the_summary_survives_missing_keys():
    from resume_agent.matching import MatchResult

    text = render_match(MatchResult(comparison={}))
    assert "Match summary" in text and "none" in text
