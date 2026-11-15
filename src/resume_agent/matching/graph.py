"""LangGraph pipeline: extract and validate the resume skills, classify them next to the job's skills (in
parallel), compare, and optionally ask a judge to review the comparison."""

import json
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm import LLMError
from .prompts import (
    compare_skills,
    extracted_skill_validator,
    judge_comparison_prompt,
    skill_fetch_prompt,
    skill_type_classification,
)


class State(TypedDict):
    run_judge: bool
    resume_doc: str
    job_description_doc: str
    extracted_skills_json: str
    validated_extracted_skills_json: str
    resume_classified_skills_json: str
    jd_classified_skills_json: str
    comparison_result_json: str
    judge_feedback_json: str


def parse_json(text):
    """Parse a model answer that may be wrapped in a markdown code fence."""
    body = (text or "").strip()
    if body.startswith("```"):
        body = body[body.find("\n") + 1 :] if "\n" in body else ""
        body = body[: body.rfind("```")] if "```" in body else body
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model did not return valid JSON: {body[:120]!r}") from exc


def build_graph(llm):
    """Build the compiled graph; every node asks ``llm.complete(prompt)``."""

    def extract(state):
        return {"extracted_skills_json": llm.complete(skill_fetch_prompt(state["resume_doc"]))}

    def validate(state):
        prompt = extracted_skill_validator(state["resume_doc"], state["extracted_skills_json"])
        return {"validated_extracted_skills_json": llm.complete(prompt)}

    def classify_resume(state):
        return {"resume_classified_skills_json": llm.complete(skill_type_classification(state["validated_extracted_skills_json"]))}

    def classify_jd(state):
        return {"jd_classified_skills_json": llm.complete(skill_type_classification(state["job_description_doc"]))}

    def compare(state):
        prompt = compare_skills(state["resume_classified_skills_json"], state["jd_classified_skills_json"])
        return {"comparison_result_json": llm.complete(prompt)}

    def judge(state):
        return {"judge_feedback_json": llm.complete(judge_comparison_prompt(state["comparison_result_json"] or "{}"))}

    workflow = StateGraph(State)
    for name, node in [
        ("extract_skills", extract),
        ("validate_skills", validate),
        ("classify_resume_skills", classify_resume),
        ("classify_jd_skills", classify_jd),
        ("compare_skills", compare),
        ("judge", judge),
    ]:
        workflow.add_node(name, node)

    # The resume branch and the job branch run in parallel and meet at the comparison.
    workflow.add_edge(START, "extract_skills")
    workflow.add_edge("extract_skills", "validate_skills")
    workflow.add_edge("validate_skills", "classify_resume_skills")
    workflow.add_edge(START, "classify_jd_skills")
    workflow.add_edge(["classify_resume_skills", "classify_jd_skills"], "compare_skills")
    workflow.add_conditional_edges("compare_skills", lambda state: "judge" if state["run_judge"] else END, ["judge", END])
    workflow.add_edge("judge", END)
    return workflow.compile()


@dataclass
class MatchResult:
    comparison: dict
    judge: dict | None = None


def run_match(resume_text, job_description_text, llm, judge=False):
    """Compare a resume with a job description. Raises ``LLMError`` if the model fails or answers badly."""
    final = build_graph(llm).invoke(
        {
            "run_judge": judge,
            "resume_doc": resume_text,
            "job_description_doc": job_description_text,
            "extracted_skills_json": "",
            "validated_extracted_skills_json": "",
            "resume_classified_skills_json": "",
            "jd_classified_skills_json": "",
            "comparison_result_json": "",
            "judge_feedback_json": "",
        }
    )
    return MatchResult(
        comparison=parse_json(final["comparison_result_json"]),
        judge=parse_json(final["judge_feedback_json"]) if judge else None,
    )
