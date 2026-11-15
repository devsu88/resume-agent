"""Turn the model's JSON results into readable Markdown."""


def _bullets(items):
    return "\n".join(f"- {item}" for item in items) or "- none"


def render_match(result):
    """Markdown summary of a ``MatchResult``. Missing keys are skipped: the model's JSON is not guaranteed."""
    comparison = result.comparison
    summary = comparison.get("matching_summary", {})
    insights = comparison.get("assessment_insights", {})
    lines = ["## Match summary"]
    if "overall_fit_score" in insights:
        lines.append(f"**Overall fit: {insights['overall_fit_score']}/100**")
    if summary:
        lines.append(f"{summary.get('total_matches', '?')} of {summary.get('total_jd_skills', '?')} required skills matched ({summary.get('match_percentage', '?')}%).")
    if insights.get("recommendation"):
        lines.append(f"\n{insights['recommendation']}")
    lines += ["\n### Strengths", _bullets(insights.get("strengths", [])), "\n### Gaps", _bullets(insights.get("gaps", []))]

    matched = comparison.get("matched_skills", [])
    if matched:
        lines += ["\n### Matched skills", "| Job skill | Candidate skill | Strength |", "|---|---|---|"]
        lines += [f"| {m.get('jd_skill', '')} | {m.get('candidate_skill', '')} | {m.get('match_strength', '')} |" for m in matched]
    missing = comparison.get("unmatched_jd_skills", [])
    if missing:
        lines += ["\n### Skills the job asks for and the resume does not show"]
        lines += [f"- **{m.get('skill', '')}** ({m.get('criticality', '')})" for m in missing]

    if result.judge:
        lines += ["\n## Judge review", f"**Quality score:** {result.judge.get('quality_score', '?')}  \n**Verdict:** {result.judge.get('overall_verdict', '?')}"]
        if result.judge.get("suggested_improvements"):
            lines += ["\n### Suggested improvements", _bullets(result.judge["suggested_improvements"])]
    return "\n".join(lines)
