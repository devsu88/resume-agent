"""The person the assistant speaks for. The real profile lives outside the repository."""

import json
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Profile:
    name: str
    summary: str
    resume: str


def _read(path):
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    return path.read_text(encoding="utf-8").strip()


def load_profile(directory):
    """Read ``resume.(pdf|txt|md)``, optional ``summary.(txt|md)`` and optional ``profile.json`` ({"name": ...})."""
    folder = Path(directory)
    resume_file = next((folder / f"resume{ext}" for ext in (".pdf", ".txt", ".md") if (folder / f"resume{ext}").exists()), None)
    if resume_file is None:
        raise FileNotFoundError(f"No resume.pdf, resume.txt or resume.md in {folder}")
    summary_file = next((folder / f"summary{ext}" for ext in (".txt", ".md") if (folder / f"summary{ext}").exists()), None)
    meta_file = folder / "profile.json"
    name = json.loads(meta_file.read_text(encoding="utf-8")).get("name", "") if meta_file.exists() else ""
    return Profile(name=name or "the candidate", summary=_read(summary_file) if summary_file else "", resume=_read(resume_file))
