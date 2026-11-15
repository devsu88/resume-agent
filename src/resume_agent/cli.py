"""``resume-agent serve`` runs the web app; ``resume-agent match`` scores a job offer from the terminal."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .chat import ResumeChat
from .llm import LLM
from .matching import run_match
from .notifier import notifier_from_env
from .profile import load_profile
from .render import render_match


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="resume-agent")
    parser.add_argument("--profile", default=os.getenv("RESUME_PROFILE_DIR", "profile"), help="folder with resume.pdf|txt|md, summary.txt, profile.json")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the web app")
    serve.add_argument("--port", type=int, default=7860)
    match = sub.add_parser("match", help="compare the resume with a job offer")
    match.add_argument("job", type=Path, help="text file with the job offer")
    match.add_argument("--judge", action="store_true", help="also run the judge review")
    match.add_argument("--json", action="store_true", help="print the raw JSON instead of Markdown")
    return parser.parse_args(argv)


def main(argv=None):
    load_dotenv()
    args = parse_args(argv)
    profile = load_profile(args.profile)
    if args.command == "serve":
        from .app import build_app

        build_app(profile, ResumeChat(profile, notifier=notifier_from_env())).launch(server_port=args.port)
        return
    result = run_match(profile.resume, args.job.read_text(encoding="utf-8"), LLM(), judge=args.judge)
    if args.json:
        print(json.dumps({"comparison": result.comparison, "judge": result.judge}, indent=2, ensure_ascii=False))
    else:
        print(render_match(result))
