"""Gradio app: a chat about the profile, and a match between the profile's resume and a pasted job offer."""

import gradio as gr

from .chat import ResumeChat
from .llm import LLM, LLMError
from .matching import run_match
from .render import render_match

KEY_LABEL = "Your API key (used only for your requests, never stored)"


def build_app(profile, chat: ResumeChat):
    def answer(message, history, api_key):
        try:
            return chat.reply(message, history, api_key=(api_key or "").strip() or None)
        except LLMError as exc:
            return f"Sorry: {exc}"

    def match(job_text, run_judge, api_key):
        if not (job_text or "").strip():
            return "Paste a job offer first."
        try:
            llm = LLM(api_key=(api_key or "").strip() or None)
            return render_match(run_match(profile.resume, job_text, llm, judge=run_judge))
        except LLMError as exc:
            return f"Sorry: {exc}"

    with gr.Blocks(title=f"Ask about {profile.name}") as app:
        gr.Markdown(f"# Ask about {profile.name}\nAn AI assistant that answers from {profile.name}'s resume. It can be wrong: check what matters.")
        with gr.Tab("Chat"):
            gr.ChatInterface(
                answer,
                additional_inputs=[gr.Textbox(label=KEY_LABEL, type="password")],
                examples=[["What are the main skills?", ""], ["Summarise the most recent role.", ""]],
            )
        with gr.Tab("Match a job offer"):
            job = gr.Textbox(label="Job offer", lines=12, placeholder="Paste the job description here")
            judge = gr.Checkbox(label="Also run the judge (a second model call that reviews the result)")
            key = gr.Textbox(label=KEY_LABEL, type="password")
            button = gr.Button("Compare", variant="primary")
            output = gr.Markdown()
            button.click(match, [job, judge, key], output)
    return app
