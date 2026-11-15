"""One small wrapper over an OpenAI-compatible chat API.

The key is passed in, never written to ``os.environ`` (which all sessions of a web app share), so a
visitor can bring their own. The endpoint comes from the standard ``OPENAI_BASE_URL`` variable, so the same
code runs on OpenAI, OpenRouter or any compatible server. Failures raise ``LLMError`` instead of returning
``None`` (the earlier matcher passed a failed call straight into the next prompt).
"""

import os
import time

import openai


class LLMError(RuntimeError):
    """The model could not be reached or did not answer."""


RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


class LLM:
    def __init__(self, api_key=None, model=None, max_retries=3, retry_delay=2.0, client=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if client is None and not self.api_key:
            raise LLMError("An API key is required: enter it in the app, or set OPENAI_API_KEY.")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.max_retries, self.retry_delay = max_retries, retry_delay
        self._client = client or openai.OpenAI(api_key=self.api_key)

    def chat(self, messages, tools=None):
        """One chat completion; returns the first choice. Retries transient errors with backoff."""
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs).choices[0]
            except openai.AuthenticationError as exc:
                raise LLMError("The API key was rejected.") from exc
            except RETRYABLE as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"The model is unavailable ({type(exc).__name__}).") from exc
                time.sleep(delay)
                delay *= 2
        raise LLMError("The model did not answer.")  # pragma: no cover

    def complete(self, prompt, system=None):
        """Single-turn text completion."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self.chat(messages).message.content
        if not text:
            raise LLMError("The model returned an empty answer.")
        return text
