import httpx
import openai
import pytest

from resume_agent.llm import LLM, LLMError
from tests.conftest import choice

REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"choices": [outcome]})()


def rate_limit():
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)


def llm(client, **kw):
    return LLM(client=client, retry_delay=0, **kw)


def test_complete_returns_the_text_and_sends_the_system_prompt():
    client = FakeClient(choice("hello"))
    assert llm(client).complete("hi", system="be brief") == "hello"
    assert client.requests[0]["messages"][0] == {"role": "system", "content": "be brief"}


def test_a_rate_limit_is_retried_then_succeeds():
    client = FakeClient(rate_limit(), rate_limit(), choice("ok"))
    assert llm(client).complete("hi") == "ok"
    assert len(client.requests) == 3


def test_it_gives_up_after_the_last_retry_with_a_clear_error():
    client = FakeClient(rate_limit(), rate_limit(), rate_limit())
    with pytest.raises(LLMError, match="unavailable"):
        llm(client).complete("hi")


def test_a_rejected_key_fails_at_once_without_retrying():
    bad = openai.AuthenticationError("no", response=httpx.Response(401, request=REQUEST), body=None)
    client = FakeClient(bad)
    with pytest.raises(LLMError, match="rejected"):
        llm(client).complete("hi")
    assert len(client.requests) == 1


def test_an_empty_answer_is_an_error_not_silence():
    with pytest.raises(LLMError, match="empty"):
        llm(FakeClient(choice(None))).complete("hi")


def test_a_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="API key"):
        LLM()


def test_the_visitor_key_is_not_written_to_the_shared_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    LLM(api_key="visitor-key")
    import os

    assert "OPENAI_API_KEY" not in os.environ


def test_the_model_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some/model")
    assert LLM(client=FakeClient()).model == "some/model"
