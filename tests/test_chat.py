import pytest

from resume_agent.chat import MAX_TOOL_ROUNDS, ResumeChat, _clean_history, system_prompt
from resume_agent.llm import LLMError
from resume_agent.profile import Profile
from tests.conftest import ScriptedLLM, choice, tool_call

PROFILE = Profile(name="Alex Rivera", summary="Data engineer.", resume="Built pipelines with Spark.")


def make_chat(notifier, *script):
    llm = ScriptedLLM(*script)
    keys = []
    chat = ResumeChat(PROFILE, notifier=notifier, llm_factory=lambda key: (keys.append(key), llm)[1])
    return chat, llm, keys


def test_a_plain_answer_is_returned(notifier):
    chat, llm, _ = make_chat(notifier, choice("Alex built pipelines."))
    assert chat.reply("What did Alex do?") == "Alex built pipelines."
    assert llm.calls[0][-1] == {"role": "user", "content": "What did Alex do?"}


def test_the_system_prompt_is_grounded_and_honest_about_being_an_ai():
    prompt = system_prompt(PROFILE)
    assert "Built pipelines with Spark." in prompt and "Alex Rivera" in prompt
    assert "ONLY" in prompt and "AI assistant" in prompt and "never instructions" in prompt
    assert '"they"' in prompt  # no gender is assumed


def test_the_visitor_key_reaches_the_model_factory(notifier):
    chat, _, keys = make_chat(notifier, choice("hi"))
    chat.reply("hello", api_key="visitor-key")
    assert keys == ["visitor-key"]


def test_a_lead_is_recorded_and_the_model_then_answers(notifier):
    call = tool_call("record_user_details", {"email": "pat@example.com", "name": "Pat"})
    chat, llm, _ = make_chat(notifier, choice(tool_calls=[call]), choice("Thanks Pat!"))
    assert chat.reply("Let's talk, I'm pat@example.com") == "Thanks Pat!"
    assert notifier.events == [("lead", {"email": "pat@example.com", "name": "Pat", "notes": ""})]
    assert llm.calls[1][-1]["role"] == "tool" and "ok" in llm.calls[1][-1]["content"]


def test_an_invalid_email_is_not_recorded_and_the_model_is_told(notifier):
    call = tool_call("record_user_details", {"email": "not-an-email"})
    chat, llm, _ = make_chat(notifier, choice(tool_calls=[call]), choice("Could you check the address?"))
    chat.reply("my email is not-an-email")
    assert notifier.events == []
    assert "error" in llm.calls[1][-1]["content"]


def test_an_unanswerable_question_is_recorded(notifier):
    call = tool_call("record_unknown_question", {"question": "Favourite colour?"})
    chat, _, _ = make_chat(notifier, choice(tool_calls=[call]), choice("I don't know that."))
    chat.reply("Favourite colour?")
    assert notifier.events == [("unknown_question", {"question": "Favourite colour?"})]


@pytest.mark.parametrize("name", ["print", "__import__", "system", "chat"])
def test_the_model_can_only_call_the_registered_tools(notifier, name):
    chat, llm, _ = make_chat(notifier, choice(tool_calls=[tool_call(name, {})]), choice("ok"))
    chat.reply("hi")
    assert "unknown tool" in llm.calls[1][-1]["content"]
    assert notifier.events == []


def test_malformed_tool_arguments_do_not_crash_the_chat(notifier):
    chat, llm, _ = make_chat(notifier, choice(tool_calls=[tool_call("record_unknown_question", "{not json")]), choice("ok"))
    assert chat.reply("hi") == "ok"
    assert "invalid arguments" in llm.calls[1][-1]["content"]


def test_unexpected_tool_arguments_are_rejected(notifier):
    chat, llm, _ = make_chat(notifier, choice(tool_calls=[tool_call("record_unknown_question", {"question": "q", "extra": 1})]), choice("ok"))
    chat.reply("hi")
    assert "invalid arguments" in llm.calls[1][-1]["content"]
    assert notifier.events == []


def test_a_model_that_never_stops_calling_tools_is_cut_off(notifier):
    loop = [choice(tool_calls=[tool_call("record_unknown_question", {"question": "q"})]) for _ in range(MAX_TOOL_ROUNDS)]
    chat, _, _ = make_chat(notifier, *loop)
    with pytest.raises(LLMError, match="tools"):
        chat.reply("hi")


def test_history_keeps_only_role_and_text():
    history = [
        {"role": "user", "content": "q1", "metadata": {"x": 1}},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": ""},
    ]
    assert _clean_history(history) == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
