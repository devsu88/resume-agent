import json
from types import SimpleNamespace

import pytest


def choice(content=None, tool_calls=None):
    """An OpenAI-style chat choice."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")


def tool_call(name, arguments, call_id="call_1"):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


class ScriptedLLM:
    """Replays prepared choices and records every request."""

    def __init__(self, *choices):
        self.script = list(choices)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        return self.script.pop(0)


class RecordingNotifier:
    def __init__(self):
        self.events = []

    def notify(self, kind, payload):
        self.events.append((kind, payload))


@pytest.fixture
def notifier():
    return RecordingNotifier()
