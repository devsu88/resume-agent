"""Where the assistant reports leads (visitors who left an email) and questions it could not answer."""

import json
import os
import time
from pathlib import Path

import httpx


class JsonlNotifier:
    """Appends one JSON line per event to a local file."""

    def __init__(self, path="leads.jsonl"):
        self.path = Path(path)

    def notify(self, kind, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **payload}, ensure_ascii=False) + "\n")


class PushoverNotifier:
    """Also sends a push notification through Pushover. A failure never breaks the conversation."""

    def __init__(self, token, user, timeout=5):
        self.token, self.user, self.timeout = token, user, timeout

    def notify(self, kind, payload):
        text = f"{kind}: " + ", ".join(f"{k}={v}" for k, v in payload.items())
        try:
            httpx.post(
                "https://api.pushover.net/1/messages.json",
                data={"token": self.token, "user": self.user, "message": text[:1000]},
                timeout=self.timeout,
            )
        except httpx.HTTPError:
            pass


class MultiNotifier:
    def __init__(self, *notifiers):
        self.notifiers = notifiers

    def notify(self, kind, payload):
        for notifier in self.notifiers:
            notifier.notify(kind, payload)


def notifier_from_env(leads_file="leads.jsonl"):
    notifiers = [JsonlNotifier(leads_file)]
    if os.getenv("PUSHOVER_TOKEN") and os.getenv("PUSHOVER_USER"):
        notifiers.append(PushoverNotifier(os.environ["PUSHOVER_TOKEN"], os.environ["PUSHOVER_USER"]))
    return MultiNotifier(*notifiers)
