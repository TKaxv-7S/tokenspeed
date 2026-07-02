#!/usr/bin/env python3
"""Run EvalScope with prefix-stable plain completion prompts."""

from typing import Any

from evalscope.cli.cli import run_cmd
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.registry import register_api

API_NAME = "openai_plain_completion"


def serialize_plain_messages(messages: list[dict[str, Any]]) -> str:
    """Join message contents while preserving the prior generation boundary."""
    if not messages:
        raise ValueError("plain completion serialization requires messages")

    contents = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be a dict")
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError(f"message {index} content must be a string")
        contents.append(content)
    return "\n".join(contents) + "\n"


@register_api(API_NAME)
class PlainCompletionOpenaiPlugin(OpenaiPlugin):
    """Send message contexts as prefix-stable OpenAI completion prompts."""

    def build_request(self, messages, param: Arguments | None = None):
        param = param or self.param
        if (
            not param.tokenize_prompt
            and not param.apply_chat_template
            and param.query_template is None
            and isinstance(messages, list)
            and messages
            and isinstance(messages[0], dict)
        ):
            messages = serialize_plain_messages(messages)
        return super().build_request(messages, param)


if __name__ == "__main__":
    run_cmd()
