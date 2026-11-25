from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tools.add_memory import AddMemoryTool


def _tool() -> AddMemoryTool:
    return AddMemoryTool(runtime=MagicMock(), session=MagicMock())


def test_build_messages_parses_json_array() -> None:
    tool = _tool()
    raw_messages = json.dumps(
        [
            ["user", " Hi "],
            ["assistant", "Hello"],
        ]
    )

    assert tool._build_messages({"messages": raw_messages}) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


@pytest.mark.parametrize("raw_messages", ["not JSON", '{"role":"user","message":"Hi"}'])
def test_build_messages_rejects_invalid_json_input(raw_messages: str) -> None:
    tool = _tool()

    with pytest.raises(ValueError, match="messages must be"):
        tool._build_messages({"messages": raw_messages})


@pytest.mark.parametrize(
    "raw_messages",
    [
        '[["system","Hi"]]',
        '[["user"]]',
        '[["assistant",1]]',
        '[{"role":"user","message":"Hi"}]',
    ],
)
def test_build_messages_rejects_invalid_tuple_items(raw_messages: str) -> None:
    tool = _tool()

    with pytest.raises(ValueError, match=r"messages\[0\]"):
        tool._build_messages({"messages": raw_messages})
