"""Repeat counters must leave terminal progress closing fences intact."""

import asyncio
import importlib
import itertools
import sys
import threading
import types

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import CodeBlockProgressAdapter, _make_runner


COMMAND = "cd /tmp/example"
FENCED = f"💻 terminal\n```\n{COMMAND}\n```"


async def _run_repeated_progress(
    monkeypatch, tmp_path, *, drain, tool_name, preview, args, repetitions, count
):
    """Exercise the real sender, holding its adapter boundary for final drain."""
    loop = asyncio.get_running_loop()
    first_sent = threading.Event()
    counter_sent = threading.Event()
    release_sender = asyncio.Event()
    counter = f"(×{count})"

    class CaptureAdapter(CodeBlockProgressAdapter):
        draining = False
        counted_text = None

        async def send_typing(self, chat_id, metadata=None):
            # This callback runs after the initial send assigned a message id.
            # Block here so the worker can enqueue repeats deterministically.
            first_sent.set()
            try:
                await release_sender.wait()
            except asyncio.CancelledError:
                self.draining = True
                raise

        async def edit_message(self, chat_id, message_id, content):
            result = await super().edit_message(chat_id, message_id, content)
            if counter in content:
                self.counted_text = content
                counter_sent.set()
            return result

    class RepeatingAgent:
        def __init__(self, **kwargs):
            self.tool_progress_callback = kwargs.get("tool_progress_callback")
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None):
            callback = self.tool_progress_callback
            callback("tool.started", tool_name, preview, args)
            assert first_sent.wait(10), "initial progress was not delivered"
            for _ in range(repetitions - 1):
                callback("tool.started", tool_name, preview, args)
            if not drain:
                loop.call_soon_threadsafe(release_sender.set)
                # The normal-path counter must be emitted before completion
                # can cancel the sender and execute the final-drain branch.
                assert counter_sent.wait(10), "normal progress counter was not delivered"
            return {"final_response": "done", "messages": [], "api_calls": 1}

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = RepeatingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal display metadata

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    # Make every edit eligible without changing asyncio's global clock.
    clock = types.SimpleNamespace(**vars(gateway_run.time))
    ticks = itertools.count(100.0, 2.0)
    clock.monotonic = lambda: next(ticks)
    monkeypatch.setattr(gateway_run, "time", clock)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
        session_id="sess-progress-dedup",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "done"
    assert adapter.draining is drain
    assert adapter.counted_text is not None
    return adapter


def _assert_fenced_counter(adapter):
    assert adapter.sent[0]["content"] == FENCED
    actual = adapter.counted_text
    assert actual.endswith("```\n(×2)"), actual
    assert all(line == "```" for line in actual.splitlines() if line.startswith("```"))
    assert actual.count(COMMAND) == 2


@pytest.mark.asyncio
async def test_normal_progress_counter_preserves_code_fence(monkeypatch, tmp_path):
    adapter = await _run_repeated_progress(
        monkeypatch, tmp_path, drain=False, tool_name="terminal", preview=COMMAND,
        args={"command": COMMAND}, repetitions=3, count=2,
    )
    _assert_fenced_counter(adapter)


@pytest.mark.asyncio
async def test_final_drain_counter_preserves_code_fence(monkeypatch, tmp_path):
    adapter = await _run_repeated_progress(
        monkeypatch, tmp_path, drain=True, tool_name="terminal", preview=COMMAND,
        args={"command": COMMAND}, repetitions=3, count=2,
    )
    _assert_fenced_counter(adapter)


@pytest.mark.asyncio
async def test_plain_progress_counters_stay_inline(monkeypatch, tmp_path):
    for drain, count in ((False, 3), (True, 2)):
        adapter = await _run_repeated_progress(
            monkeypatch, tmp_path, drain=drain, tool_name="execute_code", preview=None,
            args={}, repetitions=count, count=count,
        )
        original = adapter.sent[0]["content"]
        assert "execute_code" in original
        assert adapter.counted_text == f"{original} (×{count})"
