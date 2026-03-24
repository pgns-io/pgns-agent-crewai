# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CrewAI adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pgns_agent import Adapter, AgentServer
from pgns_agent_crewai import CrewAIAdapter

# ---------------------------------------------------------------------------
# Helpers — mock CrewAI objects
# ---------------------------------------------------------------------------


def _make_crew_output(
    *,
    raw: str = "Hello from crew",
    tasks_output: list[Any] | None = None,
    token_usage: dict[str, int] | None = None,
) -> MagicMock:
    """Create a mock CrewOutput."""
    result = MagicMock()
    result.raw = raw
    result.model_dump.return_value = {"raw": raw}
    result.tasks_output = tasks_output or []
    result.token_usage = token_usage
    return result


def _make_task_output(
    *, description: str = "research", agent: str = "researcher", raw: str = "done"
) -> MagicMock:
    t = MagicMock()
    t.description = description
    t.agent = agent
    t.raw = raw
    return t


def _make_crew(kickoff_return: Any = None) -> MagicMock:
    crew = MagicMock()
    crew.kickoff_async = AsyncMock(return_value=kickoff_return or _make_crew_output())
    return crew


# ---------------------------------------------------------------------------
# Adapter base class contract
# ---------------------------------------------------------------------------


class TestAdapterContract:
    def test_is_adapter_subclass(self) -> None:
        adapter = CrewAIAdapter(crew=_make_crew())
        assert isinstance(adapter, Adapter)


# ---------------------------------------------------------------------------
# handle() — kickoff_async path
# ---------------------------------------------------------------------------


class TestHandle:
    async def test_basic_run(self) -> None:
        crew = _make_crew(_make_crew_output(raw="result text"))
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({"topic": "webhooks"})

        crew.kickoff_async.assert_awaited_once_with(inputs={"topic": "webhooks"})
        assert result["output"] == {"raw": "result text"}

    async def test_tasks_output_in_metadata(self) -> None:
        tasks = [_make_task_output(description="step 1", agent="agent-a", raw="done")]
        crew = _make_crew(_make_crew_output(tasks_output=tasks))
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})

        assert "metadata" in result
        assert result["metadata"]["tasks"] == [
            {"description": "step 1", "agent": "agent-a", "raw": "done"}
        ]

    async def test_token_usage_dict(self) -> None:
        usage = {"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40}
        crew = _make_crew(_make_crew_output(token_usage=usage))
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})

        assert result["metadata"]["token_usage"] == usage

    async def test_token_usage_object(self) -> None:
        usage = MagicMock()
        usage.total_tokens = 100
        usage.prompt_tokens = 60
        usage.completion_tokens = 40
        crew = _make_crew(_make_crew_output(token_usage=usage))
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})

        assert result["metadata"]["token_usage"]["total_tokens"] == 100

    async def test_no_tasks_no_metadata_tasks_key(self) -> None:
        crew = _make_crew(_make_crew_output(tasks_output=[]))
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})

        assert "tasks" not in result.get("metadata", {})

    async def test_string_output_passthrough(self) -> None:
        crew = _make_crew(kickoff_return="plain string")
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})

        assert result["output"] == "plain string"

    async def test_dict_output_passthrough(self) -> None:
        crew = _make_crew(kickoff_return={"key": "value"})
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})

        assert result["output"] == {"key": "value"}


# ---------------------------------------------------------------------------
# _normalize_output edge cases
# ---------------------------------------------------------------------------


class TestNormalizeOutput:
    async def test_none_passthrough(self) -> None:
        crew = _make_crew(kickoff_return=None)
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})
        assert result["output"] is None

    async def test_list_passthrough(self) -> None:
        crew = _make_crew(kickoff_return=[1, 2, 3])
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})
        assert result["output"] == [1, 2, 3]

    async def test_raw_attr_extraction(self) -> None:
        obj = MagicMock(spec=[])  # no model_dump
        obj.raw = "raw text"
        crew = _make_crew(kickoff_return=obj)
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})
        assert result["output"] == "raw text"

    async def test_str_fallback(self) -> None:
        class Custom:
            def __str__(self) -> str:
                return "custom-str"

        crew = _make_crew(kickoff_return=Custom())
        adapter = CrewAIAdapter(crew=crew)
        result = await adapter.handle({})
        assert result["output"] == "custom-str"


# ---------------------------------------------------------------------------
# Integration with AgentServer.use()
# ---------------------------------------------------------------------------


class TestAgentServerIntegration:
    def test_registers_default_handler(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(CrewAIAdapter(crew=_make_crew()))
        assert "default" in agent.handlers

    def test_registers_named_skill(self) -> None:
        agent = AgentServer("a", "b")
        agent.use(CrewAIAdapter(crew=_make_crew()), skill="crewai")
        assert "crewai" in agent.handlers
