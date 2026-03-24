# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""CrewAI adapter for pgns-agent."""

from __future__ import annotations

__all__ = ["CrewAIAdapter"]

from typing import TYPE_CHECKING, Any

from pgns_agent import Adapter

if TYPE_CHECKING:
    from crewai import Crew


def _normalize_output(result: Any) -> Any:
    """Convert a CrewAI output to a JSON-serializable value.

    Handles the common return types from CrewAI:

    * Objects with a ``.raw`` attribute — ``CrewOutput`` / ``TaskOutput``;
      extracts the raw string.
    * Objects with a ``.model_dump()`` method — Pydantic models from
      structured output mode.
    * ``dict``, ``str``, and other primitives — passed through as-is.
    * Everything else — ``str()`` fallback.
    """
    if isinstance(result, dict | str | int | float | bool | list | type(None)):
        return result
    # Pydantic structured-output objects (e.g. crew.kickoff() with output_pydantic=...).
    # CrewOutput itself has model_dump() but is handled here before _build_result
    # reads its .raw — this branch targets task-level Pydantic outputs.
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "raw"):
        return result.raw
    return str(result)


def _build_result(result: Any) -> dict[str, Any]:
    """Build a pgns-agent result dict from a CrewAI output.

    Extracts task-level outputs into metadata when available (``CrewOutput``
    exposes a ``tasks_output`` list with per-task results).
    """
    out: dict[str, Any] = {"output": _normalize_output(result)}
    if hasattr(result, "tasks_output") and result.tasks_output:
        out["metadata"] = {
            "tasks": [
                {
                    "description": getattr(t, "description", None),
                    "agent": getattr(t, "agent", None),
                    "raw": getattr(t, "raw", None),
                }
                for t in result.tasks_output
            ],
        }
    if hasattr(result, "token_usage") and result.token_usage:
        usage = result.token_usage
        if isinstance(usage, dict):
            out.setdefault("metadata", {})["token_usage"] = usage
        elif hasattr(usage, "total_tokens"):
            out.setdefault("metadata", {})["token_usage"] = {
                "total_tokens": usage.total_tokens,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
    return out


class CrewAIAdapter(Adapter):
    """Wraps a CrewAI ``Crew`` for single-shot execution via pgns-agent.

    Calls the crew's :meth:`~crewai.Crew.kickoff` method with the task input
    as the crew's input variables and returns the result as a pgns-agent task
    result dict.

    Example::

        from crewai import Agent, Crew, Task
        from pgns_agent_crewai import CrewAIAdapter

        researcher = Agent(role="Researcher", goal="Find facts", ...)
        task = Task(description="Research {topic}", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[task])

        server = AgentServer("my-crew", "A CrewAI-powered agent")
        server.use(CrewAIAdapter(crew))
        server.listen(3000)

    Args:
        crew: A configured ``crewai.Crew`` instance.
    """

    def __init__(self, crew: Crew) -> None:
        self._crew = crew

    async def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:
        result = await self._crew.kickoff_async(inputs=task_input)
        return _build_result(result)
