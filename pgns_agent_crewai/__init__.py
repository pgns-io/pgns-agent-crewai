# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""pgns-agent-crewai — CrewAI adapter for pgns-agent."""

from pgns_agent_crewai._adapter import CrewAIAdapter
from pgns_agent_crewai._version import __version__

__all__ = [
    "CrewAIAdapter",
    "__version__",
]
