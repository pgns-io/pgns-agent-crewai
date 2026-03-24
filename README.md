# pgns-agent-crewai

CrewAI adapter for [pgns-agent](https://pypi.org/project/pgns-agent/). Run CrewAI crews as pgns agents with automatic correlation ID propagation and signed webhook delivery.

## Quick Start

```python
from crewai import Agent, Crew, Task
from pgns_agent import AgentServer
from pgns_agent_crewai import CrewAIAdapter

# Define your CrewAI crew
researcher = Agent(role="Researcher", goal="Find key facts about a topic")
task = Task(description="Research {topic}", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])

# Wrap it in a pgns agent
server = AgentServer("my-crew", "A CrewAI-powered research agent")
server.use(CrewAIAdapter(crew))
server.listen(3000)
```

## How It Works

The adapter calls `crew.kickoff(inputs=task_input)` with the incoming task payload as input variables. The `CrewOutput` is normalized into a pgns-agent result dict with:

- `output` — the crew's raw output string (or structured Pydantic model)
- `metadata.tasks` — per-task results (description, agent, raw output)
- `metadata.token_usage` — aggregate token usage when available

## Installation

```bash
pip install pgns-agent-crewai
```

## License

Apache-2.0
