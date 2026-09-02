<div align="center">

# Indoor Emergency System
### Adaptive Multi-Hazard Emergency Response Framework for Indoor Environments

An asynchronous, LLM-driven agent swarm for real-time fire and earthquake risk assessment in smart buildings

[![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/dependency%20manager-uv-de5fe9)](https://docs.astral.sh/uv/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1c3c3c)](https://github.com/langchain-ai/langgraph)
[![Camunda](https://img.shields.io/badge/workflow-Camunda-fb610b)](https://camunda.com/)
[![Memgraph](https://img.shields.io/badge/graph%20db-Memgraph-ff5000)](https://memgraph.com/)
[![Redis](https://img.shields.io/badge/queue-Redis-dc382d)](https://redis.io/)
[![Node--RED](https://img.shields.io/badge/ingestion-Node--RED-8f0000)](https://nodered.org/)
[![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-2496ed)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)]()
[![License](https://img.shields.io/badge/license-unspecified-lightgrey.svg)]()

</div>

> **Research project / prototype.** This system is not intended to be the sole safety mechanism of a real building and should not replace certified fire/seismic safety infrastructure.

---

## Description

Traditional building safety systems rely on single-hazard, threshold-based logic: a smoke detector triggers an alarm, a seismic sensor trips a cutoff - each in isolation, with no shared understanding of the building's spatial structure or of how one hazard might interact with or accelerate another. The Emergency Agent Worker implements an adaptive, multi-hazard emergency response framework for indoor environments, designed to move beyond isolated threshold alarms toward a context-aware, graph-informed reasoning process.

The framework treats hazard detection as a collaborative reasoning problem rather than a static rule-evaluation problem. Real-time telemetry from IoT sensors (temperature, smoke, gas, vibration, and other environmental signals) is streamed through Node-RED, buffered in a Redis priority queue, and consumed by an asynchronous Python worker. Each room's sensor history is handed to a swarm of language model agents, orchestrated with LangGraph using a map-reduce (triage → specialized experts) pattern:

- A triage agent performs an initial, lightweight assessment of each room, cross-referencing live telemetry against the building's topology.
- Only when an anomaly is detected does the system escalate to domain-specific expert agents - currently a fire safety agent and an earthquake safety agent - which reason more deeply about hazard-specific dynamics.
- Both triage and expert agents are grounded in a Memgraph graph database encoding the building's topology (rooms, adjacency, ventilation paths, structural dependencies), allowing the agents to reason about how a hazard could propagate between physically or structurally connected spaces - a core requirement for adaptive, spatially-aware indoor emergency response.
- The resulting structured risk assessments are published to an MQTT topic, ready for consumption by alerting dashboards, building management systems, or evacuation-guidance applications.

This escalate-only-when-needed design reduces unnecessary inference cost while preserving deeper, explainable reasoning for genuinely ambiguous or dangerous situations - a practical trade-off central to deploying adaptive reasoning systems in latency- and resource-constrained indoor monitoring contexts.

---

## Key Features

- Multi-agent hazard reasoning - a triage agent and specialized fire/earthquake expert agents built with LangGraph, following a map-reduce escalation pattern.
- Topology-aware risk propagation - hazard reasoning is grounded in a Memgraph graph of the building's rooms, adjacencies, and structural paths.
- Real-time IoT ingestion - sensor data collected via Node-RED (e.g. from ThingsBoard) and buffered as timestamp-ordered snapshots in a Redis priority queue (`room_pq`).
- Asynchronous worker core - non-blocking Python worker (`main.py`).
- Pluggable LLM backend - any OpenAI-compatible endpoint (e.g. OpenRouter), with configurable primary and fallback models.
- Custom Memgraph tool layer - purpose-built tools (`memgraph_custom_tools.py`) exposed to agents for querying building topology and current sensor state.
- Structured MQTT output - each assessment is a validated `ThreatAssessment` object (room, warning level, danger level/type, score, justification).
- Containerized supporting stack - Docker Compose definitions for Node-RED, Redis, Memgraph (MAGE), Memgraph Lab, and Memgraph MCP.

---

## Tech Stack

| Component | Role |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://github.com/langchain-ai/langchain) | Orchestration of the agent swarm (triage → experts) |
| LLM via OpenAI-compatible API (e.g. [OpenRouter](https://openrouter.ai/)) | Agent reasoning |
| [Memgraph](https://memgraph.com/) (+ MAGE / Lab / MCP) | Building topology graph, queried by the agents |
| [Redis](https://redis.io/) | Priority queue of sensor snapshots to process |
| [Node-RED](https://nodered.org/) | Ingestion/routing of IoT sensor data |
| Python ≥ 3.14, [uv](https://docs.astral.sh/uv/) | Runtime and dependency management |

---

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- Docker and Docker Compose for the supporting services (Redis, Memgraph, Node-RED, etc.)
- An OpenAI-compatible API key for the LLM (e.g. OpenRouter)
- A reachable MQTT broker for publishing results

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nicol-buratti/multi-hazard-emergency-framework.git
cd multi-hazard-emergency-framework

# 2. Install Python dependencies with uv
uv sync

# 3. Copy the environment template and configure it
cp .env.example .env
```

### Configuration

Edit `.env` with your environment values. Key variables (see `app_settings.py` and `main.py`):

| Variable | Description | Default |
|---|---|---|
| `LLM_MODEL` | Primary LLM model (OpenRouter-compatible format) | `gpt-4` |
| `LLM_API_KEY` | API key for the LLM provider | *required* |
| `LLM_BASE_URL` | Base URL of the OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `EXTRA_LLM_MODELS` | JSON list of fallback/alternative models | - |
| `REDIS_HOST` | Redis server host | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `QUEUE_NAME` | Name of the Redis priority queue to consume | `room_pq` |
| `MQTT_BROKER_ADDRESS` | MQTT broker address | `localhost` |
| `MQTT_PUBLISH_TOPIC` | MQTT topic to publish results to | `emergency/agent/data` |

### Start the supporting services

```bash
docker compose up -d
```

This launches Node-RED, Redis, Memgraph (MAGE), Memgraph Lab and the Memgraph MCP server.

Note: some containers use `network_mode: host` to reach sensor data over a VPN during development - adjust for your environment.

| Service | Port | Notes |
|---|---|---|
| Node-RED | `1880` | Ingestion flows UI |
| Redis | `6379` | Priority queue |
| Memgraph (Bolt) | `7687` | Graph database |
| Memgraph (log/monitor) | `7444` | - |
| Memgraph Lab | `3000` | Graph administration UI |
| Memgraph MCP | `8001` | Read-only MCP tool for querying Memgraph |

---

## Usage

Once the supporting services are running and `.env` is configured, start the worker:

```bash
uv run main.py
```

The worker:
1. Pops the highest-priority room from the Redis queue (`QUEUE_NAME`).
2. Rebuilds the sensor snapshot history for that room.
3. Passes it to the LangGraph agent swarm (triage → fire/earthquake experts if escalated).
4. Publishes the resulting risk assessment to the configured MQTT topic.

Shutdown is handled gracefully on `SIGINT` / `SIGTERM`.

### Example output

Each assessment published to MQTT follows the `ThreatAssessment` schema:

```json
{
  "room": "room-101",
  "warning": "none | pre-alert",
  "danger": "none | low | medium | high",
  "danger_type": "none | fire | smoke | heat | earthquake | other",
  "danger_score": 0.0,
  "justification": "Brief justification for the assessment"
}
```

### Loading building topology

The building's spatial and structural model is described in `place_graph.json` and loaded into Memgraph, enabling propagation-aware reasoning across adjacent rooms and shared ventilation/structural paths.

---

## Project Structure
```
multi-hazard-emergency-framework/
├── .vscode/ # Editor configuration
├── camunda/ # Process-orchestration / workflow definitions
├── node-red/ # Node-RED ingestion flows and configuration
│ ├── nodered_config.yml # Node-RED flow configuration
│ └── node_red_data/ # Persistent Node-RED container data
├── worker-ai/ # Core Python agent worker
│ ├── main.py # Worker entry point: Redis consumption, orchestration, MQTT publishing
│ ├── agent_graph.py # LangGraph graph definition (HazardMapReduceManager): states, schemas, triage/fire/earthquake nodes
│ ├── agent_swarm_prebuilts.py # Prebuilt components/agents reused within the swarm
│ ├── memgraph_custom_tools.py # Custom tools exposed to agents for querying Memgraph
│ ├── app_settings.py # Application configuration (pydantic-settings, reads .env)
│ ├── place_graph.json # Building topology dataset loaded into Memgraph
│ └── .env.example # Example required environment variables
├── .pre-commit-config.yaml # Pre-commit hook definitions
├── .secrets.baseline # Baseline for secrets-scanning tooling
├── docker-compose.yml # Supporting services stack (Node-RED, Redis, Memgraph, Memgraph Lab, Memgraph MCP)
└── README.md # Project documentation
```

---

## License

No license file is currently present in this repository. Until one is added, all rights are reserved by the repository owner, and no reuse, modification, or distribution rights are implicitly granted. Contributors and users should contact the maintainer for clarification before reuse.

---

<div align="center">

Built as part of research into adaptive multi-hazard emergency response for indoor environments.

</div>
