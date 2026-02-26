---
title: "Agentic SDKs in 2025: OpenAI Agents SDK & Google ADK — Build AI Agents with Python"
date: 2025-03-11
tags: Agents, OpenAI, Google, SDK, Python, LLM, Multi-Agent, Orchestration, Production
summary: "Deep dive into the two major agentic SDKs of 2025: OpenAI Agents SDK and Google Agent Development Kit (ADK). Architecture, primitives, multi-agent orchestration, tool calling, guardrails, tracing, and production patterns — with full Python code examples."
---

# Agentic SDKs in 2025: OpenAI Agents SDK & Google ADK — Build AI Agents with Python

## The Agentic Shift

2025 marks the year AI moves from "chat-with-a-model" to **autonomous agents** that plan, use tools, coordinate with each other, and execute multi-step workflows. Two major players have released production-grade SDKs to make this accessible to Python developers:

- **OpenAI Agents SDK** (March 2025) — the successor to Swarm, now a fully supported production framework
- **Google Agent Development Kit (ADK)** (April 2025) — a modular, model-agnostic framework optimized for Gemini but compatible with any LLM

This article provides a deep technical comparison of both SDKs: their architectures, core primitives, how they handle multi-agent orchestration, tool calling, guardrails, memory, tracing, and deployment. Every concept is illustrated with working Python code.

---

## 1. Installation & First Agent

### OpenAI Agents SDK

```bash
pip install openai-agents
export OPENAI_API_KEY=sk-...
```

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant that answers concisely."
)

result = Runner.run_sync(agent, "What is retrieval-augmented generation?")
print(result.final_output)
```

The SDK is **Python-first**: you use standard Python constructs (functions, classes, async/await) rather than YAML configs or DSLs. The `Runner` manages the agent loop: it sends the prompt to the LLM, processes tool calls, feeds results back, and repeats until the agent produces a final output.

### Google ADK

```bash
pip install google-adk
export GOOGLE_API_KEY=...
```

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

agent = Agent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant that answers concisely.",
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)

session = session_service.create_session(app_name="my_app", user_id="user1")

from google.genai import types
user_msg = types.Content(
    role="user",
    parts=[types.Part(text="What is retrieval-augmented generation?")]
)

for event in runner.run(user_id="user1", session_id=session.id, new_message=user_msg):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

ADK follows a more **enterprise-oriented** pattern with explicit session management, event-driven architecture, and structured content types. It's more verbose but gives you fine-grained control over every step.

---

## 2. Core Primitives Comparison

| Concept | OpenAI Agents SDK | Google ADK |
|---------|-------------------|------------|
| **Agent definition** | `Agent(name, instructions, tools, model)` | `Agent(name, model, instruction, tools)` |
| **Execution** | `Runner.run_sync()` / `Runner.run()` (async) | `runner.run()` (generator of events) |
| **Tool calling** | Python functions with type hints → auto-schema | `FunctionTool`, `AgentTool`, `LongRunningFunctionTool` |
| **Multi-agent** | Handoffs + agents-as-tools | Sub-agents via `sub_agents=[]` parameter |
| **Memory/State** | Sessions (built-in) | `SessionService` (in-memory, database, Vertex AI) |
| **Guardrails** | `InputGuardrail` / `OutputGuardrail` classes | Callback system + `before_model_callback` |
| **Tracing** | Built-in OpenAI tracing dashboard | Cloud Trace integration |
| **Streaming** | `Runner.run_streamed()` | Event stream with `runner.run()` |
| **Voice/Realtime** | `RealtimeAgent` for voice | Bidirectional streaming via Live API |
| **MCP support** | Built-in MCP server tool calling | MCP tool integration |

---

## 3. Tool Calling In Depth

Tools are the fundamental mechanism that transforms a language model from a text generator into an agent. Both SDKs make it remarkably easy to expose Python functions as tools.

### OpenAI: Function Tools

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In production: call a weather API
    return f"The weather in {city} is 22°C and sunny."

@function_tool
def search_database(query: str, limit: int = 5) -> list[dict]:
    """Search the internal knowledge base."""
    # In production: vector search, SQL query, etc.
    return [{"title": "RAG Guide", "score": 0.95}]

agent = Agent(
    name="ResearchAssistant",
    instructions="Help users find information. Use tools when needed.",
    tools=[get_weather, search_database],
)

result = Runner.run_sync(agent, "What's the weather in Tokyo and find me docs about RAG?")
```

The `@function_tool` decorator inspects type hints and docstrings to automatically generate the JSON schema that the LLM uses to decide when and how to call the function. **Pydantic models** are fully supported for complex input/output types.

### Google ADK: FunctionTool

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temperature": "22°C", "condition": "sunny"}

def search_database(query: str, limit: int = 5) -> dict:
    """Search the internal knowledge base."""
    return {"results": [{"title": "RAG Guide", "score": 0.95}]}

agent = Agent(
    name="research_assistant",
    model="gemini-2.0-flash",
    instruction="Help users find information. Use tools when needed.",
    tools=[get_weather, search_database],
)
```

ADK also auto-generates schemas from type hints. Additionally, ADK provides `LongRunningFunctionTool` for tools that take significant time (API calls, file processing) — it automatically handles polling and progress updates.

### MCP Server Integration

Both SDKs support the **Model Context Protocol (MCP)**, allowing agents to connect to external tool servers:

```python
# OpenAI Agents SDK — MCP
from agents import Agent
from agents.mcp import MCPServerStdio

server = MCPServerStdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])

agent = Agent(
    name="FileAgent",
    instructions="Help users manage files.",
    mcp_servers=[server],
)
```

```python
# Google ADK — MCP
from google.adk.tools.mcp_tool import MCPToolset

tools, exit_stack = await MCPToolset.from_server(
    connection_params=StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
)

agent = Agent(name="file_agent", model="gemini-2.0-flash", tools=tools)
```

---

## 4. Multi-Agent Orchestration

This is where agentic SDKs truly shine: composing multiple specialized agents into a coordinated system.

### OpenAI: Handoffs

The OpenAI SDK uses **handoffs** — an agent can delegate the entire conversation to another agent:

```python
from agents import Agent, Runner

triage_agent = Agent(
    name="Triage",
    instructions="""You are a triage agent. Based on the user's question:
    - Technical questions → hand off to TechAgent
    - Billing questions → hand off to BillingAgent
    - Otherwise, answer directly.""",
    handoffs=["tech_agent", "billing_agent"],
)

tech_agent = Agent(
    name="TechAgent",
    instructions="You are a senior engineer. Answer technical questions in detail.",
    tools=[search_documentation],
)

billing_agent = Agent(
    name="BillingAgent",
    instructions="You handle billing and subscription questions.",
    tools=[lookup_invoice, process_refund],
)

# Wire up handoffs
triage_agent.handoffs = [tech_agent, billing_agent]

result = Runner.run_sync(triage_agent, "My GPU instance won't start, error code E-4012")
print(result.final_output)  # Answered by TechAgent
```

Handoffs transfer full control: the receiving agent gets the conversation history and takes over. The triage agent doesn't get control back unless explicitly handed back.

You can also use **agents-as-tools**, where the parent agent calls a sub-agent like a function and gets the result back:

```python
research_agent = Agent(
    name="Researcher",
    instructions="Find relevant papers on a given topic.",
    tools=[arxiv_search],
)

writer_agent = Agent(
    name="Writer",
    instructions="Write clear technical summaries. Use the Researcher tool to find sources first.",
    tools=[research_agent.as_tool(
        tool_name="research",
        tool_description="Search academic papers on a topic"
    )],
)
```

### Google ADK: Sub-Agents & Sequential/Parallel Agents

ADK uses a **hierarchical** approach with explicit sub-agent declarations:

```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent

researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="Research the given topic thoroughly.",
    tools=[web_search, arxiv_search],
)

writer = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="Write a clear technical summary based on the research.",
)

# Sequential: researcher runs first, writer uses the output
pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[researcher, writer],
)

# Or parallel: both agents run simultaneously
fact_checker = Agent(name="fact_checker", model="gemini-2.0-flash", ...)
parallel = ParallelAgent(
    name="parallel_review",
    sub_agents=[writer, fact_checker],
)
```

ADK also supports **LoopAgent** for iterative refinement patterns (agent A produces output → agent B critiques → agent A refines → repeat until quality threshold met).

```python
from google.adk.agents import LoopAgent

refiner = LoopAgent(
    name="iterative_refiner",
    sub_agents=[draft_agent, critic_agent],
    max_iterations=3,
)
```

---

## 5. Guardrails & Safety

### OpenAI: Input/Output Guardrails

```python
from agents import Agent, Runner, InputGuardrail, OutputGuardrail, GuardrailFunctionOutput

# Input guardrail: runs in parallel with the agent
async def check_prompt_injection(ctx, agent, input_text):
    # Use a classifier or another LLM to check
    result = await Runner.run(
        injection_detector_agent,
        input_text,
        context=ctx,
    )
    is_injection = result.final_output.lower().startswith("yes")
    return GuardrailFunctionOutput(
        output_info={"is_injection": is_injection},
        tripwire_triggered=is_injection,
    )

agent = Agent(
    name="SafeAssistant",
    instructions="You are a helpful assistant.",
    input_guardrails=[
        InputGuardrail(guardrail_function=check_prompt_injection)
    ],
)
```

Guardrails run **in parallel** with the main agent execution. If a guardrail triggers, execution is immediately halted — this means you get both low latency (no sequential check) and safety.

### Google ADK: Callbacks

```python
from google.adk.agents import Agent
from google.genai import types

def safety_callback(callback_context, llm_request):
    """Check input before sending to model."""
    user_text = llm_request.contents[-1].parts[0].text
    if "ignore previous instructions" in user_text.lower():
        return types.Content(
            role="model",
            parts=[types.Part(text="I cannot process that request.")]
        )
    return None  # None = proceed normally

agent = Agent(
    name="safe_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
    before_model_callback=safety_callback,
)
```

ADK's callback system is synchronous and integrates at multiple points: `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`.

---

## 6. Memory & Sessions

### OpenAI Sessions

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You remember previous interactions in this session.",
)

# First turn
result1 = await Runner.run(agent, "My name is Michail and I work on RAG systems.")

# Second turn — same thread, agent remembers context
result2 = await Runner.run(
    agent,
    "What do I work on?",
    previous_response_id=result1.last_response_id,
)
print(result2.final_output)  # "You work on RAG systems."
```

### Google ADK Sessions

ADK provides multiple session backends:

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

# In-memory (development)
session_service = InMemorySessionService()

# Database-backed (production)
session_service = DatabaseSessionService(db_url="postgresql://...")

# Create a session with initial state
session = session_service.create_session(
    app_name="mirror",
    user_id="user1",
    state={"preferences": {"language": "fr"}, "history_count": 0}
)
```

ADK sessions carry **state dictionaries** that persist across turns, enabling complex stateful workflows.

---

## 7. Tracing & Observability

### OpenAI Tracing

Every agent run automatically generates a trace viewable in the **OpenAI Dashboard**:

```python
from agents import Agent, Runner, trace

# Custom trace grouping
with trace("research-workflow"):
    result = await Runner.run(triage_agent, user_query)

# Traces capture: agent calls, tool invocations, LLM requests/responses,
# handoff decisions, guardrail results, timing, token usage
```

Traces feed directly into OpenAI's evaluation and fine-tuning pipeline — you can use production traces to evaluate agent quality and distill behavior into smaller models.

### Google ADK Tracing

ADK integrates with **Google Cloud Trace** and **OpenTelemetry**:

```python
# Automatic tracing when deployed to Vertex AI Agent Engine
# For local development, use the ADK web UI:
# adk web -- starts a local dev server with trace visualization
```

---

## 8. Streaming & Realtime

### OpenAI Streaming

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are helpful.")

async for event in Runner.run_streamed(agent, "Explain transformers"):
    if event.type == "raw_response_event":
        # Token-level streaming
        print(event.data, end="", flush=True)
    elif event.type == "agent_updated_stream_event":
        print(f"\n[Agent: {event.new_agent.name}]")
    elif event.type == "run_item_stream_event":
        # Tool calls, handoffs
        print(f"\n[Event: {event.item.type}]")
```

### OpenAI Realtime Voice Agents

```python
from agents.voice import RealtimeAgent, RealtimeSession

agent = RealtimeAgent(
    name="VoiceAssistant",
    instructions="You are a friendly voice assistant.",
    tools=[get_weather, search_calendar],
)

async with RealtimeSession(agent) as session:
    # Handles: audio input → transcription → agent → TTS → audio output
    # Built-in: interruption detection, turn-taking, context management
    await session.run()
```

---

## 9. Production Patterns

### Pattern 1: RAG Agent with Source Citations

```python
from agents import Agent, Runner, function_tool

@function_tool
def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """Search the vector store for relevant documents."""
    results = qdrant_client.search(collection="docs", query=embed(query), limit=top_k)
    return [{"text": r.payload["text"], "source": r.payload["source"],
             "page": r.payload["page"], "score": r.score} for r in results]

rag_agent = Agent(
    name="RAGAssistant",
    instructions="""Answer questions using the knowledge base.
    Always search first. Cite sources as [Source: name, p.X].
    If no relevant documents found, say so.""",
    tools=[search_knowledge_base],
    model="gpt-4o",
)
```

### Pattern 2: Agentic Pipeline with Validation

```python
from agents import Agent, Runner, OutputGuardrail, GuardrailFunctionOutput

# Step 1: Research agent gathers information
researcher = Agent(
    name="Researcher",
    instructions="Find all relevant information about the query.",
    tools=[web_search, database_search],
)

# Step 2: Writer synthesizes
writer = Agent(
    name="Writer",
    instructions="Write a comprehensive answer based on the Researcher's findings.",
    tools=[researcher.as_tool("research", "Research a topic")],
)

# Step 3: Fact-checker validates
async def fact_check(ctx, agent, output):
    check = await Runner.run(fact_checker_agent, f"Verify: {output}")
    is_valid = "VALID" in check.final_output
    return GuardrailFunctionOutput(
        output_info={"valid": is_valid},
        tripwire_triggered=not is_valid,
    )

writer.output_guardrails = [OutputGuardrail(guardrail_function=fact_check)]
```

### Pattern 3: Human-in-the-Loop

```python
# OpenAI SDK
from agents import Agent, Runner

@function_tool
def approve_action(action: str, details: str) -> str:
    """Request human approval for a sensitive action."""
    # In production: send notification, wait for approval via webhook
    print(f"APPROVAL NEEDED: {action} — {details}")
    response = input("Approve? (yes/no): ")
    return f"{'Approved' if response == 'yes' else 'Rejected'}"

agent = Agent(
    name="CautiousAgent",
    instructions="Before executing any destructive action, use the approve_action tool.",
    tools=[approve_action, delete_resource, modify_config],
)
```

---

## 10. Deployment

### OpenAI Agents SDK

The SDK is a pure Python library — deploy anywhere Python runs:

- **FastAPI / Flask** wrapper for HTTP API
- **Docker** containerization
- **Serverless** (AWS Lambda, Cloud Functions)
- Direct integration with **OpenAI's hosted infrastructure** for managed tracing and evaluation

### Google ADK

ADK provides first-class deployment to **Vertex AI Agent Engine**:

```python
# Deploy to Vertex AI
from google.adk.deploy import VertexAIDeployer

deployer = VertexAIDeployer(project="my-project", location="us-central1")
deployer.deploy(agent=my_agent, display_name="production-agent")
```

It also runs anywhere as a standard Python app, with a built-in local dev server (`adk web`) for testing.

---

## 11. When to Choose Which

| Criteria | OpenAI Agents SDK | Google ADK |
|----------|-------------------|------------|
| **Best for** | OpenAI API users, rapid prototyping | Google Cloud users, enterprise multi-agent |
| **Model lock-in** | OpenAI models (GPT-4o, o3) | Model-agnostic (Gemini, OpenAI, Anthropic, local) |
| **Learning curve** | Very low — minimal primitives | Medium — more concepts but more control |
| **Multi-agent** | Handoffs + agents-as-tools | Sub-agents, Sequential, Parallel, Loop agents |
| **Voice/Realtime** | RealtimeAgent (built-in) | Live API integration |
| **Deployment** | Any Python environment | Vertex AI Agent Engine + any environment |
| **Pricing** | OpenAI API costs | Gemini API costs (or self-hosted models) |
| **Tracing** | OpenAI Dashboard (built-in) | Cloud Trace / OpenTelemetry |
| **MCP support** | Yes | Yes |
| **Open source** | MIT License | Apache 2.0 |

### Recommendation

- **Start with OpenAI Agents SDK** if you're already using GPT-4o/o3, want the fastest path to a working agent, and prefer minimal boilerplate.
- **Start with Google ADK** if you need model flexibility (swap between Gemini, GPT, Claude, local models), complex multi-agent orchestration patterns (sequential, parallel, loop), or deep Google Cloud integration.
- **Use both**: they're not mutually exclusive. ADK can call OpenAI models, and both support MCP for shared tooling.

---

## 12. Other Notable Agentic Frameworks (2025)

The SDK landscape is rich:

- **LangGraph** (LangChain) — graph-based agent orchestration with state machines, checkpointing, and human-in-the-loop. More complex but extremely flexible for custom topologies.
- **CrewAI** — role-based multi-agent framework. Agents have roles, goals, and backstories. Good for structured team simulations.
- **AutoGen** (Microsoft) — conversation-based multi-agent with code execution. Strong for research and code-heavy workflows.
- **Semantic Kernel** (Microsoft) — enterprise-focused, integrates with Azure AI, supports .NET and Python.
- **Smolagents** (HuggingFace) — lightweight, code-first agents with built-in support for HuggingFace models and tools.

---

## Conclusion

The release of production-grade agentic SDKs in early 2025 represents a fundamental shift in how we build AI applications. We're moving from stateless prompt-response patterns to **stateful, multi-step, multi-agent systems** that can plan, use tools, validate their own outputs, and coordinate with each other.

The key insight: **agents are not just smarter chatbots**. They're software systems where the LLM is the orchestration layer. The SDKs from OpenAI and Google provide the primitives to build these systems with proper engineering practices: typing, testing, tracing, and deployment.

The best time to start building agents was yesterday. The second best time is `pip install openai-agents` or `pip install google-adk`.

---

*Michail Berjaoui — March 2025*
*Sources: OpenAI Agents SDK documentation (openai.github.io/openai-agents-python), Google ADK documentation (google.github.io/adk-docs), official release announcements.*
