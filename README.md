# Langgraph-Tutorial

A hands-on, project-by-project learning repository for building with **LangGraph**. It takes you from the absolute basics of a `StateGraph` all the way up to production-style capstone systems (Corrective RAG, Self-RAG, and a multi-agent blog-writing pipeline), with a fully working chatbot, MCP integration, and LangSmith observability along the way.

The repo is intentionally organized as a **curriculum**: each top-level folder is a learning module that builds on the concepts introduced in the previous one. Read this README top to bottom before you start — it tells you what to install, what to configure, and in what order to work through the folders.

---

## Table of Contents

1. [What You'll Learn](#what-youll-learn)
2. [Prerequisites](#prerequisites)
3. [Setup & Installation](#setup--installation)
4. [Environment Variables](#environment-variables)
5. [How to Run an Example](#how-to-run-an-example)
6. [Recommended Learning Path](#recommended-learning-path)
7. [Repository Structure — Module by Module](#repository-structure--module-by-module)
8. [Notes & Gotchas](#notes--gotchas)

---

## What You'll Learn

By working through this repo in order, you will go from "what is a `StateGraph`" to building agentic, tool-using, memory-aware, retrieval-augmented systems, including:

- Core LangGraph primitives: `StateGraph`, `TypedDict` state, nodes, edges, compilation
- Conditional / branching workflows and routing functions
- Iterative generate → evaluate → optimize loops
- Parallel (fan-out / fan-in) workflows
- Persistence, checkpointing, and fault-tolerant recovery ("time travel")
- Composing graphs with subgraphs
- Tool calling inside a graph
- Human-in-the-loop workflows using `interrupt` / `Command`
- Short-term and long-term memory patterns (including a Postgres-backed store)
- Retrieval-Augmented Generation (RAG) inside a LangGraph workflow
- Model Context Protocol (MCP) — building/consuming MCP tool servers from a graph
- A full Streamlit chatbot with streaming, persistence, tools, and PDF-based RAG
- LangSmith tracing/observability across chains, graphs, and RAG pipelines
- Capstone systems: **Corrective RAG (CRAG)**, **Self-RAG**, and a **multi-agent planning/blog-writing agent** with parallel section workers and optional image generation

---

## Prerequisites

- Python 3.10+ (or compatible)
- A Google AI Studio API key (`GOOGLE_API_KEY`) — most examples use Gemini models (`gemini-2.5-flash`, embeddings, etc.) via `langchain-google-genai`
- (Optional, for tracing) A [LangSmith](https://smith.langchain.com/) account and API key
- (Optional, for web-search-enabled examples such as CRAG and the research-enabled blog agent) A [Tavily](https://tavily.com/) API key — these examples call `TavilySearchResults` / `TAVILY_API_KEY` directly in code
- Basic familiarity with Python and LangChain concepts is helpful but not required — the early modules assume no LangGraph knowledge

---

## Setup & Installation

```powershell
# 1. Clone the repository
git clone <this-repo-url>
cd Langgraph-Tutorial

# 2. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\Activate.ps1        # Windows PowerShell
# source myenv/bin/activate       # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> The repo already ships with a local `myenv/` folder in some setups — if you'd rather use your own environment, just create a fresh `venv`/`myenv` as shown above and install `requirements.txt` into it.

---

## Environment Variables

Copy `.env.example` to `.env` in the repo root and fill in your keys:

```env
# Google API Key
GOOGLE_API_KEY = ''            # Required — powers Gemini chat + embedding models

# Langsmith CREDS
LANGSMITH_API_KEY = ''         # Required only for the Langsmith/ module examples
LANGCHAIN_TRACING = true       # Set to true to enable tracing
LANGCHAIN_ENDPOINT = https://api.smith.langchain.com
LANGCHAIN_PROJECT = ''         # Any project name you want traces grouped under
```

A few additional examples read extra environment variables directly in code — add these to your `.env` as needed when you reach them:

- `TAVILY_API_KEY` — required by the CRAG capstone (`Capstone Projects/CRAG/corrective_rag_single_shot.py`, `graph/nodes.py`, `retrieval/web_retriever.py`) and by the research/web-search-enabled blog agents (`Capstone Projects/Planning AI Agent/blog_writting_agent_with_research.py`, `blog_writting_agent_with_image_capabilities.py`) for their web-search steps.

**Never commit your `.env` file** — it holds your personal API keys.

---

## How to Run an Example

Most examples are standalone scripts:

```powershell
python <path\to\example>.py
```

For example:

```powershell
python "Sequential Workflows\prompt_chaining.py"
python "Conditional Workflows\review_handling.py"
python "Capstone Projects\Self Rag\self-rag.py"
```

The Streamlit-based chatbot is run with `streamlit run`:

```powershell
streamlit run "Chatbot Projects\chatbot_streamlit_frontend_streaming.py"
```

Several scripts contain a commented-out line for saving a Mermaid PNG of the compiled graph (e.g. `app.get_graph().draw_mermaid_png()`) — uncomment it if you want to visualize the graph you just built.

---

## Recommended Learning Path

Work through the folders in this order. Each step assumes you understand the concepts from the steps before it.

| Step | Folder | What it teaches |
|---|---|---|
| 1 | `Sequential Workflows/` | The absolute basics: `StateGraph`, `TypedDict` state, linear node chains, and LLM-based chains |
| 2 | `Conditional Workflows/` | Branching logic with routing functions and `add_conditional_edges` |
| 3 | `Iterative Workflows/` | Generate → evaluate → optimize loops with accumulating state |
| 4 | `Parallel Workflows/` | Fan-out/fan-in parallel node execution and result aggregation |
| 5 | `Persistence/` | Checkpointing, resuming interrupted runs, and "time travel" through history |
| 6 | `SubGraphs-in-Langgraph/` | Composing larger graphs out of smaller, reusable graphs |
| 7 | `Tools in Langraph/` | Binding and invoking tools from inside a graph node |
| 8 | `Human-in-the-Loop/` | Pausing a graph for human approval using `interrupt` / `Command` |
| 9 | `Memory/` | Short-term (STM) and long-term (LTM) memory patterns, including trimming, summarization, deletion, and a Postgres-backed store |
| 10 | `RAG-in-Langgraph/` | Wiring a basic Retrieval-Augmented Generation pipeline into a graph |
| 11 | `MCP/` | Connecting a LangGraph chatbot to tools served over the Model Context Protocol |
| 12 | `Chatbot Projects/` | Putting it together: a persistent, streaming, tool- and RAG-enabled Streamlit chatbot |
| 13 | `Langsmith/` | Observability — tracing chains, sequential runs, RAG pipelines, and agents |
| 14 | `Capstone Projects/` | Advanced, end-to-end systems: Corrective RAG, Self-RAG, and a multi-agent planning/blog-writing agent |

---

## Repository Structure — Module by Module

### 1. `Sequential Workflows/`
Your entry point into LangGraph.
- `bmi_workflow.py` — a pure-Python (no LLM) linear graph computing BMI, to learn `StateGraph` mechanics without any model calls.
- `LLM_based_workflow.py` — a linear graph that calls an LLM as part of the flow.
- `prompt_chaining.py` — chaining multiple prompts/LLM calls sequentially through graph nodes.

### 2. `Conditional Workflows/`
Branching and routing.
- `solving_quadratic_equation.py` — a pure-Python graph that computes a discriminant and branches into real roots / repeated roots / no real roots, demonstrating conditional routing without any LLM.
- `review_handling.py` — an LLM-driven review-sentiment workflow: detects sentiment, then branches into a thank-you response (positive) or a diagnosis + empathetic resolution (negative), using Pydantic structured outputs.

### 3. `Iterative Workflows/`
- `post_generation.py` — generates a tweet, evaluates it with structured feedback, and optionally optimizes it in a loop until approved or a max-iteration limit is hit. Demonstrates state accumulation (`tweet_history`, `feedback_history`).

### 4. `Parallel Workflows/`
- `simple_parallel_worflow.py` — computes cricket performance metrics (strike rate, balls per boundary, boundary %) in parallel branches, then combines them in a summary node.
- `parallelization_workflow.py` — evaluates an essay on language, analysis, and clarity in parallel, then consolidates feedback and computes an average score.
- `essay.txt` — sample input text used by the parallel essay-evaluation workflow.

### 5. `Persistence/`
- `fault_tollerance.py` — simulates a long-running/hanging step and demonstrates resuming execution after interruption using `InMemorySaver`.
- `persistence_in_langgraph.py` — generates a joke + explanation, inspects final state and intermediate checkpoints, and demonstrates "time travel" by reloading an earlier checkpoint and re-invoking from that point.

### 6. `SubGraphs-in-Langgraph/`
- `subgraph-using-Add-a graph-as-a-node.py` — composing a subgraph by adding a compiled graph directly as a node in a parent graph.
- `subgraph-using-Invoke-a-graph-from-a-node.py` — composing a subgraph by invoking a separately compiled graph from within a node function.
- `subgraph.md` — accompanying notes on subgraph patterns.

### 7. `Tools in Langraph/`
- `tool_implementation.py` — binding tools to a model and routing tool calls through a graph.

### 8. `Human-in-the-Loop/`
- `basic_HITL.py` — a minimal graph that pauses via `interrupt()` to ask for human approval before letting the LLM answer, resuming via `Command`.
- `chatbot-with-HITL.py` — applying the same human-in-the-loop approval pattern inside a chatbot-style graph.
- `HITL.md` — accompanying notes on the human-in-the-loop pattern.

### 9. `Memory/`
- `stm.py`, `stm_trimming.py`, `stm_deletion.py`, `stm_summarization.py`, `stm_persistence.py` — short-term memory patterns for a conversation: keeping, trimming, deleting, summarizing, and persisting message history.
- `ltm_basic.py`, `ltm_implementation.py`, `ltm_postgressql.py` — long-term memory patterns, including a Postgres-backed long-term memory store.
- `llm-memory.md` — accompanying notes on memory strategies for LLM applications.

### 10. `RAG-in-Langgraph/`
- `rag_langgraph.py` — a basic Retrieval-Augmented Generation pipeline wired into a LangGraph workflow.

### 11. `MCP/`
- `mcp_server.py` — a standalone MCP server (used elsewhere in the repo as a calculator tool source, e.g. by the chatbot backend).
- `chatbot_mcp_client.py` — a client-side integration connecting to MCP server(s).
- `chatbot_async.py` — an async chatbot example built around MCP tool usage.

### 12. `Chatbot Projects/`
A complete, production-style chatbot.
- `chat_bot_langgraph_backend.py` — a fully async LangGraph backend: Gemini chat model, per-thread PDF ingestion + FAISS RAG tool, a stock-price tool, a DuckDuckGo search tool, and MCP-sourced tools (calculator server from `MCP/mcp_server.py`), all run on a dedicated background event loop with `AsyncSqliteSaver` checkpointing.
- `chatbot_streamlit_frontend_streaming.py` — the Streamlit UI: thread-based conversation history, PDF upload/indexing per thread, and token-by-token streaming with live tool-call status indicators, bridged from the backend's async event loop via a queue-based generator.
- `chatbot-features.md` — a full technical write-up of the chatbot's architecture, features, and data flow — read this alongside the two Python files.

### 13. `Langsmith/`
Observability and tracing, to be studied alongside the two docs below.
- `Langsmith-Overview.md` — high-level concepts: observability, trace structure, monitoring, evaluation.
- `Langsmith-in-Practise.md` — implementation-focused walkthroughs for the code examples in this folder.
- `simple_llm_call.py` — a minimal traced LangChain call to Gemini.
- `sequential_chain.py` — a chained workflow (output of one call feeds the next) with named runs, tags, and metadata.
- `rag_v1.py` — a PDF-based RAG pipeline using HuggingFace embeddings, FAISS, and Gemini.
- `rag_v2.py` — adds `@traceable` instrumentation to the RAG pipeline (loading, splitting, indexing, querying as visible spans).
- `rag_v3.py` — extends the RAG example with index fingerprinting, cached index directories, and richer traceable metadata.
- `agent.py` — an LLM agent with search and weather tools, powered by Gemini, as an observable execution flow.
- `langgraph_code_via_langsmith_traceability.py` — combines LangGraph state with LangSmith tracing, evaluating an essay across multiple dimensions in parallel.

### 14. `Capstone Projects/`
End-to-end, advanced systems that combine everything above.

**`CRAG/`** — Corrective RAG, built up in phases.
- `CRAG_Technical_Learning_Guide.md` — **start here** for this module. A full conceptual walkthrough of RAG vs. CRAG, pros/cons, when to use which, and a step-by-step build of Phase 1 (Knowledge Refinement): retrieve → decompose into sentence "strips" → score each strip with a fine-tuned T5 relevance evaluator → keep/drop → recompose → generate.
- `corrective_rag_single_shot.py` — the complete single-file CRAG pipeline: FAISS retrieval, T5-based retrieval evaluation (CORRECT / AMBIGUOUS / INCORRECT), Gemini-based query rewriting, Tavily web search fallback, T5-based knowledge-strip refinement, and final answer generation — all wired together as a LangGraph graph.
- `application.py`, `main.py` — a refactored, dependency-injected version of the same CRAG pipeline (`CRAGApplication`), assembled from the modules below.
- `ai_model/` — model loader interfaces and implementations for the T5 relevance evaluator and the Gemini model.
- `generate_embedding/` — a pluggable embedding pipeline (PDF loader, splitter, embedding provider, FAISS vector store).
- `retrieval/` — retriever interfaces/factory for semantic (FAISS) and web (Tavily) retrieval.
- `graph/` — the CRAG `StateGraph` definition: state, nodes (retrieve, evaluate, rewrite query, web search, refine, generate), router, and graph builder.
- `prompts/` — the query-rewrite and answer-generation prompt templates.
- `test_*.py` — standalone test scripts for the embedding pipeline, the T5 filtering model, the model loaders, and retrieval.

**`Self Rag/`** — Self-Reflective RAG.
- `self-rag.md` — the core idea: instead of blindly trusting retrieval, the model checks (1) whether retrieval is even needed, (2) whether retrieved documents are relevant, (3) whether the generated answer is grounded in the retrieved evidence, and (4) whether the answer actually addresses the question.
- `self-rag.py` — the full implementation: a decide-retrieval router, direct-answer path, FAISS retrieval, per-document relevance filtering, grounded-answer generation, an `IsSUP` grounding-verification loop that revises unsupported answers, an `IsUSE` usefulness check, and a query-rewrite-and-retry loop when the answer isn't useful.

**`Planning AI Agent/`** — multi-agent, parallel section-writing pipeline.
- `planning-ai-agent.md` — what a planning agent is: it builds a structured plan before acting, instead of jumping straight to a response.
- `blog_writting_agent.py` — the base version: an orchestrator plans a blog outline (Pydantic `Plan`/`Task` schema), fans out one worker per section in parallel via `Send`, and a reducer merges the sections and saves the Markdown file.
- `blog_writting_agent_with_research.py` — adds a router that decides whether web research is needed (closed-book / hybrid / open-book), a Tavily-based research node that produces structured `EvidenceItem`s, and citation-aware section writing.
- `blog_writting_agent_with_image_capabilities.py` — further extends the research version with an image-planning step and Gemini image generation, inserting generated diagrams/illustrations into the final Markdown at planned placeholders.
- `output/` and the two top-level `.md` files — example generated blog posts produced by running these agents, useful as reference output.

---

## Notes & Gotchas

- **Model names**: most examples use `gemini-2.5-flash` via `ChatGoogleGenerativeAI`; a few of the capstone agents reference newer Gemini model names directly in code (e.g. `gemini-3-flash-preview`, `gemini-2.5-flash-image`) — check the specific script if you hit a model-not-found error, as available models change over time.
- **CRAG's T5 model**: `Capstone Projects/CRAG/CRAG_Technical_Learning_Guide.md` documents a required pretrained T5 filtering model that must be downloaded separately and placed under `./filtering_model/` before running the CRAG scripts — see that guide for the expected folder contents.
- **MCP paths**: the chatbot backend (`Chatbot Projects/chat_bot_langgraph_backend.py`) points to a local, machine-specific Python executable and script path to launch the MCP calculator server — update these paths to match your own environment before running it.
- **Graph visualizations**: many scripts include a commented-out `draw_mermaid_png()` call to export the compiled graph as an image — uncomment it if you want a visual diagram of what you just built.
- **`.env` values**: never commit real API keys. Use `.env.example` as your template.

---

This README is intended to help new developers navigate `Langgraph-Tutorial` from first principles to advanced, production-style agentic systems, one folder at a time.
