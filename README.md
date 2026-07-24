# Langgraph-Tutorial

This repo contains a set of Langgraph-based workflow examples arranged by pattern and learning objective. The goal is to help developers learn how to build state graph workflows with conditional routing, iterative improvement, parallel evaluation, persistence, and chatbot integrations.

## Overview

`Langgraph-Tutorial` demonstrates how to use the Langgraph state graph API together with Google Gemini models and LangChain-style prompt handling. The examples include:

- Chatbot construction with a Langgraph backend and Streamlit UIs
- Conditional workflows and branch routing
- Iterative generation + evaluation loops
- Parallel workflow execution
- Persistence, checkpointing, and fault-tolerance
- LangSmith-enabled observability, traceable RAG workflows, and agent integrations

## Prerequisites

- Python 3.10+ or compatible
- Install dependencies using a virtual environment
- A `.env` file with your Google API key set as `GOOGLE_API_KEY`

Example:

```powershell
cd "d:\AI Engineering\Agentic AI\Langgraph-Tutorial"
myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Note: This folder includes a local virtual environment at `myenv/`. If you want to create your own environment instead, use `python -m venv venv` and install the necessary packages.

## Folder structure

### Chatbot Projects

Contains a simple Langgraph-powered chatbot example.

- `chat_bot_langgraph_backend.py`
  - Defines a minimal Langgraph `StateGraph` with one node, `chat_node`.
  - Uses `ChatGoogleGenerativeAI` to invoke Gemini and generate a chatbot response.
  - Demonstrates message passing through `TypedDict` state and basic thread-aware config.

- `chatbot_streamlit_frontend.py`
  - Streamlit UI for the chatbot backend.
  - Sends user messages to the Langgraph workflow and renders chat history.

- `chatbot_streamlit_frontend_streaming.py`
  - Streamlit UI with streaming output.
  - Demonstrates how to stream assistant responses incrementally from the Langgraph workflow.

### Conditional Workflows

Shows how to route execution conditionally using different branches.

- `review_handling.py`
  - A review sentiment workflow that first detects sentiment and then branches.
  - Positive reviews get a thank-you response.
  - Negative reviews are diagnosed for issue type, tone, and urgency, then generate an empathetic resolution.
  - Uses structured output schemas via Pydantic models.

- `solving_quadratic_equation.py`
  - A pure Python workflow that computes the discriminant of a quadratic equation.
  - Branches into real roots, repeated roots, or no real roots.
  - Demonstrates conditional graph routing without any LLM involvement.

### Iterative Workflows

Demonstrates a generate-evaluate-optimize loop.

- `post_generation.py`
  - Generates a tweet, evaluates it with structured feedback, and optionally optimizes it.
  - Repeats until the tweet is approved or a maximum number of iterations is reached.
  - Illustrates iterative improvement and state accumulation using `tweet_history` and `feedback_history`.

### Parallel Workflows

Illustrates parallel execution of independent graph branches.

- `simple_parallel_worflow.py`
  - Computes cricket performance metrics in parallel: strike rate, balls per boundary, and boundary percentage.
  - Collects results into a summary node.
  - Great for learning how to combine independent computations in a graph.

- `parallelization_workflow.py`
  - Evaluates an essay on language, analysis, and clarity in parallel.
  - Then consolidates the feedback and computes an average score.
  - Uses structured output to capture scores and feedback from each evaluator.

### Persistence

Covers state persistence, checkpoint recovery, and fault tolerance.

- `fault_tollerance.py`
  - Simulates a long-running hang in a workflow step.
  - Demonstrates resuming execution after interruption using `InMemorySaver`.
  - Shows how Langgraph can recover intermediate state from a configurable thread checkpoint.

- `persistence_in_langgraph.py`
  - Example of using persistence to inspect workflow state and history.
  - Generates a joke and explanation, then prints final state and intermediate checkpoints.
  - Demonstrates "time travel" by reloading an earlier checkpoint and re-invoking from that point.

### Langsmith Examples

Contains LangSmith-enabled LangChain and Langgraph examples for observability, tracing, evaluation, and RAG.

- `simple_llm_call.py`
  - A minimal LangChain runnable that sends a single prompt to Gemini and prints the result.
  - Demonstrates how LangSmith automatically traces prompt construction, model calls, output parsing, latency, and token usage.

- `sequential_chain.py`
  - Shows a chained workflow where the output of one Gemini call becomes the input to another.
  - Demonstrates named runs, tags, metadata, and sequential trace structure.

- `rag_v1.py`
  - Builds a PDF-based RAG pipeline using HuggingFace embeddings, FAISS, and Gemini.
  - Reads a local PDF, chunks text, indexes it, retrieves context, and answers questions from the document.

- `rag_v2.py`
  - Adds LangSmith `@traceable` instrumentation to the RAG setup and query pipeline.
  - Traces loading, splitting, indexing, and query execution as visible spans in LangSmith.

- `rag_v3.py`
  - Extends the RAG example with index fingerprinting, cached index directories, and traceable load/build runs.
  - Demonstrates reusable vector store management and richer LangSmith metadata.

- `agent.py`
  - Creates an LLM agent with search and weather tools, powered by Gemini.
  - Demonstrates LangChain agent invocation and tool usage as part of an observable execution flow.

- `langgraph_code_via_langsmith_traceability.py`
  - Combines Langgraph workflow state with LangSmith tracing.
  - Evaluates an essay across language, analysis, and clarity dimensions in parallel before summarizing overall feedback.

### Studying LangSmith

If you want to learn LangSmith in depth, start with the two documentation files in the `Langsmith` folder:

- `Langsmith/Langsmith-Overview.md` — high-level concepts, observability, trace structure, monitoring, evaluation, and practical benefits.
- `Langsmith/Langsmith-in-Practise.md` — implementation-focused walkthroughs for the provided Python examples.

These docs are designed to be read alongside the `Langsmith` code examples, so you can pair concept study with hands-on scripts.

## Learning outcomes

After exploring these examples, developers should understand:

- Basic Langgraph API usage with `StateGraph`, nodes, edges, and compilation
- How to define typed workflow state using `TypedDict` and Pydantic schemas
- Conditional edge routing for branching logic
- Iterative workflows for repeated improvement cycles
- Parallel workflow design for concurrent evaluation
- Persistence and checkpoint-based fault recovery
- Integrating Langgraph with Google Gemini via LangChain-style models
- Building simple Streamlit frontends for conversational workflows

## How to run examples

General command pattern:

```powershell
python <example_file>.py
```

Streamlit chatbot example:

```powershell
streamlit run Chatbot Projects\chatbot_streamlit_frontend.py
```

Streaming chatbot example:

```powershell
streamlit run Chatbot Projects\chatbot_streamlit_frontend_streaming.py
```

Langsmith examples:

```powershell
python Langsmith\simple_llm_call.py
python Langsmith\sequential_chain.py
python Langsmith\rag_v1.py
python Langsmith\rag_v2.py
python Langsmith\rag_v3.py
python Langsmith\agent.py
python Langsmith\langgraph_code_via_langsmith_traceability.py
```

## Notes

- Many examples use `gemini-2.5-flash` via `ChatGoogleGenerativeAI`.
- The `.env` file should contain your Google API key and should not be shared publicly.
- The included `.py` files often contain commented graph visualization lines for generating Mermaid PNGs.

---

This README is intended to help new developers understand the purpose, structure, and learning goals of the `Langgraph-Tutorial` examples.
