# LangSmith Overview

## Table of Contents

1. [What is Observability?](#1-what-is-observability)
2. [What is LangSmith?](#2-what-is-langsmith)
3. [Getting Started: Setting Up Your First Project](#3-getting-started-setting-up-your-first-project)
4. [What Does a LangSmith Trace Include?](#4-what-does-a-langsmith-trace-include)
5. [Why is LangSmith Important for LLM Applications?](#5-why-is-langsmith-important-for-llm-applications)
6. [How LangSmith Helps Improve a RAG Application](#6-how-langsmith-helps-improve-a-rag-application)
7. [Key Benefits of LangSmith](#7-key-benefits-of-langsmith)
8. [Monitoring in LangSmith](#8-monitoring-in-langsmith)
9. [Prompt Playground and Prompt Experimentation](#9-prompt-playground-and-prompt-experimentation)
10. [Evaluation Using LangSmith](#10-evaluation-using-langsmith)
11. [Dataset Creation in LangSmith](#11-dataset-creation-in-langsmith)
12. [Data Annotation in LangSmith (Annotation Queues)](#12-data-annotation-in-langsmith-annotation-queues)
13. [User Feedback Integration in LangSmith](#13-user-feedback-integration-in-langsmith)
14. [Collaboration Using LangSmith](#14-collaboration-using-langsmith)
15. [Summary](#15-summary)

---

## 1. What is Observability?

Observability is the ability to understand the internal state of a system by examining its external outputs, such as logs, metrics, and traces. In software systems — especially complex AI applications — observability helps teams diagnose issues, analyze performance, and improve reliability. It is essentially the practice of answering the question: **"Why did this happen?"**

For LLM-based applications, observability is especially important because model behavior can be non-deterministic and difficult to reason about without detailed execution data.

---

## 2. What is LangSmith?

**LangSmith** is a unified observability, debugging, evaluation, and monitoring platform for AI applications. It is designed for teams building applications powered by large language models, agents, and retrieval-augmented generation (RAG) systems.

It is built around three core concepts:

| Concept     | Description                                                                                                                                                                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project** | The workspace for whatever you're building — a RAG application, an agent application, or any other LLM-based application. All related traces are organized under a project.                                                                                                                                         |
| **Trace**   | A record of a single end-to-end execution of an AI workflow. It captures the input, output, intermediate steps, latency, token usage, cost, errors, and other metadata. Traces help developers understand what happened during a run and pinpoint where issues occurred. Structurally, a trace is a _tree of runs_. |
| **Run**     | A single step within a trace — for example, a model call, a tool invocation, or a retrieval step. Runs can be nested (parent/child) to represent complex, multi-step workflows such as agents or chains.                                                                                                            |

LangSmith helps developers:

- Debug application behavior in both development and production
- Inspect prompts, model calls, and intermediate steps
- Evaluate output quality over time
- Monitor token usage, latency, and cost
- Improve the reliability and performance of AI systems

---

## 3. Getting Started: Setting Up Your First Project

Before you can trace, evaluate, or annotate anything, you need a LangSmith workspace connected to your application. At a high level, the setup flow looks like this:

1. **Create an account.** Sign up at LangSmith using an email/password, GitHub, or Google/SSO login, depending on what your organization allows.
2. **Create an organization and workspace.** An _organization_ is the top-level billing/account entity; a _workspace_ sits inside it and is where your projects, datasets, and annotation queues actually live. Most individuals and small teams work inside a single default workspace.
3. **Generate an API key.** From the workspace **Settings** page, create an API key. This key is what your application uses to authenticate and send trace data to LangSmith. Treat it like a secret — store it in an environment variable or secrets manager rather than hardcoding it.
4. **Connect your application.** Once tracing is enabled in your application (via environment variables and the LangSmith SDK, or the LangChain/LangGraph integration), every run of your app automatically creates a **Project** in LangSmith and starts populating it with **Traces**.
5. **Open the Tracing Projects view.** This is your home base — from here you can drill into individual traces, filter runs, build datasets from real traffic, set up monitoring dashboards, and send runs to annotation queues.

> This document intentionally stays at the _what-and-why_ level for setup — the code-level implementation (SDK installation, environment variables, wrapping LLM calls, etc.) is covered separately in the accompanying scripts.

---

## 4. What Does a LangSmith Trace Include?

A LangSmith trace captures the full execution path of an AI workflow. It allows developers to inspect what happened during a single run and understand where a problem occurred.

A typical trace can include:

1. Input and output
2. All intermediate steps
3. Latency
4. Token usage
5. Cost
6. Errors and exceptions
7. Tags
8. Metadata
9. Feedback

These details make it possible to analyze both technical performance and output quality from a single view.

---

## 5. Why is LangSmith Important for LLM Applications?

Modern AI applications often involve multiple stages, such as prompt construction, model inference, tool usage, retrieval, and response generation. When something goes wrong, it can be difficult to identify whether the issue came from the model, the prompt, the retrieval layer, or another component.

LangSmith provides a structured way to inspect each of these steps and understand the full lifecycle of a request — turning an opaque, multi-stage pipeline into something you can actually debug.

---

## 6. How LangSmith Helps Improve a RAG Application

RAG applications commonly fail in one of two ways:

1. **Retriever error** — irrelevant or incorrect documents are retrieved
2. **Generator error** — the model hallucinates or fails to use the retrieved context correctly

In production, it is often unclear which part of the pipeline caused the failure. Was the retrieval step ineffective, or did the model ignore the retrieved context?

LangSmith helps answer this by automatically recording key information, including:

1. The user query
2. The retrieved documents
3. The LLM prompt, including the inserted retrieved context
4. The LLM response

This makes it much easier to identify whether the issue lies in retrieval quality, prompting strategy, or model behavior.

---

## 7. Key Benefits of LangSmith

LangSmith provides a practical foundation for building and maintaining production-grade AI systems. Its main benefits include:

- Better debugging of complex AI workflows
- Clearer visibility into prompt and model behavior
- Improved evaluation and testing
- More effective monitoring of cost and latency
- Stronger support for the iterative improvement of RAG and agent-based applications

---

## 8. Monitoring in LangSmith

**What it does:**
Monitoring in LangSmith looks across many traces at once to track the overall health of an LLM application. It aggregates key metrics such as latency (P50, P95, P99), token usage, cost, error rates, and success rates onto dashboards that update as new traces come in. You can filter these dashboards by time range, tag, or metadata (e.g., by feature, environment, or model), and set up alerts to notify you when a metric drifts outside an acceptable range — for example, a spike in latency, a rise in error rates, or unexpected cost growth.

**Why it matters:**
In production, issues often appear first as patterns across multiple runs rather than in a single trace. Monitoring helps you catch these early, before they impact users at scale. Instead of waiting for customer complaints, you're proactively alerted when performance degrades or costs spike — enabling a faster response and more reliable applications.

---

## 9. Prompt Playground and Prompt Experimentation

**What it does:**
The **Prompt Playground** is LangSmith's interactive environment for writing, testing, and iterating on prompts without needing to touch your application code. Inside the Playground you can:

- Edit a prompt's messages and variables and immediately re-run it against a model
- Switch between model providers (e.g., OpenAI, Anthropic, Google) to compare how the same prompt performs across models
- Run several prompt or model variants **side by side** and compare their outputs, latency, and token cost in one view
- Save an edited prompt as a new **commit**, so every change to a prompt creates a versioned history you can review, diff, or roll back
- Promote a specific commit to a named **environment** (e.g., "staging" or "production") so your application always pulls the version you've approved
- Pull prompts from the **Prompt Hub**, a shared/public library of reusable prompts, if you want a starting template instead of writing one from scratch

**Prompt experimentation** builds on this by letting you go beyond manual spot-checks:

- Run an A/B-style comparison of two or more prompt versions against the _same dataset_ of inputs
- Score each version against your evaluation metrics (see the next section) so comparisons are objective, not just a gut feeling
- Keep a stored history of every experiment, so you can see exactly which prompt version performed best and under what conditions — and avoid silently regressing a prompt that used to work well

---

## 10. Evaluation Using LangSmith

**What it does:**
Evaluation in LangSmith helps you systematically measure the quality of your LLM outputs. You can run tests against gold-standard datasets or apply custom evaluation metrics, such as faithfulness, relevance, or completeness. LangSmith supports multiple evaluator types:

- **LLM-as-a-Judge** — an LLM scores outputs against criteria you define
- **Heuristic / rule-based checks** — e.g., does the output parse as valid JSON, does generated code compile
- **Semantic similarity checks** — comparing an output to a reference answer
- **Human evaluation** — via annotation queues (see Section 12)
- **Pairwise comparisons** — judging which of two outputs is better, rather than scoring each in isolation
- **Custom evaluators** — your own Python or TypeScript scoring logic for business-specific correctness

Evaluations can be run both **offline** (as a batch test against a fixed dataset before deployment) and **online** (as a continuous check against live production traffic).

**Why it matters:**
LLM behavior can be unpredictable — a small change in a prompt, model, or retrieval logic may improve some cases while breaking others. Evaluation provides an objective, repeatable way to track performance over time, ensuring that new versions are actually better and preventing regressions.

**Example:**
For a RAG chatbot, you might evaluate:

- **Faithfulness** — Are answers grounded in the retrieved documents?
- **Relevance** — Did the response actually address the user's question?

By running the same dataset across GPT-4, Claude, and LLaMA, you can directly compare which model (or pipeline setup) performs best.

---

## 11. Dataset Creation in LangSmith

Datasets are the backbone of evaluation in LangSmith — they're the fixed, reusable set of inputs (and optionally expected outputs) you test prompts, models, and pipelines against. LangSmith stores dataset examples as JSON, and you can optionally define a schema so every example conforms to the same structure.

There are three practical ways to build a dataset, all doable entirely from the LangSmith UI:

1. **From existing traces (most common for real projects):**
   Go to your **Tracing Project**, open the **Runs** table, and multi-select the runs you want to keep. Click **Add to Dataset**. This is the fastest way to turn real production or testing traffic into a benchmark — you're literally curating your dataset from cases your application has already handled (including edge cases and failures you want to guard against in the future).

2. **By uploading a file:**
   Datasets can be created directly from a CSV (or similar tabular file) upload in the UI, mapping columns to input/output fields. This is useful when you already have a spreadsheet of questions and expected answers from a domain expert or support team.

3. **By adding examples manually:**
   You can also create an empty dataset and add examples one at a time directly in the UI — typing in the input and, optionally, the expected/reference output. This is the right approach for small, hand-crafted "golden" test sets.

Once a dataset exists, you can keep growing it over time — for example, by routing interesting or problematic runs from an **annotation queue** straight into a dataset as corrected reference examples (see Section 12).

---

## 12. Data Annotation in LangSmith (Annotation Queues)

Automated metrics and LLM-as-a-Judge evaluators don't catch everything — some quality judgments genuinely need a human to look at the output. LangSmith's **Annotation Queues** feature is the tool built for this: it gives reviewers (you, teammates, or dedicated annotators) a focused, one-at-a-time interface for scoring runs against a rubric, without needing to dig through the full traces table.

There are two queue styles:

### a) Single-run annotation queues

Show reviewers **one run at a time**, along with whatever feedback rubric you've configured.

**How to create one:**

1. Go to **Annotation Queues** in the left-hand navigation and click **+ Annotation Queue**.
2. Fill in a **Name** and **Description**, and optionally pick a **default dataset** — this lets you push reviewed/corrected examples straight into a dataset later.
3. Under the **Annotation Rubric**, write instructions for your annotators (shown in a sidebar on every run) and add one or more **feedback keys** — the specific things you want scored (e.g., "Correctness," "Tone," "Grounded in context"), each with a short description so reviewers know exactly what they're judging.
4. Configure **collaborator settings** — how many reviewers must review each run before it's considered done, whether runs are "reserved" for a set time while someone reviews them, and whether every workspace member must review every run or only a specific number/set of assigned people.

**How to populate the queue with runs to review:**

- From any trace's **Details** view, click **Add to Annotation Queue**.
- From the **Runs table**, multi-select several runs and click **Add to Annotation Queue** at the bottom.
- Automatically, via an **automation rule** that routes runs matching a filter (e.g., all runs with errors, or all runs where a user gave a thumbs-down) straight into a queue.
- From **Datasets & Experiments**, by selecting one or more experiment runs and choosing **Annotate**.

**How to actually review/annotate:**

1. Open the queue from **Annotation Queues** — you'll land in a focused, one-run-at-a-time view.
2. For each run, read the input/output, leave **Reviewer Notes**, and score each rubric item.
3. If the output needs correction, you can edit it directly and click **Add to Dataset** — this turns your review into a high-quality reference example for future evaluations.
4. Mark the run **Done** (keyboard shortcuts are available to speed this up) to move to the next item in the queue.

### b) Pairwise annotation queues (PAQs)

Instead of scoring one run in isolation, PAQs show **two runs side-by-side** (typically a baseline vs. a new/candidate version) so a reviewer can quickly judge which one is better.

**How to create one:**

1. Go to **Datasets & Experiments**, open a dataset, and select **exactly two experiments** you want to compare.
2. Click **Annotate → Add to Pairwise Annotation Queue**.
3. Fill in the queue's basic details, rubric, and collaborator settings (same idea as single-run queues).
4. LangSmith automatically pairs up matching runs from the two experiments and populates the queue.

**How to review:**

- For each pair, and for every rubric item, choose **A is better**, **B is better**, or **Equal** (with keyboard shortcuts `A`, `B`, `E`).
- Once all rubric items for a pair are scored, submit to move to the next comparison.

**Why annotation matters:**
Annotation queues are how you scale human judgment without it becoming chaotic — every reviewer sees a consistent rubric, progress is tracked automatically, and the results of a review can feed directly back into your datasets and evaluations, closing the loop between "a human noticed a problem" and "the system is now tested against it."

---

## 13. User Feedback Integration in LangSmith

**What it does:**

- Lets you capture thumbs up/down, ratings, or structured feedback from users in production
- Logs feedback alongside traces, tying it to the exact prompt, model, and state that produced it
- Supports bulk analysis of what users like and dislike, and can be used to automatically route poorly-rated runs into an annotation queue for deeper human review

---

## 14. Collaboration Using LangSmith

**What it does:**

- Team members can view, share, and comment on traces, datasets, and evaluations
- Provides a web UI where non-engineers (PMs, QA, annotators) can inspect and annotate runs
- Enables shared experiment dashboards

---

## 15. Summary

LangSmith is a powerful observability and evaluation platform for AI applications. It enables developers to track the full execution of LLM workflows, understand failures, and improve system quality over time. Beyond tracing, it provides a complete UI-driven workflow — setting up a project, building datasets from real traffic, experimenting with prompts, running automated evaluations, and routing runs to human reviewers through annotation queues — so that both engineers and non-engineers can participate in improving an AI application. For RAG applications in particular, it helps connect user queries with retrieved documents and generated responses in a way that makes debugging much more precise and actionable.
