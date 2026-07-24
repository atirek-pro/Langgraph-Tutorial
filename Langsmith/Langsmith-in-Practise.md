# LangSmith in Practice

## Understanding LangSmith Through Real Python Implementations

---

# Objective

The previous document introduced the theoretical concepts behind LangSmith, including tracing, observability, monitoring, evaluation, datasets, annotation queues, and prompt experimentation.

This companion guide focuses entirely on implementation.

Instead of discussing concepts, we will analyze seven progressively complex Python applications and understand:

- What each script builds
- What happens internally during execution
- What LangSmith automatically traces
- What additional tracing we manually create
- How the trace appears inside the LangSmith UI
- What practical LangSmith concept each project teaches

By the end of these examples, readers should understand not only how to integrate LangSmith into an application but also how to use tracing effectively for debugging increasingly complex AI systems.

---

# Example 1 — Simple LLM Call

**Source:** `simple_llm_call.py`

## Objective

This is the smallest possible LangChain application.

It demonstrates how a single prompt flows through a model and returns a response.

The application creates a simple chain consisting of:

```text
Prompt
   │
   ▼
Gemini Model
   │
   ▼
Output Parser
   │
   ▼
Final Output
```

Although the application itself is extremely small, it is the foundation upon which every larger LangChain application is built.

## What the Script Does

The script performs four steps:

1. Loads environment variables.
2. Creates a Gemini model.
3. Creates a prompt template.
4. Connects everything into a LangChain Runnable sequence.

Finally, the following question is sent to Gemini:

> **"What is the capital of India?"**

The response is then printed.

---

## What LangSmith Automatically Traces

When tracing is enabled, LangSmith records the complete execution.

A typical trace looks like:

```text
Root Chain
│
├── PromptTemplate
│
├── ChatGoogleGenerativeAI
│      ├── Input Prompt
│      ├── Output
│      ├── Latency
│      └── Tokens
│
└── StrOutputParser
```

---

## Information Visible Inside LangSmith

Opening the trace allows you to inspect:

- Complete user input
- Final prompt sent to Gemini
- Model response
- Output parser result
- Latency
- Token usage
- Cost (when supported)
- Errors (if any)

Although the pipeline is small, it introduces the tree structure that every future trace builds upon.

---

## What This Example Teaches About LangSmith

This first example teaches:

- Automatic tracing
- Traces vs. runs
- How LangChain components appear individually
- Prompt inspection
- Model inspection
- Latency tracking
- Token monitoring

It establishes the mental model that every LangChain Runnable becomes an observable execution step.

---

# Example 2 — Sequential Chain

**Source:** `sequential_chain.py`

## Objective

Real applications rarely make only one LLM call.

Often, the output of one model becomes the input of another.

This script demonstrates that workflow.

Pipeline:

```text
Topic
   │
   ▼
Generate Report
   │
   ▼
Detailed Report
   │
   ▼
Summarize Report
   │
   ▼
Final Summary
```

---

## What the Script Does

The application first generates a detailed report.

That report is immediately passed into another prompt asking the model to summarize it into five key points.

Unlike the previous example, this application performs two independent model calls connected together.

---

## Additional LangSmith Features Demonstrated

Unlike the previous script, this example introduces:

- Run Name
- Tags
- Metadata

through the invocation configuration.

These make traces much easier to organize inside a production project.

---

## Expected Trace Structure

```text
Sequential Chain
│
├── Prompt 1
├── Gemini Call #1
├── Parser
├── Prompt 2
├── Gemini Call #2
└── Parser
```

Each LLM call becomes its own child run.

You can inspect them independently.

---

## What You Can Learn From the Trace

Inside LangSmith you can compare:

- Prompt 1
- Prompt 2
- First LLM response
- Second LLM response
- Individual latency
- Token usage of each model call
- End-to-end latency

This immediately demonstrates why tracing is valuable.

If the summary is poor, you can determine whether:

- Report generation failed

or

- Summarization failed

instead of guessing.

---

## LangSmith Concepts Learned

- Nested runs
- Multiple model calls
- Run naming
- Tags
- Metadata
- Execution hierarchy

---

# Example 3 — Building a Basic RAG Pipeline

**Source:** `rag_v1.py`

## Objective

The first two examples only interact with an LLM.

This example introduces Retrieval-Augmented Generation (RAG).

The pipeline performs:

```text
PDF
 │
 ▼
Load Pages
 │
 ▼
Chunk Documents
 │
 ▼
Create Embeddings
 │
 ▼
FAISS Index
 │
 ▼
Retriever
 │
 ▼
Relevant Chunks
 │
 ▼
Prompt
 │
 ▼
Gemini
 │
 ▼
Answer
```

---

## What the Script Does

The script:

- Loads a PDF
- Splits it into chunks
- Creates sentence embeddings
- Builds a FAISS vector database
- Retrieves the top four similar chunks for a question
- Inserts those chunks into the prompt
- Generates an answer grounded in the retrieved context

---

## What LangSmith Traces Automatically

Without any manual instrumentation, LangSmith primarily traces the runnable chain used during query execution:

```text
Retriever
   │
   ▼
Prompt
   │
   ▼
Gemini
   │
   ▼
Parser
```

You can inspect:

- The user's question
- Retrieved context
- Final prompt
- Model output

However, preprocessing steps such as PDF loading, chunking, embedding creation, and FAISS index construction occur outside the runnable execution path, so they are not represented as separate trace nodes.

---

## What This Example Teaches

- Tracing RAG inference
- Inspecting retrieved context
- Debugging hallucinations by comparing retrieved documents and generated answers
- Understanding the limitations of automatic tracing for setup workflows

---

# Example 4 — Manual Trace Instrumentation

**Source:** `rag_v2.py`

## Objective

This version introduces explicit tracing using the `@traceable` decorator.

Instead of observing only the query pipeline, setup operations are also captured.

---

## What the Script Does

The RAG functionality remains the same, but key setup functions are wrapped with `@traceable`, including:

- PDF loading
- Document splitting
- Vector store creation
- Overall setup routine

---

## Expected Trace Structure

```text
setup_pipeline
│
├── load_pdf
├── split_documents
└── build_vectorstore

pdf_rag_query
│
├── Retriever
├── Prompt
├── Gemini
└── Parser
```

---

## Why This Matters

In production, preprocessing can be a major source of issues:

- Incorrect PDFs
- Poor chunk sizes
- Embedding failures
- Vector database creation errors

By tracing these operations, developers gain visibility into the entire pipeline rather than only the inference stage.

---

## LangSmith Concepts Learned

- `@traceable`
- Parent-child spans
- Tracing custom Python functions
- Separating setup from inference
- Improved debugging for RAG pipelines

---

# Example 5 — Production-Ready RAG with Index Caching

**Source:** `rag_v3.py`

## Objective

This example evolves the RAG application toward a production-oriented design by introducing deterministic index caching and explicit tracing for both cache hits and index creation.

---

## What the Script Does

The script fingerprints the PDF and indexing configuration to derive a unique cache key.

If a matching FAISS index already exists, it is loaded.

Otherwise, the PDF is:

- Processed
- Embedded
- Indexed
- Saved for reuse

The entire setup and query execution are wrapped in traced functions, producing a clear end-to-end execution history.

---

## Expected Trace Structure

```text
pdf_rag_full_run
│
├── setup_pipeline
│   │
│   ├── load_index
│   │
│   └── build_index
│       │
│       ├── load_pdf
│       ├── split_documents
│       └── build_vectorstore
│
├── Retriever
├── Prompt
├── Gemini
└── Parser
```

Depending on whether the index already exists, the trace will show either **load_index** or **build_index**, making cache behavior immediately visible.

---

## What This Example Teaches

- Tracing complex workflows
- Observing cache hits versus rebuilds
- Separating expensive setup from runtime inference
- Adding tags and metadata to improve trace organization

---

# Example 6 — Tool-Calling Agent

**Source:** `agent.py`

## Objective

This example introduces autonomous agents capable of selecting and invoking tools before producing a final answer.

---

## What the Script Does

The agent is configured with two tools:

- DuckDuckGo Search Tool
- Custom Weather API Tool

Depending on the user's request, the LLM decides whether to:

- Answer directly
- Invoke one or more tools
- Compose the final response

---

## Expected Trace Structure

```text
Agent
│
├── LLM Decision
├── DuckDuckGo Tool
├── Weather Tool
├── LLM Reasoning
└── Final Response
```

The exact sequence depends on the agent's decisions at runtime.

---

## What This Example Teaches

- Agent execution traces
- Tool invocation visibility
- Reasoning flow through multiple model calls
- Debugging incorrect tool selection or tool outputs

---

# Example 7 — LangGraph Workflow

**Source:** `langgraph_code_via_langsmith_traceability.py`

## Objective

The final example demonstrates tracing a graph-based workflow where multiple evaluation branches execute before converging into a final aggregation step.

---

## What the Script Does

An essay is evaluated independently across three dimensions:

- Language quality
- Analytical depth
- Clarity of thought

Each evaluator produces structured feedback and a score.

A final node aggregates these results into an overall assessment and computes the average score.

---

## Expected Trace Structure

```text
evaluate_upsc_essay
│
├── evaluate_language_fn
├── evaluate_analysis_fn
├── evaluate_thought_fn
│
└── final_evaluation_fn
```

Because the graph fans out and later joins, the trace exposes the execution topology much more clearly than a simple linear chain.

---

## What This Example Teaches

- Tracing LangGraph applications
- Tracing parallel branches
- Visualizing graph execution
- Structured outputs
- Tags and metadata on graph nodes
- Debugging complex multi-step AI workflows

---

# Summary

These seven progressively complex examples demonstrate how LangSmith evolves alongside an AI application:

| Example | Focus                  | Primary LangSmith Concept   |
| ------- | ---------------------- | --------------------------- |
| 1       | Single LLM Call        | Automatic tracing           |
| 2       | Sequential Chain       | Nested runs, tags, metadata |
| 3       | Basic RAG              | Retrieval tracing           |
| 4       | Manual Instrumentation | `@traceable` decorator      |
| 5       | Production RAG         | Cache-aware tracing         |
| 6       | Tool-Calling Agent     | Tool execution visibility   |
| 7       | LangGraph Workflow     | Graph execution tracing     |

Together, they provide a practical roadmap for understanding how LangSmith can be used to observe, debug, and optimize increasingly sophisticated AI systems—from a single prompt to production-grade RAG pipelines, autonomous agents, and graph-based workflows.

---

# 📊 What You Should Now Understand

After completing this guide, you should be comfortable with:

- Understanding the hierarchy of **Projects**, **Traces**, and **Runs**
- Inspecting prompts, responses, latency, token usage, and metadata
- Debugging sequential chains and multi-step workflows
- Tracing Retrieval-Augmented Generation (RAG) applications
- Instrumenting custom Python functions using the `@traceable` decorator
- Organizing traces with meaningful run names, tags, and metadata
- Observing cache behavior and production setup workflows
- Understanding how agents invoke tools and how those tool calls appear inside LangSmith
- Visualizing complex LangGraph executions with parallel branches and node-level traces
- Using traces to identify bottlenecks, failures, hallucinations, and incorrect tool usage

---

# 💡 Best Practices When Using LangSmith

As your AI applications become more complex, adopting a few best practices can significantly improve the usefulness of your traces.

- Assign meaningful run names to important workflows.
- Use tags to categorize traces by feature, environment, or application.
- Attach metadata such as model version, retrieval parameters, or user identifiers (where appropriate).
- Trace important custom business logic using the `@traceable` decorator instead of relying solely on automatic instrumentation.
- Keep setup workflows separate from runtime inference to simplify debugging.
- Regularly review traces to identify latency bottlenecks, prompt issues, retrieval failures, and unnecessary model calls.
- Combine tracing with LangSmith's monitoring, evaluation, and annotation capabilities to build a continuous improvement workflow.

---

# 🚀 What's Next?

This guide focused exclusively on **LangSmith Tracing**—the foundation of observability for AI applications.

Once you're comfortable reading and understanding traces, the next step is to explore LangSmith's broader capabilities:

- **Monitoring** for production metrics such as latency, token usage, and error rates.
- **Prompt Playground** for experimenting with prompt variations.
- **Datasets** for creating reusable evaluation benchmarks.
- **Experiments** for comparing prompts, models, or application versions.
- **Evaluation** using automated metrics or LLM-as-a-Judge.
- **Annotation Queues** for incorporating human feedback into your development workflow.
- **User Feedback Integration** to capture production feedback and continuously improve your application.

Together, these features transform LangSmith from a tracing platform into a complete observability, evaluation, and quality assurance solution for modern AI systems.

---

# 🎉 Conclusion

Tracing is often the first capability developers explore in LangSmith, but it is also the foundation upon which every other feature is built. Whether you're building a simple chatbot, a Retrieval-Augmented Generation (RAG) application, an autonomous AI agent, or a complex LangGraph workflow, understanding how your application executes is essential for debugging, optimization, and long-term maintenance.

The seven examples presented in this guide demonstrate how tracing evolves alongside application complexity. Beginning with a single LLM invocation and progressing through sequential chains, RAG pipelines, custom instrumentation, production-ready indexing, tool-calling agents, and graph-based workflows, each example introduces new observability concepts while reinforcing the previous ones.

By following these examples and running the accompanying Python scripts, you should now be able to:

- Confidently instrument AI applications with LangSmith.
- Navigate and interpret traces within the LangSmith UI.
- Understand the relationship between runs, spans, and traces.
- Debug multi-step AI workflows effectively.
- Identify performance bottlenecks and application failures.
- Build AI systems that are observable, maintainable, and production-ready.

Observability is no longer optional for modern AI applications. As systems become increasingly complex, tools like LangSmith provide the visibility required to understand, debug, evaluate, and continuously improve every stage of an application's lifecycle.

---

## Happy Learning! 🚀
