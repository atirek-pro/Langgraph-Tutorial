# Corrective RAG — Technical Learning Guide

> **Learning approach:** We are building CRAG step by step instead of implementing the entire architecture at once.  
> **Current phase:** Phase 1 — Knowledge Refinement  
> **Goal:** Understand what we are building, why we are building it, and how the code implements it.

---

# Section 1 — Understanding RAG and CRAG

## 1.1 Introduction

Large Language Models already contain a large amount of knowledge in their parameters, but that knowledge is not always enough for an application.

A model may:

- not know about your private documents,
- have outdated knowledge,
- misunderstand a question,
- or generate an answer that is not supported by reliable evidence.

**Retrieval-Augmented Generation (RAG)** solves part of this problem by retrieving relevant information and giving it to the LLM before generating the answer.

But RAG has another problem:

> **What happens when the retriever retrieves the wrong information?**

That is where **Corrective Retrieval-Augmented Generation (CRAG)** comes in.

---

## 1.2 Learning Objective

After this section, you should be able to explain:

- What RAG is.
- How a traditional RAG pipeline works.
- Why retrieval quality matters.
- What CRAG adds to RAG.
- How CRAG differs architecturally from traditional RAG.
- When CRAG is useful.
- When traditional RAG may still be the better choice.

After Section 1, we move into the first implementation phase:

> **Phase 1 — Knowledge Refinement**

---

## 1.3 What Is RAG?

RAG stands for:

> **Retrieval-Augmented Generation**

The basic idea is simple:

Instead of asking an LLM to answer a question only from its internal knowledge, we first retrieve relevant information and provide that information as context.

### Traditional RAG

```text
User Question
      │
      ▼
Create Query Embedding
      │
      ▼
Vector Database
      │
      ▼
Retrieve Top-k Documents
      │
      ▼
LLM + Retrieved Context
      │
      ▼
Answer
```

### Example

Suppose we have a company knowledge base containing:

```text
employee_handbook.pdf
product_documentation.pdf
engineering_guidelines.pdf
```

The user asks:

> "What is our company's remote-work policy?"

RAG does roughly this:

```text
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
Relevant chunks from employee_handbook.pdf
   ↓
LLM
   ↓
Answer
```

The important part is that the LLM gets information from the company's documents rather than relying only on its pretrained knowledge.

---

## 1.4 The Problem With Traditional RAG

Traditional RAG generally assumes that the retrieved information is useful.

But retrieval can fail.

For example:

```text
Question:
"What is the bias-variance tradeoff?"

Retrieved chunks:

Chunk 1 → Bias-variance explanation      ✅
Chunk 2 → Overfitting                    🟡
Chunk 3 → Transformer architecture      ❌
Chunk 4 → Attention mechanisms           ❌
```

A traditional RAG pipeline may pass all four chunks to the LLM:

```text
Question
   +
Chunk 1
Chunk 2
Chunk 3
Chunk 4
   ↓
LLM
```

Now the LLM has to work with both useful and irrelevant information.

This creates two problems:

1. **Context pollution** — irrelevant information consumes context.
2. **Bad evidence** — the model may use irrelevant or incorrect information when generating the answer.

CRAG addresses this retrieval problem by introducing a correction mechanism.

---

# 1.5 What Is CRAG?

CRAG stands for:

> **Corrective Retrieval-Augmented Generation**

CRAG extends RAG by adding mechanisms that evaluate and improve retrieved knowledge before final answer generation.

The original CRAG architecture introduces a **retrieval evaluator** that assesses the overall quality of retrieved documents and uses that result to trigger different corrective actions.

It also introduces a **decompose-then-recompose** approach that breaks retrieved knowledge into smaller pieces and filters irrelevant information.

The official CRAG repository describes these as core parts of the approach.

The simplified idea is:

```text
Traditional RAG:

Retrieve → Generate


CRAG:

Retrieve
   ↓
Evaluate
   ↓
Correct / Refine
   ↓
Generate
```

So the key difference is:

> **Traditional RAG retrieves information. CRAG also checks and corrects what it retrieved.**

---

# 1.6 Traditional RAG vs CRAG

|                                  | Traditional RAG       | CRAG                               |
| -------------------------------- | --------------------- | ---------------------------------- |
| Retrieval                        | ✅                    | ✅                                 |
| Retrieval quality evaluation     | Usually ❌            | ✅                                 |
| Knowledge refinement             | Usually ❌            | ✅                                 |
| Irrelevant information filtering | Limited               | ✅                                 |
| Corrective actions               | ❌                    | ✅                                 |
| External knowledge fallback      | Application-dependent | Part of the original CRAG approach |
| Architecture                     | Simpler               | More complex                       |
| Latency                          | Lower                 | Usually higher                     |
| Implementation effort            | Lower                 | Higher                             |
| Robustness to bad retrieval      | Lower                 | Higher                             |

---

# 1.7 Architecture Comparison

## Traditional RAG

```text
              ┌──────────────┐
              │ User Query   │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   Retriever  │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ Top-k Chunks │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │     LLM      │
              └──────┬───────┘
                     │
                     ▼
                  Answer
```

### Main characteristic

The retrieved context is generally passed directly to the generator.

---

## CRAG

```text
                    ┌──────────────┐
                    │  User Query  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Retriever  │
                    └──────┬───────┘
                           │
                           ▼
                  Retrieved Knowledge
                           │
                           ▼
                 ┌──────────────────┐
                 │ Retrieval        │
                 │ Evaluator        │
                 └────────┬─────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
           Correct     Ambiguous   Incorrect
              │           │           │
              └─────┬─────┴─────┬─────┘
                    │           │
                    ▼           ▼
              Correction / External
               Refinement    Knowledge
                    │           │
                    └─────┬─────┘
                          │
                          ▼
                   Refined Context
                          │
                          ▼
                         LLM
                          │
                          ▼
                        Answer
```

This is a simplified learning diagram. The complete CRAG implementation contains separate knowledge-preparation paths for correct, incorrect, and ambiguous retrieval situations.

---

# 1.8 Why Is CRAG Better Than Traditional RAG?

CRAG is **not automatically better for every application**.

It is better when retrieval quality is a significant source of failure.

### Traditional RAG

```text
Bad Retrieval
     ↓
Bad Context
     ↓
LLM
     ↓
Potentially Bad Answer
```

### CRAG

```text
Bad Retrieval
     ↓
Evaluate
     ↓
Correction
     ↓
Better Evidence
     ↓
LLM
     ↓
More Robust Answer
```

The major advantage is therefore:

> **CRAG reduces the amount of blind trust placed in the retriever.**

The original CRAG paper was specifically motivated by the fact that RAG relies heavily on the relevance of retrieved documents and can behave poorly when retrieval goes wrong.

---

# 1.9 Pros and Cons

## Traditional RAG

### Pros

- Simple architecture
- Easy to implement
- Lower latency
- Lower infrastructure cost
- Easier to debug
- Good enough for many internal knowledge-base applications

### Cons

- Retrieval errors can directly affect generation
- Irrelevant chunks may remain in context
- No built-in corrective mechanism
- Quality depends heavily on retrieval

---

## CRAG

### Pros

- Evaluates retrieved knowledge
- Can filter irrelevant information
- Can react to poor retrieval
- Can use external knowledge when necessary
- More robust to retrieval failures
- Better suited to systems where retrieval quality is critical

### Cons

- More complex architecture
- Additional model inference
- Higher latency
- Higher compute/API cost
- More components to monitor and debug
- Requires careful threshold and routing design

---

# 1.10 Which Is Better for Production?

There is no universal answer.

The correct production choice depends on the application.

### Use traditional RAG when:

- Your knowledge base is small and well controlled.
- Retrieval quality is already high.
- Documents change infrequently.
- Low latency is important.
- The cost of occasional retrieval mistakes is relatively low.
- You want the simplest production architecture.

### Consider CRAG when:

- Retrieval mistakes are expensive.
- The knowledge base is large or noisy.
- Documents can contain unrelated information.
- Questions can fall outside the knowledge base.
- The application requires higher reliability.
- External knowledge may be needed when internal retrieval fails.

A useful production rule is:

> **Do not add CRAG just because it is more sophisticated. Add it when retrieval failures are important enough to justify the additional complexity and latency.**

---

# 1.11 When Should You Use CRAG?

CRAG becomes particularly useful when:

```text
Retrieval Quality
      ↓
has a large impact on
      ↓
Answer Quality
```

Examples include:

- Enterprise knowledge assistants
- Research assistants
- Technical documentation assistants
- Long-running knowledge bases
- Systems with noisy document collections
- Systems where outdated or incorrect retrieval is costly
- Applications that may need external knowledge when local retrieval is insufficient

For a simple FAQ system with a small, highly curated knowledge base, traditional RAG may be enough.

---

# Section 2 — Phase 1: Knowledge Refinement

## 2.1 What Are We Building?

In Phase 1, we are implementing **one part of CRAG**:

> **Knowledge Refinement**

We are intentionally not implementing the complete CRAG architecture yet.

Our current pipeline is:

```text
Retrieve
   ↓
Knowledge Refinement
   ↓
Generate
```

Later phases will add retrieval evaluation and corrective routing.

This makes the project easier to understand because we can learn each CRAG component independently.

---

# 2.2 What Is Knowledge Refinement?

Knowledge refinement means:

> **Taking retrieved information, breaking it into smaller pieces, removing information that does not help answer the question, and rebuilding the context from the useful pieces.**

Imagine retrieval returns:

```text
Sentence 1 → Directly answers the question       ✅
Sentence 2 → Useful supporting information       ✅
Sentence 3 → Completely unrelated                ❌
Sentence 4 → Another unrelated topic             ❌
```

Instead of giving all four sentences to the LLM:

```text
All retrieved information
        ↓
       LLM
```

we do:

```text
All retrieved information
        ↓
Break into strips
        ↓
Evaluate strips
        ↓
Remove irrelevant strips
        ↓
Refined context
        ↓
LLM
```

---

# 2.3 Why Do We Need Knowledge Refinement?

Suppose the retriever returns a chunk like:

```text
The bias-variance tradeoff describes the relationship
between model complexity and generalization.

High bias can lead to underfitting.

Transformers use self-attention to process tokens.

Attention mechanisms are important in NLP.

High variance can lead to overfitting.
```

The question is:

> "What is the bias-variance tradeoff?"

Not every sentence is useful.

The relevant information is:

```text
The bias-variance tradeoff describes the relationship
between model complexity and generalization.

High bias can lead to underfitting.

High variance can lead to overfitting.
```

The Transformer and attention sentences are unrelated.

Knowledge refinement removes them before generation.

---

# 2.4 Phase 1 Architecture

Our implementation is:

```text
                  User Question
                       │
                       ▼
                 FAISS Retriever
                       │
                       ▼
                Retrieved Chunks
                       │
                       ▼
             Sentence Decomposition
                       │
                       ▼
                Knowledge Strips
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
             T5       T5       T5
          Evaluator Evaluator Evaluator
              │        │        │
              ▼        ▼        ▼
            Score    Score    Score
              │        │        │
              ▼        ▼        ▼
          KEEP/DROP KEEP/DROP KEEP/DROP
              │        │        │
              └────────┼────────┘
                       ▼
                 Kept Strips
                       │
                       ▼
                Refined Context
                       │
                       ▼
                 Gemini 2.5 Flash
                       │
                       ▼
                     Answer
```

This is the first part of CRAG that we are implementing.

---

# 2.5 Step 1 — Retrieve Documents

We first retrieve candidate documents using:

```text
Gemini Embeddings
        ↓
FAISS
        ↓
Top-k documents
```

Our implementation:

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### Why?

We need candidate knowledge before we can refine it.

At this stage we are **not deciding whether the retrieval is correct overall**.

We are simply collecting candidate evidence.

That larger retrieval-evaluation problem will be handled in a later phase.

---

# 2.6 Step 2 — Decompose Retrieved Knowledge

After retrieval, we combine the retrieved documents and split them into sentences.

```python
def decompose_to_sentences(text: str) -> List[str]:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]
```

The purpose is simple:

```text
Large retrieved chunk
        ↓
Smaller knowledge strips
```

Why?

Because it is easier to decide:

> "Is this specific piece useful?"

than:

> "Is this entire large chunk useful?"

The original CRAG implementation supports several decomposition modes. Our learning implementation starts with sentence-level decomposition because it makes the refinement process easy to see and understand.

---

# 2.7 Step 3 — Evaluate Each Strip With T5

We use a pretrained T5 sequence-classification model.

The model receives:

```text
Question + Strip
```

For example:

```text
Question:
What is the bias-variance tradeoff?

Strip:
High bias can cause underfitting.
```

The model produces a single scalar score:

```text
0.73
```

Another strip:

```text
Strip:
Transformers use self-attention.
```

might produce:

```text
-0.88
```

The score is the model's relevance signal.

---

# 2.8 Why T5 Instead of the Final LLM?

We intentionally separate the two jobs.

### T5

```text
Question + Strip
        ↓
Relevance Score
```

### Gemini

```text
Question + Refined Context
        ↓
Final Answer
```

This gives us:

```text
Retriever
   ↓
T5 Evaluator
   ↓
Refined Knowledge
   ↓
Gemini Generator
```

The evaluator does not generate the answer.

The generator does not decide which strips should be kept.

Each model has a specific job.

---

# 2.9 Step 4 — Apply the Filtering Threshold

Our tested T5 model produces a single scalar.

We use:

```text
Threshold = -0.5
```

The logic is:

```python
score = outputs.logits.squeeze().item()

keep = score >= -0.5
```

Therefore:

```text
             T5 Score
                 │
          ┌──────┴──────┐
          │             │
      >= -0.5         < -0.5
          │             │
        KEEP           DROP
```

Example:

```text
Relevant strip
Score = 0.730544
        ↓
      KEEP
```

```text
Irrelevant strip
Score = -0.887776
        ↓
       DROP
```

---

# 2.10 Important: This Is Not a Probability

Our model has:

```text
num_labels = 1
```

Therefore its output is one scalar.

We should use:

```python
score = outputs.logits.squeeze().item()
```

We should **not** do:

```python
softmax(outputs.logits)
```

and interpret the result as confidence.

A softmax over one value will always produce:

```text
1.0
```

That would not tell us whether the strip is relevant.

---

# 2.11 Step 5 — Recompose the Knowledge

After filtering, we have:

```text
All strips:

Strip 1 → KEEP
Strip 2 → DROP
Strip 3 → KEEP
Strip 4 → DROP
```

We keep only:

```text
Strip 1
Strip 3
```

and rebuild the context:

```python
refined_context = "\n".join(
    kept_strips
)
```

Now the generator sees:

```text
Refined Context
      ↓
Gemini 2.5 Flash
      ↓
Answer
```

instead of the original noisy retrieval.

---

# 2.12 The Complete Phase 1 Flow

Putting everything together:

```text
User Question
      │
      ▼
Gemini Embedding
      │
      ▼
FAISS Retrieval
      │
      ▼
Top-k Retrieved Chunks
      │
      ▼
Sentence Decomposition
      │
      ▼
Knowledge Strips
      │
      ▼
T5 Relevance Evaluator
      │
      ▼
Relevance Score
      │
      ▼
Score >= -0.5 ?
      │
   ┌──┴──┐
   │     │
 YES     NO
   │     │
 KEEP  DROP
   │
   ▼
Kept Strips
   │
   ▼
Refined Context
   │
   ▼
Gemini 2.5 Flash
   │
   ▼
Final Answer
```

---

# 2.13 Project Setup

## Folder Structure

Use:

```text
Corrective RAG/
│
├── corrective_rag.py
├── test_filtering_model.py
├── .env
│
├── documents/
│   ├── book1.pdf
│   ├── book2.pdf
│   └── book3.pdf
│
└── filtering_model/
    ├── added_tokens.json
    ├── config.json
    ├── model-001.safetensors
    ├── special_tokens_map.json
    ├── spiece.model
    └── tokenizer_config.json
```

---

## Install Dependencies

Create a virtual environment:

```bash
python -m venv myenv
```

Windows:

```bash
myenv\Scripts\activate
```

Install the required packages:

```bash
pip install torch
pip install transformers
pip install safetensors
pip install python-dotenv

pip install langchain-community
pip install langchain-google-genai
pip install langchain-text-splitters
pip install langgraph
pip install faiss-cpu
pip install pypdf
```

---

# 2.14 Environment Variables

Create:

```text
.env
```

Add:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

---

# 2.15 Download the T5 Pretrained Model

The knowledge-refinement implementation requires the pretrained T5 evaluator.

### T5 Model Download

> **TODO:** [Pretrained T5 Model for filtering knowledge strips](https://drive.google.com/drive/folders/1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7)

```text
[PLACEHOLDER — ADD T5 MODEL DOWNLOAD LINK]
```

Place the downloaded files inside:

```text
./filtering_model/
```

Expected structure:

```text
filtering_model/
├── added_tokens.json
├── config.json
├── model-001.safetensors
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json
```

---

# 2.16 Test the T5 Model First

Before running the full RAG pipeline, test the evaluator independently.

Run:

```bash
python test_filtering_model.py
```

You should see results similar to:

```text
Relevant strip:
Score = 0.730544
Decision = KEEP

Irrelevant strip:
Score = -0.887776
Decision = DROP
```

The exact numbers can change with different inputs.

What matters is the behavior:

```text
score >= -0.5 → KEEP
score <  -0.5 → DROP
```

---

# 2.17 Run Phase 1

Once the T5 model works independently:

```bash
python corrective_rag.py
```

The LangGraph flow is:

```text
START
  ↓
retrieve
  ↓
refine
  ↓
generate
  ↓
END
```

---

# 2.18 What We Have Finished

At the end of Phase 1, we have:

```text
✅ Document loading
        ↓
✅ Document chunking
        ↓
✅ Gemini embeddings
        ↓
✅ FAISS retrieval
        ↓
✅ Sentence decomposition
        ↓
✅ T5 strip evaluation
        ↓
✅ KEEP / DROP filtering
        ↓
✅ Refined context
        ↓
✅ Gemini answer generation
```

But we have **not** completed the full CRAG architecture yet.

---

# 2.19 What Comes Next?

The next phase will answer a different question.

### Phase 1

> **"Is this particular strip useful?"**

```text
Question + Strip
       ↓
T5
       ↓
Relevance Score
       ↓
KEEP / DROP
```

### Next Phase

> **"How good is the retrieved knowledge overall?"**

```text
Question + Retrieved Documents
       ↓
Retrieval Evaluator
       ↓
Retrieval Quality
       ↓
Correct / Ambiguous / Incorrect
       ↓
Corrective Action
```

This distinction is important.

**Knowledge refinement** cleans the retrieved knowledge.

**Retrieval evaluation** decides whether the retrieval itself is good enough and what the system should do next.

The official CRAG architecture combines these ideas with different corrective paths, including external knowledge search for cases where internal retrieval is insufficient.

---

# Phase 1 Summary

We started with:

```text
Traditional RAG

Retrieve → Generate
```

and implemented the first corrective step:

```text
Retrieve
   ↓
Decompose
   ↓
Evaluate
   ↓
Filter
   ↓
Recompose
   ↓
Generate
```

The key idea to remember is:

> **Don't give the generator everything the retriever found. First identify the pieces that actually help answer the question.**

That is the foundation of the knowledge-refinement phase.

---

## References

- [Corrective Retrieval Augmented Generation — Paper](https://arxiv.org/abs/2401.15884)
- [Official CRAG Repository](https://github.com/HuskyInSalt/CRAG)

The official repository describes CRAG as a corrective extension to RAG that evaluates retrieved knowledge, supports corrective retrieval actions, and uses a decompose-then-recompose process to filter irrelevant retrieved information.

---

# Learning Roadmap

This document is intentionally structured so that future phases can be added without rewriting Phase 1.

```text
CRAG Learning Guide
│
├── Section 1 — Understanding RAG & CRAG
│
├── Section 2 — Phase 1: Knowledge Refinement
│
├── Section 3 — Phase 2: Retrieval Evaluation
│
├── Section 4 — Phase 3: Corrective Routing
│
├── Section 5 — Phase 4: External Knowledge Search
│
└── Section 6 — Complete CRAG Implementation
```

**Current progress:**

```text
Section 1    ████████████████████  Complete
Phase 1      ████████████████████  Complete
Phase 2      ░░░░░░░░░░░░░░░░░░░░  Next
Phase 3      ░░░░░░░░░░░░░░░░░░░░  Planned
Phase 4      ░░░░░░░░░░░░░░░░░░░░  Planned
Complete     ░░░░░░░░░░░░░░░░░░░░  Planned
```
