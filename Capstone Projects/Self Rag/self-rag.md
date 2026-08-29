## Problems with Normal RAG

1. Indiscriminate or Unnecessary Retrieval
2. RAG blindly trust retrieved documents
3. RAG doesn't verify it's own answers

## What self-RAG is?

Self-RAG stands for self reflective RAG where the LLM actively judges its own retrieval, evidence, and answers instead of blindly trustion retrieved documents

## Checks that the self-RAG system do

1. Should retrieval happen?
2. Are the retrieved documenst relevent?
3. Is the generated response grounded in the retrieved documents
4. Does the response actually answer the user's question
