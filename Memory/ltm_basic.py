from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Creating Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -----------------------------
# Creating Store
# -----------------------------

store = InMemoryStore(
    index={
        "embed": embedding_model,
        "dims": 384
    }
)

# -----------------------------
# Creating Namespace
# -----------------------------

namespace = ("user", "u1")

# -----------------------------
# Adding Memories
# -----------------------------

store.put(namespace,"1",{"data": "User likes pizza"})
store.put(namespace,"2",{"data": "User prefers dark mode"})
store.put(namespace, "1", {"data": "User prefers concise answers over long explanations"})
store.put(namespace, "2", {"data": "User likes examples in Python"})
store.put(namespace, "3", {"data": "User usually works late at night"})
store.put(namespace, "4", {"data": "User prefers dark mode in applications"})
store.put(namespace, "5", {"data": "User is learning machine learning"})
store.put(namespace, "6", {"data": "User dislikes overly theoretical explanations"})
store.put(namespace, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "8", {"data": "User is based in India"})
store.put(namespace, "9", {"data": "User likes real-world analogies"})
store.put(namespace, "10", {"data": "User prefers bullet points over paragraphs"})

# -----------------------------
# Another Namespace
# -----------------------------

namespace2 = ("user", "u2")

store.put(
    namespace2,
    "1",
    {"data": "User likes pasta"}
)

store.put(
    namespace2,
    "2",
    {"data": "User prefers grid style navigation"}
)

# -----------------------------
# Retrieve Specific Memory
# -----------------------------

item = store.get(namespace2, "2")

print("Specific memory:")
print(item.value)


# -----------------------------
# Retrieve All Memories
# -----------------------------

print("\nAll memories for u2:")

items = store.search(namespace2)

for item in items:
    print(item.value)


# -----------------------------
# Semantic Search
# -----------------------------

print("\nSemantic search: learning")

items = store.search(
    namespace2,
    query="what is the user currently learning",
    limit=1
)

for item in items:
    print(item.value)


print("\nSemantic search: preferences")

items = store.search(
    namespace2,
    query="what are user's preferences",
    limit=3
)

for item in items:
    print(item.value)