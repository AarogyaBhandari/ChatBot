from ollama import Client
import chromadb
import json


# 1. Setup ChromaDB + Ollama

chroma_client = chromadb.Client()
ollama_client = Client(host="http://localhost:11434")

collection = chroma_client.get_or_create_collection(
    name="simple_knowledge"
)


# 2. Build Vector DB (LINE LEVEL)

print("Building vector database from simple.txt ...")

with open("simple.txt", "r", encoding="utf-8") as f:
    articles = json.load(f)

doc_id = 0

for article in articles:
    title = article.get("title", "")
    content = article.get("content", "")

    # Split article into individual lines
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    for line_no, line in enumerate(lines):
        embedding = ollama_client.embed(
            model="nomic-embed-text",
            input=line
        )["embeddings"][0]

        collection.add(
            ids=[f"doc_{doc_id}"],
            documents=[line],
            embeddings=[embedding],
            metadatas=[{
                "title": title,
                "line_no": line_no
            }]
        )

        doc_id += 1

print(" Vector DB built successfully")


# 3. Query (ONE LINE ANSWER)

query = input("Enter a query: ")

query_embedding = ollama_client.embed(
    model="nomic-embed-text",
    input=query
)["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1  #return top 1 result for simplicity
)


# 4. Output

answer = results["documents"][0][0]
source = results["metadatas"][0][0]

print("\n Question:")
print(query)

print("\n Answer:")
print(answer)

print("\n Source:")
print(f"Title: {source['title']}, Line: {source['line_no']}")
