import load_dataset
import SentenceTransformer
import chromadb
import ollama

# 1. Load dataset
dataset = load_dataset("Skysparko/battlefield", split="train")

# 2. Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Setup Chroma vector database
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("battlefield")

# Add dataset to vector DB
for i, row in enumerate(dataset):
    text = row["instruction"] + " " + row["output"]
    embedding = embedder.encode(text).tolist()
    collection.add(documents=[text], embeddings=[embedding], ids=[str(i)])

# 4. Chat function with RAG
def chat_with_model(query):
    # Embed query
    query_embedding = embedder.encode(query).tolist()
    # Retrieve top 3 matches
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = " ".join(results["documents"][0])

    # Send context + query to Ollama
    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant using the Battlefield dataset."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response["message"]["content"]

# 5. Run chatbot loop
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! 👋")
            break
        answer = chat_with_model(user_input)
        print("Chatbot:", answer)
