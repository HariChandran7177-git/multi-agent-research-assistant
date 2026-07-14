
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = "research_docs"


def ensure_collection():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )


def retriever_node(state):
    ensure_collection()

    # Store this run's research results into Qdrant so they're searchable
    texts = state["research_results"]
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": texts[i]})
        for i in range(len(texts))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # Now search back using the original query
    query_vector = embeddings.embed_query(state["query"])
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
    )

    state["retrieved_docs"] = [point.payload["text"] for point in results.points]
    return state


if __name__ == "__main__":
    test_state = {
        "query": "effects of climate change on coral reefs",
        "research_results": [
            "Rising ocean temperatures cause coral bleaching.",
            "Ocean acidification weakens coral skeletal structures.",
            "Coral reefs support 25% of marine species despite covering less than 1% of the ocean floor."
        ]
    }
    result = retriever_node(test_state)
    for doc in result["retrieved_docs"]:
        print(doc)
        print("---")