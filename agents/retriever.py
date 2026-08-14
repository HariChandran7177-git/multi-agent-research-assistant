import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def get_qdrant_client():
    """Initialize Qdrant client with fallback to in‑memory if remote connection fails."""
    try:
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        logger.info("Connected to remote Qdrant instance.")
        return client
    except Exception as e:
        logger.warning(f"Remote Qdrant connection failed ({e}); falling back to in‑memory client.")
        return QdrantClient(":memory:")

qdrant = get_qdrant_client()
COLLECTION_NAME = "research_docs"


def ensure_collection():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        logger.info(f"Collection '{COLLECTION_NAME}' not found — creating it")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )
        # Payload index is only needed for remote Qdrant; ignore errors for in‑memory client
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="session_id",
                field_schema="keyword",
            )
        except Exception as e:
            logger.debug(f"Payload index creation skipped or failed ({e}); may be in‑memory client.")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upsert_with_retry(qdrant, collection_name, points):
    return qdrant.upsert(collection_name=collection_name, points=points)


def retriever_node(state):
    ensure_collection()

    session_id = str(uuid.uuid4())  # unique tag for THIS run only

    texts = state["research_results"]
    logger.info(f"Embedding {len(texts)} research results")
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={"text": texts[i], "session_id": session_id}
        )
        for i in range(len(texts))
    ]

    try:
        upsert_with_retry(qdrant, COLLECTION_NAME, points)
        logger.info(f"Upserted {len(points)} points into Qdrant")
    except Exception as e:
        logger.error(f"Qdrant upsert failed after retries: {e}")
        state["retrieved_docs"] = texts[:5]
        return state

    query_vector = embeddings.embed_query(state["query"])
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(
                key="session_id", match=MatchValue(value=session_id))]
        ),
    )

    logger.info(f"Retrieved {len(results.points)} relevant documents")
    state["retrieved_docs"] = [point.payload["text"]
                               for point in results.points]
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
