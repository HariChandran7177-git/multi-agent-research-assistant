import os
import asyncio
import uuid
from typing import List
from functools import partial
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from tenacity import retry, stop_after_attempt, wait_exponential
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics
from core.config import MAX_ITERATIONS, AGENT_TIMEOUT

load_dotenv()

logger = get_logger(__name__)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def get_qdrant_client():
    """Initialize Qdrant client."""
    try:
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        # Verify the connection actually works, not just that the client object was created
        client.get_collections()
        logger.info("Connected to remote Qdrant instance.")
        return client
    except Exception as e:
        logger.error(f"CRITICAL: Qdrant connection failed: {e}")
        raise RuntimeError(
            f"Qdrant is unreachable. Original error: {e}"
        )


qdrant = None
_qdrant_initialized = False


def _initialize_qdrant():
    """Lazily initialize Qdrant client if not already initialized."""
    global qdrant, _qdrant_initialized
    if _qdrant_initialized:
        return qdrant
    try:
        qdrant = get_qdrant_client()
        _qdrant_initialized = True
        return qdrant
    except RuntimeError:
        _qdrant_initialized = True
        return None


COLLECTION_NAME = "research_docs"


def ensure_collection():
    """Ensure collection exists - returns False if Qdrant is unavailable."""
    global qdrant
    if qdrant is None:
        return False
    try:
        collections = qdrant.get_collections().collections
        if COLLECTION_NAME not in [c.name for c in collections]:
            logger.info(f"Collection '{COLLECTION_NAME}' not found — creating it")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
            # Payload index is only needed for remote Qdrant; ignore errors for in-memory client
            try:
                qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="session_id",
                    field_schema="keyword",
                )
                qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="user_id",
                    field_schema="keyword",
                )
            except Exception as e:
                logger.debug(f"Payload index creation skipped or failed ({e}); may be in-memory client.")
        return True
    except Exception as e:
        logger.warning(f"ensure_collection failed: {e}")
        qdrant = None
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def upsert_with_retry(qdrant, collection_name, points):
    return qdrant.upsert(collection_name=collection_name, points=points)


def _fallback_retrieval(state: ResearchState, texts: List[str], error_msg: str = "") -> ResearchState:
    """Return empty results when Qdrant is unavailable."""
    if error_msg:
        logger.warning(f"Qdrant unavailable, using fallback: {error_msg}")
    else:
        logger.warning("Qdrant unavailable, using fallback (no error details)")

    state["retrieved_docs"] = []
    state["qdrant_scores"] = []
    state["retrieval_available"] = False
    return state


async def retriever_node(state: ResearchState) -> ResearchState:
    """Async retriever node with timeout and metrics."""
    loop = asyncio.get_event_loop()

    # Initialize Qdrant
    global qdrant
    qdrant = _initialize_qdrant()

    # If Qdrant is unavailable, use fallback
    if qdrant is None:
        return _fallback_retrieval(state, state.get("research_results", []), "Qdrant client not available")

    session_id = str(uuid.uuid4())  # unique tag for THIS run only
    user_id = state.get("user_id", "default_user")
    texts = state.get("research_results", [])

    # Ensure collection exists
    try:
        if not ensure_collection():
            return _fallback_retrieval(state, texts, "Failed to ensure collection")
    except Exception as e:
        logger.warning(f"ensure_collection raised exception: {e}")
        return _fallback_retrieval(state, texts, f"ensure_collection error: {e}")

    logger.info(f"Embedding {len(texts)} research results")

    try:
        # Embedding with timeout protection
        vectors = await asyncio.wait_for(
            loop.run_in_executor(None, embeddings.embed_documents, texts),
            timeout=AGENT_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"Embedding timed out after {AGENT_TIMEOUT}s")
        return _fallback_retrieval(state, texts, "embedding_timeout")
    except Exception as e:
        logger.error(f"Error embedding content ({type(e).__name__}): {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.warning("Gemini rate limit hit during embedding. Using fallback.")
            return _fallback_retrieval(state, texts, "rate_limit")
        return _fallback_retrieval(state, texts, f"embedding error: {e}")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={"text": texts[i], "session_id": session_id, "user_id": user_id}
        )
        for i in range(len(texts))
    ]

    # Upsert to Qdrant
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, upsert_with_retry, qdrant, COLLECTION_NAME, points),
            timeout=AGENT_TIMEOUT
        )
        logger.info(f"Upserted {len(points)} points into Qdrant")
    except asyncio.TimeoutError:
        logger.error(f"Qdrant upsert timed out after {AGENT_TIMEOUT}s")
        return _fallback_retrieval(state, texts, "upsert_timeout")
    except Exception as e:
        logger.warning(f"Qdrant upsert failed after retries: {e}")
        return _fallback_retrieval(state, texts, f"upsert error: {e}")

    # Embed query
    try:
        query_vector = await asyncio.wait_for(
            loop.run_in_executor(None, embeddings.embed_query, state["query"]),
            timeout=AGENT_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"Query embedding timed out after {AGENT_TIMEOUT}s")
        return _fallback_retrieval(state, texts, "query_embedding_timeout")
    except Exception as e:
        logger.error(f"Error embedding query ({type(e).__name__}): {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.warning("Gemini rate limit hit during query embedding. Using fallback.")
            return _fallback_retrieval(state, texts, "rate_limit")
        return _fallback_retrieval(state, texts, f"query embedding error: {e}")

    # Query Qdrant for relevant documents
    try:
        query_filter = Filter(
            must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
        )
        query_fn = partial(qdrant.query_points,
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            query_filter=query_filter,
        )
        results = await asyncio.wait_for(
            loop.run_in_executor(None, query_fn),
            timeout=AGENT_TIMEOUT
        )

        logger.info(f"Retrieved {len(results.points)} relevant documents")
        state["retrieved_docs"] = [point.payload["text"] for point in results.points]
        state["qdrant_scores"] = [point.score for point in results.points]
        state["retrieval_available"] = True

        # Record metrics
        metrics.end_agent("retriever", input_tokens=len(state["query"]), output_tokens=len(state["retrieved_docs"]))
    except asyncio.TimeoutError:
        logger.error(f"Query points timed out after {AGENT_TIMEOUT}s")
        return _fallback_retrieval(state, texts, "query_timeout")
    except Exception as e:
        logger.warning(f"Qdrant query failed: {e}")
        return _fallback_retrieval(state, texts, f"query error: {e}")

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
