import os
import asyncio
import uuid
import threading
import warnings
import re
from typing import List, Dict, Any
from functools import partial
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from tenacity import retry, stop_after_attempt, wait_exponential
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics
from core.config import AGENT_TIMEOUT, GROQ_API_KEY, MODEL_CACHE_DIR, DATA_DIR

load_dotenv()
logger = get_logger(__name__)

# ── SentenceTransformer Embeddings ─────────────
_st_model = None
_st_lock = threading.Lock()


def _background_warmup():
    """Background worker to import PyTorch and warm up embeddings concurrently."""
    try:
        get_embeddings()
    except Exception:
        pass


def get_embeddings():
    """Get SentenceTransformer engine (thread-safe lazy-init with background warm-up)."""
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                logger.info("Initializing SentenceTransformer (all-MiniLM-L6-v2, 384-dim)...")
                
                os.environ["HF_HOME"] = MODEL_CACHE_DIR
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                
                from sentence_transformers import SentenceTransformer
                import glob
                snapshots = glob.glob(os.path.join(MODEL_CACHE_DIR, "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots", "*"))
                if snapshots:
                    _st_model = SentenceTransformer(snapshots[0], local_files_only=True)
                else:
                    _st_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=MODEL_CACHE_DIR)
    return _st_model


if os.getenv("PREWARM_EMBEDDINGS", "0") == "1":
    threading.Thread(target=_background_warmup, daemon=True, name="st-warmup").start()


class LazyEmbeddingsWrapper:
    """Thin wrapper that lazily initialises the embedding client with torch.no_grad() safety."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            import torch
            with torch.no_grad():
                res = get_embeddings().encode(texts, show_progress_bar=False, batch_size=32)
                return res.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer embedding error ({e})")
            raise RuntimeError(f"EMBEDDING FAILURE: {str(e)}")

    def embed_query(self, text: str) -> list[float]:
        try:
            import torch
            with torch.no_grad():
                res = get_embeddings().encode([text], show_progress_bar=False)
                return res[0].tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer query embedding error ({e})")
            raise RuntimeError(f"EMBEDDING FAILURE: {str(e)}")


embeddings = LazyEmbeddingsWrapper()
EMBEDDING_DIM = 384


def get_qdrant_client():
    """Initialize Qdrant client: remote if reachable, otherwise in-memory (zero lock contention)."""
    remote_url = os.getenv("QDRANT_URL")
    remote_key = os.getenv("QDRANT_API_KEY")
    if remote_url:
        try:
            client = QdrantClient(url=remote_url, api_key=remote_key, timeout=1, check_compatibility=False)
            client.get_collections()
            logger.info("Connected to remote Qdrant instance.")
            return client
        except Exception as e:
            logger.debug(f"Remote Qdrant unavailable ({e}), using in-memory client.")

    try:
        client = QdrantClient(":memory:", check_compatibility=False)
        logger.info("Connected to transient in-memory Qdrant.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize in-memory Qdrant: {e}")
        return None


qdrant = None
_qdrant_initialized = False


def _initialize_qdrant():
    """Lazily initialize Qdrant client."""
    global qdrant, _qdrant_initialized
    if _qdrant_initialized and qdrant is not None:
        return qdrant
    _qdrant_initialized = False
    try:
        qdrant = get_qdrant_client()
        _qdrant_initialized = True
        return qdrant
    except Exception:
        qdrant = None
        return None


COLLECTION_NAME = "research_docs"


def ensure_collection():
    """Ensure collection exists with the correct vector dimension."""
    global qdrant
    if qdrant is None:
        return False
    try:
        collections = qdrant.get_collections().collections
        existing_names = [c.name for c in collections]

        if COLLECTION_NAME in existing_names:
            try:
                info = qdrant.get_collection(COLLECTION_NAME)
                existing_dim = info.config.params.vectors.size
                if existing_dim != EMBEDDING_DIM:
                    qdrant.delete_collection(COLLECTION_NAME)
                else:
                    return True
            except Exception:
                try:
                    qdrant.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}")
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def upsert_with_retry(qdrant_instance, collection_name, points):
    return qdrant_instance.upsert(collection_name=collection_name, points=points)


def _split_into_chunks(text: str, max_chars: int = 1100, overlap: int = 150) -> List[str]:
    """Split text into ~300-token chunks with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    if overlap >= max_chars:
        overlap = max(0, max_chars // 4)

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            break_pt = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if break_pt > start + (max_chars // 2):
                end = break_pt + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap if end < len(text) else len(text)
        if next_start <= start:
            start = end
        else:
            start = next_start
    return chunks


def _parse_entry(entry: str) -> Dict[str, str]:
    """Parse a structured research result entry into components."""
    meta = {
        "subtask": "",
        "title": "Untitled Source",
        "domain": "",
        "date": "Recent",
        "url": "",
        "content": entry
    }
    
    subtask_m = re.search(r"^Subtask:\s*(.+)$", entry, re.MULTILINE)
    title_m = re.search(r"^Title:\s*(.+)$", entry, re.MULTILINE)
    domain_m = re.search(r"^Domain:\s*(.+)$", entry, re.MULTILINE)
    date_m = re.search(r"^Date:\s*(.+)$", entry, re.MULTILINE)
    url_m = re.search(r"^URL:\s*(https?://[^\s]+)$", entry, re.MULTILINE)
    content_m = re.search(r"^Content:\s*(.+)$", entry, re.MULTILINE | re.DOTALL)

    if subtask_m: meta["subtask"] = subtask_m.group(1).strip()
    if title_m: meta["title"] = title_m.group(1).strip()
    if domain_m: meta["domain"] = domain_m.group(1).strip()
    if date_m: meta["date"] = date_m.group(1).strip()
    if url_m: meta["url"] = url_m.group(1).strip()
    if content_m: meta["content"] = content_m.group(1).strip()

    return meta


def _fallback_retrieval(state: ResearchState, texts: List[str], error_msg: str = "") -> ResearchState:
    """Fallback: formats raw snippets directly with numbered citation tags."""
    logger.warning(f"Retriever using direct citation fallback: {error_msg}")
    chunks = []
    retrieved_docs = []
    subtask_coverage = {}

    idx = 1
    for raw in texts:
        parsed = _parse_entry(raw)
        chunk_obj = {
            "id": idx,
            "text": parsed["content"][:800],
            "url": parsed["url"],
            "title": parsed["title"],
            "domain": parsed["domain"],
            "date": parsed["date"],
            "subtask": parsed["subtask"],
            "score": 0.75
        }
        chunks.append(chunk_obj)
        retrieved_docs.append(
            f"[{idx}] {chunk_obj['text']}\n(Source: {chunk_obj['domain']} | Date: {chunk_obj['date']} | URL: {chunk_obj['url']})"
        )
        if parsed["subtask"]:
            subtask_coverage[parsed["subtask"]] = True
        idx += 1
        if idx > 16:
            break

    state["retrieved_chunks"] = chunks
    state["retrieved_docs"] = retrieved_docs
    state["qdrant_scores"] = [0.75] * len(chunks)
    state["retrieval_available"] = False
    state["subtask_coverage"] = subtask_coverage
    return state


async def retriever_node(state: ResearchState) -> ResearchState:
    """
    Sub-task Aware Retriever Node.
    1. Chunks research passages into ~300-token chunks.
    2. Stores each chunk with URL, title, date, domain, and sub-task in Qdrant.
    3. Retrieves top 4 chunks PER sub-task in parallel vector lookups.
    4. Numbers each chunk [1], [2], [3] for strict inline citations.
    """
    loop = asyncio.get_running_loop()
    global qdrant
    qdrant = _initialize_qdrant()

    texts = state.get("research_results", [])
    plan = state.get("plan", [])
    if not plan:
        plan = [state.get("query", "")]

    if not texts:
        logger.info("No research results to embed")
        return _fallback_retrieval(state, texts, "No research results to embed")

    if qdrant is None or not ensure_collection():
        return _fallback_retrieval(state, texts, "Qdrant unavailable")

    session_id = str(uuid.uuid4())
    user_id = state.get("user_id", "default_user")

    # 1. Chunk all parsed items
    all_chunks_payload = []
    chunk_texts_to_embed = []

    for entry in texts:
        parsed = _parse_entry(entry)
        text_chunks = _split_into_chunks(parsed["content"])
        for ch in text_chunks:
            chunk_texts_to_embed.append(ch)
            all_chunks_payload.append({
                "text": ch,
                "url": parsed["url"],
                "title": parsed["title"],
                "domain": parsed["domain"],
                "date": parsed["date"],
                "subtask": parsed["subtask"],
                "session_id": session_id,
                "user_id": user_id
            })

    logger.info(f"Embedding {len(chunk_texts_to_embed)} chunks into Qdrant for session {session_id[:8]}")

    # 2. Embed all chunks in threadpool
    try:
        vectors = await asyncio.wait_for(
            loop.run_in_executor(None, embeddings.embed_documents, chunk_texts_to_embed),
            timeout=AGENT_TIMEOUT
        )
    except Exception as e:
        return _fallback_retrieval(state, texts, f"embedding error: {e}")

    # 3. Upsert to Qdrant
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload=all_chunks_payload[i]
        )
        for i in range(len(chunk_texts_to_embed))
    ]

    try:
        qdrant.upsert(COLLECTION_NAME, points)
        logger.info(f"Upserted {len(points)} points into Qdrant")
        print(f"[DEBUG RETRIEVER] Upserted {len(points)} points into Qdrant successfully.", flush=True)
    except Exception as e:
        return _fallback_retrieval(state, texts, f"upsert error: {e}")

    # 4. BATCH EMBED SUB-TASKS & RETRIEVE TOP 4 CHUNKS PER SUB-TASK
    query_filter = Filter(
        must=[
            FieldCondition(key="session_id", match=MatchValue(value=session_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
    )

    try:
        subtask_vectors = await loop.run_in_executor(None, embeddings.embed_documents, plan)
    except Exception as e:
        logger.warning(f"Batch subtask embedding failed ({e}), falling back to direct chunks")
        subtask_vectors = [[0.1] * EMBEDDING_DIM] * len(plan)

    selected_chunks: List[Dict[str, Any]] = []
    seen_texts = set()
    all_scores = []
    subtask_coverage = {}
    citation_id = 1

    for idx, subtask in enumerate(plan):
        sub_vec = subtask_vectors[idx] if idx < len(subtask_vectors) else subtask_vectors[0]
        points_found = []

        # 1. Targeted retrieval: match chunks collected specifically for this subtask
        subtask_filter = Filter(
            must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="subtask", match=MatchValue(value=subtask))
            ]
        )
        try:
            res = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=sub_vec,
                limit=4,
                query_filter=subtask_filter,
            )
            points_found = list(res.points)
        except Exception as e:
            points_found = []

        # 2. If subtask filter returned fewer than 4 points, search session points
        if len(points_found) < 4:
            try:
                fallback_res = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=sub_vec,
                    limit=6,
                    query_filter=query_filter,
                )
                existing_ids = {getattr(p, "id", None) for p in points_found}
                for fp_pt in fallback_res.points:
                    if getattr(fp_pt, "id", None) not in existing_ids:
                        points_found.append(fp_pt)
                        existing_ids.add(getattr(fp_pt, "id", None))
            except Exception:
                pass

        # 3. Payload fallback if Qdrant returned nothing
        if not points_found:
            matching_payloads = [p for p in all_chunks_payload if p.get("subtask") == subtask]
            if not matching_payloads:
                matching_payloads = all_chunks_payload[idx*4:(idx+1)*4]
            if not matching_payloads:
                matching_payloads = all_chunks_payload[:4]
            points_found = [type("MockPoint", (), {"id": f"fb-{idx}-{i}", "payload": p, "score": 0.80 - (i * 0.02)})() for i, p in enumerate(matching_payloads[:4])]

        subtask_found_count = 0
        for pt in points_found:
            p_text = pt.payload.get("text", "")
            fp = p_text[:120].strip().lower()
            if fp in seen_texts and subtask_found_count >= 2:
                continue
            seen_texts.add(fp)

            chunk_obj = {
                "id": citation_id,
                "text": p_text,
                "url": pt.payload.get("url", ""),
                "title": pt.payload.get("title", "Untitled Source"),
                "domain": pt.payload.get("domain", ""),
                "date": pt.payload.get("date", "Recent"),
                "subtask": subtask,
                "score": round(float(pt.score), 3)
            }
            selected_chunks.append(chunk_obj)
            all_scores.append(float(pt.score))
            citation_id += 1
            subtask_found_count += 1
            if subtask_found_count >= 4:
                break

        subtask_coverage[subtask] = (subtask_found_count > 0)
        logger.info(f"Subtask '{subtask[:40]}...': retrieved {subtask_found_count} chunks")

    # Format retrieved docs by sub-task for clear citation & writer presentation
    formatted_docs = []
    current_subtask = None
    for ch in selected_chunks:
        if ch["subtask"] != current_subtask:
            current_subtask = ch["subtask"]
            formatted_docs.append(f"\n### Evidence for Sub-task: {current_subtask}")
        formatted_docs.append(
            f"[{ch['id']}] \"{ch['text']}\"\n(Source: {ch['domain']} | Date: {ch['date']} | URL: {ch['url']})"
        )

    state["retrieved_chunks"] = selected_chunks
    state["retrieved_docs"] = formatted_docs
    state["qdrant_scores"] = all_scores if all_scores else [0.75]
    state["retrieval_available"] = True
    state["subtask_coverage"] = subtask_coverage

    metrics.end_agent("retriever", input_tokens=len(state["query"]), output_tokens=len(selected_chunks))
    return state
