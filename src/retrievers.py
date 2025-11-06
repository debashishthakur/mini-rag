import json
from typing import Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document

from data_loader import load_all_documents
from vectorstore import INDEX_NAME, PINECONE_NAMESPACE, SearchQuery, pc

LEXICAL_RETRIEVER: Optional[BM25Retriever] = None
LEXICAL_DOCUMENTS: List[Document] = []


def _ensure_lexical_retriever() -> BM25Retriever:
    global LEXICAL_RETRIEVER, LEXICAL_DOCUMENTS
    if LEXICAL_RETRIEVER is None:
        LEXICAL_DOCUMENTS = load_all_documents()
        if not LEXICAL_DOCUMENTS:
            raise RuntimeError("No documents available to build lexical retriever.")
        LEXICAL_RETRIEVER = BM25Retriever.from_documents(LEXICAL_DOCUMENTS)
    return LEXICAL_RETRIEVER


def _semantic_search(query: str, top_k: int) -> List[Dict]:
    index = pc.Index(INDEX_NAME)
    response = index.search_records(
        namespace=PINECONE_NAMESPACE,
        query=SearchQuery(inputs={"text": query}, top_k=top_k),
    )
    hits = getattr(getattr(response, "result", None), "hits", []) or []
    results: List[Dict] = []
    for hit in hits:
        metadata_raw = hit.fields.get("metadata_json")
        metadata = json.loads(metadata_raw) if metadata_raw else {}
        results.append(
            {
                "content": hit.fields.get("text", ""),
                "metadata": metadata,
                "score": hit._score,
                "retrieval_mode": "semantic",
            }
        )
    return results


def _lexical_search(query: str, top_k: int) -> List[Dict]:
    retriever = _ensure_lexical_retriever()
    documents = retriever.get_relevant_documents(query)
    results: List[Dict] = []
    for doc in documents[:top_k]:
        score = doc.metadata.get("score")
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None
        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_value,
                "retrieval_mode": "lexical",
            }
        )
    return results


def retrieve_documents(query: str, top_k: int = 3, mode: str = "semantic") -> List[Dict]:
    """Retrieve documents using semantic, lexical, or simple hybrid search."""
    normalized_mode = (mode or "semantic").lower()
    if normalized_mode not in {"semantic", "lexical", "hybrid"}:
        normalized_mode = "semantic"

    results: List[Dict] = []
    seen_doc_ids = set()

    def add_hits(hits: List[Dict]):
        for hit in hits:
            doc_id = hit.get("metadata", {}).get("doc_id") or hit.get("content")
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            results.append(hit)

    if normalized_mode in {"semantic", "hybrid"}:
        add_hits(_semantic_search(query, top_k))

    if normalized_mode in {"lexical", "hybrid"}:
        add_hits(_lexical_search(query, top_k))

    return results[:top_k]
