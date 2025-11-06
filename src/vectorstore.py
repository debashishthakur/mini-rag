import json
import os
import time
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone, SearchQuery, ServerlessSpec

from data_loader import load_all_documents

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = (
    os.getenv("PINECONE_REGION")
    or os.getenv("PINECONE_ENV")
    or "us-east-1"
)
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "quonfido-llama-index")
EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "missing credentials."
    )


pc = Pinecone(api_key=PINECONE_API_KEY)

def wait_for_index(name: str, timeout: int = 180) -> None:
    """Poll Pinecone until the index reports ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        details = pc.describe_index(name).to_dict()
        status = (details.get("status") or {})
        if status.get("ready"):
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for Pinecone index '{name}' to be ready.")

def create_index_if_not_exists() -> None:
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' already exists.")
        return
    print(f"Creating Pinecone index '{INDEX_NAME}' with integrated model '{EMBEDDING_MODEL}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        embed={"model": EMBEDDING_MODEL, "field_map": {"text": "text"}},
    )
    wait_for_index(INDEX_NAME)
    print("Index created and ready.")

def build_records(documents) -> List[dict]:
    records: List[dict] = []
    for idx, doc in enumerate(documents):
        record = {
            "_id": f"doc-{idx}",
            "text": doc.page_content,
        }
        if doc.metadata:
            record["metadata_json"] = json.dumps(doc.metadata)
            for key, value in doc.metadata.items():
                record[f"meta_{key}"] = value
        records.append(record)
    return records

def upsert_documents():
    create_index_if_not_exists()
    index = pc.Index(INDEX_NAME)
    documents = load_all_documents()
    if not documents:
        print("No documents were loaded; aborting upsert.")
        return
    records = build_records(documents)
    index.upsert_records(namespace=PINECONE_NAMESPACE, records=records)
    print(f"Upserted {len(records)} records into index '{INDEX_NAME}' (namespace '{PINECONE_NAMESPACE}').")

def search(query: str, top_k: int = 3):
    index = pc.Index(INDEX_NAME)
    response = index.search_records(
        namespace=PINECONE_NAMESPACE,
        query=SearchQuery(inputs={"text": query}, top_k=top_k),
    )
    result = getattr(response, "result", None)
    hits = getattr(result, "hits", []) or []
    for position, hit in enumerate(hits, start=1):
        text = hit.fields.get("text", "")
        metadata_raw = hit.fields.get("metadata_json")
        metadata = json.loads(metadata_raw) if metadata_raw else {}
        score = hit._score
        print(f"\nResult {position}: score={score:.4f}")
        print(f"Text: {text}")
        if metadata:
            print(f"Metadata: {metadata}")
