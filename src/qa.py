from typing import Dict, List

from llm import get_llm, get_qa_prompt
from retrievers import retrieve_documents


def _build_context_block(results: List[Dict]) -> str:
    blocks = []
    for idx, res in enumerate(results, start=1):
        metadata = res.get("metadata", {}) or {}
        meta_lines = "\n".join(f"- {key}: {value}" for key, value in metadata.items())
        block = [
            f"Source {idx} ({res.get('retrieval_mode', 'unknown')})",
            f"Score: {res.get('score')}",
        ]
        if meta_lines:
            block.append("Metadata:")
            block.append(meta_lines)
        block.append("Content:")
        block.append(res.get("content", ""))
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _extract_fund_highlights(results: List[Dict]) -> List[Dict]:
    highlights = []
    for res in results:
        metadata = res.get("metadata", {}) or {}
        if metadata.get("source") != "fund":
            continue
        highlights.append(
            {
                "fund_name": metadata.get("fund_name"),
                "category": metadata.get("category"),
                "sharpe_ratio": metadata.get("sharpe_ratio"),
                "cagr": metadata.get("cagr"),
                "retrieval_mode": res.get("retrieval_mode"),
                "doc_id": metadata.get("doc_id"),
            }
        )
    highlights.sort(
        key=lambda item: (
            item.get("sharpe_ratio") if item.get("sharpe_ratio") is not None else -float("inf"),
            item.get("cagr") if item.get("cagr") is not None else -float("inf"),
        ),
        reverse=True,
    )
    return highlights


def answer_question_with_pinecone_llama(query: str, top_k: int = 3, mode: str = "semantic"):
    results = retrieve_documents(query, top_k=top_k, mode=mode)
    context = _build_context_block(results)

    prompt_template = get_qa_prompt()
    prompt_text = prompt_template.format(context=context, question=query)

    llm = get_llm()
    response = llm.invoke(prompt_text)
    answer_text = response.content if hasattr(response, "content") else str(response)

    fund_highlights = _extract_fund_highlights(results)

    payload = {
        "answer": answer_text.strip(),
        "sources": results,
        "retrieval_mode": mode,
    }
    if fund_highlights:
        payload["fund_highlights"] = fund_highlights
    return payload
