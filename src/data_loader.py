"""Utilities for loading CSV-backed datasets into LangChain documents."""

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

try:
    from langchain_core.documents import Document
except ImportError:  # Fallback for pre-0.2 LangChain versions
    from langchain.docstore.document import Document  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def _resolve_path(path: Union[str, Path], default_filename: str) -> Path:
    if path:
        return Path(path)
    return DATA_DIR / default_filename


def load_faq_document(faq_path: Union[str, Path, None] = None) -> Iterable[Document]:
    csv_path = _resolve_path(faq_path, "faqs.csv")
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        documents.append(
            Document(
                page_content=f"Q: {row['question']}\nA: {row['answer']}",
                metadata={
                    "source": "faq",
                    "question": row["question"],
                    "doc_id": f"faq_{idx}",
                },
            )
        )
    return documents


def load_funds_document(fund_path: Union[str, Path, None] = None) -> Iterable[Document]:
    csv_path = _resolve_path(fund_path, "funds.csv")
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        content = (
            f"Fund: {row['fund_name']}\n"
            f"Category: {row['category']}\n"
            f"3-Year CAGR: {row['cagr_3yr (%)']}%\n"
            f"Volatility: {row['volatility (%)']}%\n"
            f"Sharpe Ratio: {row['sharpe_ratio']}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "fund",
                    "fund_id": row["fund_id"],
                    "fund_name": row["fund_name"],
                    "category": row["category"],
                    "sharpe_ratio": float(row["sharpe_ratio"]),
                    "cagr": float(row["cagr_3yr (%)"]),
                    "doc_id": f"fund_{idx}",
                },
            )
        )
    return documents


def load_all_documents() -> Iterable[Document]:
    faq_doc = load_faq_document()
    funds_doc = load_funds_document()
    return list(faq_doc) + list(funds_doc)
