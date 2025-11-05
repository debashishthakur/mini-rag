""" This py file will load both the csv data into langchain documents """

import pandas as pd
from langchain.docstore.document import Document

def load_faq_document(faq_path="/Users/debashishthakur/Desktop/Masking_Testing/mini-rag/data/faqs.csv"):
    df = pd.read_csv(faq_path)
    documents = []

    for idx, row in df.iterrows():
        doc = Document(
    page_content=f"Q: {row['question']}\nA: {row['answer']}",
    metadata={
                "source": "faq",
                "question": row['question'],
                "doc_id": f"faq_{idx}"
            }
    )
    documents.append(doc)
    return documents

def load_funds_document(fund_path="/Users/debashishthakur/Desktop/Masking_Testing/mini-rag/data/funds.csv"):
    df = pd.read_csv(fund_path)
    documents = []

    for idx, row in df.iterrows():
        content = f"""
Fund: {row['fund_name']}
Category: {row['category']}
3-Year CAGR: {row['cagr_3yr (%)']}%
Volatility: {row['volatility (%)']}%
Sharpe Ratio: {row['sharpe_ratio']}
        """.strip()
        doc = Document(
    page_content=content,
    metadata={
                "source": "fund",
                "fund_id": row['fund_id'],
                "fund_name": row['fund_name'],
                "category": row['category'],
                "sharpe_ratio": float(row['sharpe_ratio']),
                "cagr": float(row['cagr_3yr (%)']),
                "doc_id": f"fund_{idx}"
            }
    )
    documents.append(doc)
    return documents

def load_all_documents():
    faq_doc = load_faq_document()
    funds_doc = load_funds_document()
    return faq_doc + funds_doc

# print(load_faq_document())
# print(load_funds_document())
print(load_all_documents())