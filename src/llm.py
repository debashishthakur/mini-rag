"""Gemini LLM configuration"""
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_qa_prompt():
    """Simple prompt template for answering questions from retrieved context."""
    return (
        "You are a helpful financial assistant. Use the provided context to answer the question.\n"
        "If the answer is not in the context, say you do not know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def get_llm(temperature=0, model="gemini-2.5-flash"):
    """
    Get Gemini LLM instance
    """
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True  # Gemini requirement
    )
