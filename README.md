1. Clone the repository locally.
  2. Install dependencies: `pip install -r requirements.txt`
  3. Copy `.env.example` to `.env` and fill in:
     - `GOOGLE_API_KEY`
     - `PINECONE_API_KEY`
     - `PINECONE_ENV`
  4. Start the FastAPI server: `uvicorn src.main:app --reload`
  5. Open `http://localhost:8000/docs` in your browser.
  6. Use the Swagger UI to test the `/ask` endpoint. Switch retrieval modes via the JSON payload by setting `"mode": "lexical"`
  or `"hybrid"` or `semantic` and adjust `top_k` to control how many results are returned.
