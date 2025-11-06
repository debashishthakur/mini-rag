Steps to run the FastAPI endpoint locally and test the mini-RAG

Step 1 : Clone the repo to local
Step 2 : Install the dependencies using ``` pip instal -r requirements.txt```
Step 3 : Copy `.env.example` to `.env` and fill in:
     6 +  - `GOOGLE_API_KEY`
     7 +  - `PINECONE_API_KEY`, `PINECONE_ENV`
Step 4 : launch the fast api endpoint by `uvicorn main:app --reload`
Step 5 : go to the localhost url and add /docs in the end
Step 7 : try out the api by sending payload. Switch retrieval modes by passing `"mode": "lexical"` or `"hybrid"` in the JSON payload; use `top_k` to control result count.
