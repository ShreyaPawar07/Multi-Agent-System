# Multi-Agent-System
Repo contains code to basic multi agent setup in Langchain and Open-Ai

## Policy Assistant – Quick Start

Simple demo that:
- Ingests `Sample Policies.pdf` into a FAISS vector store
- Lets agents and a Streamlit chat app answer policy questions

### 1. Setup

- **Python**: 3.10+ recommended  
- In a terminal, from the project folder:

```bash
python -m venv venv
venv\Scripts\activate          # On Windows
pip install -r requirements.txt
```

Set your API key (for OpenAI / OpenRouter with Grok):

```bash
setx OPENAI_API_KEY "your-key-here"
# optional if using OpenRouter:
setx OPENAI_BASE_URL "https://openrouter.ai/api/v1"
```

Restart the terminal so env vars are picked up.

### 2. Build / reuse the vector store

The first time you run anything that calls `query_vector_db`, it will:
- Read `Sample Policies.pdf`
- Create embeddings
- Save a FAISS index to `policy_faiss_index/`

Next runs will **reuse** that saved index automatically (no re‑ingest needed).

### 3. CLI agent test

Ask a sample question via the agents pipeline:

```bash
python agents.py
```

It will print a concise answer based on the policies.

### 4. Streamlit chat app

Start the chat UI:

```bash
streamlit run streamlit_app.py
```

In the browser:
- Enter your **name** once
- Ask policy questions in the chat
- Use the **Exit Chat** button to end the session

The sidebar shows a simple memory log of your past questions.

