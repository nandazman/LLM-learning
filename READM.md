A production‑ready template that turns **Strapi forum threads** into an AI‑powered Q\&A experience using **FastAPI + LangChain/LangGraph** in a single repository.

---

## ✨ Features

* **Lifecycle hooks** in Strapi trigger automatic answers for new threads and follow‑up comments.
* **Agent routing** selects the correct domain agent (Market, Tool, General) before calling an LLM.
* **Multi‑model LLM router** chooses the best provider based on task and cost.
* **Vector & Graph retrieval** (Neo4j + your preferred vector store) for hybrid search.
* **FastAPI API** secured with an \`\` header.
* **Docker‑Compose** development stack.
* **Pytest** test suite with coverage.

---

## 🏗 Architecture

```mermaid
flowchart LR
  subgraph Strapi CMS
    A[Thread Created / Comment Added]
  end

  subgraph FastAPI Server
    B[POST /agent/answer]\nvalidate API key
    C[Agent Router]\nselect domain agent
    D[Domain Agent]\nMarket / Tool / General
    E[LLM Router]\npick model & prompt
    F[LLM Provider]
    G[Neo4j]
    H[Vector Store]
  end

  A --lifecycle hook--> B --> C --> D --> E --> F
  D --> G
  D --> H
  D --answer--> B
  B --POST comment--> A
```

---

## 📂 Project Structure

```
ai_service/
├─ main.py               # FastAPI entry‑point
├─ api/
│  └─ routes.py          # /agent/answer endpoint
├─ agents/               # LangChain agent logic
│  ├─ base.py
│  ├─ market.py
│  ├─ tool.py
│  └─ general.py
├─ llm/
│  ├─ router.py          # model selection
│  └─ prompts/
├─ data/
│  ├─ neo4j_connector.py
│  └─ vector_store.py
├─ utils/
│  └─ extract_html.py
├─ tests/
│  └─ test_agents.py
├─ Dockerfile
└─ docker‑compose.yml
```

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
# clone repo
$ git clone https://github.com/your‑org/9knows‑forum‑assistant.git
$ cd 9knows‑forum‑assistant

# copy env template
$ cp .env.example .env
# edit .env with your keys
```

Key variables:

| Var                                       | Description                                   |
| ----------------------------------------- | --------------------------------------------- |
| `STRAPI_AGENT_KEY`                        | Secret sent from Strapi in `x‑api‑key` header |
| `OPENAI_API_KEY`                          | (or `ANTHROPIC_API_KEY`)                      |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASS` | Graph DB auth                                 |
| `VECTOR_DB_URL`                           | Your vector store endpoint                    |

### 2. Run Locally (Docker Compose)

```bash
$ docker‑compose up --build
```

* **FastAPI** → [http://localhost:8000/docs](http://localhost:8000/docs)
* **Neo4j**    → [http://localhost:7474](http://localhost:7474) (optional)

### 3. Wire Up Strapi

1. Install lifecycle hook in `/src/api/thread/content‑types/thread/lifecycles.js`:

```js
module.exports = {
  async afterCreate(event) {
    const { topic, question } = event.params.data;
    const res = await strapi.plugins["axios"].axios.post(
      "http://fastapi:8000/agent/answer",
      { topic, question, threadId: event.result.id },
      { headers: { "x-api-key": process.env.STRAPI_AGENT_KEY } }
    );
    // FastAPI will post the answer back as comment
  },
};
```

2. Ensure \`\` in both Strapi and FastAPI env files match.

### 4. Run Tests

```bash
$ pytest --cov
```

---

## 🛡 Security Notes

* All calls to FastAPI require `x‑api‑key` header.
* Limit key scope & rotate periodically.
* FastAPI validates key via dependency injection (see `deps.py`).

---

## 🧩 Agent Router Example

```python
# agents/__init__.py
from .market import MarketAgent
from .tool import ToolAgent
from .general import GeneralAgent

def get_agent(topic: str):
    if "market" in topic.lower():
        return MarketAgent()
    if "tool" in topic.lower():
        return ToolAgent()
    return GeneralAgent()
```

---

## 📜 License

MIT © 2025 Your Name / Your Org

---

## 📰 Article Ingestion & Summarization

You can extend the API with **two new endpoints** that let the agent learn from first‑party articles:

| Endpoint             | Method | Purpose                                                                                                                                                                               |
| -------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/article/summarize` | `POST` | Fetch the given URL (must belong to your allowed domain list), strip boilerplate, return a JSON summary plus extracted chunks (no db write).                                          |
| `/article/store`     | `POST` | Same fetch logic; in addition **embeds** the cleaned chunks and **upserts** them into the vector store (+ optionally Neo4j). Returns the summary and a `doc_id` for future reference. |

### Domain Whitelist

```python
ALLOWED_DOMAINS = {"example.com", "knowledge.example.com"}
```

If the hostname of the submitted link isn’t in the set, FastAPI returns **HTTP 403**.

### Minimal Route Skeleton

```python
from fastapi import APIRouter, HTTPException
from utils.scraper import fetch_clean_html
from agents.summariser import Summariser
from data.vector_store import vector_store

router = APIRouter(prefix="/article", tags=["Article"])

@router.post("/summarize")
async def summarize_link(url: HttpUrl, x_api_key: str = Header(...)):
    validate_api_key(x_api_key)
    if url.host not in ALLOWED_DOMAINS:
        raise HTTPException(403, "domain not allowed")
    html = await fetch_clean_html(str(url))
    summary, chunks = Summariser().run(html)
    return {"summary": summary, "chunks": chunks}

@router.post("/store")
async def store_link(url: HttpUrl, x_api_key: str = Header(...)):
    validate_api_key(x_api_key)
    if url.host not in ALLOWED_DOMAINS:
        raise HTTPException(403, "domain not allowed")
    html = await fetch_clean_html(str(url))
    summary, chunks = Summariser().run(html)
    doc_id = vector_store.add_documents(chunks)
    return {"summary": summary, "doc_id": doc_id}
```

### Strapi Workflow Example

1. Content editor pastes the canonical article URL into a field (or dedicated collection type).
2. Strapi lifecycle hook calls `/article/store` once the entry is **Published**.
3. The endpoint embeds + stores the article so future forum questions can retrieve it.

### Security Tips

* **Always require the same **\`\`**.**
* **Enforce domain check** *before* making the HTTP request.
* Add rate‑limiting or background queue if downloads are heavy.
