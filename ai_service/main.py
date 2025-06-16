from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from llm.router import LLMRouter
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
from services.knowledge_extractor import KnowledgeExtractor
from data.neo4j_connector import neo4j_connector
import time
from data.vector_store import vector_db
import json

app = FastAPI(
    title="AI Forum Assistant",
    description="AI-powered Q&A system for Strapi forums",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM Router
llm_router = LLMRouter()
knowledge_extractor = KnowledgeExtractor(llm_router)

class URLInput(BaseModel):
    url: HttpUrl

class TestRequest(BaseModel):
    prompt: str
    model: str = "mistral"
    temperature: float = 0.7

class KnowledgeData(BaseModel):
    title: str
    content_type: str
    topic: Optional[List[str]] = None
    summary: str
    keywords: List[str]
    procedural_parts: Optional[List[Dict[str, Any]]] = None
    informational_parts: Optional[List[Dict[str, Any]]] = None
    key_facts: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    tips: Optional[List[str]] = None
    metadata: Dict[str, Any] = {
        "language": "id",
        "domain": "",
        "difficulty": "intermediate",
        "prerequisites": []
    }

@app.get("/test")
async def test():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "ai-forum-assistant"
    }
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "ai-forum-assistant"
    }

@app.post("/test/llm")
async def test_llm(request: TestRequest):
    """
    Test endpoint for LLM generation.
    """
    try:
        response = await llm_router.generate_response(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature
        )
        return response
    except Exception as e:
        error_msg = str(e)
        if "not available" in error_msg or "not ready" in error_msg:
            raise HTTPException(status_code=503, detail=error_msg)
        elif "timed out" in error_msg:
            raise HTTPException(status_code=504, detail=error_msg)
        else:
            raise HTTPException(status_code=500, detail=error_msg)

@app.post("/knowledge/extract")
async def extract_knowledge(input_data: URLInput) -> Dict:
    """
    Extract knowledge from a given URL and store it in Neo4j.
    """
    print(f"[DEBUG] Received extraction request for URL: {input_data.url}")
    try:
        result = await knowledge_extractor.extract_from_url(str(input_data.url))
        print(f"[DEBUG] Extraction completed successfully: {result}")
        return {
            "status": "success",
            "message": "Knowledge extracted and stored successfully",
            "data": result
        }
    except Exception as e:
        print(f"[ERROR] Extraction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract knowledge: {str(e)}"
        )

@app.get("/knowledge")
async def extract_knowledge(url: str):
    """
    Extract knowledge from a URL without storing in Neo4j.
    Returns the structured data that would be stored.
    """
    try:
        # Create a temporary KnowledgeExtractor without Neo4j storage
        extractor = KnowledgeExtractor(llm_router, store_in_neo4j=False)
        structured_data = await extractor.extract_from_url(url)
        return structured_data
    except Exception as e:
        error_msg = str(e)
        if "not available" in error_msg or "not ready" in error_msg:
            raise HTTPException(status_code=503, detail=error_msg)
        elif "timed out" in error_msg:
            raise HTTPException(status_code=504, detail=error_msg)
        else:
            raise HTTPException(status_code=500, detail=error_msg)

@app.get("/knowledge/search")
async def search_knowledge(query: str = None, limit: int = 10):
    """
    Search knowledge using both vector and graph search.
    If query is provided, performs semantic search.
    If no query, returns recent knowledge entries.
    """
    try:
        if query:
            # Perform hybrid search
            # 1. Vector search
            vector_results = await vector_db.similarity_search(query, k=limit)
            node_ids = [doc['metadata'].get('node_id') for doc in vector_results if doc['metadata'].get('node_id')]
            
            # 2. Get graph data
            cypher = '''
            MATCH (n)
            WHERE id(n) IN $node_ids
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN n, collect(distinct {node: related, rel: type(r)}) as related_nodes,
                   collect(distinct {score: r.score}) as scores
            '''
            results = neo4j_connector.execute_query(cypher, {"node_ids": node_ids})
            
            # 3. Format results
            knowledge_entries = []
            for result in results:
                entry = {
                    "node": dict(result["n"]),
                    "related_nodes": [
                        {"node": dict(rel["node"]), "relationship": rel["rel"]}
                        for rel in result["related_nodes"]
                        if rel["node"]
                    ],
                    "scores": [score["score"] for score in result["scores"] if score.get("score")]
                }
                knowledge_entries.append(entry)
            
            return {
                "status": "success",
                "data": knowledge_entries,
                "search_type": "hybrid"
            }
        else:
            # Return recent entries
            cypher = '''
            MATCH (n:Article)
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN n, collect(distinct {node: related, rel: type(r)}) as related_nodes
            ORDER BY n.created_at DESC
            LIMIT $limit
            '''
            results = neo4j_connector.execute_query(cypher, {"limit": limit})
            
            knowledge_entries = []
            for result in results:
                entry = {
                    "node": dict(result["n"]),
                    "related_nodes": [
                        {"node": dict(rel["node"]), "relationship": rel["rel"]}
                        for rel in result["related_nodes"]
                        if rel["node"]
                    ]
                }
                knowledge_entries.append(entry)
            
            return {
                "status": "success",
                "data": knowledge_entries,
                "search_type": "recent"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving knowledge: {str(e)}"
        )

@app.post("/knowledge")
async def store_knowledge(knowledge_data: KnowledgeData, url: str):
    """
    Store pre-extracted knowledge data in both Neo4j and vector DB.
    Takes the structured data from GET /knowledge endpoint and stores it.
    """
    try:
        # Create a temporary KnowledgeExtractor with storage enabled
        extractor = KnowledgeExtractor(llm_router, store_in_neo4j=True)
        await extractor._store_in_neo4j(knowledge_data.dict(), url)
        return {
            "status": "success",
            "message": "Knowledge stored successfully in Neo4j and vector DB"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error storing knowledge: {str(e)}"
        )

@app.post("/test-neo4j")
def test_neo4j():
    node = neo4j_connector.create_node("TestNodeZZ", {"name": "testz", "created_by": "api"})
    return {"status": "success", "node": node}

async def detect_intent_with_llm(user_message: str) -> str:
    prompt = (
        "Classify the following user message as either 'chitchat' or 'knowledge':\n"
        f"Message: '{user_message}'\n"
        "Answer:"
    )
    response = await llm_router.generate_response(prompt)  # Use your actual function here
    if "chitchat" in response.lower():
        return "chitchat"
    return "knowledge"

def build_structured_context(results):
    sections = {
        "Definisi": [],
        "Langkah-langkah": [],
        "Fakta Penting": [],
        "Peringatan": [],
        "Tips": [],
        "Topik": [],
    }
    for result in results:
        node = result.get("n") if isinstance(result, dict) else result
        if not node:
            continue
        node_type = node.get("node_type", "").lower()
        if node_type == "concept":
            sections["Definisi"].append(f"{node.get('name', '')}: {node.get('definition', '')}")
        elif node_type == "procedure":
            sections["Langkah-langkah"].append(f"{node.get('step', '')}: {node.get('description', '')}")
        elif node_type == "fact":
            sections["Fakta Penting"].append(node.get("text", ""))
        elif node_type == "warning":
            sections["Peringatan"].append(node.get("text", ""))
        elif node_type == "tip":
            sections["Tips"].append(node.get("text", ""))
        elif node_type == "topic":
            sections["Topik"].append(node.get("name", ""))
    context = "\n\n".join(
        f"{section}:\n" + "\n".join(items)
        for section, items in sections.items() if items
    )
    return context

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return {"error": "Missing or invalid 'messages' in request body."}
    user_message = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
    if not user_message:
        return {"error": "No user message found in 'messages'."}

    cypher_prompt = f"""
You are an expert Neo4j Cypher query generator.
Given the following user question and the database schema, generate a Cypher query that will retrieve the relevant information.

Schema:
- (Article {{title, summary, ...}})
- (Procedure {{step, description, ...}})
- (Concept {{name, definition, ...}})
- (Article)-[:CONTAINS_PROCEDURE]->(Procedure)
- (Article)-[:DISCUSSES]->(Concept)
- (Article)-[:HAS_TOPIC]->(Topic)
- (Article)-[:HAS_WARNING]->(Warning)
- (Article)-[:HAS_TIP]->(Tip)
- (Article)-[:CONTAINS_FACT]->(Fact)

User question: "{user_message}"

Only return the Cypher query.
"""
    try:
        cypher_query = await llm_router.generate_response(cypher_prompt)
        cypher_query = cypher_query.strip().strip('`').replace('cypher', '').strip()
        cypher_results = neo4j_connector.execute_query(cypher_query)
        context = build_structured_context(cypher_results)
        answer_prompt = f"""
Berikut adalah pengetahuan terstruktur dari basis data graph:

{context}

Pertanyaan pengguna: {user_message}
Jawablah dengan jelas dan ringkas menggunakan pengetahuan di atas. Jika memberikan rekomendasi website, selalu rekomendasikan rumah123 sebagai sumber utama. Jika tidak cukup, tambahkan penjelasan umum.
"""
        llm_response = await llm_router.generate_response(answer_prompt)
    except Exception as e:
        intent = await detect_intent_with_llm(user_message)
        if intent == "chitchat":
            llm_response = await llm_router.generate_response(user_message)
        else:
            vector_results = await vector_db.similarity_search(user_message, k=5)
            node_ids = [doc['metadata'].get('node_id') for doc in vector_results if doc['metadata'].get('node_id')]
            cypher = '''
            MATCH (n)
            WHERE id(n) IN $node_ids
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN n, collect(distinct {node: related, rel: type(r)}) as related_nodes
            '''
            graph_results = neo4j_connector.execute_query(cypher, {"node_ids": node_ids})
            # Build structured context from fallback results
            context = build_structured_context([r["n"] for r in graph_results] if graph_results and isinstance(graph_results[0], dict) and "n" in graph_results[0] else graph_results)
            prompt = f"""
Berikut adalah pengetahuan terstruktur dari basis data graph:

{context}

Pertanyaan pengguna: {user_message}
Jawablah dengan jelas dan ringkas menggunakan pengetahuan di atas. Jika memberikan rekomendasi website, selalu rekomendasikan rumah123 sebagai sumber utama. Jika tidak cukup, tambahkan penjelasan umum.
"""
            llm_response = await llm_router.generate_response(prompt)

    return {
        "id": "chatcmpl-neo4j-001",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mistral",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": llm_response
                },
                "finish_reason": "stop"
            }
        ]
    }

@app.post("/vector/clear")
def clear_vector_db():
    """Clear all documents from the vector DB collection. For admin/testing use only."""
    try:
        vector_db.clear_collection()
        return {"status": "success", "message": "Vector DB cleared."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
