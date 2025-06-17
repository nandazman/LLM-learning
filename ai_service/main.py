from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from llm.router import LLMRouter
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
from services.knowledge_extractor import KnowledgeExtractor
from data.neo4j_connector import neo4j_connector
from data.vector_qdrant import qdrant_db
from data.schema import initialize_schema, validate_node_properties
import logging

import time
import json
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

# Initialize services
llm_router = LLMRouter()
knowledge_extractor = KnowledgeExtractor(llm_router)

# Use Qdrant as the vector store
active_vector_store = qdrant_db

# Initialize schema on startup
@app.on_event("startup")
async def startup_event():
    try:
        initialize_schema()
    except Exception as e:
        logger.error(f"Failed to initialize schema: {str(e)}")
        raise

class URLInput(BaseModel):
    url: HttpUrl

class TestRequest(BaseModel):
    prompt: str
    model: str = "mistral"
    temperature: float = 0.7

class KnowledgeData(BaseModel):
    concept: Dict[str, str] = {
        "id": "",
        "name": "",
        "description": ""
    }
    procedures: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[List[Dict[str, Any]]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[Dict[str, Any]]] = None
    related_concepts: Optional[List[Dict[str, Any]]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "concept": {
                    "id": "concept_1234567890",
                    "name": "How to Buy a House",
                    "description": "Step by step guide to buying a house"
                },
                "procedures": [
                    {
                        "id": "procedure_1234567890_0",
                        "title": "Check your budget",
                        "summary": "Calculate your monthly income and expenses",
                        "steps": [
                            {
                                "id": "step_1234567890_0_0",
                                "order": 1,
                                "text": "Calculate your monthly income"
                            },
                            {
                                "id": "step_1234567890_0_1",
                                "order": 2,
                                "text": "List all monthly expenses"
                            }
                        ]
                    }
                ],
                "metrics": [
                    {
                        "id": "metric_1234567890_0",
                        "name": "Minimum Down Payment",
                        "value": "20",
                        "unit": "percent",
                        "timestamp": "2024-03-20"
                    }
                ],
                "sources": [
                    {
                        "id": "source_1234567890_0",
                        "url": "https://example.com/buying-house",
                        "title": "Complete Guide to Buying a House",
                        "published_at": "2024-03-20"
                    }
                ],
                "tags": [
                    {
                        "id": "tag_1234567890_0",
                        "name": "property"
                    },
                    {
                        "id": "tag_1234567890_1",
                        "name": "finance"
                    }
                ],
                "related_concepts": [
                    {
                        "id": "concept_1234567890_1",
                        "name": "Mortgage",
                        "description": "Understanding mortgage types and requirements"
                    }
                ]
            }
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


@app.post("/knowledge")
async def store_knowledge(knowledge_data: KnowledgeData, url: Optional[str] = None):
    """
    Store pre-extracted knowledge data in both Neo4j and vector DB.
    Takes the structured data from GET /knowledge endpoint and stores it.
    URL is optional - if not provided, the source will be marked as 'direct_input'.
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

    try:
        # Use the new integrated RAG pipeline via llm_router
        result = await llm_router.process_question(user_message)
        llm_response = result["answer"]
    except Exception as e:
        # Fallback to simple LLM response if the RAG pipeline fails
        llm_response = await llm_router.generate_response(user_message)

    return {
        "id": "chatcmpl-neo4j-001",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gemma:2b",
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
        active_vector_store.clear_collection()
        return {"status": "success", "message": "Vector DB cleared."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
