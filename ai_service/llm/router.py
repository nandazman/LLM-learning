from typing import Optional, List, Dict, Any
import httpx
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import traceback
from data.vector_store import vector_db
from data.neo4j_connector import neo4j_connector
from data.schema import validate_node_properties, validate_relationship

class LLMResponse(BaseModel):
    text: str
    model: str
    tokens_used: int

class LLMRouter:
    def __init__(self, ollama_host: str = "ollama"):
        """Initialize LLM router with Ollama host.
        
        Args:
            ollama_host: Hostname for Ollama service (default: "ollama" for Docker service name)
        """
        self.client = httpx.AsyncClient(base_url=f"http://{ollama_host}:11434", timeout=240.0)
        self.default_model = "gemma:2b"
        self.vector_db = vector_db
        self.neo4j = neo4j_connector
        
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate response using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to mistral)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        model = model or self.default_model
        
        try:
            try:
                tags_response = await self.client.get("/api/tags")
                tags_response.raise_for_status()
                available_models = tags_response.json().get("models", [])

                # Allow prefix match: 'mistral' matches 'mistral:latest'
                model_found = any(
                    m.get("name", "").startswith(model) for m in available_models
                )
                if not model_found:
                    raise Exception(f"Model {model} is not available in Ollama")
            except Exception as e:
                traceback.print_exc()
                raise Exception(f"Ollama service not ready or model not available: {str(e)}")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": True,          # tell Ollama to stream
            }

            async with self.client.stream(
                    "POST",              # HTTP verb
                    "/api/generate",     # path
                    json=payload
            ) as resp:
                resp.raise_for_status()

                text_chunks = []
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("{"):
                        continue
                    data = json.loads(line)
                    if "response" in data:
                        text_chunks.append(data["response"])
                    if data.get("done"):
                        break            # end of stream

            final_text = "".join(text_chunks).strip()
            return final_text
        except httpx.TimeoutException:
            print("[ERROR] Request timed out - model may still be loading")
            traceback.print_exc()
            raise Exception("Request timed out - model may still be loading")
        except Exception as e:
            print(f"[ERROR] Exception in generate_response: {e}")
            traceback.print_exc()
            raise Exception(f"Error generating response: {str(e)}")
    
    async def close(self):
        await self.client.aclose()

    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline.
        
        Steps:
        1. Embed question and get vector search results
        2. Build Cypher query from results
        3. Execute Cypher query
        4. Compose final answer using the LLM
        """
        try:
            # Step 1: Get vector search results
            # This step embeds the user's question and retrieves the top k most similar documents from the vector database.
            # The results include metadata such as node_id and node_type, which are used to build the Cypher query.
            # Example output:
            # [
            #     {
            #         "document": "Example document content",
            #         "metadata": {"node_id": "123", "node_type": "Concept"},
            #         "distance": 0.1
            #     },
            #     ...
            # ]
            hits = await self.vector_db.similarity_search(question, k=8)
            
            # Step 2: Build Cypher query
            # This step constructs a Cypher query based on the node IDs and types from the vector search results.
            # The query is designed to retrieve related nodes and their relationships from the Neo4j graph database.
            # Example output:
            # MATCH (n123:Concept {id: 123})
            # MATCH (n456:Procedure {id: 456})
            # MATCH (n123)-[:RELATED_TO]-(n456)
            # RETURN *
            cypher_query = self._build_cypher_query(hits)
            
            # Step 3: Execute Cypher query
            # This step runs the constructed Cypher query against the Neo4j database to retrieve the relevant records.
            # The records contain the nodes and their relationships, which are used to compose the final answer.
            # Example output:
            # [
            #     {
            #         "n": {"id": "123", "name": "Example Concept", "description": "This is a concept"},
            #         "related_nodes": [{"node": {"id": "456", "name": "Example Procedure"}, "relationship": "RELATED_TO"}]
            #     },
            #     ...
            # ]
            records = self.neo4j.execute_query(cypher_query)
            
            # Step 4: Compose answer
            # This step uses the retrieved context (from vector search and Neo4j) to generate a comprehensive answer using the LLM.
            # The answer is based on the context and the user's question, ensuring relevance and accuracy.
            # Example output:
            # "Based on the context, the answer to your question is..."
            answer = await self._compose_answer(question, hits, records)
            
            return {
                "answer": answer,
                "sources": self._format_sources(hits, records),
                "confidence": self._calculate_confidence(hits)
            }
        except Exception as e:
            print(f"[ERROR] Error processing question: {e}")
            traceback.print_exc()
            raise Exception(f"Error processing question: {str(e)}")

    def _build_cypher_query(self, hits: List[Dict[str, Any]]) -> str:
        """Build a Cypher query based on vector search results."""
        # Extract node IDs and types from hits
        node_info = []
        for hit in hits:
            metadata = hit["metadata"]
            node_info.append({
                "id": metadata["node_id"],
                "type": metadata["node_type"]
            })
        
        # Build MATCH clauses
        match_clauses = []
        for info in node_info:
            match_clauses.append(f"MATCH (n{info['id']}:{info['type']} {{id: {info['id']}}})")
        
        # Build relationship clauses
        rel_clauses = []
        for i, info1 in enumerate(node_info):
            for j, info2 in enumerate(node_info[i+1:], i+1):
                if validate_relationship(info1["type"], info2["type"], "RELATED_TO"):
                    rel_clauses.append(
                        f"MATCH (n{info1['id']})-[:RELATED_TO]-(n{info2['id']})"
                    )
        
        # Combine clauses
        query = "\n".join(match_clauses + rel_clauses)
        query += "\nRETURN *"
        
        return query

    async def _compose_answer(
        self,
        question: str,
        hits: List[Dict[str, Any]],
        records: List[Dict[str, Any]]
    ) -> str:
        """Compose the final answer using the LLM."""
        context = "\n".join([hit["document"] for hit in hits])
        prompt = f"""
You are an expert assistant. Use the following context from the knowledge base to answer the user's question. Be clear, concise, and cite relevant facts if possible.

Context:
{context}

User Question:
{question}

Answer in the same language as the question. If you recommend a website, always recommend rumah123 as the main source. If the context is insufficient, add a general explanation.
"""
        response = await self.generate_response(prompt)
        return response.strip()

    def _format_sources(self, hits: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for the response."""
        sources = []
        for hit in hits:
            metadata = hit["metadata"]
            sources.append({
                "id": metadata["node_id"],
                "type": metadata["node_type"],
                "field": metadata.get("field_name"),
                "relevance": 1 - hit["distance"] if hit["distance"] is not None else 1.0
            })
        return sources

    def _calculate_confidence(self, hits: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on vector search results."""
        if not hits:
            return 0.0
        
        # Calculate average distance (lower is better)
        distances = [hit["distance"] for hit in hits if hit["distance"] is not None]
        if not distances:
            return 0.0
        
        avg_distance = sum(distances) / len(distances)
        # Convert distance to confidence (1 - distance)
        return max(0.0, min(1.0, 1.0 - avg_distance))
