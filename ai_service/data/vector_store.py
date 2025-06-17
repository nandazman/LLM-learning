from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any, Optional
import logging
import shutil
import json
from datetime import datetime

logger = logging.getLogger(__name__)

RECOMMENDED_EMBEDDING_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",    # 🧩 General-purpose multilingual, smaller size
    "distiluse-base-multilingual-cased-v2",     # ✅ Balanced multilingual embeddings
    "intfloat/multilingual-e5-small",           # 🧠 Instruction-tuned, great for RAG
    "bge-m3",                                    # 🔥 High-quality, multilingual, multi-task (if RAM allows)
    "LaBSE",                                     # 🌍 Deep multilingual support, heavier
    "all-MiniLM-L6-v2"                           # ⚠️ Lightweight but English-optimized; okay fallback
]

SELECTED_EMBEDDING_MODEL_INDEX = 0  # Change this index to select a different model

CHUNK_RULES = {
    "Concept.description": {"max_tokens": 2048, "priority": 1},
    "Procedure.summary": {"max_tokens": 2048, "priority": 1},
    "Step.text": {"max_tokens": 300, "priority": 2},
    "Metric": {
        "format": "Metric {name}: {value}{unit} as of {timestamp}",
        "priority": 3
    },
    "Tag": {
        "format": "Tag: {name}",
        "priority": 4
    },
    "Source": {
        "format": "Source: {title} ({url}) - Published: {published_at}",
        "priority": 4
    },
    "RelatedConcept": {
        "format": "Related Concept: {name} - {description}",
        "priority": 2
    }
}

class VectorDB:
    def __init__(self):
        self.persist_directory = "/app/data/chroma"
        self._ensure_directory()
        try:
            self.client = PersistentClient(path=self.persist_directory)
            model_name = RECOMMENDED_EMBEDDING_MODELS[SELECTED_EMBEDDING_MODEL_INDEX]
            # Configure collection with comprehensive HNSW settings
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name  # Model selected by index
                ),
                metadata={"description": f"Knowledge base for forum Q&A system (model: {model_name})"},
                configuration={
                    "hnsw": {
                        "space": "cosine",
                        "ef_search": 100,
                        "ef_construction": 100,
                        "max_neighbors": 16,
                        "num_threads": 4
                    }
                }
            )
            logger.info(f"Successfully initialized ChromaDB with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def _ensure_directory(self):
        """Ensure the ChromaDB directory exists and has proper permissions"""
        try:
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory, mode=0o777, exist_ok=True)
                logger.info(f"Created ChromaDB directory at {self.persist_directory}")
            
            # Test write permissions
            test_file = os.path.join(self.persist_directory, ".test")
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                logger.error(f"Directory {self.persist_directory} is not writable: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB directory: {str(e)}")
            raise

    def _format_metric_text(self, metric_data: Dict[str, Any]) -> str:
        """Format metric data according to the specified format."""
        return CHUNK_RULES["Metric"]["format"].format(**metric_data)

    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        node_id: str,
        node_type: str,
        field_name: Optional[str] = None
    ):
        """Add a document to the vector store with enhanced metadata."""
        try:
            # Validate inputs
            if not text or not isinstance(text, str):
                raise ValueError("Text must be a non-empty string")
            if not metadata or not isinstance(metadata, dict):
                raise ValueError("Metadata must be a non-empty dictionary")
            if not node_id or not isinstance(node_id, str):
                raise ValueError("Node ID must be a non-empty string")

            # Get chunk rule for this field
            chunk_rule = None
            if field_name:
                field_key = f"{node_type}.{field_name}"
                chunk_rule = CHUNK_RULES.get(field_key)

            # Prepare metadata
            clean_metadata = {
                **metadata,
                "node_id": node_id,
                "node_type": node_type,
                "field_name": field_name,
                "priority": chunk_rule.get("priority", 0) if chunk_rule else 0,
                "created_at": datetime.utcnow().isoformat()
            }

            # Convert list values to comma-separated strings
            for k, v in clean_metadata.items():
                if isinstance(v, list):
                    clean_metadata[k] = ', '.join(str(item) for item in v)

            # Add document with upsert=True to handle duplicates
            self.collection.upsert(
                documents=[text],
                metadatas=[clean_metadata],
                ids=[f"doc_{node_id}_{field_name or 'main'}"]
            )
            logger.info(f"Added document for node {node_id} ({node_type}) to vector store")
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise

    async def similarity_search(
        self,
        query: str,
        k: int = 8,
        node_types: Optional[List[str]] = None,
        min_priority: Optional[int] = None
    ) -> List[Dict]:
        """Enhanced similarity search with filtering capabilities."""
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be a positive integer")

            # Prepare where clause for filtering
            where_clause = {}
            if node_types:
                where_clause["node_type"] = {"$in": node_types}
            if min_priority is not None:
                where_clause["priority"] = {"$gte": min_priority}

            # Perform search with filtering
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            docs = []
            for i in range(len(results.get('ids', [[]])[0])):
                docs.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            return docs
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def clear_collection(self):
        """Clear all documents from the collection by deleting and recreating it."""
        try:
            self.client.delete_collection("knowledge_base")
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=RECOMMENDED_EMBEDDING_MODELS[SELECTED_EMBEDDING_MODEL_INDEX]
                ),
                metadata={"description": f"Knowledge base for forum Q&A system (model: {RECOMMENDED_EMBEDDING_MODELS[SELECTED_EMBEDDING_MODEL_INDEX]})"},
                configuration={
                    "hnsw": {
                        "space": "cosine",
                        "ef_search": 100,
                        "ef_construction": 100,
                        "max_neighbors": 16,
                        "num_threads": 4
                    }
                }
            )
            logger.info("Cleared vector store collection by deleting and recreating it.")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise

# Singleton instance
vector_db = VectorDB()
