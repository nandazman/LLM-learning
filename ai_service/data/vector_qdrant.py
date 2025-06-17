from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

RECOMMENDED_EMBEDDING_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",    # 🧩 General-purpose multilingual, smaller size
    "distiluse-base-multilingual-cased-v2",     # ✅ Balanced multilingual embeddings
    "intfloat/multilingual-e5-small",           # 🧠 Instruction-tuned, great for RAG
    "bge-m3",                                   # 🔥 High-quality, multilingual, multi-task (if RAM allows)
    "LaBSE",                                    # 🌍 Deep multilingual support, heavier
    "all-MiniLM-L6-v2"                         # ⚠️ Lightweight but English-optimized; okay fallback
]

SELECTED_EMBEDDING_MODEL_INDEX = 0  # Change this index to select a different model
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 384  # This is the size for paraphrase-multilingual-MiniLM-L12-v2

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

class QdrantDB:
    def __init__(self):
        # Get Qdrant URL from environment or use default
        self.qdrant_url = os.getenv("VECTOR_DB_URL", "http://qdrant:6333")
        
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Initialize the embedding model
            model_name = RECOMMENDED_EMBEDDING_MODELS[SELECTED_EMBEDDING_MODEL_INDEX]
            self.embedding_model = SentenceTransformer(model_name)
            
            # Create collection if it doesn't exist
            self._ensure_collection()
            logger.info(f"Successfully initialized Qdrant with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists with proper configuration"""
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == COLLECTION_NAME for collection in collections)
            
            if not exists:
                # Create new collection with optimal configuration
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,  # Optimize for larger datasets
                        memmap_threshold=20000
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,                      # Number of edges per node in HNSW graph
                        ef_construct=100,          # Size of the beam search during construction
                        full_scan_threshold=10000, # Threshold for full-scan search
                        max_indexing_threads=4     # Parallel indexing
                    )
                )
                logger.info(f"Created new Qdrant collection: {COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {str(e)}")
            raise

    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        node_id: str,
        node_type: str,
        field_name: Optional[str] = None
    ):
        """Add a document to Qdrant with enhanced metadata."""
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

            # Prepare payload (metadata)
            clean_metadata = {
                **metadata,
                "node_id": node_id,
                "node_type": node_type,
                "field_name": field_name,
                "priority": chunk_rule.get("priority", 0) if chunk_rule else 0,
                "created_at": datetime.utcnow().isoformat(),
                "text": text  # Store the text in the payload for easier retrieval
            }

            # Convert list values to comma-separated strings
            for k, v in clean_metadata.items():
                if isinstance(v, list):
                    clean_metadata[k] = ', '.join(str(item) for item in v)

            # Generate embedding
            embedding = self.embedding_model.encode(text)

            # Add point to Qdrant
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=f"doc_{node_id}_{field_name or 'main'}",
                        vector=embedding.tolist(),
                        payload=clean_metadata
                    )
                ]
            )
            logger.info(f"Added document for node {node_id} ({node_type}) to Qdrant")
            
        except Exception as e:
            logger.error(f"Error adding document to Qdrant: {str(e)}")
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

            # Generate query embedding
            query_vector = self.embedding_model.encode(query)

            # Prepare filter conditions
            filter_conditions = []
            if node_types:
                filter_conditions.append(
                    models.FieldCondition(
                        key="node_type",
                        match=models.MatchAny(any=node_types)
                    )
                )
            if min_priority is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="priority",
                        range=models.Range(
                            gte=min_priority
                        )
                    )
                )

            # Perform search with filtering
            search_result = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector.tolist(),
                limit=k,
                query_filter=models.Filter(
                    must=filter_conditions
                ) if filter_conditions else None
            )

            # Format results
            docs = []
            for hit in search_result:
                docs.append({
                    'id': hit.id,
                    'document': hit.payload.get('text', ''),
                    'metadata': {k: v for k, v in hit.payload.items() if k != 'text'},
                    'distance': hit.score
                })
            return docs
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            raise

    def clear_collection(self):
        """Clear all documents from the collection by recreating it."""
        try:
            # Delete existing collection
            self.client.delete_collection(collection_name=COLLECTION_NAME)
            
            # Recreate collection
            self._ensure_collection()
            logger.info("Cleared Qdrant collection by deleting and recreating it.")
            
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {str(e)}")
            raise

# Singleton instance
qdrant_db = QdrantDB() 