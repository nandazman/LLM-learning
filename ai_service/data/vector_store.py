from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any
import logging
import shutil

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.persist_directory = "/app/data/chroma"
        self._ensure_directory()
        try:
            self.client = PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            logger.info("Successfully initialized ChromaDB")
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

    async def add_document(self, text: str, metadata: Dict[str, Any], node_id: int):
        """Add a document to the vector store"""
        try:
            # Validate inputs
            if not text or not isinstance(text, str):
                raise ValueError("Text must be a non-empty string")
            if not metadata or not isinstance(metadata, dict):
                raise ValueError("Metadata must be a non-empty dictionary")
            if not node_id or not isinstance(node_id, int):
                raise ValueError("Node ID must be a non-zero integer")

            # Convert list values in metadata to comma-separated strings
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, list):
                    clean_metadata[k] = ', '.join(str(item) for item in v)
                else:
                    clean_metadata[k] = v

            # Add document
            self.collection.add(
                documents=[text],
                metadatas=[{**clean_metadata, "node_id": node_id}],
                ids=[f"doc_{node_id}"]
            )
            logger.info(f"Added document for node {node_id} to vector store")
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise

    async def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be a positive integer")

            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            # Chroma returns a dict with keys: 'ids', 'documents', 'metadatas', 'distances'
            # We'll return a list of dicts with 'metadata', 'document', and 'id'
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
        """Clear all documents from the collection by deleting and recreating it (ChromaDB >=0.4.22)."""
        try:
            self.client.delete_collection("knowledge_base")
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            logger.info("Cleared vector store collection by deleting and recreating it.")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise

# Singleton instance
vector_db = VectorDB()
