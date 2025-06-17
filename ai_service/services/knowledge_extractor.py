from bs4 import BeautifulSoup
import httpx
from typing import Dict, List, Optional
import json
from llm.router import LLMRouter
from data.neo4j_connector import neo4j_connector
from data.vector_qdrant import qdrant_db
import logging
import time
from datetime import datetime
import re
import os

logger = logging.getLogger(__name__)

class KnowledgeExtractor:
    def __init__(self, llm_router: LLMRouter, store_in_neo4j: bool = True):
        self.llm_router = llm_router
        self.client = httpx.AsyncClient()
        self.store_in_neo4j = store_in_neo4j

    async def extract_from_text(self, text: str, title: str, content_type: Optional[str] = None, url: Optional[str] = None) -> Dict:
        """Extract knowledge directly from text input"""
        try:
            # If content type is not provided, detect it
            if not content_type:
                content_type = await self._detect_content_type(text)
            logger.info(f"Content type detected: {content_type}")

            # Extract knowledge based on content type
            if content_type == "procedural":
                structured_data = await self._extract_procedural_knowledge(text, title)
            elif content_type == "informational":
                structured_data = await self._extract_informational_knowledge(text, title)
            else:  # mixed
                structured_data = await self._extract_mixed_knowledge(text, title)

            # Transform the data to match our schema
            transformed_data = {
                "concept": {
                    "id": f"concept_{int(time.time())}",
                    "name": structured_data.get("concept", {}).get("name", title),
                    "description": structured_data.get("concept", {}).get("description", "")
                },
                "procedures": [
                    {
                        "id": procedure.get("id", f"procedure_{int(time.time())}_{i}"),
                        "title": procedure.get("title", ""),
                        "summary": procedure.get("summary", ""),
                        "steps": [
                            {
                                "id": step.get("id", f"step_{int(time.time())}_{i}_{j}"),
                                "order": step.get("order", j + 1),
                                "text": step.get("text", "")
                            }
                            for j, step in enumerate(procedure.get("steps", []))
                        ]
                    }
                    for i, procedure in enumerate(structured_data.get("procedures", []))
                ],
                "metrics": [
                    {
                        "id": metric.get("id", f"metric_{int(time.time())}_{i}"),
                        "name": metric.get("name", ""),
                        "value": metric.get("value", ""),
                        "unit": metric.get("unit", ""),
                        "timestamp": metric.get("timestamp", datetime.now().strftime("%Y-%m-%d"))
                    }
                    for i, metric in enumerate(structured_data.get("metrics", []))
                ],
                "sources": [
                    {
                        "id": source.get("id", f"source_{int(time.time())}_{i}"),
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "published_at": source.get("published_at", datetime.now().strftime("%Y-%m-%d"))
                    }
                    for i, source in enumerate(structured_data.get("sources", []))
                ],
                "tags": [
                    {
                        "id": tag.get("id", f"tag_{int(time.time())}_{i}"),
                        "name": tag.get("name", "")
                    }
                    for i, tag in enumerate(structured_data.get("tags", []))
                ],
                "related_concepts": [
                    {
                        "id": concept.get("id", f"concept_{int(time.time())}_{i}"),
                        "name": concept.get("name", ""),
                        "description": concept.get("description", "")
                    }
                    for i, concept in enumerate(structured_data.get("related_concepts", []))
                ]
            }

            # Add source URL if provided
            if url:
                transformed_data["sources"].append({
                    "id": f"source_{int(time.time())}",
                    "url": url,
                    "title": title,
                    "published_at": datetime.now().strftime("%Y-%m-%d")
                })

            if self.store_in_neo4j:
                logger.info("Storing structured data in Neo4j...")
                await self._store_in_neo4j(transformed_data, url)
                logger.info("Data stored in Neo4j successfully.")

            return transformed_data

        except Exception as e:
            logger.error(f"Error extracting knowledge: {str(e)}")
            raise

    async def extract_from_url(self, url: str) -> Dict:
        """Extract knowledge from a given URL"""
        try:
            print(f"[DEBUG] Fetching URL: {url}")
            response = await self.client.get(url)
            response.raise_for_status()
            print(f"[DEBUG] Page fetched successfully")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            article = soup.find('article') or soup.find('div', class_='article-content')
            if not article:
                print("[ERROR] Could not find main article content")
                raise ValueError("Could not find main article content")

            title = article.find('h1')
            title_text = title.text.strip() if title else "Untitled Article"

            for element in article.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            content_elements = []
            for element in article.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text(strip=True)
                if text and len(text) > 10:
                    content_elements.append(text)
            content = "\n".join(content_elements)
            print(f"[DEBUG] Extracted content length: {len(content)}")

            # Use the new extract_from_text method
            return await self.extract_from_text(content, title_text)

        except Exception as e:
            print(f"[ERROR] Error extracting knowledge: {str(e)}")
            raise

    async def _detect_content_type(self, content: str) -> str:
        """Detect if content is procedural, informational, or mixed"""
        prompt = f"""
        Analyze this article and classify its type:
        1. procedural - if it contains step-by-step instructions or procedures
        2. informational - if it contains general information, facts, or explanations
        3. mixed - if it contains both procedures and information
        
        Content: {content[:1000]}
        
        Return only one word: procedural, informational, or mixed
        """
        print(f"[DEBUG] trying to extract content type")
        response = await self.llm_router.generate_response(prompt)
        return response.strip().lower()

    async def _extract_procedural_knowledge(self, text: str, title: str) -> Dict:
        """Extract procedural knowledge from text"""
        try:
            prompt = f"""Extract procedural knowledge from the following text. 
            Format the response as a JSON object with these exact fields:
            {{
                "concept": {{
                    "id": "concept_[timestamp]",
                    "name": "Main concept or topic name",
                    "description": "Brief summary of the concept"
                }},
                "procedures": [
                    {{
                        "id": "procedure_[timestamp]_[index]",
                        "title": "Step title or name",
                        "summary": "Detailed description of the step",
                        "steps": [
                            {{
                                "id": "step_[timestamp]_[index]",
                                "order": [step_number],
                                "text": "Detailed step description"
                            }}
                        ]
                    }}
                ],
                "metrics": [
                    {{
                        "id": "metric_[timestamp]_[index]",
                        "name": "Metric name",
                        "value": "Metric value",
                        "unit": "Unit of measurement",
                        "timestamp": "YYYY-MM-DD"
                    }}
                ],
                "sources": [
                    {{
                        "id": "source_[timestamp]_[index]",
                        "url": "Source URL",
                        "title": "Source title",
                        "published_at": "YYYY-MM-DD"
                    }}
                ],
                "tags": [
                    {{
                        "id": "tag_[timestamp]_[index]",
                        "name": "Tag name"
                    }}
                ],
                "related_concepts": [
                    {{
                        "id": "concept_[timestamp]_[index]",
                        "name": "Related concept name",
                        "description": "Related concept description"
                    }}
                ]
            }}

            Text to analyze:
            {text}

            Respond with ONLY the JSON object, no additional text."""

            response = await self.llm_router.generate_response(prompt)
            return self._parse_llm_response(response)

        except Exception as e:
            logger.error(f"Error extracting procedural knowledge: {str(e)}")
            raise

    async def _extract_informational_knowledge(self, text: str, title: str) -> Dict:
        """Extract informational knowledge from text"""
        try:
            prompt = f"""Extract informational knowledge from the following text. 
            Format the response as a JSON object with these exact fields:
            {{
                "concept": {{
                    "id": "concept_[timestamp]",
                    "name": "Main concept or topic name",
                    "description": "Brief summary of the concept"
                }},
                "procedures": [
                    {{
                        "id": "procedure_[timestamp]_[index]",
                        "title": "Step title or name",
                        "summary": "Detailed description of the step",
                        "steps": [
                            {{
                                "id": "step_[timestamp]_[index]",
                                "order": [step_number],
                                "text": "Detailed step description"
                            }}
                        ]
                    }}
                ],
                "metrics": [
                    {{
                        "id": "metric_[timestamp]_[index]",
                        "name": "Metric name",
                        "value": "Metric value",
                        "unit": "Unit of measurement",
                        "timestamp": "YYYY-MM-DD"
                    }}
                ],
                "sources": [
                    {{
                        "id": "source_[timestamp]_[index]",
                        "url": "Source URL",
                        "title": "Source title",
                        "published_at": "YYYY-MM-DD"
                    }}
                ],
                "tags": [
                    {{
                        "id": "tag_[timestamp]_[index]",
                        "name": "Tag name"
                    }}
                ],
                "related_concepts": [
                    {{
                        "id": "concept_[timestamp]_[index]",
                        "name": "Related concept name",
                        "description": "Related concept description"
                    }}
                ]
            }}

            Text to analyze:
            {text}

            Respond with ONLY the JSON object, no additional text."""

            response = await self.llm_router.generate_response(prompt)
            return self._parse_llm_response(response)

        except Exception as e:
            logger.error(f"Error extracting informational knowledge: {str(e)}")
            raise

    async def _extract_mixed_knowledge(self, text: str, title: str) -> Dict:
        """Extract mixed knowledge from text"""
        try:
            prompt = f"""Extract both procedural and informational knowledge from the following text. 
            Format the response as a JSON object with these exact fields:
            {{
                "concept": {{
                    "id": "concept_[timestamp]",
                    "name": "Main concept or topic name",
                    "description": "Brief summary of the concept"
                }},
                "procedures": [
                    {{
                        "id": "procedure_[timestamp]_[index]",
                        "title": "Step title or name",
                        "summary": "Detailed description of the step",
                        "steps": [
                            {{
                                "id": "step_[timestamp]_[index]",
                                "order": [step_number],
                                "text": "Detailed step description"
                            }}
                        ]
                    }}
                ],
                "metrics": [
                    {{
                        "id": "metric_[timestamp]_[index]",
                        "name": "Metric name",
                        "value": "Metric value",
                        "unit": "Unit of measurement",
                        "timestamp": "YYYY-MM-DD"
                    }}
                ],
                "sources": [
                    {{
                        "id": "source_[timestamp]_[index]",
                        "url": "Source URL",
                        "title": "Source title",
                        "published_at": "YYYY-MM-DD"
                    }}
                ],
                "tags": [
                    {{
                        "id": "tag_[timestamp]_[index]",
                        "name": "Tag name"
                    }}
                ],
                "related_concepts": [
                    {{
                        "id": "concept_[timestamp]_[index]",
                        "name": "Related concept name",
                        "description": "Related concept description"
                    }}
                ]
            }}

            Text to analyze:
            {text}

            Respond with ONLY the JSON object, no additional text."""

            response = await self.llm_router.generate_response(prompt)
            return self._parse_llm_response(response)

        except Exception as e:
            logger.error(f"Error extracting mixed knowledge: {str(e)}")
            raise

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse and validate LLM response"""
        try:
            # Extract JSON from response
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["concept", "procedures", "metrics", "sources", "tags", "related_concepts"]
            for field in required_fields:
                if field not in data:
                    data[field] = [] if field != "concept" else {
                        "id": f"concept_{int(time.time())}",
                        "name": "",
                        "description": ""
                    }
            
            # Validate concept fields
            if not all(field in data["concept"] for field in ["id", "name", "description"]):
                data["concept"] = {
                    "id": f"concept_{int(time.time())}",
                    "name": data["concept"].get("name", ""),
                    "description": data["concept"].get("description", "")
                }
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise Exception("Failed to parse knowledge extraction response")

    def _validate_data_quality(self, data: Dict) -> Dict:
        """Validate and clean data before storage."""
        validated = data.copy()
        
        # Clean and validate concept
        if "concept" in validated:
            concept = validated["concept"]
            concept["name"] = self._clean_text(concept.get("name", ""))
            concept["description"] = self._clean_text(concept.get("description", ""))
            # Calculate confidence score based on content quality
            concept["confidence_score"] = self._calculate_confidence_score(concept)
        
        # Clean and validate procedures
        if "procedures" in validated:
            for procedure in validated["procedures"]:
                procedure["title"] = self._clean_text(procedure.get("title", ""))
                procedure["summary"] = self._clean_text(procedure.get("summary", ""))
                procedure["confidence_score"] = self._calculate_confidence_score(procedure)
                
                # Clean and validate steps
                if "steps" in procedure:
                    for step in procedure["steps"]:
                        step["text"] = self._clean_text(step.get("text", ""))
                        step["confidence_score"] = self._calculate_confidence_score(step)
        
        # Clean and validate metrics
        if "metrics" in validated:
            for metric in validated["metrics"]:
                metric["name"] = self._clean_text(metric.get("name", ""))
                metric["value"] = self._clean_text(metric.get("value", ""))
                metric["unit"] = self._clean_text(metric.get("unit", ""))
                metric["confidence_score"] = self._calculate_confidence_score(metric)
        
        # Clean and validate tags
        if "tags" in validated:
            for tag in validated["tags"]:
                tag["name"] = self._clean_text(tag.get("name", ""))
                tag["confidence_score"] = self._calculate_confidence_score(tag)
        
        return validated

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()

    def _calculate_confidence_score(self, data: Dict) -> float:
        """Calculate confidence score based on data quality."""
        score = 1.0
        
        # Check for empty or very short content
        for key, value in data.items():
            if isinstance(value, str):
                if not value:
                    score *= 0.8
                elif len(value) < 10:
                    score *= 0.9
        
        # Check for required fields
        required_fields = {
            "concept": ["name", "description"],
            "procedure": ["title", "summary"],
            "step": ["text"],
            "metric": ["name", "value", "unit"],
            "tag": ["name"]
        }
        
        node_type = next((k for k in required_fields.keys() if k in str(data)), None)
        if node_type:
            for field in required_fields[node_type]:
                if field not in data or not data[field]:
                    score *= 0.7
        
        return round(score, 2)

    def _chunk_vector_text(self, data: Dict, source_url: Optional[str] = None) -> List[Dict]:
        """Split content into semantically meaningful chunks for vector storage."""
        chunks = []
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        # Add semantic similarity threshold to metadata
        similarity_threshold = 0.75  # Default threshold for semantic matching
        
        # Main concept chunk with enhanced metadata
        concept_chunk = {
            "text": f"""
Concept: {data['concept']['name']}
Description: {data['concept']['description']}
Confidence Score: {data['concept'].get('confidence_score', 1.0)}
""",
            "type": "concept",
            "metadata": {
                "node_id": data["concept"]["id"],
                "node_type": "Concept",
                "source_url": source_url if source_url else "",
                "timestamp": timestamp,
                "confidence_score": data["concept"].get("confidence_score", 1.0),
                "similarity_threshold": similarity_threshold
            }
        }
        chunks.append(concept_chunk)

        # Procedures chunk - one chunk per procedure with its steps
        if data.get('procedures'):
            for procedure in data['procedures']:
                procedure_text = f"""
Procedure: {procedure['title']}
Summary: {procedure['summary']}
Steps:
{chr(10).join([f"{step['order']}. {step['text']}" for step in procedure.get('steps', [])])}
"""
                chunks.append({
                    "text": procedure_text,
                    "type": "procedure",
                    "metadata": {
                        "node_id": procedure["id"],
                        "node_type": "Procedure",
                        "source_url": source_url if source_url else "",
                        "timestamp": timestamp,
                        "parent_concept_id": data["concept"]["id"]
                    }
                })

                # Individual step chunks for more granular search
                for step in procedure.get('steps', []):
                    step_text = f"""
Step {step['order']} of Procedure: {procedure['title']}
{step['text']}
"""
                    chunks.append({
                        "text": step_text,
                        "type": "step",
                        "metadata": {
                            "node_id": step["id"],
                            "node_type": "Step",
                            "source_url": source_url if source_url else "",
                            "timestamp": timestamp,
                            "parent_procedure_id": procedure["id"],
                            "parent_concept_id": data["concept"]["id"],
                            "order": step["order"]
                        }
                    })

        # Metrics chunk
        if data.get('metrics'):
            metrics_text = "Metrics:\n" + "\n".join([
                f"- {m['name']}: {m['value']} {m['unit']} ({m['timestamp']})"
                for m in data.get('metrics', [])
            ])
            chunks.append({
                "text": metrics_text,
                "type": "metrics",
                "metadata": {
                    "node_id": data["concept"]["id"],
                    "node_type": "Concept",
                    "source_url": source_url if source_url else "",
                    "timestamp": timestamp
                }
            })

            # Individual metric chunks for more granular search
            for metric in data.get('metrics', []):
                metric_text = f"""
Metric: {metric['name']}
Value: {metric['value']} {metric['unit']}
Timestamp: {metric['timestamp']}
"""
                chunks.append({
                    "text": metric_text,
                    "type": "metric",
                    "metadata": {
                        "node_id": metric["id"],
                        "node_type": "Metric",
                        "source_url": source_url if source_url else "",
                        "timestamp": metric["timestamp"],
                        "parent_concept_id": data["concept"]["id"]
                    }
                })

        # Sources chunk
        if data.get('sources') or source_url:
            sources = data.get('sources', [])
            if source_url:
                sources.append({
                    "url": source_url,
                    "title": data["concept"]["name"],
                    "published_at": timestamp
                })
            
            sources_text = "Sources:\n" + "\n".join([
                f"- {s['title']} ({s['url']}) - Published: {s.get('published_at', 'N/A')}"
                for s in sources
            ])
            chunks.append({
                "text": sources_text,
                "type": "sources",
                "metadata": {
                    "node_id": data["concept"]["id"],
                    "node_type": "Concept",
                    "source_url": source_url if source_url else "",
                    "timestamp": timestamp
                }
            })

        # Tags chunk
        if data.get('tags'):
            tags_text = "Tags:\n" + "\n".join([
                f"- {tag['name']}"
                for tag in data.get('tags', [])
            ])
            chunks.append({
                "text": tags_text,
                "type": "tags",
                "metadata": {
                    "node_id": data["concept"]["id"],
                    "node_type": "Concept",
                    "source_url": source_url if source_url else "",
                    "timestamp": timestamp
                }
            })

        # Related concepts chunk
        if data.get('related_concepts'):
            concepts_text = "Related Concepts:\n" + "\n".join([
                f"- {c['name']}: {c['description']}"
                for c in data.get('related_concepts', [])
            ])
            chunks.append({
                "text": concepts_text,
                "type": "related_concepts",
                "metadata": {
                    "node_id": data["concept"]["id"],
                    "node_type": "Concept",
                    "source_url": source_url if source_url else "",
                    "timestamp": timestamp
                }
            })

        return chunks

    def _normalize_data(self, data: Dict) -> Dict:
        """Normalize the input data to ensure consistent structure and types."""
        # Deep copy the input data to avoid modifying the original
        normalized = data.copy()
        
        # Ensure concept exists with required fields
        if "concept" not in normalized:
            normalized["concept"] = {
                "id": f"concept_{int(time.time())}",
                "name": "",
                "description": ""
            }
        else:
            if "id" not in normalized["concept"]:
                normalized["concept"]["id"] = f"concept_{int(time.time())}"
            if "name" not in normalized["concept"]:
                normalized["concept"]["name"] = ""
            if "description" not in normalized["concept"]:
                normalized["concept"]["description"] = ""
        
        # Ensure all list fields exist and contain proper objects
        list_fields = {
            "procedures": {
                "id": lambda i: f"procedure_{int(time.time())}_{i}",
                "title": "",
                "summary": "",
                "steps": []
            },
            "metrics": {
                "id": lambda i: f"metric_{int(time.time())}_{i}",
                "name": "",
                "value": "",
                "unit": "",
                "timestamp": datetime.now().strftime("%Y-%m-%d")
            },
            "sources": {
                "id": lambda i: f"source_{int(time.time())}_{i}",
                "url": "",
                "title": "",
                "published_at": datetime.now().strftime("%Y-%m-%d")
            },
            "tags": {
                "id": lambda i: f"tag_{int(time.time())}_{i}",
                "name": ""
            },
            "related_concepts": {
                "id": lambda i: f"concept_{int(time.time())}_{i}",
                "name": "",
                "description": ""
            }
        }
        
        for field, template in list_fields.items():
            if field not in normalized:
                normalized[field] = []
            else:
                # Ensure each item in the list has all required fields
                for i, item in enumerate(normalized[field]):
                    if not isinstance(item, dict):
                        normalized[field][i] = {}
                    for key, default_value in template.items():
                        if key not in normalized[field][i]:
                            normalized[field][i][key] = default_value(i) if callable(default_value) else default_value
        
        return normalized

    async def _store_in_neo4j(self, data: Dict, source_url: Optional[str] = None):
        """Store the extracted knowledge in Neo4j and vector DB"""
        try:
            # Validate and clean data before storage
            data = self._validate_data_quality(data)
            
            # Create indexes if they don't exist
            await self._ensure_indexes()
            
            # Normalize data before storage
            data = self._normalize_data(data)
            
            # Create main Concept node
            concept_props = {
                "id": f"concept_{int(time.time())}",
                "name": data["concept"]["name"],
                "description": data["concept"]["description"]
            }
            concept_node = neo4j_connector.create_node("Concept", concept_props)
            if not concept_node:
                raise Exception("Failed to create concept node")

            # Store in vector DB with chunked content
            chunks = self._chunk_vector_text(data, source_url)
            for chunk in chunks:
                await qdrant_db.add_document(
                    chunk["text"],
                    metadata={
                        **chunk["metadata"],
                        "chunk_type": chunk["type"]
                    },
                    node_id=concept_node["id"],
                    node_type="Concept",
                    field_name="description"
                )

            # Handle Sources - Check for existing sources before creating new ones
            sources_to_link = []
            
            # Handle URL source if provided
            if source_url:
                # Check if source with this URL already exists
                existing_source = neo4j_connector.execute_query(
                    "MATCH (s:Source) WHERE s.url = $url RETURN s",
                    {"url": source_url}
                )
                
                if existing_source and len(existing_source) > 0:
                    # Use existing source
                    source_node = existing_source[0]["s"]
                    logger.info(f"Using existing source: {source_url}")
                else:
                    # Create new source
                    source_props = {
                        "id": f"source_{int(time.time())}",
                        "url": source_url,
                        "title": data["concept"]["name"],
                        "published_at": data["sources"][0]["published_at"] if data.get("sources") else datetime.now().strftime("%Y-%m-%d")
                    }
                    source_node = neo4j_connector.create_node("Source", source_props)
                    logger.info(f"Created new source: {source_url}")
                
                sources_to_link.append(source_node)

            # Handle other sources from data
            for source in data.get("sources", []):
                if source.get("url"):
                    # Check if source with this URL already exists
                    existing_source = neo4j_connector.execute_query(
                        "MATCH (s:Source) WHERE s.url = $url RETURN s",
                        {"url": source["url"]}
                    )
                    
                    if existing_source and len(existing_source) > 0:
                        # Use existing source
                        source_node = existing_source[0]["s"]
                        logger.info(f"Using existing source: {source['url']}")
                    else:
                        # Create new source
                        source_props = {
                            "id": source["id"],
                            "url": source["url"],
                            "title": source["title"],
                            "published_at": source["published_at"]
                        }
                        source_node = neo4j_connector.create_node("Source", source_props)
                        logger.info(f"Created new source: {source['url']}")
                    
                    sources_to_link.append(source_node)

            # Create relationships for all sources
            for source_node in sources_to_link:
                neo4j_connector.create_relationship(
                    "Concept", "Source", "CITED_IN",
                    concept_props, {"id": source_node["id"]}, {}
                )

            # Handle Tags - Check for existing tags before creating new ones
            for tag in data.get("tags", []):
                # First, try to find an existing tag with the same name
                existing_tag = neo4j_connector.execute_query(
                    "MATCH (t:Tag) WHERE t.name = $name RETURN t",
                    {"name": tag["name"]}
                )
                
                if existing_tag and len(existing_tag) > 0:
                    # Use existing tag
                    tag_node = existing_tag[0]["t"]
                    logger.info(f"Using existing tag: {tag['name']}")
                else:
                    # Create new tag if it doesn't exist
                    tag_props = {
                        "id": tag["id"],
                        "name": tag["name"]
                    }
                    tag_node = neo4j_connector.create_node("Tag", tag_props)
                    logger.info(f"Created new tag: {tag['name']}")
                
                if tag_node:
                    neo4j_connector.create_relationship(
                        "Concept", "Tag", "TAGGED_AS",
                        concept_props, {"id": tag_node["id"]}, {}
                    )

            # Create Procedure nodes for procedural parts
            previous_proc = None
            for i, procedure in enumerate(data.get("procedures", [])):
                procedure_props = {
                    "id": procedure["id"],
                    "title": procedure["title"],
                    "summary": procedure["summary"]
                }
                procedure_node = neo4j_connector.create_node("Procedure", procedure_props)
                if procedure_node:
                    # Link procedure to concept
                    neo4j_connector.create_relationship(
                        "Concept", "Procedure", "HAS_PROCEDURE",
                        concept_props, procedure_props, {}
                    )
                    
                    # Create Steps for each procedure
                    for step in procedure.get("steps", []):
                        step_props = {
                            "id": step["id"],
                            "order": step["order"],
                            "text": step["text"]
                        }
                        step_node = neo4j_connector.create_node("Step", step_props)
                        if step_node:
                            neo4j_connector.create_relationship(
                                "Procedure", "Step", "HAS_STEP",
                                procedure_props, step_props, {}
                            )
                    
                    # Link to previous procedure if exists
                    if previous_proc:
                        neo4j_connector.create_relationship(
                            "Procedure", "Procedure", "RELATED_TO",
                            previous_proc, procedure_props, {}
                        )
                    previous_proc = procedure_props

            # Create Metric nodes for key facts
            for i, metric in enumerate(data.get("metrics", [])):
                metric_props = {
                    "id": metric["id"],
                    "name": metric["name"],
                    "value": metric["value"],
                    "unit": metric["unit"],
                    "timestamp": metric["timestamp"]
                }
                metric_node = neo4j_connector.create_node("Metric", metric_props)
                if metric_node:
                    neo4j_connector.create_relationship(
                        "Concept", "Metric", "MEASURED_BY",
                        concept_props, metric_props, {}
                    )

            # Create related Concept nodes for informational parts
            for concept in data.get("related_concepts", []):
                # Check if related concept already exists
                existing_concept = neo4j_connector.execute_query(
                    "MATCH (c:Concept) WHERE c.name = $name RETURN c",
                    {"name": concept["name"]}
                )
                
                if existing_concept and len(existing_concept) > 0:
                    # Use existing concept
                    related_concept_node = existing_concept[0]["c"]
                    logger.info(f"Using existing related concept: {concept['name']}")
                else:
                    # Create new concept if it doesn't exist
                    related_concept_props = {
                        "id": concept["id"],
                        "name": concept["name"],
                        "description": concept["description"]
                    }
                    related_concept_node = neo4j_connector.create_node("Concept", related_concept_props)
                    logger.info(f"Created new related concept: {concept['name']}")
                
                if related_concept_node:
                    neo4j_connector.create_relationship(
                        "Concept", "Concept", "RELATED_TO",
                        concept_props, {"id": related_concept_node["id"]}, {}
                    )

            return concept_node

        except Exception as e:
            logger.error(f"Error storing knowledge in Neo4j: {str(e)}")
            raise

    async def _ensure_indexes(self):
        """Ensure all necessary indexes exist for optimal query performance."""
        indexes = [
            # Concept indexes
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_description IF NOT EXISTS FOR (c:Concept) ON (c.description)",
            "CREATE FULLTEXT INDEX concept_text IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.description]",
            
            # Procedure indexes
            "CREATE INDEX procedure_title IF NOT EXISTS FOR (p:Procedure) ON (p.title)",
            "CREATE FULLTEXT INDEX procedure_text IF NOT EXISTS FOR (p:Procedure) ON EACH [p.title, p.summary]",
            
            # Step indexes
            "CREATE INDEX step_order IF NOT EXISTS FOR (s:Step) ON (s.order)",
            "CREATE FULLTEXT INDEX step_text IF NOT EXISTS FOR (s:Step) ON EACH [s.text]",
            
            # Metric indexes
            "CREATE INDEX metric_name IF NOT EXISTS FOR (m:Metric) ON (m.name)",
            "CREATE INDEX metric_timestamp IF NOT EXISTS FOR (m:Metric) ON (m.timestamp)",
            
            # Tag indexes
            "CREATE INDEX tag_name IF NOT EXISTS FOR (t:Tag) ON (t.name)",
            
            # Source indexes
            "CREATE INDEX source_url IF NOT EXISTS FOR (s:Source) ON (s.url)",
            "CREATE INDEX source_published_at IF NOT EXISTS FOR (s:Source) ON (s.published_at)"
        ]
        
        for index in indexes:
            try:
                neo4j_connector.execute_query(index)
            except Exception as e:
                logger.warning(f"Failed to create index: {str(e)}")

    async def search_similar_concepts(self, query: str, threshold: float = 0.75, limit: int = 10) -> List[Dict]:
        """Search for similar concepts using semantic similarity."""
        try:
            # Get vector representation of the query
            query_vector = await qdrant_db.get_embedding(query)
            
            # Search in vector store with threshold
            results = await qdrant_db.search(
                query_vector,
                threshold=threshold,
                limit=limit,
                filter={"node_type": "Concept"}
            )
            
            # Enhance results with additional context
            enhanced_results = []
            for result in results:
                # Get related nodes
                related_nodes = neo4j_connector.execute_query(
                    """
                    MATCH (c:Concept {id: $id})
                    OPTIONAL MATCH (c)-[:HAS_PROCEDURE]->(p:Procedure)
                    OPTIONAL MATCH (c)-[:MEASURED_BY]->(m:Metric)
                    OPTIONAL MATCH (c)-[:TAGGED_AS]->(t:Tag)
                    RETURN c, collect(distinct p) as procedures, 
                           collect(distinct m) as metrics,
                           collect(distinct t) as tags
                    """,
                    {"id": result["node_id"]}
                )
                
                if related_nodes:
                    node_data = related_nodes[0]
                    enhanced_results.append({
                        "concept": node_data["c"],
                        "procedures": node_data["procedures"],
                        "metrics": node_data["metrics"],
                        "tags": node_data["tags"],
                        "similarity_score": result["similarity_score"]
                    })
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise 