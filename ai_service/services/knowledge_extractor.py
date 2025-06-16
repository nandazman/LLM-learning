from bs4 import BeautifulSoup
import httpx
from typing import Dict, List, Optional
import json
from llm.router import LLMRouter
from data.neo4j_connector import neo4j_connector
from data.vector_store import vector_db

class KnowledgeExtractor:
    def __init__(self, llm_router: LLMRouter, store_in_neo4j: bool = True):
        self.llm_router = llm_router
        self.client = httpx.AsyncClient()
        self.store_in_neo4j = store_in_neo4j

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

            # First, detect the type of content
            content_type = await self._detect_content_type(content)
            print(f"[DEBUG] Detected content type: {content_type}")

            # Then extract knowledge based on content type
            if content_type == "procedural":
                structured_data = await self._extract_procedural_knowledge(content, title_text)
            elif content_type == "informational":
                structured_data = await self._extract_informational_knowledge(content, title_text)
            else:  # mixed
                structured_data = await self._extract_mixed_knowledge(content, title_text)

            if self.store_in_neo4j:
                print(f"[DEBUG] Storing structured data in Neo4j...")
                await self._store_in_neo4j(structured_data, url)
                print(f"[DEBUG] Data stored in Neo4j successfully.")

            return structured_data

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

    async def _extract_procedural_knowledge(self, content: str, title: str) -> Dict:
        """Extract procedural knowledge from the article using the new, detailed prompt."""
        prompt = f"""
You are an expert at extracting structured knowledge for graph databases.

Your task is to extract procedural knowledge from the following article:

Title: {title}

Content:
{content}

Return a JSON object with the following structure (all fields must be filled as accurately as possible, in Bahasa Indonesia):

{{
  "title": "{title}",
  "content_type": "procedural",          // "procedural", "informational", or "mixed"
  "topic": ["string"],               // Pilih dari: ["Jual Properti", "Beli Properti", "Sewa Properti", "Agen Properti", "KPR", "Iklan di Rumah123", "Hukum Properti", "Tips & Lainnya", "Take Over KPR"]
  "summary": "string",               // Ringkasan singkat dari artikel
  "keywords": ["string"],           // Kata kunci penting dari seluruh artikel

  "procedural_parts": [
    {{
      "step": "string",             // Langkah dalam proses
      "description": "string",      // Penjelasan langkah
      "requirements": ["string"],   // Persyaratan / dokumen / alat yang dibutuhkan
      "warnings": ["string"],       // Peringatan atau risiko di langkah ini
      "tips": ["string"],           // Saran untuk menjalankan langkah ini lebih baik
      "keywords": ["string"]        // Kata kunci khusus untuk langkah ini
    }}
  ],

  "informational_parts": [
    {{
      "concept": "string",          // Istilah atau konsep yang dijelaskan
      "definition": "string",       // Penjelasan konsep tersebut
      "importance": "high/medium/low", // Seberapa penting informasi ini
      "keywords": ["string"]        // Kata kunci relevan dengan konsep ini
    }}
  ],

  "key_facts": ["string"],         // Fakta penting dan mudah diingat
  "warnings": ["string"],          // Peringatan umum dari artikel
  "tips": ["string"],              // Tips umum dari artikel

  "metadata": {{
    "language": "id",              // Bahasa konten
    "domain": "string",            // Contoh: "properti", "hukum", "keuangan"
    "difficulty": "beginner/intermediate/advanced", // Tingkat pemahaman yang diperlukan
    "prerequisites": ["string"]    // Pengetahuan dasar yang diperlukan
  }}
}}

Only return the JSON. Don't include explanations or additional commentary.
"""
        return json.loads(await self.llm_router.generate_response(prompt))

    async def _extract_informational_knowledge(self, content: str, title: str) -> Dict:
        """Extract informational knowledge from the article using the new, detailed prompt."""
        prompt = f"""
You are an expert at extracting structured knowledge for graph databases.

Your task is to extract informational knowledge from the following article:

Title: {title}

Content:
{content}

Return a JSON object with the following structure (all fields must be filled as accurately as possible, in Bahasa Indonesia):

{{
  "title": "{title}",
  "content_type": "informational",          // "procedural", "informational", or "mixed"
  "topic": ["string"],               // Pilih dari: ["Jual Properti", "Beli Properti", "Sewa Properti", "Agen Properti", "KPR", "Iklan di Rumah123", "Hukum Properti", "Tips & Lainnya", "Take Over KPR"]
  "summary": "string",               // Ringkasan singkat dari artikel
  "keywords": ["string"],           // Kata kunci penting dari seluruh artikel

  "procedural_parts": [
    {{
      "step": "string",             // Langkah dalam proses
      "description": "string",      // Penjelasan langkah
      "requirements": ["string"],   // Persyaratan / dokumen / alat yang dibutuhkan
      "warnings": ["string"],       // Peringatan atau risiko di langkah ini
      "tips": ["string"],           // Saran untuk menjalankan langkah ini lebih baik
      "keywords": ["string"]        // Kata kunci khusus untuk langkah ini
    }}
  ],

  "informational_parts": [
    {{
      "concept": "string",          // Istilah atau konsep yang dijelaskan
      "definition": "string",       // Penjelasan konsep tersebut
      "importance": "high/medium/low", // Seberapa penting informasi ini
      "keywords": ["string"]        // Kata kunci relevan dengan konsep ini
    }}
  ],

  "key_facts": ["string"],         // Fakta penting dan mudah diingat
  "warnings": ["string"],          // Peringatan umum dari artikel
  "tips": ["string"],              // Tips umum dari artikel

  "metadata": {{
    "language": "id",              // Bahasa konten
    "domain": "string",            // Contoh: "properti", "hukum", "keuangan"
    "difficulty": "beginner/intermediate/advanced", // Tingkat pemahaman yang diperlukan
    "prerequisites": ["string"]    // Pengetahuan dasar yang diperlukan
  }}
}}

Only return the JSON. Don't include explanations or additional commentary.
"""
        return json.loads(await self.llm_router.generate_response(prompt))

    async def _extract_mixed_knowledge(self, content: str, title: str) -> Dict:
        """Extract both procedural and informational knowledge from the article using the new, detailed prompt."""
        prompt = f"""
You are an expert at extracting structured knowledge for graph databases.

Your task is to extract both procedural and informational knowledge from the following article:

Title: {title}

Content:
{content}

Return a JSON object with the following structure (all fields must be filled as accurately as possible, in Bahasa Indonesia):

{{
  "title": "{title}",
  "content_type": "string",          // "procedural", "informational", or "mixed"
  "topic": ["string"],               // Pilih dari: ["Jual Properti", "Beli Properti", "Sewa Properti", "Agen Properti", "KPR", "Iklan di Rumah123", "Hukum Properti", "Tips & Lainnya", "Take Over KPR"]
  "summary": "string",               // Ringkasan singkat dari artikel
  "keywords": ["string"],           // Kata kunci penting dari seluruh artikel

  "procedural_parts": [
    {{
      "step": "string",             // Langkah dalam proses
      "description": "string",      // Penjelasan langkah
      "requirements": ["string"],   // Persyaratan / dokumen / alat yang dibutuhkan
      "warnings": ["string"],       // Peringatan atau risiko di langkah ini
      "tips": ["string"],           // Saran untuk menjalankan langkah ini lebih baik
      "keywords": ["string"]        // Kata kunci khusus untuk langkah ini
    }}
  ],

  "informational_parts": [
    {{
      "concept": "string",          // Istilah atau konsep yang dijelaskan
      "definition": "string",       // Penjelasan konsep tersebut
      "importance": "high/medium/low", // Seberapa penting informasi ini
      "keywords": ["string"]        // Kata kunci relevan dengan konsep ini
    }}
  ],

  "key_facts": ["string"],         // Fakta penting dan mudah diingat
  "warnings": ["string"],          // Peringatan umum dari artikel
  "tips": ["string"],              // Tips umum dari artikel

  "metadata": {{
    "language": "id",              // Bahasa konten
    "domain": "string",            // Contoh: "properti", "hukum", "keuangan"
    "difficulty": "beginner/intermediate/advanced", // Tingkat pemahaman yang diperlukan
    "prerequisites": ["string"]    // Pengetahuan dasar yang diperlukan
  }}
}}

Only return the JSON. Don't include explanations or additional commentary.
"""
        return json.loads(await self.llm_router.generate_response(prompt))

    async def _store_in_neo4j(self, data: Dict, source_url: str):
        """Store the extracted knowledge in Neo4j and vector DB"""
        try:
            # Normalize helper
            def normalize_list(lst):
                return [str(x).strip().lower() for x in lst]
            def normalize_str(s):
                return str(s).strip().lower() if s else s

            article_props = {
                "title": normalize_str(data["title"]),
                "url": source_url,
                "type": data["content_type"],
                "topic": normalize_list(data["topic"]),
                "summary": data.get("summary", ""),
                "keywords": normalize_list(data.get("keywords", [])),
                "language": data.get("metadata", {}).get("language", "id"),
                "domain": normalize_str(data.get("metadata", {}).get("domain", "")),
                "difficulty": data.get("metadata", {}).get("difficulty", "intermediate"),
                "prerequisites": data.get("metadata", {}).get("prerequisites", []),
                "node_type": "Article"
            }
            article_node = neo4j_connector.create_node("Article", article_props)
            
            # Store in vector DB
            article_text = f"""Title: {data['title']}\nType: {data['content_type']}\nTopics: {', '.join(data['topic'])}\nSummary: {data.get('summary', '')}\nKeywords: {', '.join(data.get('keywords', []))}\nDomain: {data.get('metadata', {}).get('domain', '')}\nDifficulty: {data.get('metadata', {}).get('difficulty', 'intermediate')}\n"""
            # Only Article, Concept, and Procedure nodes are added to the vector DB for semantic search.
            # Facts, Warnings, Tips, Topics, Keywords, and Prerequisites are NOT embedded or indexed in the vector DB.
            await vector_db.add_document(
                text=article_text,
                metadata=article_props,
                node_id=article_node["n"].id
            )

            # Store keywords as nodes and relationships for Article
            for keyword in article_props["keywords"]:
                keyword_props = {"name": keyword, "node_type": "Keyword"}
                keyword_node = neo4j_connector.create_node("Keyword", keyword_props)
                neo4j_connector.create_relationship(
                    "Article",
                    "Keyword",
                    "HAS_KEYWORD",
                    article_props,
                    keyword_props,
                    {}
                )
            # Store prerequisites as nodes and relationships for Article
            for prereq in article_props["prerequisites"]:
                prereq_props = {"name": prereq, "node_type": "Prerequisite"}
                prereq_node = neo4j_connector.create_node("Prerequisite", prereq_props)
                neo4j_connector.create_relationship(
                    "Article",
                    "Prerequisite",
                    "HAS_PREREQUISITE",
                    article_props,
                    prereq_props,
                    {}
                )

            # Store concepts from informational_parts
            for info in data.get("informational_parts", []):
                concept_props = {
                    "name": normalize_str(info["concept"]),
                    "definition": info.get("definition", ""),
                    "importance": info.get("importance", "medium"),
                    "keywords": normalize_list(info.get("keywords", [])),
                    "node_type": "Concept"
                }
                concept_node = neo4j_connector.create_node("Concept", concept_props)
                # Store keywords for Concept
                for keyword in concept_props["keywords"]:
                    keyword_props = {"name": keyword, "node_type": "Keyword"}
                    keyword_node = neo4j_connector.create_node("Keyword", keyword_props)
                    neo4j_connector.create_relationship(
                        "Concept",
                        "Keyword",
                        "HAS_KEYWORD",
                        concept_props,
                        keyword_props,
                        {}
                    )
                neo4j_connector.create_relationship(
                    "Article",
                    "Concept",
                    "DISCUSSES",
                    article_props,
                    concept_props,
                    {}
                )

            # Store procedures from procedural_parts
            previous_proc_props = None  # Keep track of previous procedure

            for procedure in data.get("procedural_parts", []):
                proc_props = {
                    "step": procedure["step"],
                    "description": procedure["description"],
                    "requirements": procedure.get("requirements", []),
                    "warnings": procedure.get("warnings", []),
                    "tips": procedure.get("tips", []),
                    "keywords": normalize_list(procedure.get("keywords", [])),
                    "node_type": "Procedure"
                }

                proc_node = neo4j_connector.create_node("Procedure", proc_props)

                # Link to keywords
                for keyword in proc_props["keywords"]:
                    keyword_props = {"name": keyword, "node_type": "Keyword"}
                    keyword_node = neo4j_connector.create_node("Keyword", keyword_props)
                    neo4j_connector.create_relationship(
                        "Procedure", "Keyword", "HAS_KEYWORD", proc_props, keyword_props, {}
                    )

                # Link to prerequisites
                for prereq in proc_props.get("requirements", []):
                    prereq_props = {"name": prereq, "node_type": "Prerequisite"}
                    prereq_node = neo4j_connector.create_node("Prerequisite", prereq_props)
                    neo4j_connector.create_relationship(
                        "Procedure", "Prerequisite", "HAS_PREREQUISITE", proc_props, prereq_props, {}
                    )

                # Link to Article
                neo4j_connector.create_relationship(
                    "Article", "Procedure", "CONTAINS_PROCEDURE", article_props, proc_props, {}
                )

                # 🔗 Link to previous step using :NEXT
                if previous_proc_props:
                    neo4j_connector.create_relationship(
                        "Procedure", "Procedure", "NEXT", previous_proc_props, proc_props, {}
                    )

                previous_proc_props = proc_props


            # Store facts
            for fact in data.get("key_facts", []):
                fact_props = {
                    "text": fact,
                    "source": source_url,
                    "node_type": "Fact"
                }
                fact_node = neo4j_connector.create_node("Fact", fact_props)
                neo4j_connector.create_relationship(
                    "Article",
                    "Fact",
                    "CONTAINS_FACT",
                    article_props,
                    fact_props,
                    {}
                )

            # Store warnings
            for warning in data.get("warnings", []):
                warning_props = {
                    "text": warning,
                    "node_type": "Warning"
                }
                warning_node = neo4j_connector.create_node("Warning", warning_props)
                neo4j_connector.create_relationship(
                    "Article",
                    "Warning",
                    "HAS_WARNING",
                    article_props,
                    warning_props,
                    {}
                )

            # Store tips
            for tip in data.get("tips", []):
                tip_props = {
                    "text": tip,
                    "node_type": "Tip"
                }
                tip_node = neo4j_connector.create_node("Tip", tip_props)
                neo4j_connector.create_relationship(
                    "Article",
                    "Tip",
                    "HAS_TIP",
                    article_props,
                    tip_props,
                    {}
                )

            # Store topics
            for topic in data.get("topic", []):
                topic_props = {"name": normalize_str(topic), "node_type": "Topic"}
                topic_node = neo4j_connector.create_node("Topic", topic_props)
                neo4j_connector.create_relationship(
                    "Article",
                    "Topic",
                    "HAS_TOPIC",
                    article_props,
                    topic_props,
                    {}
                )

        except Exception as e:
            print(f"[ERROR] Error storing in Neo4j: {str(e)}")
            raise 