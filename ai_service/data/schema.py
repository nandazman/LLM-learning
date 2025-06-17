from typing import Dict, List, Any
from data.neo4j_connector import neo4j_connector
import logging

logger = logging.getLogger(__name__)

SCHEMA = {
    "nodes": [
        {"Concept": {"props": ["id", "name", "description"]}},
        {"Procedure": {"props": ["id", "title", "summary"]}},
        {"Step": {"props": ["id", "order", "text"]}},
        {"Metric": {"props": ["id", "name", "value", "unit", "timestamp"]}},
        {"Source": {"props": ["id", "url", "title", "published_at"]}},
        {"Tag": {"props": ["id", "name"]}}
    ],
    "relationships": [
        {"from": "Concept", "type": "RELATED_TO", "to": "Concept"},
        {"from": "Concept", "type": "HAS_PROCEDURE", "to": "Procedure"},
        {"from": "Procedure", "type": "HAS_STEP", "to": "Step"},
        {"from": "Concept", "type": "MEASURED_BY", "to": "Metric"},
        {"from": "*", "type": "CITED_IN", "to": "Source"},
        {"from": "*", "type": "TAGGED_AS", "to": "Tag"}
    ],
    "constraints_and_indexes": [
        "UNIQUE :Concept(id)",
        "UNIQUE :Procedure(id)",
        "UNIQUE :Step(id)",
        "UNIQUE :Metric(id)",
        "UNIQUE :Source(id)",
        "UNIQUE :Tag(id)",
        "FULLTEXT [Concept, Procedure, Step]"
    ]
}

def initialize_schema():
    """Initialize the Neo4j schema with constraints and indexes."""
    try:
        # Create constraints
        for constraint in SCHEMA["constraints_and_indexes"]:
            if constraint.startswith("UNIQUE"):
                neo4j_connector.execute_query(f"CREATE CONSTRAINT IF NOT EXISTS {constraint}")
            elif constraint.startswith("FULLTEXT"):
                # Extract labels from FULLTEXT index
                labels = constraint.split("[")[1].split("]")[0].split(", ")
                for label in labels:
                    neo4j_connector.execute_query(
                        f"CREATE FULLTEXT INDEX {label.lower()}_text IF NOT EXISTS "
                        f"FOR (n:{label}) ON EACH [n.name, n.description, n.text, n.summary]"
                    )
        
        logger.info("Successfully initialized Neo4j schema")
    except Exception as e:
        logger.error(f"Failed to initialize schema: {str(e)}")
        raise

def get_node_properties(node_type: str) -> List[str]:
    """Get the properties for a given node type."""
    for node in SCHEMA["nodes"]:
        if node_type in node:
            return node[node_type]["props"]
    return []

def get_relationship_types() -> List[Dict[str, str]]:
    """Get all relationship types defined in the schema."""
    return SCHEMA["relationships"]

def validate_node_properties(node_type: str, properties: Dict[str, Any]) -> bool:
    """Validate that all required properties are present for a node type."""
    required_props = get_node_properties(node_type)
    return all(prop in properties for prop in required_props)

def validate_relationship(from_type: str, to_type: str, rel_type: str) -> bool:
    """Validate if a relationship type is allowed between two node types."""
    for rel in SCHEMA["relationships"]:
        if (rel["from"] in [from_type, "*"] and 
            rel["to"] in [to_type, "*"] and 
            rel["type"] == rel_type):
            return True
    return False 