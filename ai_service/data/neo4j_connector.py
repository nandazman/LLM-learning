from neo4j import GraphDatabase
from typing import Optional, List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self):
        self._driver = None
        self._uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = os.getenv("NEO4J_USER", "neo4j")
        self._password = os.getenv("NEO4J_PASS", "password")
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password)
            )
            # Verify connection
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query (str): Cypher query to execute
            parameters (Dict[str, Any], optional): Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        if not self._driver:
            self.connect()
            
        try:
            with self._driver.session() as session:
                print("the query is", query)
                print("the parameters are", parameters)
                print("the uri is", self._uri)
                print("the user is", self._user)
                print("the password is", self._password)
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new node with the given label and properties.
        
        Args:
            label (str): Node label
            properties (Dict[str, Any]): Node properties
            
        Returns:
            Dict[str, Any]: Created node data
        """
        query = f"""
        CREATE (n:{label})
        SET n = $properties
        RETURN n
        """
        result = self.execute_query(query, {"properties": properties})
        return result[0] if result else {}

    def create_relationship(
        self,
        from_label: str,
        to_label: str,
        relationship_type: str,
        from_properties: Dict[str, Any],
        to_properties: Dict[str, Any],
        relationship_properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            from_label (str): Label of the source node
            to_label (str): Label of the target node
            relationship_type (str): Type of relationship
            from_properties (Dict[str, Any]): Properties to match source node
            to_properties (Dict[str, Any]): Properties to match target node
            relationship_properties (Dict[str, Any], optional): Properties for the relationship
            
        Returns:
            Dict[str, Any]: Created relationship data
        """
        # Create WHERE clauses for matching nodes
        from_where = " AND ".join([f"a.{k} = ${k}" for k in from_properties.keys()])
        to_where = " AND ".join([f"b.{k} = ${k}" for k in to_properties.keys()])
        
        query = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE {from_where} AND {to_where}
        CREATE (a)-[r:{relationship_type} $rel_props]->(b)
        RETURN a, r, b
        """
        
        # Combine all properties for the query
        params = {
            **from_properties,
            **to_properties,
            "rel_props": relationship_properties or {}
        }
        
        result = self.execute_query(query, params)
        return result[0] if result else {}

    def find_nodes(self, label: str, properties: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find nodes matching the given label and properties.
        
        Args:
            label (str): Node label to search for
            properties (Dict[str, Any], optional): Properties to match
            
        Returns:
            List[Dict[str, Any]]: Matching nodes
        """
        if properties:
            where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (n:{label})
            WHERE {where_clause}
            RETURN n
            """
        else:
            query = f"""
            MATCH (n:{label})
            RETURN n
            """
        return self.execute_query(query, properties or {})

    def delete_node(self, label: str, properties: Dict[str, Any]) -> bool:
        """
        Delete a node matching the given label and properties.
        
        Args:
            label (str): Node label
            properties (Dict[str, Any]): Properties to match
            
        Returns:
            bool: True if node was deleted, False otherwise
        """
        where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"""
        MATCH (n:{label})
        WHERE {where_clause}
        DELETE n
        RETURN count(n) as deleted
        """
        result = self.execute_query(query, properties)
        return result[0]["deleted"] > 0 if result else False

# Singleton instance
neo4j_connector = Neo4jConnector()
