# tcm/graph/knowledge_graph.py
from typing import Dict, List, Optional, Set, Generator, Tuple
import networkx as nx
from pydantic import ValidationError
from pathlib import Path
import json

from tcm.core.models import Node, Relationship, Source
from tcm.core.enums import NodeType, RelationType
from tcm.core.exceptions import GraphError, ValidationError

class TCMKnowledgeGraph:
    """Main knowledge graph implementation for TCM knowledge system."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_index: Dict[str, Node] = {}
        self.relationship_index: Dict[str, Dict[str, List[Relationship]]] = {}
        
    @property
    def node_count(self) -> int:
        """Return total number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    @property
    def relationship_count(self) -> int:
        """Return total number of relationships in the graph."""
        return self.graph.number_of_edges()
    
    def add_node(self, node: Node) -> None:
        """Add a node to the knowledge graph."""
        try:
            if node.id in self.node_index:
                raise GraphError(f"Node with ID {node.id} already exists")
            
            # Add to NetworkX graph
            self.graph.add_node(
                node.id,
                type=node.type,
                data=node.dict()
            )
            
            # Add to index
            self.node_index[node.id] = node
            
        except ValidationError as e:
            raise ValidationError(f"Invalid node data: {e}")
        except Exception as e:
            raise GraphError(f"Error adding node: {e}")
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between nodes."""
        try:
            # Verify nodes exist
            if relationship.source_id not in self.node_index:
                raise GraphError(f"Source node {relationship.source_id} does not exist")
            if relationship.target_id not in self.node_index:
                raise GraphError(f"Target node {relationship.target_id} does not exist")
            
            # Add to NetworkX graph
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                type=relationship.type,
                data=relationship.dict()
            )
            
            # Add to index
            if relationship.source_id not in self.relationship_index:
                self.relationship_index[relationship.source_id] = {}
            if relationship.target_id not in self.relationship_index[relationship.source_id]:
                self.relationship_index[relationship.source_id][relationship.target_id] = []
                
            self.relationship_index[relationship.source_id][relationship.target_id].append(relationship)
            
        except ValidationError as e:
            raise ValidationError(f"Invalid relationship data: {e}")
        except Exception as e:
            raise GraphError(f"Error adding relationship: {e}")
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID."""
        return self.node_index.get(node_id)
    
    def get_relationships(self, source_id: str, target_id: str) -> List[Relationship]:
        """Get all relationships between two nodes."""
        return self.relationship_index.get(source_id, {}).get(target_id, [])
    
    def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[RelationType] = None,
        direction: str = "outgoing"
    ) -> Generator[Tuple[Node, Relationship], None, None]:
        """Get neighboring nodes and their relationships."""
        if direction not in ["outgoing", "incoming", "both"]:
            raise ValueError("Invalid direction specified")
            
        if direction in ["outgoing", "both"]:
            for _, neighbor_id, edge_data in self.graph.out_edges(node_id, data=True):
                if relationship_type is None or edge_data["type"] == relationship_type:
                    yield (self.get_node(neighbor_id), edge_data["data"])
                    
        if direction in ["incoming", "both"]:
            for neighbor_id, _, edge_data in self.graph.in_edges(node_id, data=True):
                if relationship_type is None or edge_data["type"] == relationship_type:
                    yield (self.get_node(neighbor_id), edge_data["data"])
    
    def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 3
    ) -> Generator[List[Tuple[Node, Relationship]], None, None]:
        """Find all paths between two nodes up to a maximum length."""
        try:
            for path in nx.all_simple_paths(self.graph, start_id, end_id, cutoff=max_length):
                path_with_data = []
                for i in range(len(path) - 1):
                    current_id = path[i]
                    next_id = path[i + 1]
                    node = self.get_node(current_id)
                    edge_data = self.graph.get_edge_data(current_id, next_id)[0]
                    path_with_data.append((node, edge_data["data"]))
                # Add final node
                path_with_data.append((self.get_node(path[-1]), None))
                yield path_with_data
        except nx.NetworkXNoPath:
            return
            
    def validate(self) -> List[str]:
        """Validate the graph structure and data."""
        errors = []
        
        # Check for orphaned nodes
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) == 0:
                errors.append(f"Orphaned node found: {node_id}")
                
        # Check for relationship consistency
        for source_id, target_id, edge_data in self.graph.edges(data=True):
            if source_id not in self.node_index:
                errors.append(f"Relationship references missing source node: {source_id}")
            if target_id not in self.node_index:
                errors.append(f"Relationship references missing target node: {target_id}")
                
        # Check index consistency
        for node_id in self.node_index:
            if node_id not in self.graph:
                errors.append(f"Indexed node missing from graph: {node_id}")
                
        return errors
    
    def save(self, path: Path) -> None:
        """Save the knowledge graph to disk."""
        data = {
            "nodes": [node.dict() for node in self.node_index.values()],
            "relationships": [
                rel.dict()
                for source_rels in self.relationship_index.values()
                for target_rels in source_rels.values()
                for rel in target_rels
            ]
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> 'TCMKnowledgeGraph':
        """Load a knowledge graph from disk."""
        data = json.loads(path.read_text())
        graph = cls()
        
        # Load nodes first
        for node_data in data["nodes"]:
            graph.add_node(Node(**node_data))
            
        # Then load relationships
        for rel_data in data["relationships"]:
            graph.add_relationship(Relationship(**rel_data))
            
        return graph