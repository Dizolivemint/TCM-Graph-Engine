# tcm/graph/query_engine.py
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from tcm.core.models import Node, Relationship, Source
from tcm.core.enums import NodeType, RelationType
from .knowledge_graph import TCMKnowledgeGraph

@dataclass
class QueryResult:
    """Container for query results."""
    nodes: List[Node]
    relationships: List[Relationship]
    sources: Set[Source]
    confidence: float
    metadata: Dict[str, Any]

class TCMGraphQueryEngine:
    """Query engine for TCM knowledge graph."""
    
    def __init__(self, graph: TCMKnowledgeGraph):
        self.graph = graph
    
    def find_treatments(self, pattern_id: str) -> QueryResult:
        """Find treatments for a given pattern."""
        treatments = {
            "herbs": [],
            "points": [],
            "formulas": [],
            "sources": set(),
            "confidence": 1.0
        }
        
        for node, relationship in self.graph.get_neighbors(
            pattern_id,
            relationship_type=RelationType.TREATS
        ):
            if node.type == NodeType.HERB:
                treatments["herbs"].append(node)
            elif node.type == NodeType.POINT:
                treatments["points"].append(node)
            elif node.type == NodeType.FORMULA:
                treatments["formulas"].append(node)
                
            treatments["sources"].update(relationship.sources)
            treatments["confidence"] *= relationship.confidence
            
        return QueryResult(
            nodes=treatments["herbs"] + treatments["points"] + treatments["formulas"],
            relationships=[],  # We could include the relationships if needed
            sources=treatments["sources"],
            confidence=treatments["confidence"],
            metadata={}
        )
    
    def find_patterns_from_symptoms(
        self,
        symptom_ids: List[str],
        min_confidence: float = 0.5
    ) -> QueryResult:
        """Find patterns that match a set of symptoms."""
        pattern_matches = {}
        all_sources = set()
        
        for symptom_id in symptom_ids:
            for node, relationship in self.graph.get_neighbors(
                symptom_id,
                relationship_type=RelationType.MANIFESTS
            ):
                if node.type != NodeType.PATTERN:
                    continue
                    
                if node.id not in pattern_matches:
                    pattern_matches[node.id] = {
                        "node": node,
                        "matching_symptoms": 0,
                        "confidence": 1.0,
                        "relationships": []
                    }
                    
                pattern_matches[node.id]["matching_symptoms"] += 1
                pattern_matches[node.id]["confidence"] *= relationship.confidence
                pattern_matches[node.id]["relationships"].append(relationship)
                all_sources.update(relationship.sources)
        
        # Filter and sort matches
        valid_matches = [
            match for match in pattern_matches.values()
            if match["confidence"] >= min_confidence
        ]
        valid_matches.sort(
            key=lambda x: (x["matching_symptoms"], x["confidence"]),
            reverse=True
        )
        
        return QueryResult(
            nodes=[match["node"] for match in valid_matches],
            relationships=[
                rel for match in valid_matches
                for rel in match["relationships"]
            ],
            sources=all_sources,
            confidence=max((match["confidence"] for match in valid_matches), default=0.0),
            metadata={"symptom_matches": {
                match["node"].id: match["matching_symptoms"]
                for match in valid_matches
            }}
        )