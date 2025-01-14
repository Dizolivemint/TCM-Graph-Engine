# tcm/analysis/analysis_engine.py
import re
from typing import Any, List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from tcm.core.models import Node, Relationship, Source
from tcm.core.enums import NodeType, RelationType
from tcm.graph.knowledge_graph import TCMKnowledgeGraph
from tcm.models.foundation import BaseEmbeddingModel, BaseLLM

class AnalysisType(Enum):
    PATTERN = "pattern"
    HERB = "herb"
    POINT = "point"
    FORMULA = "formula"

@dataclass
class AnalysisResult:
    """Container for TCM analysis results."""
    primary_entities: List[Node]
    interactions: List[Dict[str, any]]
    contraindications: List[Dict[str, any]]
    explanation: str
    mechanism: str
    confidence: float
    sources: Set[Source]
    metadata: Dict[str, any]

class TCMAnalysisEngine:
    """Comprehensive TCM analysis engine with foundation model integration."""
    
    def __init__(
        self,
        graph: TCMKnowledgeGraph,
        embedding_model: BaseEmbeddingModel,
        llm: BaseLLM,
        text_corpus: Dict[str, str]
    ):
        self.graph = graph
        self.embedding_model = embedding_model
        self.llm = llm
        self.text_corpus = text_corpus
        self._initialize_cache()

    def analyze_herb_combination(
        self,
        herb_ids: List[str],
        condition: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze interactions between multiple herbs."""
        # Get graph-based interactions
        graph_interactions = self._get_herb_interactions(herb_ids)
        
        # Get relevant texts about these herbs and condition
        context = self._get_herb_context(herb_ids, condition)
        
        # Generate analysis prompt
        prompt = self._create_herb_analysis_prompt(
            herbs=[self.graph.get_node(h_id) for h_id in herb_ids],
            interactions=graph_interactions,
            context=context,
            condition=condition
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={"analysis_type": "herb_combination"},
            system_prompt=self._get_system_prompt(AnalysisType.HERB)
        )
        
        return AnalysisResult(
            primary_entities=[self.graph.get_node(h_id) for h_id in herb_ids],
            interactions=self._parse_herb_interactions(analysis["response"]),
            contraindications=self._extract_contraindications(analysis["response"]),
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_confidence(graph_interactions, analysis),
            sources=self._gather_sources(herb_ids, analysis),
            metadata={"condition": condition} if condition else {}
        )

    def analyze_point_combination(
        self,
        point_ids: List[str],
        condition: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze acupuncture point combinations."""
        # Get channel relationships
        channel_info = self._get_channel_relationships(point_ids)
        
        # Get classical combinations
        classical_combos = self._get_classical_combinations(point_ids)
        
        # Get relevant texts
        context = self._get_point_context(point_ids, condition)
        
        # Generate analysis prompt
        prompt = self._create_point_analysis_prompt(
            points=[self.graph.get_node(p_id) for p_id in point_ids],
            channel_info=channel_info,
            classical_combos=classical_combos,
            context=context,
            condition=condition
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={"analysis_type": "point_combination"},
            system_prompt=self._get_system_prompt(AnalysisType.POINT)
        )
        
        return AnalysisResult(
            primary_entities=[self.graph.get_node(p_id) for p_id in point_ids],
            interactions=self._parse_point_interactions(analysis["response"]),
            contraindications=self._extract_contraindications(analysis["response"]),
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_confidence(classical_combos, analysis),
            sources=self._gather_sources(point_ids, analysis),
            metadata={"channels": channel_info, "condition": condition}
        )

    def analyze_formula_modification(
        self,
        base_formula_id: str,
        modifications: List[Dict[str, str]],
        condition: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze formula modifications."""
        base_formula = self.graph.get_node(base_formula_id)
        
        # Get original formula composition
        composition = self._get_formula_composition(base_formula_id)
        
        # Analyze modifications
        mod_analysis = self._analyze_modifications(
            composition,
            modifications,
            condition
        )
        
        # Get relevant texts
        context = self._get_formula_context(base_formula_id, modifications)
        
        # Generate analysis prompt
        prompt = self._create_formula_analysis_prompt(
            base_formula=base_formula,
            composition=composition,
            modifications=modifications,
            mod_analysis=mod_analysis,
            context=context,
            condition=condition
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={"analysis_type": "formula_modification"},
            system_prompt=self._get_system_prompt(AnalysisType.FORMULA)
        )
        
        return AnalysisResult(
            primary_entities=[base_formula],
            interactions=self._parse_formula_interactions(analysis["response"]),
            contraindications=self._extract_contraindications(analysis["response"]),
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_confidence(mod_analysis, analysis),
            sources=self._gather_sources([base_formula_id], analysis),
            metadata={
                "modifications": modifications,
                "condition": condition
            }
        )

    def _get_herb_interactions(self, herb_ids: List[str]) -> List[Dict]:
        """Get herb interactions from knowledge graph."""
        interactions = []
        
        # Check direct relationships
        for i, herb1_id in enumerate(herb_ids):
            for herb2_id in herb_ids[i+1:]:
                rels = self.graph.get_relationships(herb1_id, herb2_id)
                for rel in rels:
                    if rel.type in [RelationType.SUPPORTS, RelationType.CONTRAINDICATES]:
                        interactions.append({
                            "herbs": (herb1_id, herb2_id),
                            "type": rel.type,
                            "attributes": rel.attributes,
                            "confidence": rel.confidence
                        })
        
        # Check indirect relationships through mechanisms
        shared_mechanisms = self._find_shared_mechanisms(herb_ids)
        for mechanism in shared_mechanisms:
            interactions.append({
                "type": "shared_mechanism",
                "mechanism": mechanism,
                "herbs": herb_ids,
                "confidence": 0.7  # Lower confidence for indirect relationships
            })
        
        return interactions

    def _get_channel_relationships(self, point_ids: List[str]) -> Dict:
        """Analyze relationships between points' channels."""
        channels = {}
        relationships = []
        
        # Get channel for each point
        for point_id in point_ids:
            for node, rel in self.graph.get_neighbors(
                point_id,
                relationship_type=RelationType.LOCATED_ON
            ):
                if node.type == NodeType.CHANNEL:
                    channels[point_id] = node
        
        # Analyze channel relationships
        for i, (point1_id, channel1) in enumerate(channels.items()):
            for point2_id, channel2 in list(channels.items())[i+1:]:
                rel = {
                    "points": (point1_id, point2_id),
                    "channels": (channel1.id, channel2.id),
                    "type": self._determine_channel_relationship(channel1, channel2)
                }
                relationships.append(rel)
        
        return {
            "point_channels": channels,
            "relationships": relationships
        }

    def _get_classical_combinations(
        self,
        point_ids: List[str]
    ) -> List[Dict]:
        """Find classical point combinations from knowledge graph."""
        combinations = []
        
        # Look for established combinations in graph
        for point_id in point_ids:
            for node, rel in self.graph.get_neighbors(point_id):
                if node.type == NodeType.POINT and node.id in point_ids:
                    if "classical_combination" in rel.attributes:
                        combinations.append({
                            "points": (point_id, node.id),
                            "name": rel.attributes["classical_combination"],
                            "source": rel.sources[0] if rel.sources else None
                        })
        
        return combinations

    def _get_formula_composition(
        self,
        formula_id: str
    ) -> List[Dict]:
        """Get detailed composition of a formula."""
        composition = []
        
        # Get herb components
        for node, rel in self.graph.get_neighbors(
            formula_id,
            relationship_type=RelationType.CONTAINS
        ):
            if node.type == NodeType.HERB:
                composition.append({
                    "herb": node,
                    "attributes": rel.attributes,  # Should contain dosage, role, etc.
                    "confidence": rel.confidence
                })
        
        return composition

    def _analyze_modifications(
        self,
        composition: List[Dict],
        modifications: List[Dict[str, str]],
        condition: Optional[str]
    ) -> Dict:
        """Analyze impact of formula modifications."""
        analysis = {
            "removed_herbs": [],
            "added_herbs": [],
            "modified_herbs": [],
            "impact": {}
        }
        
        # Process each modification
        for mod in modifications:
            if mod["type"] == "add":
                herb_node = self.graph.get_node(mod["herb_id"])
                analysis["added_herbs"].append({
                    "herb": herb_node,
                    "reason": mod.get("reason"),
                    "impact": self._analyze_herb_impact(herb_node, composition)
                })
            elif mod["type"] == "remove":
                herb_node = self.graph.get_node(mod["herb_id"])
                analysis["removed_herbs"].append({
                    "herb": herb_node,
                    "reason": mod.get("reason"),
                    "impact": self._analyze_removal_impact(herb_node, composition)
                })
            elif mod["type"] == "modify":
                herb_node = self.graph.get_node(mod["herb_id"])
                analysis["modified_herbs"].append({
                    "herb": herb_node,
                    "change": mod.get("change"),
                    "reason": mod.get("reason"),
                    "impact": self._analyze_modification_impact(
                        herb_node,
                        mod["change"],
                        composition
                    )
                })
        
        # Analyze overall impact if condition is provided
        if condition:
            analysis["impact"]["condition"] = self._analyze_condition_impact(
                analysis,
                condition
            )
        
        return analysis

    def _create_herb_analysis_prompt(
        self,
        herbs: List[Node],
        interactions: List[Dict],
        context: str,
        condition: Optional[str]
    ) -> str:
        """Create prompt for herb combination analysis."""
        prompt_parts = [
            "Analyze the following herb combination:",
            "\nHerbs:",
            *[f"- {herb.name}" for herb in herbs],
            "\nKnown Interactions:",
            *[self._format_interaction(i) for i in interactions],
            "\nContext Information:",
            context
        ]
        
        if condition:
            prompt_parts.extend([
                "\nTarget Condition:",
                condition,
                "\nAnalyze effectiveness for this condition."
            ])
        
        prompt_parts.extend([
            "\nPlease provide:",
            "1. Analysis of interactions and synergies",
            "2. Potential contraindications",
            "3. Mechanism of action",
            "4. Safety considerations",
            "5. Key therapeutic principles"
        ])
        
        return "\n".join(prompt_parts)

    def _create_point_analysis_prompt(
        self,
        points: List[Node],
        channel_info: Dict,
        classical_combos: List[Dict],
        context: str,
        condition: Optional[str]
    ) -> str:
        """Create prompt for point combination analysis."""
        prompt_parts = [
            "Analyze the following acupuncture point combination:",
            "\nPoints:",
            *[f"- {point.name}" for point in points],
            "\nChannel Relationships:",
            *[self._format_channel_relationship(r) 
              for r in channel_info["relationships"]],
            "\nClassical Combinations:",
            *[self._format_classical_combo(c) for c in classical_combos],
            "\nContext Information:",
            context
        ]
        
        if condition:
            prompt_parts.extend([
                "\nTarget Condition:",
                condition,
                "\nAnalyze effectiveness for this condition."
            ])
        
        prompt_parts.extend([
            "\nPlease provide:",
            "1. Analysis of point synergies",
            "2. Channel theory analysis",
            "3. Classical principles applied",
            "4. Safety considerations",
            "5. Expected therapeutic effects"
        ])
        
        return "\n".join(prompt_parts)

    def _get_system_prompt(self, analysis_type: AnalysisType) -> str:
        """Get appropriate system prompt for analysis type."""
        base_prompt = (
            "You are a Traditional Chinese Medicine expert with deep knowledge of "
            "classical texts and modern research. Provide detailed analysis with "
            "clear reasoning and source attribution."
        )
        
        if analysis_type == AnalysisType.HERB:
            return base_prompt + """
Focus on:
- Herb properties and interactions
- Safety and contraindications
- Mechanism of action
- Evidence from classical texts and research"""
        
        elif analysis_type == AnalysisType.POINT:
            return base_prompt + """
Focus on:
- Channel theory and relationships
- Classical combinations
- Needle technique considerations
- Traditional and modern applications"""
        
        elif analysis_type == AnalysisType.FORMULA:
            return base_prompt + """
Focus on:
- Formula composition principles
- Modification rationale
- Impact on formula dynamics
- Clinical applications"""
        
        return base_prompt

    # Helper methods

    def _initialize_cache(self) -> None:
        """Initialize embedding cache for text corpus."""
        self.embedding_cache = {}
        for doc_id, text in self.text_corpus.items():
            self.embedding_cache[doc_id] = self.embedding_model.encode(text)

    def _find_shared_mechanisms(self, entity_ids: List[str]) -> List[Dict]:
        """Find shared mechanisms between entities."""
        mechanism_counts = {}
        
        for entity_id in entity_ids:
            # Get all mechanisms related to this entity
            for node, rel in self.graph.get_neighbors(
                entity_id,
                relationship_type=RelationType.INFLUENCES
            ):
                if node.type == NodeType.MECHANISM:
                    if node.id not in mechanism_counts:
                        mechanism_counts[node.id] = {
                            "mechanism": node,
                            "entities": set(),
                            "relationships": []
                        }
                    mechanism_counts[node.id]["entities"].add(entity_id)
                    mechanism_counts[node.id]["relationships"].append(rel)
        
        # Return mechanisms shared by multiple entities
        return [
            {
                "mechanism": data["mechanism"],
                "entities": list(data["entities"]),
                "count": len(data["entities"]),
                "confidence": min(rel.confidence for rel in data["relationships"])
            }
            for data in mechanism_counts.values()
            if len(data["entities"]) > 1
        ]

    def _get_relevant_text(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant text passages using embedding similarity."""
        query_embedding = self.embedding_model.encode(query)
        
        similarities = []
        for doc_id, doc_embedding in self.embedding_cache.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc_id, similarity, self.text_corpus[doc_id]))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(doc_id, text) for doc_id, _, text in similarities[:top_k]]

    def _parse_herb_interactions(self, llm_response: str) -> List[Dict]:
        """Parse herb interactions from LLM response."""
        interactions = []
        
        # Look for interaction patterns in the text
        interaction_patterns = [
            (r"synergistic interaction between ([^.]+) and ([^.]+)", "synergistic"),
            (r"antagonistic effect between ([^.]+) and ([^.]+)", "antagonistic"),
            (r"(\w+) enhances the effect of (\w+)", "enhancing"),
            (r"(\w+) reduces the effect of (\w+)", "reducing")
        ]
        
        for pattern, interaction_type in interaction_patterns:
            matches = re.finditer(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                interactions.append({
                    "herbs": (match.group(1).strip(), match.group(2).strip()),
                    "type": interaction_type,
                    "text": match.group(0)
                })
        
        return interactions

    def _parse_point_interactions(self, llm_response: str) -> List[Dict]:
        """Parse acupuncture point interactions from LLM response."""
        interactions = []
        
        # Parse different types of point relationships
        relationship_patterns = [
            (r"([A-Z]+\d+)[^\w]+([A-Z]+\d+)[^\w]+(reinforces|strengthens)", "reinforcing"),
            (r"([A-Z]+\d+)[^\w]+([A-Z]+\d+)[^\w]+(reduces|weakens)", "reducing"),
            (r"([A-Z]+\d+)[^\w]+([A-Z]+\d+)[^\w]+(balances)", "balancing")
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                interactions.append({
                    "points": (match.group(1), match.group(2)),
                    "type": rel_type,
                    "text": match.group(0)
                })
        
        return interactions

    def _parse_formula_interactions(self, llm_response: str) -> List[Dict]:
        """Parse formula modification interactions from LLM response."""
        interactions = []
        
        # Parse different types of modifications and their effects
        modification_patterns = [
            (r"Adding (\w+) ([^.]+)", "addition"),
            (r"Removing (\w+) ([^.]+)", "removal"),
            (r"Increasing (\w+) ([^.]+)", "increase"),
            (r"Decreasing (\w+) ([^.]+)", "decrease")
        ]
        
        for pattern, mod_type in modification_patterns:
            matches = re.finditer(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                interactions.append({
                    "herb": match.group(1),
                    "type": mod_type,
                    "effect": match.group(2),
                    "text": match.group(0)
                })
        
        return interactions

    def _extract_contraindications(self, llm_response: str) -> List[Dict]:
        """Extract contraindications from LLM response."""
        contraindications = []
        
        # Look for contraindication patterns
        patterns = [
            r"(?:contraindicated|avoid|caution)\s+(?:in|with|when)\s+([^.]+)",
            r"(?:not recommended|unsafe)\s+(?:in|with|when)\s+([^.]+)",
            r"(?:risk|danger|hazard)\s+(?:of|in|with)\s+([^.]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                contraindications.append({
                    "condition": match.group(1).strip(),
                    "text": match.group(0),
                    "severity": self._determine_severity(match.group(0))
                })
        
        return contraindications

    def _extract_mechanism(self, llm_response: str) -> str:
        """Extract mechanism of action description."""
        # Look for mechanism section
        mechanism_patterns = [
            r"Mechanism of action:([^.]+(?:\.[^.]+){0,2})",
            r"Mechanism:([^.]+(?:\.[^.]+){0,2})",
            r"Works by:([^.]+(?:\.[^.]+){0,2})"
        ]
        
        for pattern in mechanism_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""

    def _extract_explanation(self, llm_response: str) -> str:
        """Extract main explanation from LLM response."""
        # Look for main analysis section
        analysis_patterns = [
            r"Analysis:([^.]+(?:\.[^.]+){2,4})",
            r"Key points:([^.]+(?:\.[^.]+){2,4})",
            r"Summary:([^.]+(?:\.[^.]+){2,4})"
        ]
        
        for pattern in analysis_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no clear section found, return first few sentences
        sentences = llm_response.split('.')[:3]
        return '. '.join(sentences).strip() + '.'

    def _determine_severity(self, text: str) -> str:
        """Determine severity level from contraindication text."""
        if any(word in text.lower() for word in ["severe", "dangerous", "critical", "absolute"]):
            return "high"
        elif any(word in text.lower() for word in ["moderate", "significant", "important"]):
            return "moderate"
        return "low"

    def _calculate_confidence(
        self,
        graph_data: Any,
        llm_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on graph data presence
        if isinstance(graph_data, list) and graph_data:
            confidence += 0.2
        elif isinstance(graph_data, dict) and graph_data:
            confidence += 0.2
            
        # Adjust based on analysis completeness
        if len(llm_analysis["response"]) > 200:  # Substantial analysis
            confidence += 0.1
        if "mechanism" in llm_analysis["response"].lower():
            confidence += 0.1
        if "evidence" in llm_analysis["response"].lower():
            confidence += 0.1
            
        return min(1.0, confidence)

    def _gather_sources(
        self,
        entity_ids: List[str],
        llm_analysis: Dict[str, Any]
    ) -> Set[Source]:
        """Gather all relevant sources."""
        sources = set()
        
        # Add sources from graph entities
        for entity_id in entity_ids:
            node = self.graph.get_node(entity_id)
            if node:
                sources.update(node.sources)
        
        # Extract sources mentioned in LLM analysis
        source_patterns = [
            r"according to ([^,.]+)",
            r"cited in ([^,.]+)",
            r"referenced in ([^,.]+)"
        ]
        
        for pattern in source_patterns:
            matches = re.finditer(pattern, llm_analysis["response"], re.IGNORECASE)
            for match in matches:
                source_text = match.group(1).strip()
                # Would need logic to map text to Source objects
                pass
        
        return sources

    def _format_interaction(self, interaction: Dict) -> str:
        """Format interaction data for prompt."""
        if interaction["type"] == "shared_mechanism":
            return f"Shared mechanism: {interaction['mechanism']} among {', '.join(interaction['herbs'])}"
        return f"{interaction['type'].title()} between {interaction['herbs'][0]} and {interaction['herbs'][1]}"

    def _format_channel_relationship(self, relationship: Dict) -> str:
        """Format channel relationship for prompt."""
        return (f"Points {relationship['points'][0]} and {relationship['points'][1]} "
                f"have a {relationship['type']} relationship through channels "
                f"{relationship['channels'][0]} and {relationship['channels'][1]}")

    def _format_classical_combo(self, combo: Dict) -> str:
        """Format classical combination for prompt."""
        source_info = f" (from {combo['source'].title})" if combo['source'] else ""
        return f"Classical combination: {combo['name']} - {', '.join(combo['points'])}{source_info}"

    def _determine_channel_relationship(
        self,
        channel1: Node,
        channel2: Node
    ) -> str:
        """Determine the relationship between two channels."""
        # Implementation would include:
        # - Six channel theory relationships
        # - Interior/Exterior relationships
        # - Five element relationships
        pass

    # Additional TCMAnalysisEngine methods

    def analyze_treatment_strategy(
        self,
        pattern_id: str,
        include_herbs: bool = True,
        include_points: bool = True,
        include_formulas: bool = True
    ) -> AnalysisResult:
        """Analyze comprehensive treatment strategy for a pattern."""
        pattern = self.graph.get_node(pattern_id)
        
        # Get all treatment options
        treatments = self._get_treatment_options(
            pattern_id,
            include_herbs,
            include_points,
            include_formulas
        )
        
        # Get relevant context
        context = self._get_pattern_context(pattern_id, treatments)
        
        # Generate analysis prompt
        prompt = self._create_treatment_analysis_prompt(
            pattern=pattern,
            treatments=treatments,
            context=context
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={"analysis_type": "treatment_strategy"},
            system_prompt=self._get_system_prompt(AnalysisType.PATTERN)
        )
        
        return AnalysisResult(
            primary_entities=[pattern],
            interactions=self._parse_treatment_interactions(analysis["response"]),
            contraindications=self._extract_contraindications(analysis["response"]),
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_confidence(treatments, analysis),
            sources=self._gather_sources([pattern_id], analysis),
            metadata={
                "treatment_options": treatments,
                "pattern": pattern.name
            }
        )

    # Continuing TCMAnalysisEngine methods

    def analyze_formula_pattern_matching(
        self,
        formula_id: str,
        pattern_ids: List[str]
    ) -> AnalysisResult:
        """Analyze how well a formula matches given patterns."""
        formula = self.graph.get_node(formula_id)
        patterns = [self.graph.get_node(p_id) for p_id in pattern_ids]
        
        # Get formula composition
        composition = self._get_formula_composition(formula_id)
        
        # Analyze pattern-formula relationships
        matches = self._analyze_pattern_formula_matches(formula, patterns)
        
        # Get relevant context
        context = self._get_formula_pattern_context(formula_id, pattern_ids)
        
        # Generate analysis prompt
        prompt = self._create_pattern_matching_prompt(
            formula=formula,
            patterns=patterns,
            composition=composition,
            matches=matches,
            context=context
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={"analysis_type": "formula_pattern_matching"},
            system_prompt=self._get_system_prompt(AnalysisType.FORMULA)
        )
        
        return AnalysisResult(
            primary_entities=[formula] + patterns,
            interactions=self._parse_formula_pattern_interactions(analysis["response"]),
            contraindications=self._extract_contraindications(analysis["response"]),
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_confidence(matches, analysis),
            sources=self._gather_sources([formula_id] + pattern_ids, analysis),
            metadata={
                "pattern_matches": matches,
                "formula_composition": composition
            }
        )

    def analyze_safety_profile(
        self,
        entity_ids: List[str],
        patient_data: Optional[Dict] = None
    ) -> AnalysisResult:
        """Analyze safety considerations for herbs, points, or formulas."""
        entities = [self.graph.get_node(e_id) for e_id in entity_ids]
        
        # Get safety-related relationships
        safety_data = self._get_safety_data(entities)
        
        # Get patient-specific considerations if provided
        patient_considerations = None
        if patient_data:
            patient_considerations = self._analyze_patient_safety(
                entities,
                patient_data
            )
        
        # Get relevant context
        context = self._get_safety_context(entities)
        
        # Generate analysis prompt
        prompt = self._create_safety_analysis_prompt(
            entities=entities,
            safety_data=safety_data,
            patient_considerations=patient_considerations,
            context=context
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={
                "analysis_type": "safety_profile",
                "has_patient_data": bool(patient_data)
            },
            system_prompt=self._get_safety_system_prompt()
        )
        
        return AnalysisResult(
            primary_entities=entities,
            interactions=[],  # Safety analysis focuses on contraindications
            contraindications=self._extract_contraindications(analysis["response"]),
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_safety_confidence(safety_data, analysis),
            sources=self._gather_sources(entity_ids, analysis),
            metadata={
                "safety_data": safety_data,
                "patient_considerations": patient_considerations
            }
        )

    def analyze_pattern_progression(
        self,
        pattern_id: str,
        time_course: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze potential progression paths of a pattern."""
        pattern = self.graph.get_node(pattern_id)
        
        # Get progression relationships
        progressions = self._get_pattern_progressions(pattern_id)
        
        # Analyze progression likelihood
        progression_analysis = self._analyze_progression_likelihood(
            pattern_id,
            progressions,
            time_course
        )
        
        # Get relevant context
        context = self._get_progression_context(pattern_id, progressions)
        
        # Generate analysis prompt
        prompt = self._create_progression_analysis_prompt(
            pattern=pattern,
            progressions=progressions,
            progression_analysis=progression_analysis,
            time_course=time_course,
            context=context
        )
        
        # Get LLM analysis
        analysis = self.llm.generate_with_metadata(
            prompt=prompt,
            metadata={"analysis_type": "pattern_progression"},
            system_prompt=self._get_system_prompt(AnalysisType.PATTERN)
        )
        
        return AnalysisResult(
            primary_entities=[pattern],
            interactions=self._parse_progression_interactions(analysis["response"]),
            contraindications=[],  # Not relevant for progression analysis
            explanation=self._extract_explanation(analysis["response"]),
            mechanism=self._extract_mechanism(analysis["response"]),
            confidence=self._calculate_confidence(progressions, analysis),
            sources=self._gather_sources([pattern_id], analysis),
            metadata={
                "progressions": progressions,
                "time_course": time_course,
                "likelihood_analysis": progression_analysis
            }
        )

    def _analyze_pattern_formula_matches(
        self,
        formula: Node,
        patterns: List[Node]
    ) -> List[Dict]:
        """Analyze how formula components match pattern characteristics."""
        matches = []
        
        # Get formula's therapeutic principles
        formula_principles = self._get_therapeutic_principles(formula)
        
        for pattern in patterns:
            # Get pattern characteristics
            characteristics = self._get_pattern_characteristics(pattern)
            
            # Compare principles and characteristics
            match_analysis = {
                "pattern": pattern,
                "matching_principles": [],
                "partial_matches": [],
                "mismatches": [],
                "confidence": 0.0
            }
            
            for principle in formula_principles:
                if principle in characteristics["treatment_principles"]:
                    match_analysis["matching_principles"].append(principle)
                elif self._is_partial_match(principle, characteristics):
                    match_analysis["partial_matches"].append(principle)
                else:
                    match_analysis["mismatches"].append(principle)
            
            # Calculate confidence based on matches
            match_analysis["confidence"] = (
                len(match_analysis["matching_principles"]) * 1.0 +
                len(match_analysis["partial_matches"]) * 0.5
            ) / len(formula_principles)
            
            matches.append(match_analysis)
        
        return matches

    def _get_safety_data(self, entities: List[Node]) -> Dict:
        """Get safety-related data for entities."""
        safety_data = {
            "contraindications": [],
            "cautions": [],
            "interactions": [],
            "toxicity_data": []
        }
        
        for entity in entities:
            # Get contraindication relationships
            for node, rel in self.graph.get_neighbors(
                entity.id,
                relationship_type=RelationType.CONTRAINDICATES
            ):
                safety_data["contraindications"].append({
                    "entity": entity,
                    "contraindication": node,
                    "details": rel.attributes,
                    "confidence": rel.confidence
                })
            
            # Get cautions from attributes
            if "cautions" in entity.attributes:
                safety_data["cautions"].extend([
                    {"entity": entity, "caution": c}
                    for c in entity.attributes["cautions"]
                ])
            
            # Get toxicity data if available
            if "toxicity" in entity.attributes:
                safety_data["toxicity_data"].append({
                    "entity": entity,
                    "data": entity.attributes["toxicity"]
                })
        
        return safety_data

    def _analyze_patient_safety(
        self,
        entities: List[Node],
        patient_data: Dict
    ) -> Dict:
        """Analyze safety considerations specific to patient."""
        considerations = {
            "high_risk_factors": [],
            "contraindications": [],
            "monitoring_needed": [],
            "dosage_adjustments": []
        }
        
        # Check each entity against patient factors
        for entity in entities:
            entity_risks = self._check_patient_entity_safety(
                entity,
                patient_data
            )
            
            for category in considerations:
                considerations[category].extend(entity_risks.get(category, []))
        
        return considerations

    def _get_pattern_progressions(self, pattern_id: str) -> List[Dict]:
        """Get pattern progression relationships."""
        progressions = []
        
        # Get transformation relationships
        for node, rel in self.graph.get_neighbors(
            pattern_id,
            relationship_type=RelationType.TRANSFORMS
        ):
            if node.type == NodeType.PATTERN:
                progressions.append({
                    "target_pattern": node,
                    "relationship": rel,
                    "mechanism": rel.attributes.get("mechanism", ""),
                    "conditions": rel.attributes.get("conditions", []),
                    "time_course": rel.attributes.get("time_course", ""),
                    "confidence": rel.confidence
                })
        
        return progressions

    def _analyze_progression_likelihood(
        self,
        pattern_id: str,
        progressions: List[Dict],
        time_course: Optional[str]
    ) -> Dict:
        """Analyze likelihood of different progression paths."""
        likelihood_analysis = {
            "high_likelihood": [],
            "moderate_likelihood": [],
            "low_likelihood": [],
            "time_factors": {}
        }
        
        for prog in progressions:
            # Calculate base likelihood
            likelihood = prog["confidence"]
            
            # Adjust based on conditions
            if prog["conditions"]:
                likelihood *= self._calculate_condition_match(
                    prog["conditions"],
                    time_course
                )
            
            # Categorize based on likelihood
            if likelihood > 0.7:
                likelihood_analysis["high_likelihood"].append(prog)
            elif likelihood > 0.4:
                likelihood_analysis["moderate_likelihood"].append(prog)
            else:
                likelihood_analysis["low_likelihood"].append(prog)
            
            # Add time course analysis if available
            if prog["time_course"]:
                likelihood_analysis["time_factors"][prog["target_pattern"].id] = {
                    "course": prog["time_course"],
                    "likelihood": likelihood
                }
        
        return likelihood_analysis

    # Utility methods for pattern matching
    def _is_partial_match(
        self,
        principle: str,
        characteristics: Dict
    ) -> bool:
        """Determine if principle partially matches pattern characteristics."""
        # Implementation would include:
        # - Semantic similarity checking
        # - Related principle matching
        # - Hierarchical relationship checking
        pass

    def _get_therapeutic_principles(self, formula: Node) -> List[str]:
        """Get therapeutic principles of a formula."""
        principles = []
        
        # Get directly stated principles
        if "therapeutic_principles" in formula.attributes:
            principles.extend(formula.attributes["therapeutic_principles"])
        
        # Derive principles from composition
        composition = self._get_formula_composition(formula.id)
        for comp in composition:
            if "function" in comp["attributes"]:
                principles.extend(comp["attributes"]["function"])
        
        return list(set(principles))  # Remove duplicates

    def _get_pattern_characteristics(self, pattern: Node) -> Dict:
        """Get comprehensive characteristics of a pattern."""
        return {
            "manifestations": pattern.attributes.get("manifestations", []),
            "treatment_principles": pattern.attributes.get("treatment_principles", []),
            "etiology": pattern.attributes.get("etiology", []),
            "mechanisms": pattern.attributes.get("mechanisms", [])
        }

    def _calculate_condition_match(
        self,
        conditions: List[str],
        time_course: Optional[str]
    ) -> float:
        """Calculate how well conditions match given time course."""
        if not time_course:
            return 0.5  # Default match value
            
        match_count = sum(
            1 for condition in conditions
            if condition.lower() in time_course.lower()
        )
        
        return match_count / len(conditions) if conditions else 0.5