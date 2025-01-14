from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from tcm.core.models import Node, Relationship, Source
from tcm.core.enums import NodeType, RelationType
from tcm.graph.knowledge_graph import TCMKnowledgeGraph
from tcm.graph.query_engine import TCMGraphQueryEngine, QueryResult

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    PATTERN_DIAGNOSIS = "pattern_diagnosis"
    HERB_PROPERTIES = "herb_properties"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    POINT_COMBINATION = "point_combination"
    MECHANISM_EXPLANATION = "mechanism_explanation"
    GENERAL_INFORMATION = "general_information"

@dataclass
class ChatContext:
    """Maintains context for the chat session."""
    current_topic: Optional[str] = None
    referenced_entities: List[str] = None
    last_query_result: Optional[QueryResult] = None
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        self.referenced_entities = self.referenced_entities or []
        self.conversation_history = self.conversation_history or []

@dataclass
class RAGResponse:
    """Container for RAG-enhanced responses."""
    answer: str
    sources: List[Source]
    confidence: float
    context: List[str]
    nodes: List[Node]
    reasoning: List[str]

class TCMQueryOrchestrator:
    """Orchestrates RAG-enhanced query processing for TCM chatbot."""
    
    def __init__(
        self,
        knowledge_graph: TCMKnowledgeGraph,
        embedding_model: Any,
        llm_model: Any,
        config: Optional[Dict] = None
    ):
        self.graph = knowledge_graph
        self.query_engine = TCMGraphQueryEngine(knowledge_graph)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.config = config or {}
        
    def process_query(
        self,
        query: str,
        context: ChatContext
    ) -> RAGResponse:
        """Process a natural language query with RAG enhancement."""
        try:
            # Detect query intent
            intent = self._detect_intent(query, context)
            
            # Extract relevant entities
            entities = self._extract_entities(query)
            
            # Get graph-based knowledge
            graph_results = self._query_graph(intent, entities, context)
            
            # Retrieve relevant text passages
            text_results = self._retrieve_text_context(query, graph_results)
            
            # Generate enhanced response
            response = self._generate_response(
                query,
                intent,
                graph_results,
                text_results,
                context
            )
            
            # Update context
            self._update_context(context, intent, entities, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _detect_intent(self, query: str, context: ChatContext) -> QueryIntent:
        """Detect the intent of the query."""
        # Create prompt for intent classification
        prompt = self._create_intent_prompt(query, context)
        
        # Use LLM to classify intent
        response = self.llm_model.generate(prompt)
        
        # Map response to QueryIntent
        return self._map_to_intent(response)
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract TCM entities from the query."""
        entities = {
            "herbs": [],
            "patterns": [],
            "symptoms": [],
            "points": []
        }
        
        # Use embedding model for entity extraction
        # This could be enhanced with specialized NER models
        embeddings = self.embedding_model.encode(query)
        
        # Find closest matches in the knowledge graph
        # Implementation would depend on embedding model specifics
        
        return entities
    
    def _query_graph(
        self,
        intent: QueryIntent,
        entities: Dict[str, List[str]],
        context: ChatContext
    ) -> QueryResult:
        """Query the knowledge graph based on intent and entities."""
        if intent == QueryIntent.PATTERN_DIAGNOSIS:
            return self.query_engine.find_patterns_from_symptoms(
                entities.get("symptoms", [])
            )
            
        elif intent == QueryIntent.TREATMENT_RECOMMENDATION:
            if pattern_id := self._get_pattern_id(entities, context):
                return self.query_engine.find_treatments(pattern_id)
                
        elif intent == QueryIntent.HERB_PROPERTIES:
            # Implementation for herb property queries
            pass
            
        elif intent == QueryIntent.POINT_COMBINATION:
            # Implementation for point combination analysis
            pass
            
        # Add more intent handlers as needed
        
        return QueryResult([], [], set(), 0.0, {})
    
    def _retrieve_text_context(
        self,
        query: str,
        graph_results: QueryResult
    ) -> List[str]:
        """Retrieve relevant text passages using dense retrieval."""
        relevant_passages = []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Get text associated with graph results
        for node in graph_results.nodes:
            for source in node.sources:
                # Implementation would retrieve and rank relevant passages
                # from the source documents
                pass
        
        return relevant_passages
    
    def _generate_response(
        self,
        query: str,
        intent: QueryIntent,
        graph_results: QueryResult,
        text_results: List[str],
        context: ChatContext
    ) -> RAGResponse:
        """Generate enhanced response using LLM."""
        # Create prompt template based on intent
        prompt_template = self._get_prompt_template(intent)
        
        # Fill template with results
        prompt = prompt_template.format(
            query=query,
            graph_results=self._format_graph_results(graph_results),
            text_results=self._format_text_results(text_results),
            context=self._format_context(context)
        )
        
        # Generate response
        response = self.llm_model.generate(prompt)
        
        # Extract reasoning steps
        reasoning = self._extract_reasoning(response)
        
        return RAGResponse(
            answer=response,
            sources=list(graph_results.sources),
            confidence=graph_results.confidence,
            context=text_results,
            nodes=graph_results.nodes,
            reasoning=reasoning
        )
    
    def _update_context(
        self,
        context: ChatContext,
        intent: QueryIntent,
        entities: Dict[str, List[str]],
        response: RAGResponse
    ) -> None:
        """Update conversation context."""
        # Update current topic
        if intent in [QueryIntent.PATTERN_DIAGNOSIS, QueryIntent.TREATMENT_RECOMMENDATION]:
            context.current_topic = "diagnosis_treatment"
        
        # Update referenced entities
        for entity_type, entity_list in entities.items():
            context.referenced_entities.extend(entity_list)
        
        # Update conversation history
        context.conversation_history.append({
            "intent": intent,
            "entities": entities,
            "response": response
        })
    
    def _create_intent_prompt(self, query: str, context: ChatContext) -> str:
        """Create prompt for intent classification."""
        return f"""Classify the TCM query intent:
Query: {query}
Previous topic: {context.current_topic}
Referenced entities: {', '.join(context.referenced_entities)}

Available intents:
- Pattern diagnosis
- Treatment recommendation
- Herb properties
- Point combination
- Mechanism explanation
- General information

Intent:"""
    
    def _map_to_intent(self, response: str) -> QueryIntent:
        """Map LLM response to QueryIntent enum."""
        intent_map = {
            "pattern diagnosis": QueryIntent.PATTERN_DIAGNOSIS,
            "treatment recommendation": QueryIntent.TREATMENT_RECOMMENDATION,
            "herb properties": QueryIntent.HERB_PROPERTIES,
            "point combination": QueryIntent.POINT_COMBINATION,
            "mechanism explanation": QueryIntent.MECHANISM_EXPLANATION,
            "general information": QueryIntent.GENERAL_INFORMATION
        }
        
        response_lower = response.lower().strip()
        return intent_map.get(response_lower, QueryIntent.GENERAL_INFORMATION)
    
    def _get_pattern_id(
        self,
        entities: Dict[str, List[str]],
        context: ChatContext
    ) -> Optional[str]:
        """Get pattern ID from entities or context."""
        # Check entities first
        if patterns := entities.get("patterns"):
            return patterns[0]
            
        # Check context
        if context.last_query_result and context.last_query_result.nodes:
            for node in context.last_query_result.nodes:
                if node.type == NodeType.PATTERN:
                    return node.id
                    
        return None
    
    def _get_prompt_template(self, intent: QueryIntent) -> str:
        """Get appropriate prompt template for the intent."""
        templates = {
            QueryIntent.PATTERN_DIAGNOSIS: """
Given the following information about TCM patterns:
{graph_results}

Additional context:
{text_results}

Previous conversation:
{context}

Please diagnose the pattern(s) based on the symptoms described in the query:
{query}

Provide your reasoning and cite relevant sources.
""",
            # Add more templates for other intents
        }
        
        return templates.get(intent, "Please answer the following TCM query: {query}")
    
    def _format_graph_results(self, results: QueryResult) -> str:
        """Format graph results for prompt."""
        formatted = []
        
        for node in results.nodes:
            formatted.append(f"- {node.name}")
            for key, value in node.attributes.items():
                formatted.append(f"  {key}: {value}")
                
        return "\n".join(formatted)
    
    def _format_text_results(self, results: List[str]) -> str:
        """Format text results for prompt."""
        return "\n".join(f"- {text}" for text in results)
    
    def _format_context(self, context: ChatContext) -> str:
        """Format conversation context for prompt."""
        if not context.conversation_history:
            return "No previous context"
            
        formatted = []
        for entry in context.conversation_history[-3:]:  # Last 3 interactions
            formatted.append(f"Intent: {entry['intent']}")
            formatted.append(f"Entities: {entry['entities']}")
            formatted.append(f"Response: {entry['response'].answer}\n")
            
        return "\n".join(formatted)
    
    def _extract_reasoning(self, response: str) -> List[str]:
        """Extract reasoning steps from response."""
        # Implementation would parse reasoning steps
        # Could use structured output from LLM
        return []