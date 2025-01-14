# tcm/extractors/herbs.py
import re
from typing import Dict, List, Optional, Set, Tuple
import hashlib

from tcm.core.exceptions import ProcessingError
from tcm.core.models import Node, Source
from tcm.core.enums import NodeType, PropertyType
from .base import BaseExtractor, ExtractionResult

class HerbExtractor(BaseExtractor):
    """Extracts herb entities and their properties from text."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.herb_patterns = self._compile_herb_patterns()
        self.property_patterns = self._compile_property_patterns()
    
    def extract(self, text: str, sources: List[Source]) -> ExtractionResult:
        """Extract herb information from text."""
        try:
            # Generate cache key
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            # Check cache
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            # Extract herb mentions
            herb_mentions = self._extract_herb_mentions(text)
            
            # Extract properties for each herb
            nodes = []
            for herb_name, positions in herb_mentions.items():
                # Get surrounding context for property extraction
                contexts = self._get_contexts(text, positions)
                
                # Extract properties from contexts
                properties = {}
                for context in contexts:
                    props = self._extract_properties(context)
                    properties.update(props)
                
                # Create herb node
                node = Node(
                    id=f"herb_{self._generate_id(herb_name)}",
                    type=NodeType.HERB,
                    name=herb_name,
                    attributes=properties,
                    sources=sources,
                    confidence=self._calculate_confidence(len(contexts), properties)
                )
                nodes.append(node)
            
            result = ExtractionResult(
                nodes=nodes,
                confidence=sum(n.confidence for n in nodes) / len(nodes) if nodes else 0,
                sources=set(sources),
                metadata={"mention_counts": herb_mentions}
            )
            
            # Cache result
            if self.validate_extraction(result):
                self.cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            raise ProcessingError(f"Herb extraction failed: {e}")
    
    def _compile_herb_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for herb identification."""
        return {
            'latin': re.compile(r'\b[A-Z][a-z]+ [a-z]+\b(?=\s*\([A-Za-z\s]+\))?'),
            'pinyin': re.compile(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b'),
            'chinese': re.compile(r'[\u4e00-\u9fff]{2,}')  # Basic Chinese character matching
        }
    
    def _compile_property_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for property extraction."""
        return {
            PropertyType.TEMPERATURE: re.compile(
                r'\b(cold|cool|neutral|warm|hot)\b',
                re.IGNORECASE
            ),
            PropertyType.TASTE: re.compile(
                r'\b(bitter|sweet|sour|pungent|salty|bland)\b',
                re.IGNORECASE
            ),
            PropertyType.DIRECTION: re.compile(
                r'\b(ascending|descending|floating|sinking)\b',
                re.IGNORECASE
            ),
            PropertyType.CHANNEL_TROPISM: re.compile(
                r'\b(liver|heart|spleen|lung|kidney|stomach|gallbladder'
                r'|small intestine|large intestine|bladder|san jiao|pericardium)\b',
                re.IGNORECASE
            )
        }
    
    def _extract_herb_mentions(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Extract herb names and their positions from text."""
        mentions = {}
        
        for pattern_type, pattern in self.herb_patterns.items():
            for match in pattern.finditer(text):
                herb_name = match.group()
                if herb_name not in mentions:
                    mentions[herb_name] = []
                mentions[herb_name].append(match.span())
        
        return mentions
    
    def _get_contexts(
        self,
        text: str,
        positions: List[Tuple[int, int]]
    ) -> List[str]:
        """Get context windows around herb mentions."""
        contexts = []
        window_size = self.config.get('context_window', 100)
        
        for start, end in positions:
            context_start = max(0, start - window_size)
            context_end = min(len(text), end + window_size)
            contexts.append(text[context_start:context_end])
        
        return contexts
    
    def _extract_properties(self, context: str) -> Dict[str, List[str]]:
        """Extract herb properties from context."""
        properties = {}
        
        for prop_type, pattern in self.property_patterns.items():
            matches = pattern.findall(context)
            if matches:
                properties[prop_type] = list(set(match.lower() for match in matches))
        
        return properties
    
    def _calculate_confidence(
        self,
        mention_count: int,
        properties: Dict[str, List[str]]
    ) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on number of mentions
        if mention_count > 1:
            confidence += 0.1
            
        # Adjust based on property completeness
        if len(properties) >= 3:  # Has most property types
            confidence += 0.2
        elif len(properties) >= 2:  # Has some properties
            confidence += 0.1
            
        return min(1.0, confidence)
    
    def _generate_id(self, name: str) -> str:
        """Generate a stable ID for an herb."""
        # Remove special characters and convert to lowercase
        clean_name = re.sub(r'[^\w\s]', '', name.lower())
        # Replace spaces with underscores
        return re.sub(r'\s+', '_', clean_name)