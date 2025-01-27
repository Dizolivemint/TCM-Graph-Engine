# tcm/extractors/patterns.py
import re
from typing import Any, Dict, List, Set, Tuple, Optional
import hashlib
from enum import Enum

from tcm.core.exceptions import ProcessingError
from tcm.core.models import Node, Source
from tcm.core.enums import NodeType
from tcm.processors.text_extractor import ExtractedEntity
from .base import BaseExtractor, ExtractionResult

class PatternCategory(Enum):
    """Categories of TCM patterns."""
    EIGHT_PRINCIPLES = "eight_principles"
    ZANG_FU = "zang_fu"
    QI_BLOOD = "qi_blood"
    FLUID = "fluid"
    SIX_STAGES = "six_stages"
    FOUR_LEVELS = "four_levels"
    SAN_JIAO = "san_jiao"

class PatternExtractor(BaseExtractor):
    """Extracts TCM patterns/syndromes from text."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.pattern_patterns = self._compile_pattern_patterns()
        self.category_patterns = self._compile_category_patterns()
        self.modifier_patterns = self._compile_modifier_patterns()
    
    def extract(self, text: str, sources: List[Source], context_entities: Optional[List[ExtractedEntity]] = None) -> ExtractionResult:
        """Extract pattern information from text."""
        try:
            # Generate cache key
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            # Check cache
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            # Extract pattern mentions
            pattern_mentions = self._extract_pattern_mentions(text)
            
            # Use context entities to enhance extraction if available
            if context_entities:
                for entity in context_entities:
                    if entity.text not in pattern_mentions:
                        pattern_mentions[entity.text] = []
                    position = entity.metadata.get('position', (0, len(entity.text)))
                    pattern_mentions[entity.text].append(position)
            
            # Process each pattern mention
            nodes = []
            for pattern_name, positions in pattern_mentions.items():
                # Get surrounding context
                contexts = self._get_contexts(text, positions)
                
                # Extract pattern details
                details = self._extract_pattern_details(pattern_name, contexts)
                
                # Create pattern node
                node = Node(
                    id=f"pattern_{self._generate_id(pattern_name)}",
                    type=NodeType.PATTERN,
                    name=pattern_name,
                    attributes={
                        "category": details["category"],
                        "modifiers": details["modifiers"],
                        "characteristics": details["characteristics"]
                    },
                    sources=sources,
                    confidence=self._calculate_confidence(details)
                )
                nodes.append(node)
            
            result = ExtractionResult(
                nodes=nodes,
                confidence=sum(n.confidence for n in nodes) / len(nodes) if nodes else 0,
                sources=set(sources),
                metadata={
                    "mention_counts": pattern_mentions,
                    "categories": {
                        node.id: node.attributes["category"]
                        for node in nodes
                    }
                }
            )
            
            # Cache result
            if self.validate_extraction(result):
                self.cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            raise ProcessingError(f"Pattern extraction failed: {e}")
    
    def _compile_pattern_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for pattern identification."""
        return {
            # Eight Principles patterns
            'eight_principles': re.compile(
                r'\b((?:interior|exterior|cold|heat|excess|deficiency|yin|yang)'
                r'\s+(?:pattern|syndrome))\b',
                re.IGNORECASE
            ),
            
            # Zang-Fu patterns
            'zang_fu': re.compile(
                r'\b((?:liver|heart|spleen|lung|kidney|gallbladder|stomach|'
                r'small intestine|large intestine|bladder|san jiao|pericardium)'
                r'\s+(?:\w+\s+)*(?:pattern|syndrome))\b',
                re.IGNORECASE
            ),
            
            # Qi-Blood patterns
            'qi_blood': re.compile(
                r'\b((?:qi|blood|essence)\s+(?:deficiency|stagnation|'
                r'rebellion|sinking|collapse)\s+(?:pattern|syndrome))\b',
                re.IGNORECASE
            ),
            
            # Six Stages patterns
            'six_stages': re.compile(
                r'\b((?:tai yang|yang ming|shao yang|tai yin|shao yin|jue yin)'
                r'\s+(?:pattern|syndrome))\b',
                re.IGNORECASE
            ),
            
            # Four Levels patterns
            'four_levels': re.compile(
                r'\b((?:wei|qi|ying|blood)\s+(?:level|stage)'
                r'\s+(?:pattern|syndrome))\b',
                re.IGNORECASE
            )
        }
    
    def _compile_category_patterns(self) -> Dict[PatternCategory, re.Pattern]:
        """Compile patterns for categorizing patterns."""
        return {
            PatternCategory.EIGHT_PRINCIPLES: re.compile(
                r'\b(interior|exterior|cold|heat|excess|deficiency|yin|yang)\b',
                re.IGNORECASE
            ),
            PatternCategory.ZANG_FU: re.compile(
                r'\b(liver|heart|spleen|lung|kidney|gallbladder|stomach|'
                r'small intestine|large intestine|bladder|san jiao|pericardium)\b',
                re.IGNORECASE
            ),
            PatternCategory.QI_BLOOD: re.compile(
                r'\b(qi|blood|essence)\b',
                re.IGNORECASE
            ),
            PatternCategory.FLUID: re.compile(
                r'\b(fluid|phlegm|dampness|water)\b',
                re.IGNORECASE
            ),
            PatternCategory.SIX_STAGES: re.compile(
                r'\b(tai yang|yang ming|shao yang|tai yin|shao yin|jue yin)\b',
                re.IGNORECASE
            ),
            PatternCategory.FOUR_LEVELS: re.compile(
                r'\b(wei|qi|ying|blood)\s+(level|stage)\b',
                re.IGNORECASE
            ),
            PatternCategory.SAN_JIAO: re.compile(
                r'\b(upper|middle|lower)\s+jiao\b',
                re.IGNORECASE
            )
        }
    
    def _compile_modifier_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for pattern modifiers."""
        return {
            'severity': re.compile(
                r'\b(mild|moderate|severe|extreme)\b',
                re.IGNORECASE
            ),
            'progression': re.compile(
                r'\b(acute|chronic|recurring)\b',
                re.IGNORECASE
            ),
            'transformation': re.compile(
                r'\b(transforming|developing|evolving)\s+(?:into|to|towards)\b',
                re.IGNORECASE
            ),
            'complexity': re.compile(
                r'\b(simple|complex|complicated|combined)\b',
                re.IGNORECASE
            )
        }
    
    def _extract_pattern_mentions(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Extract pattern mentions and their positions from text."""
        mentions = {}
        
        for pattern_type, pattern in self.pattern_patterns.items():
            for match in pattern.finditer(text):
                pattern_name = match.group()
                if pattern_name not in mentions:
                    mentions[pattern_name] = []
                mentions[pattern_name].append(match.span())
        
        return mentions
    
    def _get_contexts(
        self,
        text: str,
        positions: List[Tuple[int, int]]
    ) -> List[str]:
        """Get context windows around pattern mentions."""
        contexts = []
        window_size = self.config.get('context_window', 150)  # Larger window for patterns
        
        for start, end in positions:
            context_start = max(0, start - window_size)
            context_end = min(len(text), end + window_size)
            contexts.append(text[context_start:context_end])
        
        return contexts
    
    def _extract_pattern_details(
        self,
        pattern_name: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Extract detailed information about a pattern."""
        details = {
            "category": None,
            "modifiers": set(),
            "characteristics": {}
        }
        
        # Determine pattern category
        details["category"] = self._determine_category(pattern_name)
        
        # Extract modifiers from contexts
        for context in contexts:
            # Look for modifiers
            for modifier_type, pattern in self.modifier_patterns.items():
                matches = pattern.findall(context)
                if matches:
                    details["modifiers"].update(matches)
            
            # Extract characteristics based on category
            if details["category"]:
                characteristics = self._extract_characteristics(
                    context,
                    details["category"]
                )
                details["characteristics"].update(characteristics)
        
        return details
    
    def _determine_category(self, pattern_name: str) -> Optional[PatternCategory]:
        """Determine the category of a pattern."""
        pattern_lower = pattern_name.lower()
        
        for category, pattern in self.category_patterns.items():
            if pattern.search(pattern_lower):
                return category
        
        return None
    
    def _extract_characteristics(
        self,
        context: str,
        category: PatternCategory
    ) -> Dict[str, List[str]]:
        """Extract category-specific characteristics."""
        characteristics = {}
        
        if category == PatternCategory.EIGHT_PRINCIPLES:
            characteristics.update(self._extract_eight_principles(context))
        elif category == PatternCategory.ZANG_FU:
            characteristics.update(self._extract_zang_fu_characteristics(context))
        elif category == PatternCategory.QI_BLOOD:
            characteristics.update(self._extract_qi_blood_characteristics(context))
        # Add more category-specific extraction methods
        
        return characteristics
    
    def _extract_eight_principles(self, context: str) -> Dict[str, List[str]]:
        """Extract Eight Principles characteristics."""
        pairs = [
            ("interior_exterior", r'\b(interior|exterior)\b'),
            ("cold_heat", r'\b(cold|heat)\b'),
            ("deficiency_excess", r'\b(deficiency|excess)\b'),
            ("yin_yang", r'\b(yin|yang)\b')
        ]
        
        characteristics = {}
        for name, pattern in pairs:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                characteristics[name] = [m.lower() for m in matches]
        
        return characteristics
    
    def _extract_zang_fu_characteristics(self, context: str) -> Dict[str, List[str]]:
        """Extract Zang-Fu characteristics."""
        characteristics = {
            "organs": [],
            "conditions": []
        }
        
        # Extract affected organs
        organ_pattern = re.compile(
            r'\b(liver|heart|spleen|lung|kidney|gallbladder|stomach|'
            r'small intestine|large intestine|bladder|san jiao|pericardium)\b',
            re.IGNORECASE
        )
        matches = organ_pattern.findall(context)
        if matches:
            characteristics["organs"] = [m.lower() for m in matches]
        
        # Extract organ conditions
        condition_pattern = re.compile(
            r'\b(qi deficiency|yang deficiency|yin deficiency|blood deficiency|'
            r'qi stagnation|blood stasis|fire|damp|cold|heat)\b',
            re.IGNORECASE
        )
        matches = condition_pattern.findall(context)
        if matches:
            characteristics["conditions"] = [m.lower() for m in matches]
        
        return characteristics
    
    def _extract_qi_blood_characteristics(self, context: str) -> Dict[str, List[str]]:
        """Extract Qi-Blood pattern characteristics."""
        characteristics = {
            "substance": [],
            "condition": []
        }
        
        # Extract affected substance
        substance_pattern = re.compile(r'\b(qi|blood|essence)\b', re.IGNORECASE)
        matches = substance_pattern.findall(context)
        if matches:
            characteristics["substance"] = [m.lower() for m in matches]
        
        # Extract condition
        condition_pattern = re.compile(
            r'\b(deficiency|stagnation|rebellion|sinking|collapse)\b',
            re.IGNORECASE
        )
        matches = condition_pattern.findall(context)
        if matches:
            characteristics["condition"] = [m.lower() for m in matches]
        
        return characteristics
    
    def _calculate_confidence(self, details: Dict[str, Any]) -> float:
        """Calculate confidence score for pattern extraction."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on category identification
        if details["category"]:
            confidence += 0.1
        
        # Adjust based on modifiers
        if details["modifiers"]:
            confidence += 0.1
        
        # Adjust based on characteristics
        if details["characteristics"]:
            confidence += 0.2
            
        return min(1.0, confidence)
    
    def _generate_id(self, name: str) -> str:
        """Generate a stable ID for a pattern."""
        # Remove special characters and convert to lowercase
        clean_name = re.sub(r'[^\w\s]', '', name.lower())
        # Replace spaces with underscores
        return re.sub(r'\s+', '_', clean_name)