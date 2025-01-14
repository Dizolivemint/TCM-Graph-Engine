# tcm/extractors/signs.py
import re
from typing import Dict, List, Set, Tuple, Optional
import hashlib
from dataclasses import dataclass

from tcm.core.exceptions import ProcessingError
from tcm.core.models import Node, Source
from tcm.core.enums import NodeType, SignType
from .base import BaseExtractor, ExtractionResult

@dataclass
class SignCharacteristic:
    """Container for sign characteristics."""
    primary: str
    modifiers: List[str]
    severity: Optional[str] = None
    location: Optional[str] = None

class SignExtractor(BaseExtractor):
    """Extracts TCM diagnostic signs from text."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.sign_patterns = self._compile_sign_patterns()
        self.modifier_patterns = self._compile_modifier_patterns()
        self.relationship_patterns = self._compile_relationship_patterns()
    
    def extract(self, text: str, sources: List[Source]) -> ExtractionResult:
        """Extract diagnostic sign information from text."""
        try:
            # Generate cache key
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            # Check cache
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            # Extract sign mentions by type
            nodes = []
            all_mentions = {}
            
            for sign_type in SignType:
                mentions = self._extract_sign_mentions(text, sign_type)
                all_mentions.update(mentions)
                
                # Process each sign mention
                for sign_text, positions in mentions.items():
                    # Get surrounding context
                    contexts = self._get_contexts(text, positions)
                    
                    # Extract sign details
                    details = self._extract_sign_details(sign_text, contexts, sign_type)
                    
                    # Create sign node
                    node = Node(
                        id=f"sign_{sign_type.lower()}_{self._generate_id(sign_text)}",
                        type=NodeType.SIGN,
                        name=sign_text,
                        attributes={
                            "sign_type": sign_type,
                            "characteristics": details["characteristics"].__dict__,
                            "relationships": details["relationships"],
                            "significance": details["significance"]
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
                    "mention_counts": all_mentions,
                    "sign_types": {
                        node.id: node.attributes["sign_type"]
                        for node in nodes
                    }
                }
            )
            
            # Cache result
            if self.validate_extraction(result):
                self.cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            raise ProcessingError(f"Sign extraction failed: {e}")
    
    def _compile_sign_patterns(self) -> Dict[SignType, Dict[str, re.Pattern]]:
        """Compile patterns for sign identification by type."""
        return {
            SignType.TONGUE: {
                'body': re.compile(
                    r'\b(pale|red|purple|blue|dusky|dark)\s+tongue\b'
                    r'|\btongue\s+(body|color|texture|shape)\b'
                    r'|\btongue\s+is\s+\w+',
                    re.IGNORECASE
                ),
                'coating': re.compile(
                    r'\b(thick|thin|white|yellow|gray|black)\s+(?:tongue\s+)?coating\b'
                    r'|\bcoating\s+is\s+\w+',
                    re.IGNORECASE
                ),
                'moisture': re.compile(
                    r'\b(dry|wet|moist|sticky)\s+(?:tongue|coating)\b',
                    re.IGNORECASE
                )
            },
            
            SignType.PULSE: {
                'quality': re.compile(
                    r'\b(floating|sinking|slow|rapid|string|tight|slippery|rough|thin|'
                    r'big|small|weak|strong|regular|irregular)\s+pulse\b'
                    r'|\bpulse\s+is\s+\w+',
                    re.IGNORECASE
                ),
                'rhythm': re.compile(
                    r'\b(irregular|intermittent|knotted|hurried)\s+(?:pulse|rhythm)\b',
                    re.IGNORECASE
                ),
                'position': re.compile(
                    r'\b(cun|guan|chi)\s+position\b'
                    r'|\b(superficial|middle|deep)\s+(?:level|position)\b',
                    re.IGNORECASE
                )
            },
            
            SignType.COMPLEXION: {
                'color': re.compile(
                    r'\b(pale|red|yellow|green|blue|purple|black|white)\s+(?:complexion|face)\b'
                    r'|\bcomplexion\s+is\s+\w+',
                    re.IGNORECASE
                ),
                'quality': re.compile(
                    r'\b(bright|dark|dull|shiny|moist|dry|withered|fresh)\s+(?:complexion|face)\b',
                    re.IGNORECASE
                ),
                'location': re.compile(
                    r'\b(cheeks?|forehead|nose|chin|eyes?)\s+(?:area|region|shows?|appears?)\b',
                    re.IGNORECASE
                )
            },
            
            SignType.BODY: {
                'temperature': re.compile(
                    r'\b(cold|hot|warm|cool|burning|chills?|fever)\b',
                    re.IGNORECASE
                ),
                'moisture': re.compile(
                    r'\b(sweating|perspiration|dry|damp|moist|clammy)\b',
                    re.IGNORECASE
                ),
                'pain': re.compile(
                    r'\b(sharp|dull|fixed|moving|distending|burning)\s+pain\b'
                    r'|\bpain\s+is\s+\w+',
                    re.IGNORECASE
                )
            }
        }
    
    def _compile_modifier_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for sign modifiers."""
        return {
            'severity': re.compile(
                r'\b(mild|moderate|severe|slight|pronounced|marked)\b',
                re.IGNORECASE
            ),
            'progression': re.compile(
                r'\b(acute|chronic|recurring|intermittent|persistent)\b',
                re.IGNORECASE
            ),
            'time': re.compile(
                r'\b(morning|afternoon|evening|night|daily|constant)\b',
                re.IGNORECASE
            ),
            'location': re.compile(
                r'\b(left|right|upper|lower|central|lateral|medial)\b',
                re.IGNORECASE
            )
        }
    
    def _compile_relationship_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for sign relationships."""
        return {
            'correlation': re.compile(
                r'\b(indicates|suggests|shows|reflects|corresponds to)\s+([^,.]+)',
                re.IGNORECASE
            ),
            'causation': re.compile(
                r'\b(caused by|due to|result of|because of)\s+([^,.]+)',
                re.IGNORECASE
            ),
            'progression': re.compile(
                r'\b(develops into|progresses to|transforms into)\s+([^,.]+)',
                re.IGNORECASE
            )
        }
    
    def _extract_sign_mentions(
        self,
        text: str,
        sign_type: SignType
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Extract mentions of a specific sign type."""
        mentions = {}
        
        for category, pattern in self.sign_patterns[sign_type].items():
            for match in pattern.finditer(text):
                sign_text = match.group()
                if sign_text not in mentions:
                    mentions[sign_text] = []
                mentions[sign_text].append(match.span())
        
        return mentions
    
    def _get_contexts(
        self,
        text: str,
        positions: List[Tuple[int, int]]
    ) -> List[str]:
        """Get context windows around sign mentions."""
        contexts = []
        window_size = self.config.get('context_window', 100)
        
        for start, end in positions:
            context_start = max(0, start - window_size)
            context_end = min(len(text), end + window_size)
            contexts.append(text[context_start:context_end])
        
        return contexts
    
    def _extract_sign_details(
        self,
        sign_text: str,
        contexts: List[str],
        sign_type: SignType
    ) -> Dict[str, any]:
        """Extract detailed information about a diagnostic sign."""
        details = {
            "characteristics": self._extract_characteristics(sign_text, contexts, sign_type),
            "relationships": self._extract_relationships(contexts),
            "significance": self._extract_significance(contexts)
        }
        
        return details
    
    def _extract_characteristics(
        self,
        sign_text: str,
        contexts: List[str],
        sign_type: SignType
    ) -> SignCharacteristic:
        """Extract sign characteristics based on type."""
        # Extract primary characteristic
        primary = self._extract_primary_characteristic(sign_text, sign_type)
        
        # Extract modifiers from contexts
        modifiers = []
        severity = None
        location = None
        
        for context in contexts:
            # Check modifiers
            for modifier_type, pattern in self.modifier_patterns.items():
                matches = pattern.findall(context)
                if matches:
                    if modifier_type == 'severity':
                        severity = matches[0]
                    elif modifier_type == 'location':
                        location = matches[0]
                    else:
                        modifiers.extend(matches)
        
        return SignCharacteristic(
            primary=primary,
            modifiers=list(set(modifiers)),
            severity=severity,
            location=location
        )
    
    def _extract_primary_characteristic(
        self,
        sign_text: str,
        sign_type: SignType
    ) -> str:
        """Extract the primary characteristic from sign text."""
        # Extract main descriptive word based on sign type
        if sign_type == SignType.TONGUE:
            match = re.search(r'\b(pale|red|purple|blue|dusky|dark)\b', sign_text, re.I)
        elif sign_type == SignType.PULSE:
            match = re.search(
                r'\b(floating|sinking|slow|rapid|string|tight|slippery|rough)\b',
                sign_text,
                re.I
            )
        elif sign_type == SignType.COMPLEXION:
            match = re.search(
                r'\b(pale|red|yellow|green|blue|purple|black|white)\b',
                sign_text,
                re.I
            )
        else:  # BODY
            match = re.search(
                r'\b(cold|hot|warm|cool|burning|dry|damp|painful)\b',
                sign_text,
                re.I
            )
        
        return match.group(1).lower() if match else ""
    
    def _extract_relationships(self, contexts: List[str]) -> Dict[str, List[str]]:
        """Extract relationships between signs and other findings."""
        relationships = {
            "correlations": [],
            "causes": [],
            "progressions": []
        }
        
        for context in contexts:
            # Check correlations
            for match in self.relationship_patterns['correlation'].finditer(context):
                relationships["correlations"].append(match.group(2).strip())
                
            # Check causation
            for match in self.relationship_patterns['causation'].finditer(context):
                relationships["causes"].append(match.group(2).strip())
                
            # Check progression
            for match in self.relationship_patterns['progression'].finditer(context):
                relationships["progressions"].append(match.group(2).strip())
        
        return relationships
    
    def _extract_significance(self, contexts: List[str]) -> Dict[str, List[str]]:
        """Extract clinical significance and implications."""
        significance = {
            "patterns": [],
            "conditions": [],
            "implications": []
        }
        
        # Common pattern indicators
        pattern_indicators = re.compile(
            r'\b(?:indicates|suggests|shows)\s+(?:a\s+)?(.*?pattern|.*?syndrome)',
            re.IGNORECASE
        )
        
        # Condition indicators
        condition_indicators = re.compile(
            r'\b(?:associated with|seen in|typical of)\s+(.*?)(?=\.|,|\band\b|\bor\b|$)',
            re.IGNORECASE
        )
        
        # Clinical implication indicators
        implication_indicators = re.compile(
            r'\b(?:requires|needs|should be treated with|consider)\s+(.*?)(?=\.|,|\band\b|\bor\b|$)',
            re.IGNORECASE
        )
        
        for context in contexts:
            # Extract patterns
            for match in pattern_indicators.finditer(context):
                significance["patterns"].append(match.group(1).strip())
                
            # Extract conditions
            for match in condition_indicators.finditer(context):
                significance["conditions"].append(match.group(1).strip())
                
            # Extract implications
            for match in implication_indicators.finditer(context):
                significance["implications"].append(match.group(1).strip())
        
        return significance
    
    def _calculate_confidence(self, details: Dict[str, any]) -> float:
        """Calculate confidence score for sign extraction."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on characteristics
        if details["characteristics"].primary:
            confidence += 0.1
        if details["characteristics"].modifiers:
            confidence += 0.1
        
        # Adjust based on relationships
        if any(details["relationships"].values()):
            confidence += 0.1
        
        # Adjust based on significance
        if any(details["significance"].values()):
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _generate_id(self, text: str) -> str:
        """Generate a stable ID for a sign."""
        # Remove special characters and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        # Replace spaces with underscores
        return re.sub(r'\s+', '_', clean_text)