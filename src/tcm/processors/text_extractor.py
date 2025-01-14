# tcm/processors/text_extractor.py
from typing import Any, Dict, List, Optional
import re
from dataclasses import dataclass

from .base import BaseProcessor
from tcm.core.exceptions import ProcessingError

@dataclass
class ExtractedEntity:
    """Container for extracted entities."""
    text: str
    type: str
    context: str
    confidence: float
    metadata: Dict[str, Any]

class TextExtractor(BaseProcessor):
    """Extracts TCM-specific entities and relationships from text."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.patterns = self._compile_extraction_patterns()
    
    def process(self, text_blocks: List[Dict]) -> List[ExtractedEntity]:
        """Process text blocks to extract entities and relationships."""
        extracted_entities = []
        
        for block in text_blocks:
            text = block['text']
            
            # Extract entities using patterns
            for entity_type, pattern in self.patterns.items():
                matches = pattern.finditer(text)
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]
                    
                    extracted_entities.append(ExtractedEntity(
                        text=match.group(),
                        type=entity_type,
                        context=context,
                        confidence=0.8,  # Could be adjusted based on pattern reliability
                        metadata={'position': match.span()}
                    ))
        
        return extracted_entities
    
    def _compile_extraction_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for entity extraction."""
        return {
            'herb': re.compile(r'\b[A-Z][a-z]+ [a-z]+\b(?=\s*\([A-Za-z\s]+\))?'),
            'formula': re.compile(r'[A-Z][a-z\s]+ (?:Tang|San|Wan|Gao)'),
            'point': re.compile(r'[A-Z]{2}\-\d+|[A-Z]{2}\s\d+'),
            'pattern': re.compile(r'[A-Z][a-z\s]+ (?:deficiency|excess|syndrome|pattern)'),
            # Add more patterns as needed
        }

# tcm/processors/structure_analyzer.py
from typing import Dict, List, Optional
import networkx as nx
from dataclasses import dataclass

from .base import BaseProcessor
from tcm.core.exceptions import ProcessingError
from tcm.core.models import Node, Relationship

@dataclass
class StructureAnalysis:
    """Results of document structure analysis."""
    sections: List[Dict]
    hierarchy: Dict
    relationships: List[Dict]
    metadata: Dict

class StructureAnalyzer(BaseProcessor):
    """Analyzes document structure and content relationships."""
    
    def process(self, content: Dict) -> StructureAnalysis:
        """Analyze document structure and content relationships."""
        try:
            # Extract document sections
            sections = self._extract_sections(content)
            
            # Build content hierarchy
            hierarchy = self._build_hierarchy(sections)
            
            # Analyze relationships between sections
            relationships = self._analyze_relationships(sections)
            
            return StructureAnalysis(
                sections=sections,
                hierarchy=hierarchy,
                relationships=relationships,
                metadata=self._extract_metadata(content)
            )
            
        except Exception as e:
            raise ProcessingError(f"Error analyzing document structure: {e}")
    
    def _extract_sections(self, content: Dict) -> List[Dict]:
        """Extract logical sections from document content."""
        sections = []
        current_section = None
        
        for page in content['content']:
            for block in page['blocks']:
                # Detect section headers and content
                if self._is_section_header(block):
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        'title': block['text'],
                        'content': [],
                        'level': self._determine_header_level(block)
                    }
                elif current_section:
                    current_section['content'].append(block)
        
        if current_section:
            sections.append(current_section)
            
        return sections
    
    def _build_hierarchy(self, sections: List[Dict]) -> Dict:
        """Build hierarchical structure of document sections."""
        hierarchy = {'root': []}
        stack = [('root', -1)]
        
        for section in sections:
            level = section['level']
            while stack[-1][1] >= level:
                stack.pop()
            
            parent = stack[-1][0]
            if parent not in hierarchy:
                hierarchy[parent] = []
            
            section_id = section['title']
            hierarchy[parent].append(section_id)
            stack.append((section_id, level))
            
        return hierarchy
    
    def _analyze_relationships(self, sections: List[Dict]) -> List[Dict]:
        """Analyze relationships between sections."""
        relationships = []
        
        # Build a graph of section relationships
        graph = nx.DiGraph()
        
        for i, section in enumerate(sections):
            graph.add_node(section['title'])
            
            # Look for references to other sections
            for other_section in sections[i+1:]:
                if self._has_reference(section, other_section['title']):
                    relationships.append({
                        'source': section['title'],
                        'target': other_section['title'],
                        'type': 'references'
                    })
                    graph.add_edge(section['title'], other_section['title'])
        
        return relationships
    
    def _is_section_header(self, block: Dict) -> bool:
        """Determine if a text block is a section header."""
        # Implementation would look at text properties, formatting, etc.
        pass
    
    def _determine_header_level(self, block: Dict) -> int:
        """Determine the hierarchical level of a header."""
        # Implementation would analyze formatting, numbering, etc.
        pass
    
    def _has_reference(self, section: Dict, target: str) -> bool:
        """Check if one section references another."""
        # Implementation would look for explicit references
        pass
    
    def _extract_metadata(self, content: Dict) -> Dict:
        """Extract metadata about document structure."""
        return {
            'num_sections': len(content.get('content', [])),
            'avg_section_length': 0,  # Would calculate average
            'max_depth': 0,  # Would calculate maximum hierarchy depth
            'has_references': False  # Would detect reference section
        }