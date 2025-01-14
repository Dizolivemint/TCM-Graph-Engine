# tcm/processors/reference_tracker.py
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from tcm.core.exceptions import ProcessingError

from .base import BaseProcessor
from tcm.core.models import Source
from tcm.core.enums import SourceType

@dataclass
class Reference:
    """Container for reference information."""
    text: str
    source: Source
    context: str
    page: Optional[str]
    confidence: float

class ReferenceTracker(BaseProcessor):
    """Tracks and validates source references."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.references: Dict[str, List[Reference]] = {}
    
    def process(self, content: Dict) -> Dict[str, List[Reference]]:
        """Process and track references in content."""
        try:
            # Extract references
            references = self._extract_references(content)
            
            # Validate and normalize references
            validated_refs = self._validate_references(references)
            
            # Group by source
            for ref in validated_refs:
                source_id = ref.source.id
                if source_id not in self.references:
                    self.references[source_id] = []
                self.references[source_id].append(ref)
            
            return self.references
            
        except Exception as e:
            raise ProcessingError(f"Error processing references: {e}")
    
    def _extract_references(self, content: Dict) -> List[Reference]:
        """Extract references from content."""
        references = []
        
        for page in content['content']:
            text = page['text']
            
            # Look for citation patterns
            citations = self._find_citations(text)
            
            for citation in citations:
                ref = Reference(
                    text=citation['text'],
                    source=self._parse_source(citation),
                    context=self._get_citation_context(text, citation['position']),
                    page=str(page['page_num']),
                    confidence=0.9
                )
                references.append(ref)
        
        return references
    
    def _validate_references(self, references: List[Reference]) -> List[Reference]:
        """Validate and normalize extracted references."""
        validated = []
        
        for ref in references:
            # Validate source information
            if not ref.source.id or not ref.source.title:
                continue
                
            # Normalize source information
            normalized = self._normalize_source(ref.source)
            if normalized:
                ref.source = normalized
                validated.append(ref)
        
        return validated
    
    def _find_citations(self, text: str) -> List[Dict]:
        """Find citations in text content.
        
        Handles multiple citation formats:
        1. Classical texts (e.g., "Nei Jing Chapter 3")
        2. Modern author-year (e.g., "Zhang et al., 2019")
        3. Numbered references (e.g., "[1]" or "(1)")
        4. Traditional book chapters (e.g., "《黄帝内经》第三章")
        
        Returns:
            List of dictionaries containing:
            - text: The full citation text
            - type: Citation type
            - position: (start, end) tuple of citation position
            - details: Additional parsed information
        """
        citations = []
        
        # Classical text patterns
        classical_pattern = re.compile(
            r'(?:'
            r'(?:Nei Jing|Shang Han Lun|Wen Bing|Jin Gui|Nan Jing)'  # English names
            r'|'
            r'(?:《[^》]+》)'  # Chinese book names
            r')\s*'
            r'(?:Chapter|第)?'  # Optional chapter marker
            r'[\s第]?'
            r'(?:\d+|\w+)'  # Chapter number/name
            r'(?:[\s,;:]+(?:\d+(?:\.\d+)?)?)?'  # Optional section/verse
        )
        
        # Modern citation patterns
        modern_pattern = re.compile(
            r'(?:'
            r'\(?\d{4}\)?'  # Year in parentheses
            r'|'
            r'(?:[A-Z][a-z]+(?:\set\sal\.?|\sand\s[A-Z][a-z]+)?[\s,]+\d{4})'  # Author year
            r'|'
            r'(?:\[[\d,\s-]+\])'  # Numbered reference
            r'|'
            r'(?:\([\d,\s-]+\))'  # Parenthetical numbers
            r')'
        )
        
        # Find classical citations
        for match in classical_pattern.finditer(text):
            citations.append({
                'text': match.group(),
                'type': 'classical',
                'position': match.span(),
                'details': {
                    'text_type': 'classical',
                    'full_match': match.group()
                }
            })
        
        # Find modern citations
        for match in modern_pattern.finditer(text):
            citation_text = match.group()
            
            # Determine citation type
            if re.match(r'\d{4}', citation_text):
                citation_type = 'year'
            elif 'et al.' in citation_text or 'and' in citation_text:
                citation_type = 'author_year'
            else:
                citation_type = 'numbered'
            
            citations.append({
                'text': citation_text,
                'type': 'modern',
                'position': match.span(),
                'details': {
                    'citation_type': citation_type,
                    'full_match': citation_text
                }
            })
        
        # Handle complex cases and cleanup
        citations = self._merge_adjacent_citations(citations)
        citations = self._validate_citations(citations)
        
        return citations
        
    def _merge_adjacent_citations(self, citations: List[Dict]) -> List[Dict]:
        """Merge citations that are likely part of the same reference."""
        if not citations:
            return citations
            
        merged = []
        current = citations[0]
        
        for next_citation in citations[1:]:
            # Check if citations are close together (within 3 characters)
            if (next_citation['position'][0] - current['position'][1]) <= 3:
                # Merge citations
                current['text'] = current['text'] + next_citation['text']
                current['position'] = (
                    current['position'][0],
                    next_citation['position'][1]
                )
                current['details']['merged'] = True
            else:
                merged.append(current)
                current = next_citation
                
        merged.append(current)
        return merged
        
    def _validate_citations(self, citations: List[Dict]) -> List[Dict]:
        """Validate and filter citations to remove false positives."""
        validated = []
        
        for citation in citations:
            # Skip likely false positives
            if self._is_false_positive(citation):
                continue
                
            # Add confidence score
            citation['confidence'] = self._calculate_citation_confidence(citation)
            
            if citation['confidence'] >= 0.5:  # Minimum confidence threshold
                validated.append(citation)
                
        return validated
        
    def _is_false_positive(self, citation: Dict) -> bool:
      """Check if a citation is likely a false positive.
      
      Implements comprehensive checks for common false positive patterns in TCM literature.
      
      Args:
          citation: Dictionary containing citation information
          
      Returns:
          True if the citation is likely a false positive
      """
      text = citation['text'].lower()
      
      # Common false positive patterns
      false_patterns = [
          r'^\d+$',  # Just a number
          r'^\d{1,3}(?:\.\d+)?$',  # Numbers that are likely measurements
          r'^\(\d+\)$',  # Just parenthesized numbers
          r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',  # Month abbreviations
          r'\d+(?:mg|ml|g|kg|mm|cm|m)',  # Measurements
          r'^\d{1,2}:\d{1,2}$',  # Time format
          r'chapter\s+\d+$',  # Just chapter numbers
          r'^\d{1,2}/\d{1,2}$',  # Fractions or dates
      ]
      
      # Check basic patterns
      for pattern in false_patterns:
          if re.match(pattern, text.strip()):
              return True
              
      # Context-specific checks for TCM literature
      if citation['type'] == 'classical':
          # Validate classical text references
          if not any(classic in text for classic in [
              'nei jing', 'shang han', 'wen bing', 'jin gui', 'nan jing',
              '内经', '伤寒', '温病', '金匮', '难经'
          ]):
              return True
              
      elif citation['type'] == 'modern':
          # Additional checks for modern citations
          if len(text) < 6:  # Too short to be valid
              return True
          if text.count('(') != text.count(')'):  # Unmatched parentheses
              return True
              
      return False
    
    def _slugify(self, text: str) -> str:
      """Convert text to a URL-friendly slug.
      
      Args:
          text: Text to convert to slug
          
      Returns:
          Slugified version of the text
      """
      # Convert to lowercase
      text = text.lower()
      
      # Remove Chinese characters and replace with pinyin
      # In a real implementation, you would use a proper Chinese-to-pinyin converter
      chinese_char_map = {
          '内经': 'neijing',
          '伤寒': 'shanghan',
          '金匮': 'jingui',
          '难经': 'nanjing',
          '温病': 'wenbing',
          # Add more mappings as needed
      }
      for chinese, pinyin in chinese_char_map.items():
          text = text.replace(chinese, pinyin)
      
      # Remove special characters and replace spaces with underscores
      text = re.sub(r'[^\w\s-]', '', text)
      text = re.sub(r'[-\s]+', '_', text)
      
      return text.strip('_')
    
    def _parse_source(self, citation: Dict) -> Source:
      """Parse citation into Source object.
      
      Extracts structured source information from citation text with TCM-specific handling.
      
      Args:
          citation: Dictionary containing citation information
          
      Returns:
          Source object containing parsed information
          
      Raises:
          ValueError: If citation cannot be parsed into a valid source
      """
      text = citation['text']
      
      if citation['type'] == 'classical':
          return self._parse_classical_source(text)
      else:
          return self._parse_modern_source(text)
        
    def _parse_classical_source(self, text: str) -> Source:
      """Parse classical TCM text citations."""
      # Match classical text patterns
      classical_match = re.match(
          r'(?P<title>(?:Nei Jing|Shang Han Lun|Wen Bing|Jin Gui|Nan Jing|《[^》]+》))'
          r'(?:[\s,]+(?:Chapter|第)?[\s第]?(?P<chapter>\d+))?'
          r'(?:[\s,]*(?P<section>\d+(?:\.\d+)?)?)?',
          text,
          re.IGNORECASE
      )
      
      if not classical_match:
          raise ValueError(f"Invalid classical citation format: {text}")
          
      groups = classical_match.groupdict()
      
      # Create source ID from title and chapter
      source_id = f"classical_{self.slugify(groups['title'])}_{groups.get('chapter', '')}"
      
      return Source(
          id=source_id,
          type=SourceType.CLASSICAL_TEXT,
          title=groups['title'],
          year=None,  # Historical texts often don't have precise years
          page=f"Chapter {groups.get('chapter', '')}" if groups.get('chapter') else None
      )
    
    def _parse_modern_source(self, text: str) -> Source:
      """Parse modern citation formats."""
      # Try author-year format first
      author_year_match = re.match(
          r'(?P<authors>[A-Z][a-z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-z]+)?)'
          r'[\s,]+\(?(?P<year>\d{4})\)?',
          text
      )
      
      if author_year_match:
          groups = author_year_match.groupdict()
          authors = [a.strip() for a in groups['authors'].replace('et al.', '').split('and')]
          
          return Source(
              id=f"modern_{self.slugify(authors[0])}_{groups['year']}",
              type=SourceType.RESEARCH_PAPER,
              title="",  # Title would need to be looked up
              authors=authors,
              year=int(groups['year'])
          )
          
      # Try numbered reference format
      numbered_match = re.match(r'\[(?P<ref_num>\d+)\]|\((?P<ref_num2>\d+)\)', text)
      if numbered_match:
          ref_num = numbered_match.group('ref_num') or numbered_match.group('ref_num2')
          return Source(
              id=f"reference_{ref_num}",
              type=SourceType.RESEARCH_PAPER,
              title="",  # Would need to be looked up from reference list
              authors=[],
              year=None
          )
          
      raise ValueError(f"Unable to parse citation format: {text}")
    
    def _get_citation_context(self, text: str, position: tuple) -> str:
        """Get surrounding context for a citation."""
        start, end = position
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        return text[context_start:context_end]
    
    def _normalize_source(self, source: Source) -> Optional[Source]:
      """Normalize source information.
      
      Standardizes source information by:
      - Normalizing author names
      - Standardizing publisher names
      - Validating and normalizing years
      - Handling classical text names consistently
      
      Args:
          source: Source object to normalize
          
      Returns:
          Normalized Source object or None if normalization fails
      """
      try:
          # Don't modify the original source
          normalized = Source(
              id=source.id,
              type=source.type,
              title=source.title,
              authors=source.authors.copy(),
              year=source.year,
              publisher=source.publisher,
              page=source.page
          )
          
          # Normalize title
          normalized.title = self._normalize_title(normalized.title)
          
          # Normalize authors
          normalized.authors = [
              self._normalize_author_name(author)
              for author in normalized.authors
          ]
          
          # Normalize publisher
          if normalized.publisher:
              normalized.publisher = self._normalize_publisher(normalized.publisher)
              
          # Handle classical texts specially
          if normalized.type == SourceType.CLASSICAL_TEXT:
              normalized = self._normalize_classical_text(normalized)
              
          return normalized
          
      except Exception as e:
          ProcessingError(f"Source normalization failed: {e}")
          return None
        
    def _normalize_title(self, title: str) -> str:
      """Normalize source titles."""
      # Remove extra whitespace
      title = ' '.join(title.split())
      
      # Handle Chinese text names consistently
      title = re.sub(r'《([^》]+)》', r'\1', title)
      
      # Standardize common text names
      translations = {
          "huang di nei jing": "Huangdi Neijing",
          "shang han lun": "Shanghan Lun",
          "jin gui yao lue": "Jingui Yaolue",
          # Add more standard translations
      }
      
      for key, value in translations.items():
          if title.lower().startswith(key):
              return value + title[len(key):]
              
      return title
    
    def _normalize_author_name(self, name: str) -> str:
      """Normalize author names."""
      # Remove extra whitespace
      name = ' '.join(name.split())
      
      # Handle Chinese names
      if re.search(r'[\u4e00-\u9fff]', name):
          return name  # Keep Chinese names as-is
          
      # Format Western names
      parts = name.split()
      if len(parts) > 1:
          # Convert to "Lastname, F." format
          return f"{parts[-1]}, {'. '.join(p[0].upper() for p in parts[:-1])}."
          
      return name
    
    def _normalize_publisher(self, publisher: str) -> str:
      """Normalize publisher names."""
      # Remove extra whitespace and standardize
      publisher = ' '.join(publisher.split())
      
      # Standard publisher name mappings
      mappings = {
          "people's medical": "People's Medical Publishing House",
          "china tcm": "China Traditional Chinese Medicine Publishing House",
          # Add more standard publisher names
      }
      
      # Try to match and standardize
      for key, value in mappings.items():
          if key in publisher.lower():
              return value
              
      return publisher
    
    def _normalize_classical_text(self, source: Source) -> Source:
      """Special handling for classical text normalization."""
      # Ensure consistent naming for classical texts
      source.title = self._normalize_title(source.title)
      
      # Set standard ID format for classical texts
      if not source.id.startswith('classical_'):
          source.id = f"classical_{self.slugify(source.title)}"
          
      # Clear modern-style fields
      source.authors = []  # Classical texts often don't have clear authorship
      
      return source
        
    def _calculate_citation_confidence(self, citation: Dict) -> float:
        """Calculate confidence score for a citation."""
        confidence = 0.0
        
        # Base confidence by type
        if citation['type'] == 'classical':
            confidence = 0.8  # Classical citations are usually reliable
        else:
            confidence = 0.7  # Modern citations start slightly lower
            
        # Adjust based on content
        if '《' in citation['text'] and '》' in citation['text']:
            confidence += 0.1  # Chinese book markers increase confidence
            
        if re.search(r'\d{4}', citation['text']):
            confidence += 0.1  # Years increase confidence
            
        if 'et al' in citation['text']:
            confidence += 0.1  # Author patterns increase confidence
            
        return min(1.0, confidence)  # Cap at 1.0