# tcm/extractors/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from tcm.core.models import Node, Source
from tcm.core.exceptions import ExtractionError
from tcm.processors.text_extractor import ExtractedEntity

@dataclass
class ExtractionResult:
    """Container for extraction results."""
    nodes: List[Node]
    confidence: float
    sources: Set[Source]
    metadata: Dict[str, Any]

class BaseExtractor(ABC):
    """Base class for TCM entity extractors."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._extracted_cache: Dict[str, ExtractionResult] = {}
    
    @abstractmethod
    def extract(
        self, 
        text: str, 
        sources: List[Source],
        context_entities: Optional[List[ExtractedEntity]] = None
    ) -> ExtractionResult:
        """Extract entities from text with source attribution and optional context."""
        pass
    
    def validate_extraction(self, result: ExtractionResult) -> bool:
        """Validate extraction results."""
        if not result.nodes:
            return False
            
        # Check confidence threshold
        if result.confidence < self.config.get('min_confidence', 0.5):
            return False
            
        # Ensure source attribution
        if not result.sources:
            return False
            
        return True
    
    def get_cached_result(self, cache_key: str) -> Optional[ExtractionResult]:
        """Retrieve cached extraction result."""
        return self._extracted_cache.get(cache_key)
    
    def cache_result(self, cache_key: str, result: ExtractionResult) -> None:
        """Cache extraction result."""
        self._extracted_cache[cache_key] = result