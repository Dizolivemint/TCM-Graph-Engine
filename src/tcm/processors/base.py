# tcm/processors/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

from tcm.core.exceptions import ProcessingError
from tcm.core.models import Source

class BaseProcessor(ABC):
    """Base class for all document processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    @abstractmethod
    def process(self, content: Any) -> Any:
        """Process the input content."""
        pass