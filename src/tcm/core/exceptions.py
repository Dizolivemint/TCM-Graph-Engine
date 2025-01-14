# tcm/core/exceptions.py
class TCMError(Exception):
    """Base exception for TCM knowledge system."""

class ValidationError(TCMError):
    """Raised when data validation fails."""

class GraphError(TCMError):
    """Raised for graph-related errors."""
    
class ProcessingError(TCMError):
    """Raised when document processing fails."""

class ExtractionError(TCMError):
    """Raised when entity extraction fails."""

class SourceError(TCMError):
    """Raised for source reference errors."""