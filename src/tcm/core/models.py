# tcm/core/models.py
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from .enums import NodeType, RelationType, SourceType

class Source(BaseModel):
    """Reference source information."""
    id: str
    type: SourceType
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    publisher: Optional[str] = None
    page: Optional[str] = None
    
    @field_validator('year')
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1000 or v > datetime.now().year):
            raise ValueError(f"Invalid year: {v}")
        return v

    def __hash__(self) -> int:
        """Make Source hashable based on its id."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Define equality based on id for consistent hashing."""
        if not isinstance(other, Source):
            return NotImplemented
        return self.id == other.id
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make Source immutable to ensure hash consistency

class Node(BaseModel):
    """Base node in the knowledge graph."""
    id: str
    type: NodeType
    name: str
    names: Dict[str, str] = Field(default_factory=dict)  # Alternative names/synonyms
    attributes: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Source]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip().lower()
    
    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v: List[Source]) -> List[Source]:
        if not v:
            raise ValueError("At least one source is required")
        return v

class Relationship(BaseModel):
    """Relationship between nodes in the knowledge graph."""
    source_id: str
    target_id: str
    type: RelationType
    attributes: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Source]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('source_id', 'target_id')
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("IDs cannot be empty")
        return v.strip().lower()
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        frozen = True  # Relationships are immutable