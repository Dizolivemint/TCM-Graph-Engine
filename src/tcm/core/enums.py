# tcm/core/enums.py
from enum import Enum, auto, StrEnum

class NodeType(StrEnum):
    """Types of nodes in the TCM knowledge graph."""
    HERB = auto()
    FORMULA = auto()
    POINT = auto()
    PATTERN = auto()
    SYMPTOM = auto()
    SIGN = auto()
    MECHANISM = auto()
    FUNCTION = auto()
    INDICATION = auto()
    CHANNEL = auto()

class SignType(StrEnum):
    """Types of diagnostic signs in TCM."""
    TONGUE = auto()
    PULSE = auto()
    COMPLEXION = auto()
    BODY = auto()

class RelationType(StrEnum):
    """Types of relationships between nodes."""
    TREATS = auto()            # Treatment relationship
    MANIFESTS = auto()         # Symptom/Sign -> Pattern
    CONTRAINDICATES = auto()   # Negative interaction
    SUPPORTS = auto()          # Positive interaction/synergy
    CONTAINS = auto()          # Formula -> Herb
    BELONGS_TO = auto()        # Hierarchical relationship
    TRANSFORMS = auto()        # Pattern progression
    CAUSES = auto()            # Causal relationship
    LOCATED_ON = auto()        # Point -> Channel
    INFLUENCES = auto()        # Point/Herb -> Channel/Organ

class PropertyType(StrEnum):
    """Properties of herbs and formulas."""
    TEMPERATURE = auto()
    TASTE = auto()
    DIRECTION = auto()
    CHANNEL_TROPISM = auto()

class SourceType(StrEnum):
    """Types of reference sources."""
    TEXTBOOK = auto()
    CLASSICAL_TEXT = auto()
    RESEARCH_PAPER = auto()
    CLINICAL_NOTE = auto()