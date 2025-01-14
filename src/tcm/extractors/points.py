# tcm/extractors/point_mapper.py
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import re

@dataclass
class PointNameEntry:
    """Container for point name variations."""
    standard: str  # e.g., "LI-4"
    pinyin: str    # e.g., "Hegu"
    chinese: str   # e.g., "合谷"
    english: str   # e.g., "Joining Valley"
    embedding: Optional[np.ndarray] = None

class MLPointMapper:
    """ML-based mapping for acupuncture point names."""
    
    def __init__(
        self,
        embedding_model,
        embedding_dim: int = 384,
        similarity_threshold: float = 0.85
    ):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding caches
        self.point_embeddings: Dict[str, PointNameEntry] = {}
        self.embedding_index: Dict[str, np.ndarray] = {}
    
    def add_point_entry(self, entry: PointNameEntry) -> None:
        """Add a point entry and compute its embeddings."""
        # Generate embeddings for each name variation
        names = [
            entry.standard,
            entry.pinyin,
            entry.english,
            f"{entry.pinyin} ({entry.english})"
        ]
        
        # Compute combined embedding
        embeddings = [
            self.embedding_model.encode(name)
            for name in names
        ]
        entry.embedding = np.mean(embeddings, axis=0)
        
        # Add to indices
        self.point_embeddings[entry.standard] = entry
        self.embedding_index[entry.standard] = entry.embedding
    
    def find_point_matches(
        self,
        text: str,
        return_scores: bool = False
    ) -> List[Tuple[str, float]]:
        """Find matching point names in text using similarity search."""
        # Generate embedding for input text
        text_embedding = self.embedding_model.encode(text)
        
        # Calculate similarities with all point entries
        similarities = {
            point_id: self._calculate_similarity(text_embedding, entry.embedding)
            for point_id, entry in self.point_embeddings.items()
        }
        
        # Filter and sort matches
        matches = [
            (point_id, score)
            for point_id, score in similarities.items()
            if score >= self.similarity_threshold
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if return_scores:
            return matches
        return [point_id for point_id, _ in matches]
    
    def get_point_names(self, point_id: str) -> Optional[Dict[str, str]]:
        """Get all name variations for a point."""
        entry = self.point_embeddings.get(point_id)
        if not entry:
            return None
            
        return {
            "standard": entry.standard,
            "pinyin": entry.pinyin,
            "chinese": entry.chinese,
            "english": entry.english
        }
    
    def translate_point_name(
        self,
        name: str,
        target_language: str = "english"
    ) -> Optional[str]:
        """Translate a point name to target language."""
        # Find best matching point entry
        matches = self.find_point_matches(name, return_scores=True)
        if not matches:
            return None
            
        point_id, score = matches[0]
        entry = self.point_embeddings[point_id]
        
        if target_language == "english":
            return entry.english
        elif target_language == "pinyin":
            return entry.pinyin
        elif target_language == "chinese":
            return entry.chinese
        else:
            return entry.standard
    
    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    @classmethod
    def from_training_data(
        cls,
        training_data: List[Dict[str, str]],
        embedding_model
    ) -> 'MLPointMapper':
        """Initialize mapper from training data."""
        mapper = cls(embedding_model)
        
        for point_data in training_data:
            entry = PointNameEntry(
                standard=point_data["standard"],
                pinyin=point_data["pinyin"],
                chinese=point_data["chinese"],
                english=point_data["english"]
            )
            mapper.add_point_entry(entry)
        
        return mapper