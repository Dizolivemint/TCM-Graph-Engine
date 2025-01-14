# tcm/models/foundation.py
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np

class BaseEmbeddingModel(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text into embedding vector."""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts into embedding vectors."""
        pass

class BaseLLM(ABC):
    """Base class for large language models."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate text response."""
        pass
    
    @abstractmethod
    def generate_with_metadata(
        self,
        prompt: str,
        metadata: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response with additional metadata."""
        pass

# Specific implementations
class OpenAIEmbeddings(BaseEmbeddingModel):
    """OpenAI embeddings implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def encode(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([data.embedding for data in response.data])

class AnthropicLLM(BaseLLM):
    """Anthropic Claude implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.content[0].text
    
    def generate_with_metadata(
        self,
        prompt: str,
        metadata: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        response = self.generate(prompt, system_prompt)
        return {
            "response": response,
            "metadata": metadata,
            "model": self.model
        }

# Factory for model creation
class ModelFactory:
    """Factory for creating foundation model instances."""
    
    @staticmethod
    def create_embedding_model(
        provider: str,
        api_key: str,
        **kwargs
    ) -> BaseEmbeddingModel:
        if provider.lower() == "openai":
            return OpenAIEmbeddings(api_key, **kwargs)
        # Add more providers as needed
        raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def create_llm(
        provider: str,
        api_key: str,
        **kwargs
    ) -> BaseLLM:
        if provider.lower() == "anthropic":
            return AnthropicLLM(api_key, **kwargs)
        # Add more providers as needed
        raise ValueError(f"Unsupported LLM provider: {provider}")