# tcm/processors/document_processor.py
import fitz  # PyMuPDF
from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseProcessor
from tcm.core.exceptions import ProcessingError
from tcm.core.models import Source

class DocumentProcessor(BaseProcessor):
    """Processes various document formats into normalized text content."""
    
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.docx', '.md'}
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.metadata = {}
    
    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process a document file into normalized text content."""
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")
            
        if file_path.suffix not in self.SUPPORTED_FORMATS:
            raise ProcessingError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            if file_path.suffix == '.pdf':
                return self._process_pdf(file_path)
            elif file_path.suffix == '.txt':
                return self._process_text(file_path)
            # Add handlers for other formats
            
        except Exception as e:
            raise ProcessingError(f"Error processing document: {e}")
    
    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF documents."""
        doc = fitz.open(file_path)
        content = []
        metadata = {}
        
        # Extract metadata
        metadata.update({
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', '')
        })
        
        # Process each page
        for page in doc:
            content.append({
                'page_num': page.number + 1,
                'text': page.get_text(),
                'blocks': [
                    {
                        'type': block[6],
                        'bbox': block[0:4],
                        'text': block[4]
                    }
                    for block in page.get_text("blocks")
                ]
            })
            
        return {
            'content': content,
            'metadata': metadata,
            'source': str(file_path)
        }
    
    def _process_text(self, file_path: Path) -> Dict[str, Any]:
        """Process plain text documents."""
        content = file_path.read_text(encoding='utf-8')
        return {
            'content': [{
                'page_num': 1,
                'text': content,
                'blocks': [{
                    'type': 'text',
                    'text': content
                }]
            }],
            'metadata': {},
            'source': str(file_path)
        }