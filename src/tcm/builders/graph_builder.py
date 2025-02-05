#!/usr/bin/env python3
from pathlib import Path
from typing import List, Dict, Set
import logging

from tcm.core.models import Node, Relationship, Source
from tcm.core.enums import NodeType, SourceType
from tcm.processors.document_processor import DocumentProcessor
from tcm.processors.text_extractor import TextExtractor
from tcm.extractors.herbs import HerbExtractor
from tcm.extractors.patterns import PatternExtractor
from tcm.extractors.signs import SignExtractor
from tcm.graph.knowledge_graph import TCMKnowledgeGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Builds TCM knowledge graph from source documents."""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.graph = TCMKnowledgeGraph()
        
        # Initialize processors
        self.doc_processor = DocumentProcessor()
        self.text_extractor = TextExtractor()
        
        # Initialize specialized extractors
        self.herb_extractor = HerbExtractor()
        self.pattern_extractor = PatternExtractor()
        self.sign_extractor = SignExtractor()
        
        # Track processed sources
        self.processed_sources: Set[Source] = set()
        
    def build(self) -> None:
        """Build the complete knowledge graph."""
        try:
            # Process all source documents
            for file_path in self.source_dir.glob("**/*"):
                if file_path.suffix in self.doc_processor.SUPPORTED_FORMATS:
                    self._process_document(file_path)
            
            # Validate the graph
            self._validate_graph()
            
            # Save the graph
            self._save_graph()
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise
    
    def _process_document(self, file_path: Path) -> None:
      """Process a single document and extract knowledge."""
      logger.info(f"Processing document: {file_path}")
      
      try:
          # Process document
          doc_content = self.doc_processor.process(file_path)
          
          # Create source reference
          source = Source(
              id=f"source_{file_path.stem}",
              type=SourceType.TEXTBOOK,  # Default to textbook, can be refined
              title=doc_content['metadata'].get('title', file_path.name)
          )
          self.processed_sources.add(source)
          
          # Extract text blocks
          text_blocks = self._extract_text_blocks(doc_content)
          
          # Extract general entities first
          entities = self.text_extractor.process(text_blocks)
          
          # Group entities by type for context
          entity_groups = {
              'herbs': [e for e in entities if e.type == 'herb'],
              'patterns': [e for e in entities if e.type == 'pattern'],
              'signs': [e for e in entities if e.type == 'sign']
          }
          
          # Use entities context for specialized extraction
          combined_text = "\n".join(block['text'] for block in text_blocks)
          
          # Process with specialized extractors, passing relevant entities
          herbs = self.herb_extractor.extract(
              combined_text,
              [source],
              context_entities=entity_groups['herbs']
          )
          patterns = self.pattern_extractor.extract(
              combined_text,
              [source],
              context_entities=entity_groups['patterns']
          )
          signs = self.sign_extractor.extract(
              combined_text,
              [source],
              context_entities=entity_groups['signs']
          )
          
          # Add all extracted knowledge to graph
          self._add_to_graph(entities, herbs, patterns, signs)
          
      except Exception as e:
          logger.error(f"Error processing {file_path}: {e}")
          raise
    
    def _extract_text_blocks(self, doc_content: Dict) -> List[Dict]:
        """Extract text blocks from document content."""
        blocks = []
        for page in doc_content['content']:
            blocks.extend(page['blocks'])
        return blocks
    
    def _add_to_graph(self, entities, herbs, patterns, signs) -> None:
      """Add extracted knowledge to the graph."""
      # Add nodes from general entities first
      if entities:  # Check if entities is not None
          for entity in entities:
              try:
                  self.graph.add_node(entity)
              except Exception as e:
                  logger.warning(f"Error adding entity node {entity.id}: {e}")

      # Add specialized nodes
      for extraction_name, extraction in [
          ("herbs", herbs),
          ("patterns", patterns),
          ("signs", signs)
      ]:
          if extraction is None:
              logger.warning(f"Skipping {extraction_name} extraction: Result is None")
              continue
              
          try:
              for node in extraction.nodes:
                  try:
                      self.graph.add_node(node)
                  except Exception as e:
                      logger.warning(f"Error adding {extraction_name} node {node.id}: {e}")
          except AttributeError as e:
              logger.error(f"Invalid extraction result for {extraction_name}: {e}")
      
      # Only infer relationships if we have valid extraction results
      if any(x is not None for x in [herbs, patterns, signs]):
          try:
              self._infer_relationships(entities, herbs, patterns, signs)
          except Exception as e:
              logger.error(f"Error inferring relationships: {e}")
    
    def _infer_relationships(self, herbs, patterns, signs) -> None:
        """Infer relationships between extracted entities."""
        # Implement relationship inference logic
        # This is where domain-specific rules would be applied
        # For example: connecting herbs to patterns they treat
        pass
    
    def _validate_graph(self) -> None:
        """Validate the built graph."""
        errors = self.graph.validate()
        if errors:
            logger.warning("Graph validation found issues:")
            for error in errors:
                logger.warning(f"  - {error}")
    
    def _save_graph(self) -> None:
        """Save the graph to disk."""
        output_path = self.output_dir / "tcm_knowledge_graph.json"
        self.graph.save(output_path)
        logger.info(f"Graph saved to {output_path}")
        logger.info(f"Total nodes: {self.graph.node_count}")
        logger.info(f"Total relationships: {self.graph.relationship_count}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build TCM knowledge graph from sources")
    parser.add_argument("source_dir", type=Path, help="Directory containing source documents")
    parser.add_argument("output_dir", type=Path, help="Directory for output files")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build graph
    builder = KnowledgeGraphBuilder(args.source_dir, args.output_dir)
    builder.build()

if __name__ == "__main__":
    main()