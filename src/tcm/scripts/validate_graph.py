#!/usr/bin/env python3
"""
Validate a built TCM knowledge graph.
"""
import argparse
from pathlib import Path
import logging
import sys
import json

from tcm.graph.knowledge_graph import TCMKnowledgeGraph
from tcm.core.exceptions import TCMError

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_graph(graph_path: Path) -> bool:
    """Validate a knowledge graph file."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load the graph
        graph = TCMKnowledgeGraph.load(graph_path)
        
        # Run validation
        errors = graph.validate()
        
        if not errors:
            logger.info("Graph validation successful")
            logger.info(f"Total nodes: {graph.node_count}")
            logger.info(f"Total relationships: {graph.relationship_count}")
            return True
            
        logger.error("Graph validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {graph_path}")
        return False
    except TCMError as e:
        logger.error(f"TCM-specific error: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Validate a TCM knowledge graph"
    )
    parser.add_argument(
        "graph_path",
        type=Path,
        help="Path to the knowledge graph JSON file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if not args.graph_path.exists():
        print(f"Error: File not found: {args.graph_path}")
        return 1
        
    return 0 if validate_graph(args.graph_path) else 1

if __name__ == "__main__":
    sys.exit(main())