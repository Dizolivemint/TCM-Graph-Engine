"""
Command-line tool for building TCM knowledge graph.
"""
import argparse
from pathlib import Path
import logging
import sys

from tcm.builders.graph_builder import KnowledgeGraphBuilder
from tcm.core.exceptions import TCMError

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Build TCM knowledge graph from source documents"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing source documents"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for output files"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure directories exist
        args.source_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize and run builder
        builder = KnowledgeGraphBuilder(
            source_dir=args.source_dir,
            output_dir=args.output_dir
        )
        builder.build()
        
        logger.info("Knowledge graph built successfully")
        return 0
        
    except TCMError as e:
        logger.error(f"TCM-specific error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())