"""
Document Ingestion CLI.

Usage::

    # Ingest all files from raw_docs/ into data/ingested/ (local backend)
    python scripts/ingest_documents.py

    # Specify custom paths
    python scripts/ingest_documents.py --input raw_docs/papers_pdf --output data/ingested

    # Use MinerU backend (falls back to local on failure)
    python scripts/ingest_documents.py --backend mineru

    # Force local backend
    python scripts/ingest_documents.py --backend local
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.ingestion.document_ingestor import (
    ingest_file,
    ingest_directory,
    print_ingest_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the ResearchAgent RAG knowledge base."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="raw_docs",
        help="Input file or directory (default: raw_docs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ingested",
        help="Output directory for ingested markdown (default: data/ingested)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["local", "mineru"],
        help="Ingestion backend: local (default) or mineru (optional). "
             "Overrides DOCUMENT_INGESTION_BACKEND env var.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        result = ingest_file(input_path, output_dir, backend=args.backend)
        print_ingest_summary([result])
    elif input_path.is_dir():
        results = ingest_directory(input_path, output_dir, backend=args.backend)
        print_ingest_summary(results)
    else:
        print(f"Error: input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
