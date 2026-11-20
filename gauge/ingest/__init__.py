"""Ingestion: connectors for web novel sites plus local imports, and the incremental runner."""
from .runner import IngestRunner, build_sources, source_for_url

__all__ = ["IngestRunner", "build_sources", "source_for_url"]
