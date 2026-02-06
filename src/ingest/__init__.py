"""Data ingestion clients for prediction market platforms."""

from src.ingest.polymarket import PolymarketClient
from src.ingest.metaculus import MetaculusClient
from src.ingest.manifold import ManifoldClient

__all__ = ["PolymarketClient", "MetaculusClient", "ManifoldClient"]
