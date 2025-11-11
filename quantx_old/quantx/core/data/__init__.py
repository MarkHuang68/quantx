"""Data access layer.

This subpackage contains utilities for loading, caching and persisting
market data.  At the heart of the system is the concept of a 1‑minute
"base" time series which is the single source of truth for all
aggregated timeframes.  The loader will first attempt to read data
from the on‑disk cache, and will only call the provider when
necessary.
"""

from . import loader  # noqa: F401

__all__ = ["loader"]