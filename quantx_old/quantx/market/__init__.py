"""Market interfaces.

This subpackage exposes market providers which supply raw market data
and trade execution channels.  In this environment network access to
real exchanges is unavailable, so only a simulated provider is
implemented.  Users can extend :class:`quantx.market.provider.base.MarketProvider`
to implement support for real exchanges via ccxt or other SDKs.
"""

from . import provider, stream  # noqa: F401

__all__ = ["provider", "stream"]