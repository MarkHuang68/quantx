# aggregator
try:
    from .simulated import SimulatedProvider  # existing module in your repo
except Exception:
    class SimulatedProvider:  # type: ignore
        pass
try:
    from .bybit import BybitProvider
except Exception:
    class BybitProvider:  # type: ignore
        pass

__all__ = ["SimulatedProvider", "BybitProvider"]
