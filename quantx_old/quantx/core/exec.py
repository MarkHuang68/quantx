"""Order execution simulation.

In a real environment this module would route orders to an exchange via
REST or WebSocket APIs.  For the purposes of backtesting and simple
live simulations in this project we implement a very basic order
router that records orders and updates account state.  The router
applies slippage and fees and ensures orders adhere to price/quantity
constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Order:
    timestamp: str
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    qty: float
    fee: float
    slippage: float


@dataclass
class ExecutionEngine:
    """Simple execution engine for simulation and live stubs."""

    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    slippage_bps: float = 0.0
    orders: List[Order] = field(default_factory=list)

    def execute_order(
        self,
        timestamp: str,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        maker: bool = True,
    ) -> Tuple[float, float]:
        """Execute an order and record it.

        Returns the effective execution price and the fee amount.
        Price is adjusted by slippage and fee.
        """
        # Determine slippage direction: buy -> price increases, sell -> price decreases
        slip_direction = 1 if side.lower() == "buy" else -1
        price_slippage = price * (self.slippage_bps / 10000.0) * slip_direction
        exec_price = price + price_slippage
        fee_bps = self.maker_fee_bps if maker else self.taker_fee_bps
        fee = abs(qty) * exec_price * (fee_bps / 10000.0)
        # Record order
        order = Order(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            price=exec_price,
            qty=qty,
            fee=fee,
            slippage=price_slippage,
        )
        self.orders.append(order)
        return exec_price, fee