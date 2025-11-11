# quantx/core/orderbook_utils.py
# 版本: v2 (修正：Maker Price 放置位置與精度)
# 說明:
# - 修正 calculate_optimal_maker_price 邏輯，確保 Maker 價格放置在 BBO 之外一檔，
#   並使用正確的精度進行捨入。

from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# 定義 Order Book 快照的類型結構：List[List[Price: float, Size: float]]
OrderBookSide = List[List[float]]

def get_bbo(orderbook_snapshot: Optional[Dict[str, OrderBookSide]]) -> Optional[Tuple[float, float]]:
    """
    從訂單簿快照中獲取最佳買價 (Best Bid) 和最佳賣價 (Best Offer)。
    """
    if orderbook_snapshot is None:
        return None
        
    bids = orderbook_snapshot.get('bids')
    asks = orderbook_snapshot.get('asks')

    best_bid = bids[0][0] if bids and len(bids) > 0 and bids[0] and len(bids[0]) > 0 else 0.0
    best_ask = asks[0][0] if asks and len(asks) > 0 and asks[0] and len(asks[0]) > 0 else 0.0

    if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
        return None
        
    return best_bid, best_ask

def calculate_optimal_maker_price(
    orderbook_snapshot: Optional[Dict[str, List[List[float]]]],
    side: str,
    instrument_precision: float = 0.01  # 最小價格精度
) -> Optional[float]:
    """
    計算最佳的 Maker 掛單價格（即掛在 BBO 旁邊一檔的價格）。
    
    Args:
        instrument_precision: 該交易對的最小價格變動單位 (Tick Size)。
    """
    bbo = get_bbo(orderbook_snapshot)
    if bbo is None:
        return None

    best_bid, best_ask = bbo

    precision = max(instrument_precision, 1e-8)
    
    if side == 'buy':
        # Maker Buy: 掛在 Best Bid 的下方一檔
        optimal_price = best_bid - precision

    elif side == 'sell':
        # Maker Sell: 掛在 Best Ask 的上方一檔
        optimal_price = best_ask + precision
    else:
        return None

    # 修正：使用更可靠的 round 方式，確保精度正確
    decimal_places = 0
    if precision < 1.0:
         try:
              # 計算精度位數 (例如 0.01 -> 2; 0.1 -> 1)
              decimal_places = max(0, -int(np.floor(np.log10(precision))))
         except Exception:
              decimal_places = 8
             
    optimal_price = round(optimal_price, decimal_places)


    if optimal_price <= 0:
        return None

    return optimal_price

def calculate_obi(
    orderbook_snapshot: Optional[Dict[str, List[List[float]]]],
    depth: int
) -> float:
    """
    計算指定深度的訂單簿不平衡度 (OBI)。
    """
    if orderbook_snapshot is None:
        return 0.0 # 數據缺失時，視為中性

    bids: OrderBookSide = orderbook_snapshot.get('bids', [])
    asks: OrderBookSide = orderbook_snapshot.get('asks', [])

    # 1. 截取指定深度
    bids_D = bids[:depth]
    asks_D = asks[:depth]

    # 2. 計算 OBI (Order Book Imbalance)
    # level[1] 是 Size
    bid_size_sum = sum([level[1] for level in bids_D if len(level) > 1])
    ask_size_sum = sum([level[1] for level in asks_D if len(level) > 1])
    
    total_liquidity = bid_size_sum + ask_size_sum
    
    if total_liquidity > 0:
        # OBI = (Bid Size - Ask Size) / (Bid Size + Ask Size)
        obi = (bid_size_sum - ask_size_sum) / total_liquidity
        return obi
    else:
        return 0.0 # 零流動性時 OBI 為 0 (中性)