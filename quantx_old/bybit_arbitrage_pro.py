# bybit_arbitrage_pro_safe_v3.py
import ccxt
import time
import logging
from datetime import datetime

# ========= 0) 參數 =========
API_KEY = "8BxkMF4Cms1uJryKgh"            # 實單請填
API_SECRET = "6Pa1VfDluGx9zqXXHx6XiRijOgeaKHUz47Uu"         # 實單請填
USE_TESTNET = True
DRY_RUN = False          # ← 先模擬；要實單改 False

START_USDT = 150            # 每輪投入
MIN_PROFIT_RATE = 0.004     # 0.4% 淨利
SAFETY_PAD = 0.002          # 額外安全墊 0.2%
CHECK_INTERVAL = 0.6
MAX_RETRIES = 3
OB_LIMIT = 50               # order book 深度
SLACK_SHRINK = 0.002        # 逐腿縮量 0.2%

PAIRS = {
    "USDT→BTC": "BTC/USDT",   # quote: USDT
    "BTC→ETH":  "ETH/BTC",    # quote: BTC
    "ETH→USDT": "ETH/USDT",   # quote: USDT
}

# ========= 1) 日誌 =========
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tri-arb")

# ========= 2) 交易所 =========
exchange = ccxt.bybit({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
    "timeout": 20000,
})
exchange.set_sandbox_mode(USE_TESTNET)
markets = exchange.load_markets()

for s in PAIRS.values():
    if s not in markets:
        raise RuntimeError(f"交易對不存在或不支援現貨: {s}")

def taker_fee(symbol: str) -> float:
    return float(markets[symbol].get("taker", 0.001))  # 取不到用 0.1%

def get_limits(symbol: str):
    m = markets[symbol]
    limits = m.get("limits", {}) or {}
    amount_min = (limits.get("amount", {}) or {}).get("min", 0.0) or 0.0
    cost_min = (limits.get("cost", {}) or {}).get("min", 0.0) or 0.0
    return float(amount_min), float(cost_min)

def amount_prec(symbol: str, amount: float) -> float:
    return float(exchange.amount_to_precision(symbol, amount))

def price_prec(symbol: str, price: float) -> float:
    return float(exchange.price_to_precision(symbol, price))

def fetch_free(ccy: str) -> float:
    try:
        bal = exchange.fetch_balance()
        return float((bal.get("free", {}) or {}).get(ccy, 0.0))
    except Exception:
        return float("inf") if DRY_RUN else 0.0

def print_pair_limits():
    for name, sym in PAIRS.items():
        m = markets[sym]
        amt_min = (m.get("limits", {}).get("amount", {}) or {}).get("min", None)
        cost_min = (m.get("limits", {}).get("cost", {}) or {}).get("min", None)
        prec = m.get("precision", {})
        log.info(f"[limits] {name} {sym} | minAmt={amt_min} | minCost={cost_min} | precision={prec}")

def get_orderbooks_snapshot():
    """一次抓三腿 orderbook，減少時間差"""
    obs = {}
    for _, sym in PAIRS.items():
        last_err = None
        for _ in range(MAX_RETRIES):
            try:
                obs[sym] = exchange.fetch_order_book(sym, limit=OB_LIMIT)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.2)
        if sym not in obs:
            log.error(f"OrderBook 取得失敗 {sym}: {last_err}")
            return None
    return obs

def executable_avg_from_ob(ob, side: str, quota: float):
    """
    從單一 orderbook 計算可成交均價。
    - side='buy': quota 是欲花的 quote 金額 → 回 (avg_px, base_filled, quote_used)
    - side='sell': quota 是欲賣出的 base 數量 → 回 (avg_px, base_sold, quote_got)
    """
    if not ob: return None, 0.0, 0.0
    levels = ob["asks"] if side == "buy" else ob["bids"]
    if not levels: return None, 0.0, 0.0

    filled_base, used_quote = 0.0, 0.0
    if side == "buy":
        need_quote = quota
        for price, lvl_base in levels:
            if need_quote <= 0: break
            lvl_quote = price * lvl_base
            take_quote = min(lvl_quote, need_quote)
            take_base = take_quote / price
            filled_base += take_base
            used_quote += take_quote
            need_quote -= take_quote
        if filled_base <= 0: return None, 0.0, 0.0
        avg_px = used_quote / filled_base
        return avg_px, filled_base, used_quote
    else:
        need_base = quota
        for price, lvl_base in levels:
            if need_base <= 0: break
            take_base = min(lvl_base, need_base)
            take_quote = take_base * price
            filled_base += take_base
            used_quote += take_quote
            need_base -= take_base
        if filled_base <= 0: return None, 0.0, 0.0
        avg_px = used_quote / filled_base
        return avg_px, filled_base, used_quote

def ensure_minima(symbol: str, raw_amount: float, avg_price: float):
    """
    將下單數量補到 minAmt / minCost 之上，再做 precision。
    回傳 (amount_precised, ok_bool, detail_msg)
    """
    amt_min, cost_min = get_limits(symbol)
    target = max(raw_amount, amt_min or 0.0)
    if cost_min and cost_min > 0 and avg_price and avg_price > 0:
        need_by_cost = (cost_min / avg_price) * 1.02  # +2% 緩衝
        target = max(target, need_by_cost)
    amt = amount_prec(symbol, target)
    if avg_price and cost_min and amt * avg_price < cost_min:
        tweak = (cost_min / avg_price) * 1.03
        amt = amount_prec(symbol, max(amt, tweak))
    ok = True
    reason = ""
    if amt <= 0:
        ok, reason = False, "amount 轉 precision 後為 0"
    elif amt < (amt_min or 0):
        ok, reason = False, f"amount<{amt_min}"
    elif avg_price and cost_min and amt * avg_price < cost_min:
        ok, reason = False, f"cost({amt*avg_price})<minCost({cost_min})"
    return amt, ok, reason

def debug_limits(symbol: str, amount: float, avg_price: float, tag: str):
    amt_min, cost_min = get_limits(symbol)
    cost = (amount * avg_price) if (avg_price and amount) else 0.0
    log.info(f"[limits] {tag} {symbol} | amt={amount} | avg={avg_price} | cost≈{cost} | "
             f"minAmt={amt_min} | minCost={cost_min}")

def required_start_usdt_lower_bound() -> float:
    """
    以 minCost 反推本輪最低 START_USDT，含手續費與縮量緩衝。
    回: 需要的最低 USDT（不足則建議調高 START_USDT）
    """
    s1 = PAIRS["USDT→BTC"]   # quote USDT
    s2 = PAIRS["BTC→ETH"]    # quote BTC
    s3 = PAIRS["ETH→USDT"]   # quote USDT

    m1, m2, m3 = markets[s1], markets[s2], markets[s3]
    fee1 = float(m1.get("taker", 0.001))
    fee2 = float(m2.get("taker", 0.001))
    fee3 = float(m3.get("taker", 0.001))
    minCost1 = (m1.get("limits", {}).get("cost", {}) or {}).get("min", 0.0) or 0.0  # USDT
    minCost2 = (m2.get("limits", {}).get("cost", {}) or {}).get("min", 0.0) or 0.0  # BTC
    minCost3 = (m3.get("limits", {}).get("cost", {}) or {}).get("min", 0.0) or 0.0  # USDT

    obs = get_orderbooks_snapshot()
    if not obs:
        return 0.0
    ob1, ob2, ob3 = obs[s1], obs[s2], obs[s3]
    p1 = ob1["asks"][0][0] if ob1["asks"] else None
    p2 = ob2["asks"][0][0] if ob2["asks"] else None
    p3 = ob3["bids"][0][0] if ob3["bids"] else None
    if not all([p1, p2, p3]):
        return 0.0

    need1 = float(minCost1)

    need2 = 0.0
    if minCost2 > 0:
        need2 = (minCost2 * p1) / ((1 - fee1) * (1 - SLACK_SHRINK)) * 1.02

    need3 = 0.0
    if minCost3 > 0:
        btc_budget_needed = (minCost3 / (p3 * (1 - fee3))) * p2 / (1 - fee2)
        start_needed = (btc_budget_needed * p1) / ((1 - fee1) * (1 - SLACK_SHRINK))
        need3 = start_needed * 1.02

    return max(need1, need2, need3)

def simulate_and_maybe_trade():
    """
    流程：USDT→BTC（買），BTC→ETH（買），ETH→USDT（賣）
    使用同輪 orderbook 估價與滑點 + 三腿 taker 手續費。
    下單前檢查：三腿門檻 + 各腿預算上限。
    """
    # 開倉前先檢查起始資金是否足夠跨過三腿門檻
    req_usdt = required_start_usdt_lower_bound()
    if req_usdt and START_USDT + 1e-8 < req_usdt:
        log.info(f"起始 {START_USDT} USDT 低於所需下限 {req_usdt:.2f}，本輪跳過。")
        return 0.0

    obs = get_orderbooks_snapshot()
    if obs is None:
        return 0.0

    # 1) USDT→BTC（買）：用 USDT 花掉 START_USDT
    s1 = PAIRS["USDT→BTC"]
    fee1 = taker_fee(s1)
    ob1 = obs[s1]
    avg1, btc_amt, usdt_used = executable_avg_from_ob(ob1, "buy", START_USDT)
    if not avg1 or btc_amt <= 0:
        return 0.0
    btc_after_fee = btc_amt * (1 - fee1)

    # 2) BTC→ETH（買）：用 BTC 換 ETH（關鍵：門檻 & 預算）
    s2 = PAIRS["BTC→ETH"]
    fee2 = taker_fee(s2)
    ob2 = obs[s2]
    btc_budget = btc_after_fee * (1 - SLACK_SHRINK)                 # 你可花的 BTC 上限
    amt_min_2, cost_min_2 = get_limits(s2)                          # minCost 以 BTC 計

    # 預算若無法跨越 minCost，直接跳過，避免 170140
    if btc_budget + 1e-12 < cost_min_2:
        need_usdt = (cost_min_2 * avg1) / ((1 - fee1) * (1 - SLACK_SHRINK))
        log.info(f"BTC 預算不足以跨過 ETH/BTC minCost。"
                 f"btc_budget={btc_budget:.8f}, minCost2={cost_min_2}, "
                 f"建議 START_USDT ≥ {need_usdt:.2f}（目前 {START_USDT}）")
        return 0.0

    # 先估在預算下可買到的 ETH（含滑點）
    avg2, eth_amt, btc_used_sim = executable_avg_from_ob(ob2, "buy", btc_budget)
    if not avg2 or eth_amt <= 0:
        return 0.0

    # 門檻下限（把 ETH 數量墊到達到 minCost 的最低量），但不得超過預算能買到的上限
    eth_by_min_cost = (cost_min_2 / avg2) * 1.02    # +2%
    eth_by_budget   = btc_budget / avg2
    target_eth      = max(eth_by_min_cost, amt_min_2 or 0.0)
    target_eth      = min(target_eth, eth_by_budget)

    if target_eth <= 0:
        log.error("BTC→ETH 不可行：門檻與預算無交集，跳過本輪。")
        return 0.0

    eth_after_fee_sim = eth_amt * (1 - fee2)   # 模擬用
    # 3) ETH→USDT（賣）：預算上限 = 你手上的 ETH（模擬或實際），同時要跨過 minCost(USDT)
    s3 = PAIRS["ETH→USDT"]
    fee3 = taker_fee(s3)
    ob3 = obs[s3]

    # 用模擬的 eth_after_fee_sim 當第 3 腿最大可賣量
    sell_eth_quota_sim = eth_after_fee_sim * (1 - SLACK_SHRINK)
    avg3, eth_sold_sim, usdt_back_sim = executable_avg_from_ob(ob3, "sell", sell_eth_quota_sim)
    if not avg3 or eth_sold_sim <= 0:
        return 0.0
    usdt_after_fee_sim = usdt_back_sim * (1 - fee3)

    profit = usdt_after_fee_sim - START_USDT
    rate = profit / START_USDT
    log.info(f"估算: p1≈{avg1:.8f} | p2≈{avg2:.8f} | p3≈{avg3:.4f} | 淨利率 {rate:.3%}")

    if rate < (MIN_PROFIT_RATE + SAFETY_PAD):
        return 0.0

    # ===== 觸發 =====
    if DRY_RUN:
        log.info(f"✅ 觸發（模擬）| 預期利潤 {profit:+.2f} USDT")
        return profit

    # --- 實單 1) 買 BTC（用 USDT） ---
    # 實際下單的 base 量：以估得 btc_amt 為基礎，縮量 0.2%，並確保不超過 START_USDT 預算
    max_btc_by_budget = START_USDT / avg1
    target_btc = min(btc_amt * (1 - SLACK_SHRINK), max_btc_by_budget)
    buy_btc_amt, ok1, reason1 = ensure_minima(s1, target_btc, avg1)
    # 仍需確保名目不超過預算（保守再 min 一次）
    if buy_btc_amt * avg1 > START_USDT:
        buy_btc_amt = amount_prec(s1, max_btc_by_budget * (1 - 1e-6))
    debug_limits(s1, buy_btc_amt, avg1, "leg1")
    if not ok1 or buy_btc_amt <= 0:
        log.error(f"USDT→BTC 不符最小交易額/精度：{reason1}")
        return 0.0
    try:
        o1 = exchange.create_market_buy_order(s1, buy_btc_amt)
    except Exception as e:
        log.error(f"下單失敗 leg1: {e}")
        return 0.0
    got_btc = buy_btc_amt * (1 - fee1)

    # --- 實單 2) 買 ETH（用 BTC） ---
    btc_budget = got_btc * (1 - SLACK_SHRINK)
    amt_min_2, cost_min_2 = get_limits(s2)
    if btc_budget + 1e-12 < cost_min_2:
        need_usdt = (cost_min_2 * avg1) / ((1 - fee1) * (1 - SLACK_SHRINK))
        log.info(f"BTC 預算不足以跨過 ETH/BTC minCost。"
                 f"btc_budget={btc_budget:.8f}, minCost2={cost_min_2}, "
                 f"建議 START_USDT ≥ {need_usdt:.2f}（目前 {START_USDT}）")
        return 0.0
    # 預算與門檻夾擊
    eth_by_budget = btc_budget / avg2
    eth_by_min_cost = (cost_min_2 / avg2) * 1.02
    target_eth = max(eth_by_min_cost, amt_min_2 or 0.0)
    target_eth = min(target_eth, eth_by_budget)
    if target_eth <= 0:
        log.error("BTC→ETH 不可行：門檻與預算無交集，跳過本輪。")
        return 0.0
    buy_eth_amt = amount_prec(s2, target_eth)
    notional_btc_2 = buy_eth_amt * avg2
    log.info(f"[preorder] leg2 {s2} | amt={buy_eth_amt} | avg~{avg2} | "
             f"notional(BTC)≈{notional_btc_2} | minCost={cost_min_2} | budget={btc_budget}")
    if notional_btc_2 + 1e-12 < cost_min_2 or notional_btc_2 > btc_budget:
        log.error(f"BTC→ETH 不符條件 | notional(BTC)={notional_btc_2:.8f}, "
                  f"minCost={cost_min_2}, budget={btc_budget:.8f}")
        return 0.0
    debug_limits(s2, buy_eth_amt, avg2, "leg2")
    try:
        o2 = exchange.create_market_buy_order(s2, buy_eth_amt)
    except Exception as e:
        log.error(f"下單失敗 leg2: {e}")
        return 0.0
    got_eth = buy_eth_amt * (1 - fee2)

    # --- 實單 3) 賣 ETH 換 USDT ---
    s3 = PAIRS["ETH→USDT"]
    amt_min_3, cost_min_3 = get_limits(s3)  # cost_min_3 以 USDT 計
    # 預算上限：你手上的 ETH
    sell_eth_cap = got_eth * (1 - SLACK_SHRINK)
    # 門檻下限：為了達到 minCost(USDT)，需要的 ETH
    need_eth_for_min_cost = 0.0
    if cost_min_3 > 0 and avg3 > 0:
        need_eth_for_min_cost = (cost_min_3 / avg3) * 1.02
    target_sell_eth = max(need_eth_for_min_cost, amt_min_3 or 0.0)
    target_sell_eth = min(target_sell_eth, sell_eth_cap)  # 不可超過你擁有的
    if target_sell_eth <= 0:
        log.error("ETH→USDT 不可行：門檻與預算無交集，跳過本輪。")
        return 0.0
    sell_eth_amt = amount_prec(s3, target_sell_eth)
    notional_usdt_3 = sell_eth_amt * avg3
    log.info(f"[preorder] leg3 {s3} | amt={sell_eth_amt} | avg~{avg3} | "
             f"notional(USDT)≈{notional_usdt_3} | minCost={cost_min_3} | have={sell_eth_cap}")
    if notional_usdt_3 + 1e-12 < cost_min_3 or sell_eth_amt > sell_eth_cap:
        log.error(f"ETH→USDT 不符條件 | notional={notional_usdt_3:.8f}, "
                  f"minCost={cost_min_3}, have_eth={sell_eth_cap:.8f}")
        return 0.0
    debug_limits(s3, sell_eth_amt, avg3, "leg3")
    try:
        o3 = exchange.create_market_sell_order(s3, sell_eth_amt)
    except Exception as e:
        log.error(f"下單失敗 leg3: {e}")
        return 0.0

    final_usdt = sell_eth_amt * avg3 * (1 - fee3)
    realized = final_usdt - START_USDT
    log.info(f"✅ 觸發（實單）| 實得約 {realized:+.2f} USDT")
    return realized

# ========= 3) 主程序 =========
if __name__ == "__main__":
    print_pair_limits()
    log.info(f"Bybit 三角套利 Bot 啟動 | 每次套利={START_USDT} | 門檻={MIN_PROFIT_RATE:.1%} "
             f"| 安全墊={SAFETY_PAD:.1%} | 測試網={USE_TESTNET} | DRY_RUN={DRY_RUN}")

    total_profit = 0.0
    daily_profit = 0.0
    last_day = datetime.now().day

    while True:
        try:
            p = simulate_and_maybe_trade()
            if p > 0:
                total_profit += p
                daily_profit += p
                log.info(f"當日累計：{daily_profit:.2f} USDT | 總計：{total_profit:.2f} USDT")

            if datetime.now().day != last_day:
                log.info(f"昨日收益：{daily_profit:.2f} USDT")
                daily_profit = 0.0
                last_day = datetime.now().day

            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log.info(f"\nBot 已停止 | 總收益：{total_profit:.2f} USDT")
            break
        except Exception as e:
            log.error(f"未知錯誤: {e}")
            time.sleep(2)
