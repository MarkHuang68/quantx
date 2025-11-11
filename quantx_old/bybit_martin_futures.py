# bybit_martin_futures_CORRECTED.py
import ccxt
import time
import logging
from datetime import datetime

# ==================== 1. Bybit U本位期貨設定 ====================
exchange = ccxt.bybit({
    'apiKey': 'amYOf9q40Iz7kKMWwD',      # 填入你的
    'secret': 'MBcaFWRdhQoONpwp46M5t8maf8tYLSXfp551',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',     # 正確：U本位永續合約
        'adjustForTimeDifference': True,
    },
})

# 測試網（首次建議開啟）
# exchange.set_sandbox_mode(True)

# ==================== 2. 參數 ====================
SYMBOL = 'BTCUSDT'
LEVERAGE = 3
BASE_ORDER_USDT = 100
SAFETY_ORDERS = 3
MARTIN_RATIO = 1.5
TAKE_PROFIT_PCT = 0.025
STOP_LOSS_PCT = 0.08
CHECK_INTERVAL = 15

# ==================== 3. 全局狀態 ====================
position = {'side': None, 'size': 0, 'entry': 0, 'orders': []}

# ==================== 4. 日誌 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger()

# ==================== 5. 工具函數 ====================
def get_price():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return ticker['last']
    except Exception as e:
        log.error(f"價格錯誤: {e}")
        return None

def set_leverage():
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        log.info(f"槓桿設定: {LEVERAGE}x")
    except Exception as e:
        log.warning(f"槓桿設定失敗: {e}")

def set_position_mode():
    try:
        exchange.set_position_mode(False, SYMBOL)
        log.info("倉位模式: 單向模式")
    except Exception as e:
        log.warning(f"模式設定失敗: {e}")

def place_order(side, usdt_amount):
    global position  # 必須在最前面
    try:
        price = get_price()
        if not price: return None

        contracts = round(usdt_amount * LEVERAGE / price, 3)
        if contracts < 0.001: contracts = 0.001

        order = exchange.create_order(
            symbol=SYMBOL,
            type='market',
            side=side,
            amount=contracts
        )
        log.info(f"{side.upper()} {contracts} 張 @ {price:.2f}")
        return {'price': price, 'contracts': contracts}
    except Exception as e:
        log.error(f"下單失敗: {e}")
        return None

def close_position():
    global position  # 必須在最前面
    if not position['side']: return
    try:
        side = 'sell' if position['side'] == 'long' else 'buy'
        exchange.create_order(SYMBOL, 'market', side, position['size'])
        current_price = get_price()
        profit = (current_price - position['entry']) * position['size'] if position['side'] == 'long' else (position['entry'] - current_price) * position['size']
        log.info(f"平倉成功！盈虧: {profit:+.2f} USDT")
        position = {'side': None, 'size': 0, 'entry': 0, 'orders': []}
    except Exception as e:
        log.error(f"平倉失敗: {e}")

# ==================== 6. 馬丁邏輯 ====================
def martin_strategy():
    global position  # 必須在最前面
    price = get_price()
    if not price: return

    if position['side']:
        unrealized = (price - position['entry']) * position['size']
        log.info(f"持倉中 | 浮動盈虧: {unrealized:+.2f} USDT | 目前價: {price}")

    # 無持倉 → 開多
    if not position['side']:
        order = place_order('buy', BASE_ORDER_USDT)
        if order:
            position.update({
                'side': 'long',
                'size': order['contracts'],
                'entry': order['price'],
                'orders': [order]
            })
            set_leverage()
        return

    # 持倉中
    pnl_pct = (price - position['entry']) / position['entry']

    # 止盈
    if pnl_pct >= TAKE_PROFIT_PCT:
        close_position()
        return

    # 止損
    if pnl_pct <= -STOP_LOSS_PCT:
        close_position()
        return

    # 加倉
    if len(position['orders']) < SAFETY_ORDERS:
        last_price = position['orders'][-1]['price']
        drop = (last_price - price) / last_price
        if drop >= 0.015:
            amount = BASE_ORDER_USDT * (MARTIN_RATIO ** len(position['orders']))
            order = place_order('buy', amount)
            if order:
                position['orders'].append(order)
                position['size'] += order['contracts']
                total_value = sum(o['price'] * o['contracts'] for o in position['orders'])
                position['entry'] = total_value / position['size']

def restore_position():
    global position
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if abs(pos['contracts']) > 0:  # 有持倉（abs 處理 short 負值）
                side = pos['side']
                entry = float(pos['entryPrice'])
                size = abs(float(pos['contracts']))
                
                if side == 'short':
                    log.warning(f"偵測到 short 倉位 ({size} 張)，但策略僅支援 long，自動平倉")
                    close_side = 'buy'
                    exchange.create_order(SYMBOL, 'market', close_side, size)
                    log.info("Short 倉位已平倉")
                    position = {'side': None, 'size': 0, 'entry': 0, 'orders': []}
                    return
                
                # 僅處理 long
                log.info(f"恢復 long 倉位: {size} 張 @ {entry:.2f}")
                
                # 獲取最近交易記錄重建 orders
                trades = exchange.fetch_my_trades(SYMBOL, limit=20)
                relevant_trades = [t for t in trades if t['side'] == 'buy']
                relevant_trades.sort(key=lambda t: t['timestamp'])  # 按時間升序
                
                total_trade_size = sum(t['amount'] for t in relevant_trades)
                
                if abs(total_trade_size - size) < 0.001:  # 容忍小誤差
                    orders = [{'price': float(t['price']), 'contracts': float(t['amount'])} for t in relevant_trades]
                    log.info(f"成功重建 {len(orders)} 筆訂單")
                else:
                    log.warning(f"交易總量 {total_trade_size:.3f} 不匹配倉位 {size:.3f}，使用簡化模式")
                    orders = [{'price': entry, 'contracts': size}]
                
                # 驗證平均 entry（可選，作為除錯）
                calc_entry = sum(o['price'] * o['contracts'] for o in orders) / size if size > 0 else 0
                log.info(f"計算平均 entry: {calc_entry:.2f} (交易所: {entry:.2f})")
                
                position = {
                    'side': side,
                    'size': size,
                    'entry': entry,  # 使用交易所平均價
                    'orders': orders
                }
                return
        
        log.info("無持倉，從零開始")
    except Exception as e:
        log.error(f"恢復持倉失敗: {e}")
        position = {'side': None, 'size': 0, 'entry': 0, 'orders': []}

# ==================== 7. 主循環 ====================
if __name__ == "__main__":
    log.info("Bybit U本位期貨馬丁 Bot 啟動！")
    log.info(f"基礎單: {BASE_ORDER_USDT} USDT | 加倉 {SAFETY_ORDERS} 次 | 止盈 {TAKE_PROFIT_PCT:.1%}")
    set_leverage()
    set_position_mode()        # 強制單向
    restore_position()

    while True:
        try:
            martin_strategy()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log.info("Bot 已停止")
            break
        except Exception as e:
            log.error(f"未知錯誤: {e}")
            time.sleep(5)