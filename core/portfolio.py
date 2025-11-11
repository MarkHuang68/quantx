# 檔案: core/portfolio.py

import pandas as pd

class Portfolio:
    def __init__(self, initial_capital, exchange):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.exchange = exchange
        # 倉位結構: {symbol: {'long': position, 'short': position}}
        # position 字典: {'contracts': float, 'entry_price': float, 'leverage': int, 'liquidation_price': float}
        self.positions = {}
        self.history = []

    def sync_with_exchange(self, exchange_positions):
        """
        用交易所的期貨倉位數據更新 Portfolio。
        :param exchange_positions: 來自 exchange.get_positions() 的倉位列表。
        """
        self.positions.clear() # 清空舊倉位
        for pos in exchange_positions:
            symbol = pos['symbol']
            side = pos['side'] # 'long' or 'short'

            if symbol not in self.positions:
                self.positions[symbol] = {
                    'long': self._get_empty_position(),
                    'short': self._get_empty_position()
                }

            self.positions[symbol][side] = {
                'contracts': float(pos['contracts']),
                'entry_price': float(pos['entryPrice']),
                'leverage': float(pos['leverage']),
                'liquidation_price': float(pos.get('liquidationPrice', 0))
            }

        print("--- Portfolio 倉位同步完成 ---")
        print(self.get_positions_summary())

    def _get_empty_position(self):
        """返回一個空的倉位結構。"""
        return {'contracts': 0, 'entry_price': 0, 'leverage': 0, 'liquidation_price': 0}

    def update_position(self, symbol, trade_side, amount, price, leverage):
        """
        (重構版) 在紙上交易中更新倉位，處理開/平/反向倉位，並返回已實現盈虧。
        :param trade_side: 'buy' or 'sell'
        :return: realized_pnl (float)
        """
        realized_pnl = 0
        if symbol not in self.positions:
            self.positions[symbol] = {
                'long': self._get_empty_position(),
                'short': self._get_empty_position()
            }

        long_pos = self.positions[symbol]['long']
        short_pos = self.positions[symbol]['short']

        if trade_side == 'buy':
            # 可能是開多倉，也可能是平空倉
            if short_pos['contracts'] > 0:
                # 優先平空倉
                close_amount = min(amount, short_pos['contracts'])
                realized_pnl = (short_pos['entry_price'] - price) * close_amount

                short_pos['contracts'] -= close_amount
                amount -= close_amount
                print(f"倉位更新: {symbol} [平空] {close_amount:.4f} @ {price}, 已實現盈虧: {realized_pnl:.2f} USDT")

                if short_pos['contracts'] == 0:
                    self.positions[symbol]['short'] = self._get_empty_position()

            if amount > 0:
                # 剩餘數量開多倉 (或加倉)
                self._update_side_position('long', symbol, amount, price, leverage)

        elif trade_side == 'sell':
            # 可能是開空倉，也可能是平多倉
            if long_pos['contracts'] > 0:
                # 優先平多倉
                close_amount = min(amount, long_pos['contracts'])
                realized_pnl = (price - long_pos['entry_price']) * close_amount

                long_pos['contracts'] -= close_amount
                amount -= close_amount
                print(f"倉位更新: {symbol} [平多] {close_amount:.4f} @ {price}, 已實現盈虧: {realized_pnl:.2f} USDT")

                if long_pos['contracts'] == 0:
                    self.positions[symbol]['long'] = self._get_empty_position()

            if amount > 0:
                # 剩餘數量開空倉 (或加倉)
                self._update_side_position('short', symbol, amount, price, leverage)

        return realized_pnl

    def _update_side_position(self, side, symbol, amount, price, leverage):
        """輔助函數，用於開倉或加倉特定方向的倉位。"""
        position = self.positions[symbol][side]
        current_contracts = position['contracts']
        current_avg_price = position['entry_price']

        # 計算新的平均價格 (加倉)
        new_total_contracts = current_contracts + amount
        new_avg_price = ((current_avg_price * current_contracts) + (price * amount)) / new_total_contracts

        position['contracts'] = new_total_contracts
        position['entry_price'] = new_avg_price
        position['leverage'] = leverage

        # --- 計算爆倉價格 ---
        from settings import MAINTENANCE_MARGIN_RATE
        if side == 'long':
            liquidation_price = new_avg_price * (1 - (1 / leverage) + MAINTENANCE_MARGIN_RATE)
        else: # short
            liquidation_price = new_avg_price * (1 + (1 / leverage) - MAINTENANCE_MARGIN_RATE)

        position['liquidation_price'] = liquidation_price
        log_action = "加倉" if current_contracts > 0 else "開倉"
        print(f"倉位更新: {symbol} [{log_action}{'多' if side == 'long' else '空'}] {amount:.4f} @ {price} | 新均價: {new_avg_price:.2f}, 總量: {new_total_contracts:.4f}, 爆倉價: {liquidation_price:.2f}")

    def close_position(self, symbol, side):
        """在紙上交易中手動清空一個倉位 (例如，在爆倉時)。"""
        if symbol in self.positions and side in self.positions[symbol]:
            self.positions[symbol][side] = self._get_empty_position()
            print(f"通知: {symbol} 的 {side} 倉位已被強制平倉。")

    def sync_spot_positions(self, spot_positions):
        """
        用交易所的現貨倉位數據更新 Portfolio (簡化版)。
        :param spot_positions: 一個 {asset: amount} 的字典。
        """
        self.positions.clear()
        for asset, amount in spot_positions.items():
            if amount > 0:
                symbol = f"{asset}/USDT" # 假設都以 USDT 計價
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'long': self._get_empty_position(),
                        'short': self._get_empty_position()
                    }
                # 現貨只有多頭倉位
                self.positions[symbol]['long']['contracts'] = float(amount)

        print("--- Portfolio 現貨倉位同步完成 ---")
        print(self.get_positions_summary())

    def update_position_from_trade(self, trade):
        """
        根據一筆已執行的交易來更新倉位。
        """
        symbol = trade['symbol']
        side = trade['side']  # 'buy' or 'sell'
        amount = trade['filled']
        price = trade['price']
        cost = trade['cost']
        fee = trade.get('fee', {}).get('cost', 0)

        self.cash -= fee

        # 確定是開多/平空 還是 開空/平多
        position_side = 'long' if side == 'buy' else 'short'

        if symbol not in self.positions:
            self.positions[symbol] = {
                'long': {'contracts': 0, 'entry_price': 0},
                'short': {'contracts': 0, 'entry_price': 0}
            }

        current_position = self.positions[symbol][position_side]

        # 這裡需要更複雜的邏輯來處理是開倉還是平倉，
        # 但對於 Bybit 雙向持倉，API 會直接告知是 long 還是 short
        # 所以我們主要信任 sync_with_exchange
        print(f"注意：update_position_from_trade 尚未完全實現，依賴 sync。")


    async def update(self, dt):
        """
        更新投資組合的總價值 (非同步版本)。
        """
        # 實盤中，我們更信任交易所回傳的總權益
        try:
            balance_info = await self.exchange.get_balance()
            total_value = balance_info.get('total', self.cash) # 使用 total equity
            self.cash = balance_info.get('free', self.cash) # 更新可用現金
        except Exception as e:
            print(f"無法從交易所獲取餘額，將本地計算價值: {e}")
            total_value = self.cash

            # 加上所有倉位的未實現盈虧
            for symbol, sides in self.positions.items():
                latest_price = await self.exchange.get_latest_price(symbol)
                if not latest_price: continue

                # 多頭倉位
                long_pos = sides['long']
                if long_pos['contracts'] > 0:
                    unrealized_pnl = (latest_price - long_pos['entry_price']) * long_pos['contracts']
                    total_value += (long_pos['contracts'] * long_pos['entry_price']) + unrealized_pnl

                # 空頭倉位
                short_pos = sides['short']
                if short_pos['contracts'] > 0:
                    unrealized_pnl = (short_pos['entry_price'] - latest_price) * short_pos['contracts']
                    total_value += (short_pos['contracts'] * short_pos['entry_price']) + unrealized_pnl

        self.history.append({'timestamp': dt, 'total_value': total_value})

    def get_positions(self):
        return self.positions

    def get_positions_summary(self):
        """返回一個易於閱讀的倉位摘要。"""
        summary = "當前倉位:\n"
        if not self.positions:
            return summary + "  (空倉)"

        for symbol, sides in self.positions.items():
            if sides['long']['contracts'] > 0:
                pos = sides['long']
                summary += f"  - {symbol} [多頭]: {pos['contracts']} @ {pos['entry_price']}\n"
            if sides['short']['contracts'] > 0:
                pos = sides['short']
                summary += f"  - {symbol} [空頭]: {pos['contracts']} @ {pos['entry_price']}\n"
        return summary.strip()

    def get_total_value(self):
        if not self.history:
            return self.initial_capital
        return self.history[-1]['total_value']
