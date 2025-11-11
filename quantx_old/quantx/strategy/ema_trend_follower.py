# quantx/strategy/ema_trend_follower.py

from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor
from quantx.ta.indicators import CrossDown, CrossUp, HeikinAshi, SMA, EMA, RSI, BBands, MACD, ADX, StochRSI

class ema_trend_follower(BaseExecutor):
    """
    EMA 順勢追蹤策略

    核心邏輯：
    - 使用三條 EMA (短期、中期、長期) 的排列順序來判斷主要趨勢。
    - 使用 ADX 來過濾掉趨勢強度不足的市場。
    - 使用 EMA 的交叉作為主要的出場信號。
    - (可選) 使用 MACD 和 Stochastic RSI 作為額外的進場過濾器，以增加信號的可靠性。
    """

    def __init__(self, **kwargs):
        """
        初始化策略參數
        """
        super().__init__(**kwargs)

        # EMA 週期
        self.ema_short_period = int(self.params.get('ema_short_period', 9))
        self.ema_mid_period = int(self.params.get('ema_mid_period', 21))
        self.ema_long_period = int(self.params.get('ema_long_period', 50))

        # ADX 參數
        self.adx_period = int(self.params.get('adx_period', 14))
        self.adx_threshold = float(self.params.get('adx_threshold', 23.0))

        # 可選過濾器開關
        self.use_macd_filter = bool(self.params.get('use_macd_filter', False))
        self.use_stoch_rsi_filter = bool(self.params.get('use_stoch_rsi_filter', False))

        # MACD 參數 (如果啟用)
        if self.use_macd_filter:
            self.macd_fast = int(self.params.get('macd_fast', 12))
            self.macd_slow = int(self.params.get('macd_slow', 26))
            self.macd_signal = int(self.params.get('macd_signal', 9))

        # StochRSI 參數 (如果啟用)
        if self.use_stoch_rsi_filter:
            self.stoch_rsi_period = int(self.params.get('stoch_rsi_period', 14))
            self.stoch_rsi_k = int(self.params.get('stoch_rsi_k', 3))
            self.stoch_rsi_d = int(self.params.get('stoch_rsi_d', 3))

        # 策略自帶停損百分比
        self.default_stop_loss_pct = float(self.params.get('default_stop_loss_pct', 3.0))

    def on_bar(self, ctx: ContextBase):
        """
        核心交易邏輯
        """
        # ctx.executor.accepts_global_drawdown_action = True
        # ctx.executor.default_stop_loss_pct = 2
        klines = ctx.data(ctx.symbol, ctx.tf)

        # --- 1. 計算指標 ---
        # EMA
        klines[f'ema_short'] = EMA(klines.close, length=self.ema_short_period)
        klines[f'ema_mid'] = EMA(klines.close, length=self.ema_mid_period)
        klines[f'ema_long'] = EMA(klines.close, length=self.ema_long_period)

        # ADX
        adx = ADX(klines.high, klines.low, klines.close, length=self.adx_period)
        klines['adx'] = adx['ADX']

        # MACD (如果啟用)
        if self.use_macd_filter:
            macd = MACD(klines.close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            klines['macd'] = macd['MACD_macd']
            klines['macd_signal'] = macd['MACD_signal']

        # Stochastic RSI (如果啟用)
        if self.use_stoch_rsi_filter:
            stoch_rsi = StochRSI(klines.close, length=self.stoch_rsi_period, rsi_length=self.stoch_rsi_period, k=self.stoch_rsi_k, d=self.stoch_rsi_d)
            klines['stoch_k'] = stoch_rsi['StochRSI_k']
            klines['stoch_d'] = stoch_rsi['StochRSI_d']

        # EMA 交叉信號
        klines['ema_cross_up'] = CrossUp(klines.ema_short, klines.ema_mid)
        klines['ema_cross_down'] = CrossDown(klines.ema_short, klines.ema_mid)

        # 獲取最新一根 K棒 的數據
        latest = klines.iloc[-1]

        # --- 2. 進場邏輯 ---
        if ctx.position.is_flat():
            # 多頭進場條件
            long_trend_ok = latest.ema_short > latest.ema_mid > latest.ema_long
            trend_strength_ok = latest.adx > self.adx_threshold

            macd_ok_long = True
            if self.use_macd_filter:
                macd_ok_long = latest.macd > latest.macd_signal

            stoch_rsi_ok_long = True
            if self.use_stoch_rsi_filter:
                stoch_rsi_ok_long = latest.stoch_k > latest.stoch_d

            if long_trend_ok and trend_strength_ok and macd_ok_long and stoch_rsi_ok_long:
                # 傳遞 stop_loss_pct 參數以啟用策略自帶的停損
                ctx.open_long()
                return

            # 空頭進場條件
            short_trend_ok = latest.ema_short < latest.ema_mid < latest.ema_long

            macd_ok_short = True
            if self.use_macd_filter:
                macd_ok_short = latest.macd < latest.macd_signal

            stoch_rsi_ok_short = True
            if self.use_stoch_rsi_filter:
                stoch_rsi_ok_short = latest.stoch_k < latest.stoch_d

            if short_trend_ok and trend_strength_ok and macd_ok_short and stoch_rsi_ok_short:
                # 傳遞 stop_loss_pct 參數以啟用策略自帶的停損
                ctx.open_short()
                return

        # --- 3. 出場邏輯 ---
        if ctx.position.is_long() and latest.ema_cross_down:
            ctx.close_long()
            return

        if ctx.position.is_short() and latest.ema_cross_up:
            ctx.close_short()
            return
