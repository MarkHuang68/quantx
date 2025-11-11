# -*- coding: utf-8 -*-
"""
SMC 特徵計算器 (Smart Money Concepts Feature Calculator)
[SMC課程 最終優化版本 v4.0 - 導入 CHoCH 結構反轉]

這個模組提供了用於計算一系列基於「聰明錢概念」(SMC) 的技術分析特徵的函數。
本版本已根據全系列 SMC 課程 (2NcmMfHyizM, _osqlRalupM, FYNGQudjf_k, v7hxiTpVRPw, 9vL4lmQM_Hs) 
的SMC細節規則進行了優化。

[優化點 v4.0]:
1.  find_choch (新): 新增函數，標記「SMC結構反轉 (CHoCH)」。
2.  add_smc_state_features (重大升級): 
    - S-M-C 狀態機現在「同時」由 BOS (SMC順勢) 或 CHoCH (SMC反轉) 觸發。
    - 使 ML 模型能同時學習「SMC順勢」和「SMC反轉」兩種SMC進場模型。
"""

import pandas as pd
import numpy as np

# --- 1. 基礎 SMC 特徵 (向量化) ---

def find_swing_highs_lows(df: pd.DataFrame, window: int = 3):
    """[向量化] 尋找波段高點和低點 (SMC 基礎)。"""
    df['swing_high'] = df['high'].rolling(window*2 + 1, center=True, min_periods=window*2+1).max() == df['high']
    df['swing_low'] = df['low'].rolling(window*2 + 1, center=True, min_periods=window*2+1).min() == df['low']
    return df

def find_break_of_structure(df: pd.DataFrame):
    """[向量化 & 邏輯修正] 識別市場結構的破壞 (BOS - 順勢)。"""
    if 'swing_high' not in df.columns:
        df = find_swing_highs_lows(df) 

    swing_high_price = df['high'].where(df['swing_high'])
    swing_low_price = df['low'].where(df['swing_low'])

    last_swing_high = swing_high_price.ffill().shift(1)
    last_swing_low = swing_low_price.ffill().shift(1)

    breaks_high = df['close'] > last_swing_high
    breaks_low = df['close'] < last_swing_low

    first_break_high = breaks_high & ~breaks_high.shift(1).fillna(False)
    first_break_low = breaks_low & ~breaks_low.shift(1).fillna(False)

    df['bos_bullish'] = np.where(first_break_high, last_swing_high, np.nan) 
    df['bos_bearish'] = np.where(first_break_low, last_swing_low, np.nan) 
    
    df['bos_bullish_idx'] = np.where(first_break_high, df.index, pd.NaT)
    df['bos_bearish_idx'] = np.where(first_break_low, df.index, pd.NaT)

    return df

def find_choch(df: pd.DataFrame):
    """
    [SMC課程 v4.0 新增] 識別市場慣性改變 (CHoCH - 反轉)。
    CHoCH 是「SMC結構反轉」的第一個SMC訊號。
    
    [SMC定義]:
    - 看漲 CHoCH: 市場處於「SMC下降趨勢」(LL, LH)，價格「首次」突破了「最後一個波段高點」(Last LH)。
    - 看跌 CHoCH: 市場處於「SMC上升趨勢」(HH, HL)，價格「首次」突破了「最後一個波段低點」(Last HL)。
    """
    if 'swing_high' not in df.columns:
        df = find_swing_highs_lows(df)
        
    swing_high_price = df['high'].where(df['swing_high'])
    swing_low_price = df['low'].where(df['swing_low'])
    
    last_swing_high = swing_high_price.ffill().shift(1)
    last_swing_low = swing_low_price.ffill().shift(1)
    
    # 找出SMC趨勢 (簡化SMC邏輯：比較前一個高/低點)
    prev_swing_high = swing_high_price.ffill().shift(2)
    prev_swing_low = swing_low_price.ffill().shift(2)
    
    is_uptrend = (last_swing_high > prev_swing_high) & (last_swing_low > prev_swing_low)
    is_downtrend = (last_swing_high < prev_swing_high) & (last_swing_low < prev_swing_low)
    
    # S-M-C 偵測 CHoCH (反轉)
    # 1. 看跌 CHoCH: 處於「SMC上升趨勢」，且「SMC收盤價」跌破「SMC最後一個波段低點」
    breaks_low = df['close'] < last_swing_low
    is_bearish_choch = is_uptrend & breaks_low & ~breaks_low.shift(1).fillna(False)
    
    # 2. 看漲 CHoCH: 處於「SMC下降趨勢」，且「SMC收盤價」漲破「SMC最後一個波段高點」
    breaks_high = df['close'] > last_swing_high
    is_bullish_choch = is_downtrend & breaks_high & ~breaks_high.shift(1).fillna(False)

    df['choch_bullish'] = np.where(is_bullish_choch, last_swing_high, np.nan)
    df['choch_bearish'] = np.where(is_bearish_choch, last_swing_low, np.nan)
    
    df['choch_bullish_idx'] = np.where(is_bullish_choch, df.index, pd.NaT)
    df['choch_bearish_idx'] = np.where(is_bearish_choch, df.index, pd.NaT)
    
    return df

def find_order_blocks(df: pd.DataFrame):
    """
    SMC 訂單塊 (Order Block) 定義：
    - 看漲 OB：前一根是「下跌蠟燭」，當前是「上漲蠟燭」，且「收盤 > 前一根開盤」
    - 看跌 OB：前一根是「上漲蠟燭」，當前是「下跌蠟燭」，且「收盤 < 前一根開盤」
    """
    # 蠟燭方向
    prev_down = (df['close'].shift(1) < df['open'].shift(1))
    prev_up = (df['close'].shift(1) > df['open'].shift(1))
    curr_up = (df['close'] > df['open'])
    curr_down = (df['close'] < df['open'])

    # 強勢移動：收盤突破前一根實體
    bullish_move = curr_up & (df['close'] > df['open'].shift(1))
    bearish_move = curr_down & (df['close'] < df['open'].shift(1))

    bullish_ob = prev_down & curr_up & bullish_move
    bearish_ob = prev_up & curr_down & bearish_move

    df['bullish_ob_top'] = np.where(bullish_ob, df['high'], np.nan)
    df['bullish_ob_bottom'] = np.where(bullish_ob, df['low'], np.nan)
    df['bearish_ob_top'] = np.where(bearish_ob, df['high'], np.nan)
    df['bearish_ob_bottom'] = np.where(bearish_ob, df['low'], np.nan)

    return df

def find_fair_value_gaps(df: pd.DataFrame):
    """[向量化] 尋找公允價值缺口 (FVG)。"""
    bullish_fvg_cond = (df['high'].shift(2) < df['low'])
    bearish_fvg_cond = (df['low'].shift(2) > df['high'])

    bullish_fvg_mask = bullish_fvg_cond.shift(-1).fillna(False)
    bearish_fvg_mask = bearish_fvg_cond.shift(-1).fillna(False)

    df['bullish_fvg_top'] = np.where(bullish_fvg_mask, df['low'].shift(-1), np.nan)
    df['bullish_fvg_bottom'] = np.where(bullish_fvg_mask, df['high'].shift(1), np.nan)
    df['bearish_fvg_top'] = np.where(bearish_fvg_mask, df['low'].shift(1), np.nan)
    df['bearish_fvg_bottom'] = np.where(bearish_fvg_mask, df['high'].shift(-1), np.nan)

    return df

# smc_features.py → 完整替換 find_high_quality_ob
def find_high_quality_ob(df: pd.DataFrame) -> pd.DataFrame:
    """
    [SMC v5.0 階層式 POI 系統] - 完整版
    產生所有 4 個 hq_ob 欄位 + poi + feat_has_hq_ob
    """
    # --- 1. 嚴格 HQ-OB：OB 必須與 FVG 重疊 ---
    # 看漲 HQ-OB
    bullish_hq = df['bullish_ob_top'].notna() & df['bullish_fvg_top'].notna()
    df['hq_bullish_ob_top'] = np.where(bullish_hq, df['bullish_ob_top'], np.nan)
    df['hq_bullish_ob_bottom'] = np.where(bullish_hq, df['bullish_ob_bottom'], np.nan)

    # 看跌 HQ-OB
    bearish_hq = df['bearish_ob_top'].notna() & df['bearish_fvg_top'].notna()
    df['hq_bearish_ob_top'] = np.where(bearish_hq, df['bearish_ob_top'], np.nan)
    df['hq_bearish_ob_bottom'] = np.where(bearish_hq, df['bearish_ob_bottom'], np.nan)

    # --- 2. POI 階層：優先 HQ-OB → 再用 OB ---
    df['poi_bullish'] = df['hq_bullish_ob_top'].fillna(df['bullish_ob_top'])
    df['poi_bearish'] = df['hq_bearish_ob_bottom'].fillna(df['bearish_ob_bottom'])

    # --- 3. 品質特徵：是否有 HQ-OB ---
    df['feat_has_hq_ob'] = (
        df['hq_bullish_ob_top'].notna() | 
        df['hq_bullish_ob_bottom'].notna() |
        df['hq_bearish_ob_top'].notna() | 
        df['hq_bearish_ob_bottom'].notna()
    ).astype(int)

    return df

# smc_features.py → find_order_flow 函數
def find_order_flow(df: pd.DataFrame) -> pd.DataFrame:
    """[SMC課程新增] 尋找「訂單流」 (OF) (SMC 規則: 連續2根K線)。"""
    is_down_candle = (df['close'] < df['open'])
    is_up_candle = (df['close'] > df['open'])
    is_2_down = is_down_candle.shift(1) & is_down_candle.shift(2)
    is_2_up = is_up_candle.shift(1) & is_up_candle.shift(2)
    strong_move_up = (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    strong_move_down = (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
    
    bullish_of_mask = is_2_down & strong_move_up
    bullish_of_mask = bullish_of_mask.shift(-1).fillna(False)
    
    bearish_of_mask = is_2_up & strong_move_down
    bearish_of_mask = bearish_of_mask.shift(-1).fillna(False)

    # 修正：用 rolling 取前兩根 high/low 的最大/小值
    high_prev2 = df['high'].shift(1).rolling(window=2, min_periods=1).max()
    low_prev2 = df['low'].shift(1).rolling(window=2, min_periods=1).min()

    df['bullish_of_top'] = np.where(bullish_of_mask, high_prev2, np.nan)
    df['bullish_of_bottom'] = np.where(bullish_of_mask, low_prev2, np.nan)
    df['bearish_of_top'] = np.where(bearish_of_mask, high_prev2, np.nan)
    df['bearish_of_bottom'] = np.where(bearish_of_mask, low_prev2, np.nan)

    return df


# --- 2. 核心 SMC 狀態機 (SMC 課程 v4.1 - ML 狀態優化) ---

def add_smc_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [SMC課程 v4.1 - ML 狀態優化] 創建 ML "狀態" 特徵 (SMC BOS + CHoCH 邏輯)。
    
    此SMC狀態機模擬SMC交易SOP，並為 ML 模型提供「SMC狀態」特徵。
    
    [SMC v4.1 優化點]:
    1.  [SMC 狀態特徵] 新增 'feat_smc_state_...' (0=中立, 1=看漲SMC, -1=看跌SMC)，
        讓SMC模型知道SMC當前處於哪個SMC SOP 階段。
    2.  [SMC 中立值] 將 'feat_dist_...' 的SMC默認值從 9999.0 (SMC易混淆) 改為 0.0 (SMC中立)。
    
    SMC SOP 流程:
    1.  [BOS] / [CHoCH] 發生 (SMC趨勢確認)。
    2.  [Target] 建立。
    3.  [IDM] 建立。
    4.  [Extreme] (HQ-OB) 建立。
    5.  [Grab IDM] (價格突破 IDM) -> 啟動 Extreme POI 追蹤。
    6.  [Hit Extreme] (價格觸及 Extreme) -> 啟動 Target 追蹤。
    7.  [Hit Target] (價格觸及 Target) -> 重置SMC狀態機。
    """
    
    if 'hq_bullish_ob_top' not in df.columns or 'bos_bullish_idx' not in df.columns or 'choch_bullish_idx' not in df.columns:
        raise ValueError("請先運行所有 'find_' 函數 (包括 find_choch())。")

    # --- (SMC v4.1) 初始化 ML 特徵欄位 ---
    
    # (SMC v4.1) 距離特徵: 預設為 0.0 (SMC中立)
    df['feat_dist_to_IDM'] = 0.0
    df['feat_dist_to_EXTREME'] = 0.0
    df['feat_dist_to_TARGET'] = 0.0
    
    # (SMC v4.1) 狀態特徵 (ML 關鍵): 0=中立, 1=看漲SMC, -1=看跌SMC
    df['feat_smc_state_IDM'] = 0
    df['feat_smc_state_EXTREME'] = 0
    df['feat_smc_state_TARGET'] = 0

    # --- SMC 狀態機變數 ---
    trend_state = 0         # 0=中立, 1=看漲, -1=看跌
    external_target = np.nan
    idm_level = np.nan
    extreme_poi = np.nan
    idm_taken = False
    entry_active = False
    
    # v4.0 變數：追蹤SMC訊號的來源
    last_bullish_signal_idx = pd.NaT
    last_bearish_signal_idx = pd.NaT

    for i in range(len(df)):
        
        # --- 1. 偵測SMC趨SMC變化 (BOS 或 CHoCH) ---
        
        is_new_bullish_signal = False
        if df['bos_bullish'].iloc[i] > 0:
            is_new_bullish_signal = True
            last_bullish_signal_idx = df.index[i]
        elif df['choch_bullish'].iloc[i] > 0:
            is_new_bullish_signal = True
            last_bullish_signal_idx = df.index[i]
            
        is_new_bearish_signal = False
        if df['bos_bearish'].iloc[i] > 0:
            is_new_bearish_signal = True
            last_bearish_signal_idx = df.index[i]
        elif df['choch_bearish'].iloc[i] > 0:
            is_new_bearish_signal = True
            last_bearish_signal_idx = df.index[i]

        # ---
        
        if is_new_bullish_signal:
            trend_state = 1
            external_target = np.nan # 重置SMC目標
            idm_level = np.nan       # 重置SMC IDM
            extreme_poi = np.nan     # 重置SMC POI
            idm_taken = False        # 重置SMC IDM 狀態
            entry_active = False     # 重置SMC進場狀態
            
            # 使用 poi_bullish 作為 Extreme POI
            recent_poi_idx = df['poi_bullish'].iloc[:i+1].last_valid_index()
            if recent_poi_idx is not None and recent_poi_idx < last_bullish_signal_idx:
                extreme_poi = df.at[recent_poi_idx, 'poi_bullish']
            
            continue

        if is_new_bearish_signal:
            trend_state = -1
            external_target = np.nan
            idm_level = np.nan
            extreme_poi = np.nan
            idm_taken = False
            entry_active = False

            recent_poi_idx = df['poi_bearish'].iloc[:i+1].last_valid_index()
            if recent_poi_idx is not None and recent_poi_idx < last_bearish_signal_idx:
                extreme_poi = df.at[recent_poi_idx, 'poi_bearish']

            continue

        # --- 2. SMC 狀態機運行 (看漲趨勢) ---
        if trend_state == 1:
            
            # 狀態 1: 尋找 Target
            if np.isnan(external_target) and df['swing_high'].iloc[i]:
                if df.index[i] > last_bullish_signal_idx:
                    external_target = df['high'].iloc[i]

            # 狀態 2: 尋找 IDM
            elif not np.isnan(external_target) and np.isnan(idm_level) and df['swing_low'].iloc[i]:
                idm_level = df['low'].iloc[i]

            # 狀態 3: 追蹤 IDM (SMC陷阱)
            if not np.isnan(idm_level) and not idm_taken:
                df.loc[df.index[i], 'feat_dist_to_IDM'] = df['low'].iloc[i] - idm_level
                df.loc[df.index[i], 'feat_smc_state_IDM'] = 1 # SMC v4.1: 啟動看漲 IDM 狀態
            
            # 狀態 4: 檢查 IDM 是否被「抓取」
            if not idm_taken and not np.isnan(idm_level):
                if df['low'].iloc[i] < idm_level:
                    idm_taken = True 
            
            # 狀態 5: 追蹤 Extreme (SMC真正進場點)
            if idm_taken and not entry_active and not np.isnan(extreme_poi):
                df.loc[df.index[i], 'feat_dist_to_EXTREME'] = df['low'].iloc[i] - extreme_poi
                df.loc[df.index[i], 'feat_smc_state_EXTREME'] = 1 # SMC v4.1: 啟動看漲 Extreme 狀態
                
                if df['low'].iloc[i] <= extreme_poi:
                    entry_active = True 
                    extreme_poi = np.nan  
            
            # 狀態 6: 追蹤 Target (SMC出場點)
            if entry_active and not np.isnan(external_target):
                df.loc[df.index[i], 'feat_dist_to_TARGET'] = df['high'].iloc[i] - external_target
                df.loc[df.index[i], 'feat_smc_state_TARGET'] = 1 # SMC v4.1: 啟動看漲 Target 狀態
                
                if df['high'].iloc[i] >= external_target:
                    # SMC SOP 循環完成，重置所有SMC狀態
                    trend_state = 0
                    external_target = np.nan
                    idm_level = np.nan
                    extreme_poi = np.nan
                    idm_taken = False
                    entry_active = False

        # --- 3. SMC 狀態機運行 (看跌趨勢) ---
        elif trend_state == -1:
            
            # 狀態 1: 尋找 Target
            if np.isnan(external_target) and df['swing_low'].iloc[i]:
                if df.index[i] > last_bearish_signal_idx:
                    external_target = df['low'].iloc[i]

            # 狀態 2: 尋找 IDM
            elif not np.isnan(external_target) and np.isnan(idm_level) and df['swing_high'].iloc[i]:
                idm_level = df['high'].iloc[i]

            # 狀態 3: 追蹤 IDM (SMC陷阱)
            if not np.isnan(idm_level) and not idm_taken:
                df.loc[df.index[i], 'feat_dist_to_IDM'] = idm_level - df['high'].iloc[i]
                df.loc[df.index[i], 'feat_smc_state_IDM'] = -1 # SMC v4.1: 啟動看跌 IDM 狀態
            
            # 狀態 4: 檢查 IDM 是否被「抓取」
            if not idm_taken and not np.isnan(idm_level):
                if df['high'].iloc[i] > idm_level:
                    idm_taken = True 
            
            # 狀態 5: 追蹤 Extreme (SMC真正進場點)
            if idm_taken and not entry_active and not np.isnan(extreme_poi):
                df.loc[df.index[i], 'feat_dist_to_EXTREME'] = extreme_poi - df['high'].iloc[i]
                df.loc[df.index[i], 'feat_smc_state_EXTREME'] = -1 # SMC v4.1: 啟動看跌 Extreme 狀態
                
                if df['high'].iloc[i] >= extreme_poi:
                    entry_active = True 
                    extreme_poi = np.nan  
            
            # 狀態 6: 追蹤 Target (SMC出場點)
            if entry_active and not np.isnan(external_target):
                df.loc[df.index[i], 'feat_dist_to_TARGET'] = external_target - df['low'].iloc[i]
                df.loc[df.index[i], 'feat_smc_state_TARGET'] = -1 # SMC v4.1: 啟動看跌 Target 狀態
                
                if df['low'].iloc[i] <= external_target:
                    # SMC SOP 循環完成，重置所有SMC狀態
                    trend_state = 0
                    external_target = np.nan
                    idm_level = np.nan
                    extreme_poi = np.nan
                    idm_taken = False
                    entry_active = False
        
    # (SMC v4.1) 清理因 np.nan 運算 (e.g., price - np.nan) 產生的 NaN 值
    df.fillna({
        'feat_dist_to_IDM': 0.0, 
        'feat_dist_to_EXTREME': 0.0, 
        'feat_dist_to_TARGET': 0.0,
        'feat_smc_state_IDM': 0,
        'feat_smc_state_EXTREME': 0,
        'feat_smc_state_TARGET': 0
    }, inplace=True)
               
    return df
        
    # 清理 NaN 值
    df.fillna({'feat_dist_to_IDM': 9999.0, 
               'feat_dist_to_EXTREME': 9999.0, 
               'feat_dist_to_TARGET': 9999.0}, inplace=True)
               
    return df