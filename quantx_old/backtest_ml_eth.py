# backtest_ml_eth_pro.py
import time
from datetime import datetime, timedelta, timezone

import ccxt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ======= 基本參數 =======
SYMBOL 	  = "ETH/USDT"
TIMEFRAME = "15m"
DAYS 	  = 90
EXCHANGE  = "bybit"
SEED 	  = 42

# Horizon 對齊 1h
TF_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60}.get(TIMEFRAME, 5)
HORIZON_MIN = 60
FUTURE = max(1, HORIZON_MIN // TF_MIN) # 12 根 K 棒 (12 * 5min = 60min)

# 回測參數
INITIAL_USDT   = 400
LEVERAGE 	   = 2.0
FEE_BPS 	   = 10
SLIP_BPS 	   = 4
RISK_PER_TRADE = 0.35
MIN_SIZE 	   = 0.01
MAX_HOLD_BARS  = FUTURE
ATR_LEN 	   = 14
ATR_SL 		   = 2.0 	# 保留 ATR 停損

# Walk-Forward
N_FOLDS = 4

# ======= 取數（分頁） =======
exchange = getattr(ccxt, EXCHANGE)({"enableRateLimit": True, "options": {"defaultType":"spot"}})

def fetch_ohlcv_paginated(symbol, timeframe, since_ms, until_ms, limit_per=1000, sleep=0.07):
	out, cur = [], since_ms
	print(f"Fetching data from {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)} to {datetime.fromtimestamp(until_ms/1000, tz=timezone.utc)}")
	while True:
		try:
			batch = exchange.fetch_ohlcv(symbol, timeframe, since=cur, limit=limit_per)
			if not batch:
				print("  No more data returned from exchange. Fetch complete.")
				break
			
			out += batch
			last_ts = batch[-1][0]
			print(f"  Fetched {len(batch)} bars, last ts: {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)}")
			
			if last_ts >= until_ms:
				print("  Reached target end time. Fetch complete.")
				break
			
			cur = last_ts + 1 
			time.sleep(sleep)
		except Exception as e:
			print(f"Error fetching OHLCV: {e}, retrying...")
			time.sleep(5)
	
	uniq = {r[0]: r for r in out}
	rows = [uniq[k] for k in sorted(uniq)]
	return rows

def load_ohlcv_days(symbol, timeframe, days):
	now = datetime.now(timezone.utc)
	since = int((now - timedelta(days=days+1)).timestamp()*1000)
	until = int(now.timestamp()*1000)
	raw = fetch_ohlcv_paginated(symbol, timeframe, since, until)
	if not raw: raise RuntimeError("取不到 OHLCV")
	df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
	df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
	df = df[(df['ts'] >= since) & (df['ts'] <= until)]
	df.set_index("timestamp", inplace=True)
	print(f"Total bars loaded: {len(df)}. Data range: {df.index.min()} to {df.index.max()}")
	return df[["open","high","low","close","volume"]]

df = load_ohlcv_days(SYMBOL, TIMEFRAME, DAYS)

# ======= 特徵 =======
def rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
def rsi(close, n=14):
	d = close.diff()
	up, dn = d.clip(lower=0), (-d).clip(lower=0)
	rs = rma(up,n)/ (rma(dn,n)+1e-12)
	return 100 - 100/(1+rs)

def compute_features(df):
	o = df.copy()
	o["ret1"] = o.close.pct_change()
	o["vol20"] = o.ret1.rolling(20).std() * np.sqrt(60*24*365/TF_MIN)
	o["rsi14"] = rsi(o.close,14)
	o["ema12"] = o.close.ewm(span=12, adjust=False).mean()
	o["ema26"] = o.close.ewm(span=26, adjust=False).mean()
	o["ema_up"] = (o.ema12 > o.ema26).astype(int) # 順勢特徵
	o["ma20"]  = o.close.rolling(20).mean()
	o["bb_std"]= o.close.rolling(20).std()
	o["bb_pos"]= (o.close - o.ma20) / (o.bb_std + 1e-12)
	o["bb_bw"] = (2*o.bb_std) / (o.ma20.abs()+1e-12)
	o["bar_pos"]= (o.close - o.low) / (o.high - o.low + 1e-12)
	
	tr = np.maximum(o.high-o.low, np.maximum((o.high-o.close.shift()).abs(), (o.low-o.close.shift()).abs()))
	o["atr"] = rma(tr, ATR_LEN)
	
	o.dropna(inplace=True)
	return o

df = compute_features(df)
if len(df) < 2000:
	print(f"警告：數據量可能仍然不足 ({len(df)} bars), 請檢查 DAYS 或交易所限制。")

# ======= 標籤 (Y) 定義 =======
# <--- 變更：方向三 (改回原始的順勢標籤)
print("Using Direction 3: Pro-Trend Label (y = future > close)")
df["future"] = df.close.shift(-FUTURE)
df["y"] = (df.future > df.close).astype(int)
df.dropna(inplace=True)
# <--- 變更結束

FEATURES = ["rsi14","vol20","ema_up","bb_pos","bb_bw","bar_pos"]
X_all = df[FEATURES].values
y_all = df["y"].values
p_all = df["close"].values
atr_all = df["atr"].values
ts_all = df.index.to_numpy()

# ======= 工具：策略回測（多空） =======
# (此函數保持為「方向一」的版本，使用 ATR 停損和 ML 出場)
def backtest(proba, price, side_mask, atr,
			 open_th, close_th=0.5,
			 initial_usdt=INITIAL_USDT, leverage=LEVERAGE,
			 fee_bps=FEE_BPS, slip_bps=SLIP_BPS,
			 risk=RISK_PER_TRADE, min_size=MIN_SIZE,
			 max_hold=MAX_HOLD_BARS, atr_sl=ATR_SL):
	fee = fee_bps/1e4; slip = slip_bps/1e4
	eq=[]
	
	cash = initial_usdt
	mtm_equity = initial_usdt
	pos = 0.0; entry=0.0; side=0; hold=0; trades=0

	for i in range(len(price)):
		px=float(price[i]); pr=float(proba[i]); m=int(side_mask[i]); a=float(atr[i])
		
		if side == 0:
			mtm_equity = cash
		else:
			margin = (pos * entry) / leverage
			unrealized_pnl = side * pos * (px - entry)
			mtm_equity = cash + margin + unrealized_pnl
		eq.append(mtm_equity)

		# 平倉條件
		if side!=0 and pos>0:
			exit_px = px*(1 - slip*side)
			margin = (pos*entry)/leverage
			fee_out = exit_px*pos*fee
			
			sl_atr_hit = (side==1 and px <= entry - (a * atr_sl)) or (side==-1 and px >= entry + (a * atr_sl))
			ml_exit = (side==1 and pr < close_th) or (side==-1 and pr > (1-close_th))
			time_exit = (hold>=max_hold)
			
			do_close = (sl_atr_hit or ml_exit or time_exit)
			
			if do_close:
				pnl = side*pos*(exit_px - entry)
				cash += margin + pnl - fee_out
				pos=0.0; side=0; entry=0.0; hold=0

		# 開倉
		if side==0 and m!=0:
			if (m==1 and pr>open_th) or (m==-1 and pr<(1-open_th)):
				budget = cash*risk
				in_px = px*(1 + slip*m)
				size = (budget*leverage)/in_px
				size = np.floor(size/min_size)*min_size
				margin_cost = (size*in_px)/leverage
				fee_in = (in_px*size*fee)
				total_cost = margin_cost + fee_in
				
				if size >= min_size and total_cost <= cash:
					cash -= total_cost
					pos=size; side=m; entry=in_px; trades+=1; hold=0

		if side!=0: hold+=1

	if side!=0 and pos>0:
		px=float(price[-1])
		margin = (pos*entry)/leverage
		unrealized_pnl = side*pos*(px - entry)
		mtm_equity = cash + margin + unrealized_pnl
		eq[-1] = mtm_equity
	
	final_equity = eq[-1]
	
	eq=np.array(eq,dtype=float)
	if len(eq)<3:
		return {"final":final_equity,"profit":final_equity-initial_usdt,"roi_pct":(final_equity/initial_usdt-1)*100,
				"sharpe":0.0,"mdd_pct":0.0,"trades":trades}
				
	rets=np.diff(eq)/ (eq[:-1]+1e-12)
	sharpe=(np.mean(rets)/(np.std(rets)+1e-12))*np.sqrt(24 * 365)
	peak=np.maximum.accumulate(eq); mdd=((eq-peak)/ (peak+1e-12)).min()*100
	return {"final":float(final_equity),"profit":float(final_equity-initial_usdt),"roi_pct":float((final_equity/initial_usdt-1)*100),
			"sharpe":float(sharpe),"mdd_pct":float(mdd),"trades":int(trades)}

# ======= 濾網 (Side Mask) =======
# <--- 變更：方向三 (改為順勢進場訊號)
print("Using Direction 3: Pro-Trend Entry (EMA Trend + RSI 50 Cross)")
# 趨勢濾網
ema_trend_up = (df["ema_up"] == 1)
ema_trend_down = (df["ema_up"] == 0)

# 動能訊號：RSI 剛穿越 50
rsi = df["rsi14"]
rsi_cross_up = (rsi > 50) & (rsi.shift(1).fillna(50) <= 50)
rsi_cross_down = (rsi < 50) & (rsi.shift(1).fillna(50) >= 50)

# 最終訊號：趨勢 + 動能
long_mask = (ema_trend_up & rsi_cross_up).values.astype(int)
short_mask = (ema_trend_down & rsi_cross_down).values.astype(int)

side_mask_long  = long_mask*1
side_mask_short = (short_mask* -1)
# <--- 變更結束

# ======= Walk-Forward + 機率校正 + 門檻尋優 =======
def wf_splits(n, n_folds=N_FOLDS):
	idx = np.arange(n)
	fold_size = n // (n_folds + 1)
	if fold_size < 100:
		print(f"警告：Fold size 過小 ({fold_size})，數據總量 ({n}) 不足。")
	for k in range(1, n_folds + 1):
		split = fold_size * k
		train_idx = idx[:split]
		test_idx  = idx[split: split + fold_size]
		if len(test_idx) < 30:
			continue
		yield train_idx, test_idx


def fit_xgb(X_tr, y_tr, X_va, y_va):
	pos = float(y_tr.mean()) if len(y_tr) else 0.5
	spw = ((1-pos)/(pos+1e-12)) if pos>0 else 1.0
	model = XGBClassifier(
		n_estimators=800, max_depth=4, learning_rate=0.03,
		subsample=0.9, colsample_bytree=0.9,
		reg_lambda=1.2, min_child_weight=2.0,
		tree_method="hist", objective="binary:logistic",
		random_state=SEED, scale_pos_weight=spw,
		eval_metric="auc", early_stopping_rounds=60, n_jobs=0
	)
	model.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_va,y_va)], verbose=False)
	return model

def tune_threshold(proba, price, side_mask, atr, opens):
	best_th, best_score = 0.55, -1e18
	for th in opens:
		res = backtest(proba, price, side_mask, atr, open_th=th)
		score = res["sharpe"]
		if score > best_score:
			best_score, best_th = score, th
	return best_th, best_score

def ensure_min_trades(proba, price, side_mask, atr, th, min_trades=8):
	cur = th
	while True:
		res = backtest(proba, price, side_mask, atr, open_th=cur)
		if res["trades"] >= min_trades or cur<=0.505:
			return cur
		cur -= 0.01

results=[]
for tr_idx, te_idx in wf_splits(len(df), N_FOLDS):
	X_tr, X_te = X_all[tr_idx], X_all[te_idx]
	y_tr, y_te = y_all[tr_idx], y_all[te_idx]
	p_tr, p_te = p_all[tr_idx], p_all[te_idx]
	atr_tr, atr_te = atr_all[tr_idx], atr_all[te_idx]
	sideL_tr, sideL_te = side_mask_long[tr_idx],  side_mask_long[te_idx]
	sideS_tr, sideS_te = side_mask_short[tr_idx], side_mask_short[te_idx]

	scaler = StandardScaler()
	X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)

	# 模型 + 校正（Isotonic）
	model = fit_xgb(X_tr, y_tr, X_te, y_te)
	proba_tr_raw = model.predict_proba(X_tr)[:,1]
	proba_te_raw = model.predict_proba(X_te)[:,1] 

	iso = IsotonicRegression(out_of_bounds="clip")
	iso.fit(proba_tr_raw, y_tr)
	proba_tr = iso.transform(proba_tr_raw)
	proba_te = iso.transform(proba_te_raw)

	# 長/短分開尋優門檻
	thL,_ = tune_threshold(proba_tr, p_tr, sideL_tr, atr_tr, opens=np.linspace(0.50,0.68,19))
	thS,_ = tune_threshold(1-proba_tr, p_tr, sideS_tr, atr_tr, opens=np.linspace(0.50,0.68,19))

	# 確保交易數
	thL = ensure_min_trades(proba_te, p_te, sideL_te, atr_te, thL, min_trades=6)
	thS = ensure_min_trades(1-proba_te, p_te, sideS_te, atr_te, thS, min_trades=6)

	# 測試段回測
	btL = backtest(proba_te, p_te, sideL_te, atr_te, open_th=thL)
	btS = backtest(1-proba_te, p_te, sideS_te, atr_te, open_th=thS)
	
	final = {
		"fold_start": str(ts_all[te_idx[0]]), "fold_end": str(ts_all[te_idx[-1]]),
		"L_final":btL["final"], "L_profit":btL["profit"], "L_sharpe":btL["sharpe"], "L_mdd":btL["mdd_pct"], "L_tr":btL["trades"], "L_th":thL,
		"S_final":btS["final"], "S_profit":btS["profit"], "S_sharpe":btS["sharpe"], "S_mdd":btS["mdd_pct"], "S_tr":btS["trades"], "S_th":thS,
	}
	results.append(final)

# 匯總
sumL = {"profit":0,"tr":0}; sumS={"profit":0,"tr":0}
for r in results:
	sumL["profit"]+=r["L_profit"]; sumL["tr"]+=r["L_tr"]
	sumS["profit"]+=r["S_profit"]; sumS["tr"]+=r["S_tr"]

print("\n=== Walk-Forward 摘要（多空分開） ===")
for i,r in enumerate(results,1):
	print(f"[Fold {i}] {r['fold_start']} ~ {r['fold_end']} | "
		  f"Long PnL {r['L_profit']:+.2f} ({r['L_tr']}tr, sh={r['L_sharpe']:.2f}, mdd={r['L_mdd']:.1f}%, th={r['L_th']:.3f})  ||  "
		  f"Short PnL {r['S_profit']:+.2f} ({r['S_tr']}tr, sh={r['S_sharpe']:.2f}, mdd={r['S_mdd']:.1f}%, th={r['S_th']:.3f})")

print("\n合計：")
print(f"Long 總收益: {sumL['profit']:+.2f} USDT | 交易數: {sumL['tr']}")
print(f"Short 總收益: {sumS['profit']:+.2f} USDT | 交易數: {sumS['tr']}")
print(f"\n數據範圍: {df.index.min()} to {df.index.max()} (共 {len(df)} 根 K 棒)")
print("\n提示：RSI 策略的交易訊號可能偏少。若交易數過少，可考慮在 ensure_min_trades 降低 min_trades；"
	  "若交易過多，可嘗試提高門檻搜尋範圍 (tune_threshold) 或縮小 PCT_SL/PCT_TP。")