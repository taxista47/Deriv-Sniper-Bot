import json, time, sys
import numpy as np
import pandas as pd
import websocket
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===========================
# CONFIGURATION
# ===========================
# ✅ KHTAR B-RA7TEK HNA: "CANDLES" (Chmou3) awla "LINE" (Khat)
CHART_TYPE = "CANDLES"  

SYMBOL = "R_100"          
GRANULARITY = 900         # 15 Minutes
CANDLES = 500             
REFRESH_SEC = 5           

APP_ID = 1089
API_TOKEN = "YOUR_TOKEN_HERE" 
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# PARAMETRES SNIPER
BIN_SIZE = 2.0
TOP_LEVELS = 10
LEFT_RIGHT = 5
LEVEL_TOL = 2.5
RSI_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# ===========================
# 1. FETCH DATA
# ===========================
def fetch_candles():
    ws = None
    try:
        ws = websocket.create_connection(WS_URL, timeout=10)
        ws.send(json.dumps({"authorize": API_TOKEN}))
        auth = json.loads(ws.recv())
        if "error" in auth: return pd.DataFrame()
            
        ws.send(json.dumps({
            "ticks_history": SYMBOL,
            "style": "candles",
            "granularity": GRANULARITY,
            "count": CANDLES,
            "end": "latest"
        }))
        resp = json.loads(ws.recv())
        if "error" in resp: return pd.DataFrame()
        
        candles = resp.get("candles", [])
        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["epoch"], unit="s")
        for c in ["open", "high", "low", "close"]: df[c] = df[c].astype(float)
        
        df = df.sort_values("time").reset_index(drop=True)
        df["price"] = df["close"]
        return df
    except: return pd.DataFrame()
    finally:
        if ws: ws.close()

# ===========================
# 2. INDICATEURS
# ===========================
def calculate_indicators(df):
    d = df.copy()
    delta = d['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    d['rsi'] = 100 - (100 / (1 + rs))
    
    d['ema50'] = d['price'].ewm(span=EMA_FAST, adjust=False).mean()
    d['ema200'] = d['price'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    d['bb_mid'] = d['price'].rolling(20).mean()
    d['bb_std'] = d['price'].rolling(20).std()
    d['bb_up'] = d['bb_mid'] + (d['bb_std'] * 2)
    d['bb_low'] = d['bb_mid'] - (d['bb_std'] * 2)
    return d

def find_swings(prices, left=5, right=5):
    p = np.asarray(prices, dtype=float)
    highs, lows = [], []
    for i in range(left, len(p) - right):
        wl = p[i-left:i]
        wr = p[i+1:i+1+right]
        if p[i] > wl.max() and p[i] > wr.max(): highs.append(i)
        if p[i] < wl.min() and p[i] < wr.min(): lows.append(i)
    return np.array(highs), np.array(lows)

def compute_levels(df):
    highs, lows = find_swings(df["price"], left=LEFT_RIGHT, right=LEFT_RIGHT)
    def get_top_k(indices):
        if len(indices) == 0: return []
        vals = df["price"].iloc[indices]
        binned = np.round(vals / BIN_SIZE) * BIN_SIZE
        counts = binned.value_counts()
        return counts.head(TOP_LEVELS).index.tolist()
    res = get_top_k(highs)
    sup = get_top_k(lows)
    return sorted(set(res + sup))

# ===========================
# 3. SCANNER LOGIC
# ===========================
def scan_history(df, levels):
    signals = []
    d = calculate_indicators(df)
    
    if not levels: return [], d
    
    for i in range(200, len(d)):
        curr_p = d['price'].iloc[i]
        curr_h = d['high'].iloc[i]
        curr_l = d['low'].iloc[i]
        prev_p = d['price'].iloc[i-1]
        curr_t = d['time'].iloc[i]
        rsi = d['rsi'].iloc[i]
        ema50 = d['ema50'].iloc[i]
        ema200 = d['ema200'].iloc[i]
        
        nearest = min(levels, key=lambda x: abs(x - curr_p))
        dist = abs(nearest - curr_p)
        
        is_uptrend = (ema50 > ema200) and (curr_p > ema50)
        is_downtrend = (ema50 < ema200) and (curr_p < ema50)
        
        # BUY (On Low)
        if is_uptrend and (dist < LEVEL_TOL) and (curr_p > nearest) and (rsi < 60) and (curr_p > prev_p):
            if not signals or (curr_t - signals[-1][1]).total_seconds() > (GRANULARITY * 4):
                signals.append(("BUY", curr_t, curr_l)) 
                
        # SELL (On High)
        elif is_downtrend and (dist < LEVEL_TOL) and (curr_p < nearest) and (rsi > 40) and (curr_p < prev_p):
            if not signals or (curr_t - signals[-1][1]).total_seconds() > (GRANULARITY * 4):
                signals.append(("SELL", curr_t, curr_h)) 
                
    return signals, d

# ===========================
# 4. PLOT ENGINE (CANDLES + LINE)
# ===========================
def plot_full_chart(df, levels, all_signals, full_data):
    # NOTE: On clear() ici, ce qui cause le reset du zoom. 
    # C'est inévitable en matplotlib "simple".
    plt.clf() 
    
    # Vue: 150 dernières bougies
    view = full_data.iloc[-150:].copy()
    
    x = view["time"]
    ema50 = view["ema50"]
    ema200 = view["ema200"]
    curr_p = view["price"].iloc[-1]
    
    # --- A. DESSIN DES BOUGIES (CANDLES) ---
    if CHART_TYPE == "CANDLES":
        # Largeur des bougies (Calcul dynamique)
        width = 0.006 
        
        # Séparation Vert/Rouge
        up = view[view.close >= view.open]
        down = view[view.close < view.open]
        
        # Wicks (Dyoul) - L-Khat L-R9i9
        plt.vlines(up.time, up.low, up.high, color='white', linewidth=0.8, alpha=0.6)
        plt.vlines(down.time, down.low, down.high, color='white', linewidth=0.8, alpha=0.6)
        
        # Bodies (Jssm) - L-Barra l-ghlida
        plt.bar(up.time, up.close - up.open, width, bottom=up.open, color='#00e676', alpha=0.9)
        plt.bar(down.time, down.close - down.open, width, bottom=down.open, color='#ff1744', alpha=0.9)
        
    # --- B. DESSIN LINE (Si choisi) ---
    else:
        plt.plot(x, view["price"], color='white', linewidth=1.5, label='Price')

    # --- INDICATEURS ---
    plt.plot(x, ema50, color='#e040fb', linewidth=1.2, label='EMA 50')
    plt.plot(x, ema200, color='#2979ff', linewidth=1.5, linestyle='--', label='EMA 200')

    # --- LEVELS ---
    for lv in levels:
        if lv > curr_p - 150 and lv < curr_p + 150:
            col = '#00e676' if curr_p > lv else '#ff1744'
            plt.axhline(lv, color=col, linewidth=0.8, alpha=0.5)

    # --- SIGNAUX (POSITIONS CORRIGÉES) ---
    start_view = x.iloc[0]
    visible_signals = [s for s in all_signals if s[1] >= start_view]
    
    # Calcul Offset Intelligent (5% du range actuel)
    y_min, y_max = view['low'].min(), view['high'].max()
    offset = (y_max - y_min) * 0.05 

    for sig in visible_signals:
        typ, t, exact_price = sig 
        
        if typ == 'BUY':
            col = '#00e676'
            marker = '^'
            pos_arrow = exact_price - offset
            pos_text = exact_price - (offset * 1.8)
        else:
            col = '#ff1744'
            marker = 'v'
            pos_arrow = exact_price + offset
            pos_text = exact_price + (offset * 1.8)
            
        plt.scatter(t, pos_arrow, s=150, color=col, marker=marker, zorder=20, edgecolors='white')
        plt.text(t, pos_text, f"{typ}\n{exact_price:.2f}", 
                 color='black', fontsize=7, fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc=col, alpha=0.9))

    # --- INFO HEADER ---
    plt.axhline(curr_p, color='#2979ff', linestyle=':', linewidth=1)
    rsi_val = full_data['rsi'].iloc[-1]
    
    if curr_p > ema50.iloc[-1] > ema200.iloc[-1]: trend = "HAUSSIER [UP]"
    elif curr_p < ema50.iloc[-1] < ema200.iloc[-1]: trend = "BAISSIER [DOWN]"
    else: trend = "NEUTRE"
    
    plt.title(f"SNIPER V16 ({CHART_TYPE}) | Trend: {trend} | RSI: {rsi_val:.1f} | Price: {curr_p:.2f}", color='white')
    plt.grid(True, color='#37474f', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.pause(REFRESH_SEC)

if __name__ == "__main__":
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title(f'SNIPER V16 - {CHART_TYPE} MODE')
    
    print(f">> MODE GRAPHIQUE: {CHART_TYPE}")
    print(">> CHARGEMENT...")
    
    try:
        while True:
            df = fetch_candles()
            if not df.empty:
                levels = compute_levels(df)
                all_signals, full_data = scan_history(df, levels)
                plot_full_chart(df, levels, all_signals, full_data)
            else:
                print("Connexion...")
                time.sleep(2)
    except KeyboardInterrupt: print("Stop.")