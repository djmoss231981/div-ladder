import streamlit as st
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import hashlib

st.set_page_config(page_title="div-ladder", layout="wide")

# --- AUTHENTICATION ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    p = Path("auth/users.json"); p.parent.mkdir(exist_ok=True)
    return json.loads(p.read_text()) if p.exists() else {}

def save_users(users):
    Path("auth").mkdir(exist_ok=True)
    with open("auth/users.json", "w") as f:
        json.dump(users, f)

def authenticate(u, p):
    users = load_users()
    return u in users and users[u]["password"] == hash_password(p)

# --- MODEL STORAGE ---
def save_user_model(user, name, data, public=False):
    folder = Path("models")/user; folder.mkdir(parents=True, exist_ok=True)
    data["public"] = public
    with open(folder/f"{name}.json","w") as f:
        json.dump(data, f)

def get_user_models(user):
    folder = Path("models")/user
    return [f.stem for f in folder.glob("*.json")] if folder.exists() else []

def list_public_models():
    out = []
    root = Path("models")
    if root.exists():
        for d in root.iterdir():
            if d.is_dir():
                for f in d.glob("*.json"):
                    md = json.loads(f.read_text())
                    if md.get("public"):
                        out.append((d.name, f.stem, md))
    return out

# --- DATA HELPERS ---
def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except:
        return 0.0

def get_dividend(ticker):
    try:
        dv = yf.Ticker(ticker).dividends
        return dv.iloc[-1] if not dv.empty else 0.0
    except:
        return 0.0

# --- CORE SIMULATION ---
def _simulate(tickers, holdings, periods, freq, cascade, frac, last_handling, historical=False):
    freq_map = {'Weekly':52,'Monthly':12,'Quarterly':4,'Semi-Annual':2,'Annually':1}
    date_freq = {'Weekly':'W','Monthly':'M','Quarterly':'Q','Semi-Annual':'2Q','Annually':'A'}[freq]
    steps = periods * freq_map[freq]
    dates = (pd.date_range(end=pd.Timestamp.today(), periods=steps, freq=date_freq)
             if historical else range(steps))
    current = dict(zip(tickers, holdings))
    cumulative = {t:0.0 for t in tickers}
    records = []

    for idx in dates:
        row = {}; new = current.copy()
        for i, t in enumerate(tickers):
            s = current[t]
            # price
            if historical:
                hist_p = yf.Ticker(t).history(start=idx, end=idx+pd.Timedelta(days=5))
                price = hist_p["Close"].mean() if not hist_p.empty else get_price(t)
                # dividend per share
                dv_series = yf.Ticker(t).dividends
                if not dv_series.empty:
                    dv_series.index = dv_series.index.tz_localize(None)
                    dv_series = dv_series[dv_series.index <= pd.Timestamp(idx)]
                per_share = dv_series.iloc[-1] if not dv_series.empty else 0.0
            else:
                price = get_price(t)
                per_share = get_dividend(t)/freq_map[freq]

            earned = s * per_share
            cumulative[t] += earned

            pct      = cascade[i]/100 if i < len(cascade) else 0
            cascaded = earned * pct
            reinv    = earned - cascaded

            # last security override
            if i == len(tickers)-1:
                if last_handling == "Reinvest in itself":
                    reinv, cascaded = earned, 0.0
                else:
                    reinv, cascaded = 0.0, earned

            # buy self
            bought_self = reinv/price if frac else reinv//price
            new[t] += bought_self

            # buy next or distribute
            if i < len(tickers)-1:
                np = get_price(tickers[i+1])
                bought_next = cascaded/np if frac else cascaded//np
                new[tickers[i+1]] += bought_next
            elif last_handling=="Distribute equally across chain":
                part = cascaded/len(tickers)
                for tk in tickers:
                    tp = get_price(tk)
                    new[tk] += (part/tp if frac else part//tp)

            # --- Compute Market Value & Capital Gains ---
            market_value = s * price
            prev_mv = current_metrics[t].get("last_market_value", initial_costs[t])  # see below
            cap_gain = market_value - prev_mv
            cumulative_cap[t] += cap_gain
          
            # record metrics
            row[f"{t}_MarketValue"]     = market_value
            row[f"{t}_CapGain"]         = cap_gain                     
            row[f"{t}_CumulativeCap"]   = cumulative_cap[t]            
            row[f"{t}_Earned"]          = earned
            row[f"{t}_Cascaded"]        = cascaded
            row[f"{t}_CumulativeDivs"]  = cumulative[t]
            row[f"{t}_Holdings"]        = s
            row[f"{t}_Price"]           = price

            # Compute each security’s initial cost (share count × initial price)
            initial_costs = {
                t: holdings[i] * get_price(t)
                for i, t in enumerate(tickers)
            }
            # Track last market value per security (start at initial cost)
            current_metrics = {
                t: {"last_market_value": initial_costs[t]}
                for t in tickers
            }
            # Cumulative capital gains
            cumulative_cap = {t: 0.0 for t in tickers}

            # Update last_market_value for next iteration
            for t in tickers:
                current_metrics[t]["last_market_value"] = row[f"{t}_MarketValue"]


        records.append(row)
        current = new

    df = pd.DataFrame(records, index=dates)
    df.index.name = "Date" if historical else f"{freq} Step"
    
    # Portfolio Capital Gains
    df["Total CapGain"]       = df.filter(like="_CapGain").sum(axis=1)
    df["Total CumulativeCap"] = df.filter(like="_CumulativeCap").sum(axis=1)

    return df

def simulate_forward(tickers, holdings, periods, freq, cascade, frac, last_handling):
    return _simulate(tickers, holdings, periods, freq, cascade, frac, last_handling, False)

def simulate_backtest(tickers, holdings, periods, freq, cascade, frac, last_handling):
    return _simulate(tickers, holdings, periods, freq, cascade, frac, last_handling, True)

# --- LOGIN/REGISTER UI ---
if "user" not in st.session_state:
    with st.expander("Login / Register"):
        mode = st.radio("Mode", ["Login","Register"])
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button(mode):
            us = load_users()
            if mode=="Register":
                if u in us: st.error("Username exists.")
                else:
                    us[u] = {"password":hash_password(p)}
                    save_users(us)
                    st.success("Registered—please log in.")
            else:
                if authenticate(u,p):
                    st.session_state["user"] = u
                    st.success(f"Welcome, {u}!")
                else:
                    st.error("Login failed.")
    st.stop()

user = st.session_state["user"]
st.title("div-ladder")

# --- MAIN UI ---
tab1, tab2 = st.tabs(["Create Simulation","Load Model"])

with tab1:
    with st.form("form"):
        name = st.text_input("Model Name")
        n    = st.slider("Number of Securities", 2, 6, 3)
        tickers, shares = [], []

        for i in range(n):
            c1, c2 = st.columns(2)
            with c1:
                t = st.text_input(f"Ticker {i+1}", key=f"t{i}").upper()
                if t:
                    hist = yf.Ticker(t).dividends
                    if not hist.empty:
                        hist.index = hist.index.tz_localize(None)
                        recent = hist[hist.index >= pd.Timestamp.today() - pd.DateOffset(years=1)]
                        if not recent.empty:
                            last4 = recent.sort_index(ascending=False).head(4)
                            st.markdown(f"• **Payouts (last year)**: {len(last4)}")
                            st.markdown(f"• **Avg Payout**: ${last4.mean():.4f}")
                        else:
                            st.warning("No payouts found, enter a new ticker")
                    else:
                        st.warning("No payouts found, enter a new ticker")
            with c2:
                s = st.number_input(f"Shares {i+1}", value=100.0, min_value=0.0, key=f"s{i}")
            tickers.append(t); shares.append(s)

        st.subheader("Cascade % Between Pairs")
        cascade = []
        for i in range(n-1):
            pct = st.slider(f"{tickers[i]}→{tickers[i+1]}", 0, 100, 100, step=5, key=f"c{i}")
            cascade.append(pct)

        last_hand = st.selectbox("Last Security Dividend Handling",
                                 ["Reinvest in itself","Distribute equally across chain"]
        )
        freq = st.selectbox("Frequency", ['Weekly','Monthly','Quarterly','Semi-Annual','Annually'])
        yrs  = st.number_input("Years to Simulate", 1, 30, 5)
        frac = st.checkbox("Allow Fractional Shares", True)
        pub  = st.checkbox("Make Model Public")
        mode = st.radio("Mode", ["Forward Projection","Historical Backtest"])

        go   = st.form_submit_button("Run Simulation")

    if go:
        df = (simulate_forward if mode=="Forward Projection" else simulate_backtest)(
            tickers, shares, yrs, freq, cascade, frac, last_hand
        )

        st.subheader("Simulation Results")
        st.dataframe(df)

        # Portfolio totals
        df["Total Market Value"]    = df.filter(like="_MarketValue").sum(axis=1)
        df["Total Cumulative Divs"] = df.filter(like="_CumulativeDivs").sum(axis=1)
        df["Total Portfolio Value"] = df["Total Market Value"] + df["Total Cumulative Divs"]

        st.subheader("Portfolio Totals Over Time")
        st.line_chart(df[["Total Market Value","Total Cumulative Divs"]])

        st.subheader("Compounding Growth Over Time")
        st.area_chart(df[["Total Market Value","Total Cumulative Divs"]])

        st.subheader("Total Portfolio Value Curve")
        st.line_chart(df["Total Portfolio Value"])

        st.subheader("Portfolio Capital Gains Over Time")
        st.line_chart(df[["Total CapGain", "Total CumulativeCap"]])

        st.subheader("Per-Security Breakdown")
        for t in tickers:
            with st.expander(t):
                cols = [f"{t}_{m}" for m in ["Holdings","Cost","Price","MarketValue","Earned","Cascaded","CumulativeDivs"]]
                st.dataframe(df[cols])

        if name:
            save_user_model(user, name, {
                "tickers": tickers, "holdings": shares,
                "periods": yrs, "reinvest_freq": freq,
                "cascade_matrix": cascade, "allow_fractional": frac,
                "mode": mode, "last_handling": last_hand
            }, public=pub)
            st.success(f"Model '{name}' saved.")

with tab2:
    own = get_user_models(user)
    if own:
        sel = st.selectbox("Your Models", own)
        if st.button("Load"):
            mf   = Path("models")/user/f"{sel}.json"
            data = json.loads(mf.read_text())
            fn   = simulate_forward if data["mode"]=="Forward Projection" else simulate_backtest
            df   = fn(
                data["tickers"], data["holdings"],
                data["periods"], data["reinvest_freq"],
                data["cascade_matrix"], data["allow_fractional"],
                data["last_handling"]
            )
            st.subheader("Loaded Simulation")
            st.dataframe(df)
            for t in data["tickers"]:
                with st.expander(t):
                    cols = [f"{t}_{m}" for m in ["Holdings","Cost","Price","MarketValue","Earned","Cascaded","CumulativeDivs"]]
                    st.dataframe(df[cols])
    else:
        st.info("No models yet.")

with st.expander("Browse Public Models"):
    pubs = list_public_models()
    if pubs:
        for owner,name,md in pubs:
            st.markdown(f"**{owner}/{name}**")
            st.json(md)
    else:
        st.info("No public models.")
