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
    p = Path("auth/users.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return json.loads(p.read_text()) if p.exists() else {}

def save_users(users):
    Path("auth").mkdir(exist_ok=True)
    with open("auth/users.json", "w") as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    return username in users and users[username]["password"] == hash_password(password)

# --- MODEL STORAGE ---
def save_user_model(username, name, model_data, public=False):
    folder = Path("models") / username
    folder.mkdir(parents=True, exist_ok=True)
    model_data["public"] = public
    with open(folder / f"{name}.json", "w") as f:
        json.dump(model_data, f)

def get_user_models(username):
    folder = Path("models") / username
    return [f.stem for f in folder.glob("*.json")] if folder.exists() else []

def list_public_models():
    public = []
    models_root = Path("models")
    if models_root.exists():
        for user_dir in models_root.iterdir():
            if user_dir.is_dir():
                for file in user_dir.glob("*.json"):
                    data = json.loads(file.read_text())
                    if data.get("public"):
                        public.append((user_dir.name, file.stem, data))
    return public

# --- DATA FETCHING ---
def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except:
        return 0.0

def get_dividend(ticker):
    try:
        divs = yf.Ticker(ticker).dividends
        return divs.iloc[-1] if not divs.empty else 0.0
    except:
        return 0.0

# --- FORWARD SIMULATION ---
def simulate_forward(tickers, holdings, periods, freq, cascade, frac):
    freq_map = {'Weekly': 52, 'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annually': 1}
    steps = periods * freq_map[freq]
    current = dict(zip(tickers, holdings))
    rows = []
    for _ in range(steps):
        row = {}
        new = current.copy()
        for i, t in enumerate(tickers):
            s = current[t]
            price = get_price(t)
            div = get_dividend(t) / freq_map[freq]
            earned = s * div
            pct = cascade[i] / 100 if i < len(cascade) else 0
            cascaded = earned * pct
            reinv = earned - cascaded
            bought_self = reinv / price if frac else reinv // price
            new[t] += bought_self
            if i+1 < len(tickers):
                next_price = get_price(tickers[i+1])
                bought_next = cascaded / next_price if frac else cascaded // next_price
                new[tickers[i+1]] += bought_next
            row[f"{t}_Holdings"] = s
            row[f"{t}_Cost"]     = s * price
            row[f"{t}_Earned"]   = earned
            row[f"{t}_Cascaded"] = cascaded
        rows.append(row)
        current = new
    return pd.DataFrame(rows)

# --- HISTORICAL BACKTEST ---
def simulate_backtest(tickers, holdings, periods, freq, cascade, frac):
    freq_map = {'Weekly': 52, 'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annually': 1}
    date_freq = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Semi-Annual': '2Q', 'Annually': 'A'}[freq]
    steps = periods * freq_map[freq]
    dates = pd.date_range(end=pd.Timestamp.today(), periods=steps, freq=date_freq)
    current = dict(zip(tickers, holdings))
    rows = []
    for date in dates:
        row = {}
        new = current.copy()
        for i, t in enumerate(tickers):
            s = current[t]
            # Last known dividend up to this date
            divs = yf.Ticker(t).dividends
            divs = divs[divs.index <= date]
            last_div = divs.iloc[-1] if not divs.empty else 0.0
            earned = s * last_div
            pct = cascade[i] / 100 if i < len(cascade) else 0
            cascaded = earned * pct
            reinv = earned - cascaded
            # price around date
            hist = yf.Ticker(t).history(start=date, end=date + pd.Timedelta(days=5))
            price = hist["Close"].mean() if not hist.empty else get_price(t)
            bought_self = reinv / price if frac else reinv // price
            new[t] += bought_self
            if i+1 < len(tickers):
                hist2 = yf.Ticker(tickers[i+1]).history(start=date, end=date + pd.Timedelta(days=5))
                next_price = hist2["Close"].mean() if not hist2.empty else get_price(tickers[i+1])
                bought_next = cascaded / next_price if frac else cascaded // next_price
                new[tickers[i+1]] += bought_next
            row[f"{t}_Holdings"] = s
            row[f"{t}_Cost"]     = s * price
            row[f"{t}_Earned"]   = earned
            row[f"{t}_Cascaded"] = cascaded
        rows.append(row)
        current = new
    df = pd.DataFrame(rows, index=dates)
    df.index.name = "Date"
    return df

# --- LOGIN/REGISTER UI ---
if "user" not in st.session_state:
    with st.expander("Login / Register"):
        mode = st.radio("Mode", ["Login", "Register"])
        uname = st.text_input("Username")
        pwd   = st.text_input("Password", type="password")
        if st.button(mode):
            users = load_users()
            if mode == "Register":
                if uname in users:
                    st.error("Username exists.")
                else:
                    users[uname] = {"password": hash_password(pwd)}
                    save_users(users)
                    st.success("Registered—please log in.")
            else:
                if authenticate(uname, pwd):
                    st.session_state["user"] = uname
                    st.success(f"Welcome back, {uname}!")
                else:
                    st.error("Login failed.")
    st.stop()

user = st.session_state["user"]
st.title("div-ladder")

# --- MAIN APP ---
tab1, tab2 = st.tabs(["Create Simulation", "Load Model"])

with tab1:
    with st.form("sim"):
        name = st.text_input("Model Name")
        nsec = st.slider("Number of Securities", 2, 6, 3)
        tickers, shares = [], []
        for i in range(nsec):
            c1, c2 = st.columns(2)
            with c1: t = st.text_input(f"Ticker {i+1}", key=f"t{i}").upper()
            with c2: s = st.number_input(f"Shares {i+1}", value=100.0, min_value=0.0, key=f"s{i}")
            tickers.append(t); shares.append(s)

        st.subheader("Cascade % Between Pairs")
        cascade = []
        for i in range(nsec-1):
            pct = st.slider(f"{tickers[i]}→{tickers[i+1]}", 0, 100, 100, step=5, key=f"c{i}")
            cascade.append(pct)

        freq = st.selectbox("Frequency", ['Weekly','Monthly','Quarterly','Semi-Annual','Annually'])
        yrs  = st.number_input("Years to Simulate", 1, 30, 5)
        frac = st.checkbox("Allow Fractional Shares", True)
        pub  = st.checkbox("Make Model Public")
        mode = st.radio("Mode", ["Forward Projection","Historical Backtest"])
        go   = st.form_submit_button("Run")

    if go:
        df = simulate_forward(tickers, shares, yrs, freq, cascade, frac) if mode=="Forward Projection" else simulate_backtest(tickers, shares, yrs, freq, cascade, frac)
        st.subheader("Simulation Results")
        st.dataframe(df)

        st.subheader("Per-Security Details")
        for t in tickers:
            with st.expander(t):
                cols = [f"{t}_{m}" for m in ["Holdings","Cost","Earned","Cascaded"]]
                st.dataframe(df[cols])

        if name:
            save_user_model(user, name, {
                "tickers": tickers, "holdings": shares,
                "periods": yrs, "reinvest_freq": freq,
                "cascade_matrix": cascade, "allow_fractional": frac,
                "mode": mode
            }, public=pub)
            st.success(f"Saved '{name}'")

with tab2:
    own = get_user_models(user)
    if own:
        sel = st.selectbox("Your Models", own)
        if st.button("Load"):
            mfile = Path("models")/user/f"{sel}.json"
            data  = json.loads(mfile.read_text())
            df    = simulate_forward(**data) if data["mode"]=="Forward Projection" else simulate_backtest(**data)
            st.dataframe(df)
            for t in data["tickers"]:
                with st.expander(t):
                    cols = [f"{t}_{m}" for m in ["Holdings","Cost","Earned","Cascaded"]]
                    st.dataframe(df[cols])
    else:
        st.info("No models yet.")

with st.expander("Browse Public Models"):
    pubs = list_public_models()
    if pubs:
        for owner, mname, md in pubs:
            st.markdown(f"**{owner}/{mname}**")
            st.json(md)
    else:
        st.info("None public.")
