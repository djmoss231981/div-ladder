import streamlit as st
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import hashlib

# --- 'Page Configuration' ---
st.set_page_config(page_title="div-ladder", layout="wide")

# --- 'Cached Data Fetch' ---
@st.cache_data
def load_ticker_data(tickers):
    """
    Fetch 1-year historical prices and dividends for each ticker.
    Returns two dicts: prices and dividends, each keyed by ticker.
    """
    prices, divs = {}, {}
    for t in tickers:
        tk   = yf.Ticker(t)
        hist = tk.history(period="1y")["Close"].resample("D").ffill()
        dv   = tk.dividends.resample("D").ffill()
        prices[t] = hist
        divs[t]   = dv
    return prices, divs

# --- 'Authentication Utilities' ---
def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def load_users() -> dict:
    path = Path("auth/users.json")
    path.parent.mkdir(exist_ok=True)
    return json.loads(path.read_text()) if path.exists() else {}

def save_users(users: dict) -> None:
    Path("auth").mkdir(exist_ok=True)
    Path("auth/users.json").write_text(json.dumps(users))

def authenticate(user: str, pwd: str) -> bool:
    return load_users().get(user, {}).get("password") == hash_password(pwd)

# --- 'Model Storage' ---
def save_model(user: str, name: str, model: dict, public: bool=False) -> None:
    folder = Path("models") / user
    folder.mkdir(parents=True, exist_ok=True)
    model["public"] = public
    Path(folder / f"{name}.json").write_text(json.dumps(model))

def list_models(user: str) -> list:
    folder = Path("models") / user
    return [f.stem for f in folder.glob("*.json")] if folder.exists() else []

def list_public_models() -> list:
    pubs = []
    root = Path("models")
    if root.exists():
        for udir in root.iterdir():
            if udir.is_dir():
                for f in udir.glob("*.json"):
                    m = json.loads(f.read_text())
                    if m.get("public"):
                        pubs.append((udir.name, f.stem, m))
    return pubs

# --- 'Simulation Core' ---
def simulate(
    tickers: list, shares: list, years: int, freq: str,
    cascade: list, frac: bool, last_handling: str,
    historical: bool=False
) -> pd.DataFrame:
    """
    Run forward projection or historical backtest simulation.
    """
    freq_map = {"Weekly":52, "Monthly":12, "Quarterly":4, "Semi-Annual":2, "Annually":1}
    date_freq = {"Weekly":"W", "Monthly":"M", "Quarterly":"Q", "Semi-Annual":"2Q", "Annually":"A"}

    steps = int(years * freq_map[freq])
    prices, divs = load_ticker_data(tickers)

    # --- 'Index Determination' ---
    if historical:
        index = pd.date_range(end=pd.Timestamp.today(), periods=steps, freq=date_freq[freq])
    else:
        index = range(steps)

    # --- 'Initialize State' ---
    holdings = dict(zip(tickers, shares))
    cum_div   = {t:0.0 for t in tickers}
    cum_cap   = {t:0.0 for t in tickers}
    last_vals = {t:shares[i]*prices[t].iloc[-1] for i,t in enumerate(tickers)}

    # --- 'Compute Positions' ---
    length = len(prices[tickers[0]])
    idxs   = (pd.Series(range(steps))*length//steps).clip(0,length-1).astype(int)

    records = []
    for step in range(steps):
        pos = idxs.iat[step]
        row = {}

        for i, t in enumerate(tickers):
            # --- 'Price & Dividend' ---
            price    = prices[t].iat[pos]
            dividend = divs[t].iat[pos] if historical else divs[t].iat[-1]/freq_map[freq]

            s      = holdings[t]
            earned = s * dividend
            cum_div[t] += earned

            # --- 'Capital Gains' ---
            mv   = s * price
            gain = mv - last_vals[t]
            cum_cap[t] += gain
            last_vals[t] = mv

            # --- 'Cascade & Reinvest' ---
            pct   = cascade[i]/100 if i < len(cascade) else 0
            reinv = earned*(1-pct)
            casc  = earned*pct
            if i==len(tickers)-1 and last_handling!="Reinvest in itself":
                casc, reinv = earned, 0

            holdings[t] += (reinv/price if frac else reinv//price)
            if i < len(tickers)-1:
                next_t = tickers[i+1]
                holdings[next_t] += casc/prices[next_t].iat[pos]
            elif last_handling=="Distribute equally across chain":
                part = casc/len(tickers)
                for tk in tickers:
                    holdings[tk] += part/prices[tk].iat[pos]

            # --- 'Record Metrics' ---
            row.update({
                f"{t}_Holdings":       s,
                f"{t}_Price":          price,
                f"{t}_Cost":           mv,
                f"{t}_MarketValue":    mv,
                f"{t}_CapGain":        gain,
                f"{t}_CumulativeCap":  cum_cap[t],
                f"{t}_Earned":         earned,
                f"{t}_Cascaded":       casc,
                f"{t}_CumulativeDivs": cum_div[t],
            })

        records.append(row)

    # --- 'Build DataFrame' ---
    df = pd.DataFrame(records, index=index)
    df.index.name = "Date" if historical else "Step"

    # --- 'Portfolio Aggregates' ---
    df["Total MarketValue"]    = df.filter(like="_MarketValue").sum(axis=1)
    df["Total CumulativeDivs"] = df.filter(like="_CumulativeDivs").sum(axis=1)
    df["Total CapGain"]        = df.filter(like="_CapGain").sum(axis=1)
    df["Total CumulativeCap"]  = df.filter(like="_CumulativeCap").sum(axis=1)
    df["Total PortfolioValue"] = df["Total MarketValue"] + df["Total CumulativeDivs"]

    return df

# --- 'UI: Authentication' ---
if "user" not in st.session_state:
    with st.expander("Login/Register"):
        mode     = st.radio("Mode",["Login","Register"])
        username = st.text_input("Username")
        pwd      = st.text_input("Password",type="password")
        if st.button(mode):
            users = load_users()
            if mode=="Register":
                users[username] = {"password":hash_password(pwd)}
                save_users(users)
                st.success("Registered")
            else:
                if authenticate(username,pwd):
                    st.session_state["user"] = username
                    st.success(f"Welcome {username}")
                else:
                    st.error("Login failed")
    st.stop()

# --- 'UI: Main' ---
user = st.session_state["user"]
st.title("div-ladder")

tabs = st.tabs(["Create","Load","Public"])

# --- 'UI: Create Tab' ---
with tabs[0]:
    with st.form("create"): 
        name       = st.text_input("Model Name")
        n          = st.slider("Securities",2,6,3)
        ticks      = [st.text_input(f"Ticker {i+1}",key=f"tick_{i}") for i in range(n)]
        shares     = [st.number_input(f"Shares {i+1}",100.0,key=f"share_{i}") for i in range(n)]
        cascade    = [st.slider(f"{ticks[i]}→{ticks[i+1]}",0,100,100,5,key=f"casc_{i}") for i in range(n-1)]
        last_hand  = st.selectbox("Last Handling",["Reinvest in itself","Distribute equally across chain"])
        freq       = st.selectbox("Frequency",["Weekly","Monthly","Quarterly","Semi-Annual","Annually"])
        years      = st.number_input("Years",1,30,5)
        frac       = st.checkbox("Allow Fractional Shares",True)
        pub        = st.checkbox("Make Model Public")
        mode_sel   = st.radio("Mode",["Forward","Historical"])
        run        = st.form_submit_button("Run Simulation")

    if run:
        df = simulate(ticks,shares,years,freq,cascade,frac,last_hand,mode_sel=="Historical")
        st.dataframe(df)
        st.area_chart(df[["Total MarketValue","Total CumulativeDivs"]])
        st.line_chart(df["Total PortfolioValue"])
        for t in ticks:
            with st.expander(t):
                st.dataframe(df.filter(like=f"{t}_"))
        save_model(user,name,{"tickers":ticks,"holdings":shares,"years":years,"freq":freq,"cascade":cascade,"frac":frac,"last_handling":last_hand,"mode":mode_sel},pub)

# --- 'UI: Load Tab' ---
with tabs[1]:
    models = list_models(user)
    if models:
        sel = st.selectbox("Your Models",models)
        if st.button("Load Model"):
            data = json.loads((Path("models")/user/f"{sel}.json").read_text())
            df = simulate(**{**data,"historical":data["mode"]=="Historical"})
            st.dataframe(df)
            for t in data["tickers"]:
                with st.expander(t):
                    st.dataframe(df.filter(like=f"{t}_"))
    else:
        st.info("No models saved")

# --- 'UI: Public Tab' ---
with tabs[2]:
    pubs = list_public_models()
    if pubs:
        for u,name,m in pubs:
            st.markdown(f"**{u}/{name}**")
            st.json(m)
    else:
        st.info("No public models")