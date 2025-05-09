import streamlit as st
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import hashlib

st.set_page_config(page_title="div-ladder", layout="wide")

# --- AUTH FUNCTIONS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    path = Path("auth/users.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return json.load(open(path)) if path.exists() else {}

def save_users(users):
    with open("auth/users.json", "w") as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    hashed = hash_password(password)
    return username in users and users[username]['password'] == hashed

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
    for user_dir in Path("models").iterdir():
        if user_dir.is_dir():
            for file in user_dir.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    if data.get("public"):
                        public.append((user_dir.name, file.stem, data))
    return public

# --- SIMULATION HELPERS ---
def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except: return 0.0

def get_dividend(ticker):
    try:
        divs = yf.Ticker(ticker).dividends
        return divs.iloc[-1] if not divs.empty else 0.0
    except: return 0.0

def simulate_model(tickers, holdings, periods, reinvest_freq, cascade_matrix, allow_fractional):
    freq_map = {'Weekly': 52, 'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annually': 1}
    steps = periods * freq_map[reinvest_freq]
    result = []
    current = dict(zip(tickers, holdings))

    for step in range(steps):
        row = {}
        new_holdings = current.copy()
        for i, ticker in enumerate(tickers):
            shares = current[ticker]
            price = get_price(ticker)
            dividend = get_dividend(ticker) / freq_map[reinvest_freq]
            earned = shares * dividend
            cascade_pct = cascade_matrix[i] / 100 if i < len(cascade_matrix) else 0
            cascaded = earned * cascade_pct
            reinvested = earned * (1 - cascade_pct)
            bought = reinvested / price if allow_fractional else reinvested // price
            new_holdings[ticker] += bought
            if i + 1 < len(tickers):
                next_price = get_price(tickers[i + 1])
                transfer = cascaded / next_price if allow_fractional else cascaded // next_price
                new_holdings[tickers[i + 1]] += transfer
            row[f"{ticker}_Holdings"] = shares
            row[f"{ticker}_Cost"] = shares * price
            row[f"{ticker}_Earned"] = earned
            row[f"{ticker}_Cascaded"] = cascaded
        result.append(row)
        current = new_holdings
    return pd.DataFrame(result)

# --- AUTH UI ---
if "user" not in st.session_state:
    with st.expander("Login / Register"):
        mode = st.radio("Mode", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button(mode):
            users = load_users()
            if mode == "Register":
                if username in users:
                    st.error("Username already exists.")
                else:
                    users[username] = {"password": hash_password(password)}
                    save_users(users)
                    st.success("Registration successful.")
            elif mode == "Login":
                if authenticate(username, password):
                    st.session_state["user"] = username
                    st.success(f"Welcome, {username}!")
                else:
                    st.error("Login failed.")
    st.stop()

user = st.session_state["user"]
st.title("div-ladder")

# --- MAIN APP ---
tab1, tab2 = st.tabs(["Create Simulation", "Load Model"])

with tab1:
    with st.form("form"):
        model_name = st.text_input("Model Name")
        num = st.slider("Number of Securities", 2, 6, 3)
        tickers, holdings = [], []
        for i in range(num):
            c1, c2 = st.columns(2)
            with c1:
                ticker = st.text_input(f"Ticker {i+1}", key=f"t_{i}").upper()
            with c2:
                shares = st.number_input(f"Shares {i+1}", value=100.0, min_value=0.0, key=f"s_{i}")
            tickers.append(ticker)
            holdings.append(shares)

        st.subheader("Cascade Matrix")
        cascade_matrix = []
        for i in range(num - 1):
            pct = st.slider(f"{tickers[i]} → {tickers[i+1]} %", 0, 100, 100, step=5, key=f"cm_{i}")
            cascade_matrix.append(pct)

        reinvest_freq = st.selectbox("Reinvestment Frequency", ['Weekly', 'Monthly', 'Quarterly', 'Semi-Annual', 'Annually'])
        periods = st.number_input("Years to Simulate", 1, 30, 5)
        allow_fractional = st.checkbox("Allow Fractional Shares", True)
        save_model_public = st.checkbox("Make Model Public")
        run = st.form_submit_button("Run Simulation")

    if run:
        df = simulate_model(tickers, holdings, periods, reinvest_freq, cascade_matrix, allow_fractional)
        st.subheader("Simulation Table")
        st.dataframe(df)

        st.subheader("Security Breakdown")
        for ticker in tickers:
            with st.expander(ticker):
                cols = [f"{ticker}_{m}" for m in ["Holdings", "Cost", "Earned", "Cascaded"]]
                st.dataframe(df[cols])

        if model_name:
            save_user_model(user, model_name, {
                "tickers": tickers,
                "holdings": holdings,
                "periods": periods,
                "reinvest_freq": reinvest_freq,
                "cascade_matrix": cascade_matrix,
                "allow_fractional": allow_fractional
            }, public=save_model_public)
            st.success(f"Model '{model_name}' saved.")

with tab2:
    models = get_user_models(user)
    if models:
        selected = st.selectbox("Your Models", models)
        if st.button("Load"):
            path = Path("models") / user / f"{selected}.json"
            data = json.load(open(path))
            df = simulate_model(data["tickers"], data["holdings"], data["periods"],
                                data["reinvest_freq"], data["cascade_matrix"], data["allow_fractional"])
            st.dataframe(df)
            for ticker in data["tickers"]:
                with st.expander(ticker):
                    cols = [f"{ticker}_{m}" for m in ["Holdings", "Cost", "Earned", "Cascaded"]]
                    st.dataframe(df[cols])
    else:
        st.info("No saved models found.")

with st.expander("Browse Public Models"):
    for owner, name, data in list_public_models():
        st.markdown(f"**{owner}/{name}**")
        st.json(data)