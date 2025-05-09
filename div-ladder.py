
import streamlit as st
import yfinance as yf
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Dividend Ladder Simulator", layout="wide")

def save_model(name, model_data):
    Path("models").mkdir(exist_ok=True)
    with open(f"models/{name}.json", "w") as f:
        json.dump(model_data, f)

def load_model(name):
    with open(f"models/{name}.json", "r") as f:
        return json.load(f)

def get_user_models(user):
    Path("models").mkdir(exist_ok=True)
    return [f.stem for f in Path("models").glob("*.json")]

def get_latest_dividend(ticker):
    try:
        dividends = yf.Ticker(ticker).dividends
        return dividends.iloc[-1] if not dividends.empty else 0.0
    except Exception:
        return 0.0

def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except Exception:
        return 0.0

def simulate_model(tickers, holdings, periods, reinvest_freq, allow_fractional):
    freq_map = {'Weekly': 52, 'Monthly': 12, 'Quarterly': 4, 'Semi-Annually': 2, 'Annually': 1}
    steps = periods * freq_map[reinvest_freq]

    df = pd.DataFrame(columns=tickers)
    df.loc[0] = holdings

    for step in range(1, steps + 1):
        last = df.loc[step - 1].copy()
        new = last.copy()
        for i, ticker in enumerate(tickers):
            dividend = get_latest_dividend(ticker) / freq_map[reinvest_freq]
            total_dividend = last[ticker] * dividend

            if i < len(tickers) - 1:
                next_ticker = tickers[i + 1]
                next_price = get_price(next_ticker)
                if next_price > 0:
                    shares_bought = total_dividend / next_price if allow_fractional else total_dividend // next_price
                    new[next_ticker] += shares_bought
        df.loc[step] = new

    df.index.name = f"{reinvest_freq} Period"
    return df


import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    user_path = Path("auth/users.json")
    user_path.parent.mkdir(parents=True, exist_ok=True)
    if user_path.exists():
        with open(user_path, "r") as f:
            return json.load(f)
    else:
        return {}

def save_users(users):
    with open("auth/users.json", "w") as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    hashed = hash_password(password)
    return username in users and users[username]['password'] == hashed

def get_user_models(username):
    user_folder = Path("models") / username
    return [f.stem for f in user_folder.glob("*.json")] if user_folder.exists() else []

def save_user_model(username, name, model_data, public=False):
    user_folder = Path("models") / username
    user_folder.mkdir(parents=True, exist_ok=True)
    model_data['public'] = public
    with open(user_folder / f"{name}.json", "w") as f:
        json.dump(model_data, f)

def list_public_models():
    model_path = Path("models")
    public_models = []
    for user_dir in model_path.iterdir():
        if user_dir.is_dir():
            for file in user_dir.glob("*.json"):
                with open(file, "r") as f:
                    model_data = json.load(f)
                    if model_data.get("public"):
                        public_models.append((user_dir.name, file.stem, model_data))
    return public_models


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
                    st.success("Registration successful. Please log in.")
            elif mode == "Login":
                if authenticate(username, password):
                    st.session_state["user"] = username
                    st.success(f"Welcome, {username}!")
                else:
                    st.error("Login failed.")
    st.stop()

user = st.session_state["user"]
st.markdown(f"**Logged in as:** {user}")

st.title("Dividend Ladder Simulator")

tab1, tab2 = st.tabs(["Create & Simulate", "Load Saved Model"])

with tab1:
    with st.form("model_form"):
        st.subheader("Define Your Model")
        model_name = st.text_input("Model Name")
        num_securities = st.slider("Number of Securities", 2, 6, 3)

        tickers = []
        holdings = []
        for i in range(num_securities):
            col1, col2 = st.columns(2)
            with col1:
                ticker = st.text_input(f"Ticker {i+1}", key=f"ticker_{i}")
            with col2:
                shares = st.number_input(f"Initial Shares {i+1}", min_value=0.0, value=100.0, key=f"shares_{i}")
            if ticker:
                try:
                    stock = yf.Ticker(ticker)
                    price = stock.history(period="1d")["Close"].iloc[-1]
                    dividends = stock.dividends
                    latest_dividend = dividends.iloc[-1] if not dividends.empty else 0.0
                    latest_dividend_date = dividends.index[-1].strftime("%Y-%m-%d") if not dividends.empty else "N/A"
                    st.markdown(f"**Current Price**: ${price:.2f}")
                    st.markdown(f"**Last Dividend**: ${latest_dividend:.4f} on {latest_dividend_date}")
                    st.markdown(f"**Cost (Shares × Price)**: ${shares * price:.2f}")
                except:
                    st.warning("Unable to fetch price/dividend data for this ticker.")

            tickers.append(ticker.upper())
            holdings.append(shares)

        # --- Portfolio Level Statistics ---
        try:
            total_cost = 0
            total_dividend = 0
            for ticker, shares in zip(tickers, holdings):
                stock = yf.Ticker(ticker)
                price = stock.history(period="1d")["Close"].iloc[-1]
                dividend = stock.dividends.iloc[-1] if not stock.dividends.empty else 0.0
                total_cost += shares * price
                total_dividend += shares * dividend

            st.markdown("### Portfolio Summary")
            st.markdown(f"**Total Portfolio Cost**: ${total_cost:,.2f}")
            st.markdown(f"**Total Annual Dividends (Latest)**: ${total_dividend:,.2f}")
            if total_cost > 0:
                yield_pct = (total_dividend / total_cost) * 100
                st.markdown(f"**Average Portfolio Yield**: {yield_pct:.2f}%")
        except:
            st.warning("Unable to compute portfolio statistics.")


        periods = st.number_input("Number of Years to Simulate", min_value=1, value=5)
        reinvest_freq = st.selectbox("Dividend Reinvestment Frequency", ['Monthly', 'Quarterly', 'Annually'])
        allow_fractional = st.checkbox("Allow Fractional Shares?", value=True)
        save_option = st.checkbox("Save this model")

        submitted = st.form_submit_button("Simulate")


        sim_mode = st.radio("Simulation Mode", ["Forward Projection", "Historical Backtest"])

        if sim_mode == "Historical Backtest":
            st.warning("Historical backtest will use actual dividend history if available.")
            def simulate_historical(tickers, holdings, periods, reinvest_freq, allow_fractional):
                freq_map = {'Weekly': 52, 'Monthly': 12, 'Quarterly': 4, 'Semi-Annually': 2, 'Annually': 1}
                steps = periods * freq_map[reinvest_freq]

                df = pd.DataFrame(columns=tickers)
                df.loc[0] = holdings
                dates = pd.date_range(end=pd.Timestamp.today(), periods=steps, freq={'Weekly':'W', 'Monthly':'M', 'Quarterly':'Q', 'Semi-Annually':'S', 'Annually':'Y'}[reinvest_freq])

                for step, date in enumerate(dates, 1):
                    last = df.loc[step - 1].copy()
                    new = last.copy()
                    for i, ticker in enumerate(tickers):
                        try:
                            dividends = yf.Ticker(ticker).dividends
                            div_paid = dividends[dividends.index <= date].iloc[-1] if not dividends.empty else 0.0
                        except:
                            div_paid = 0.0

                        total_dividend = last[ticker] * div_paid
                        if i < len(tickers) - 1:
                            next_ticker = tickers[i + 1]
                            try:
                                next_price = yf.Ticker(next_ticker).history(start=date, end=date + pd.Timedelta(days=5))['Close'].mean()
                            except:
                                next_price = 0.0
                            if next_price > 0:
                                shares_bought = total_dividend / next_price if allow_fractional else total_dividend // next_price
                                new[next_ticker] += shares_bought
                    df.loc[step] = new

                df.index.name = "Date"
                df.index = dates[:len(df)]
                return df

            result_df = simulate_historical(tickers, holdings, periods, reinvest_freq, allow_fractional)
        else:
            result_df = simulate_model(tickers, holdings, periods, reinvest_freq, allow_fractional)
    if submitted:
        result_df = simulate_model(tickers, holdings, periods, reinvest_freq, allow_fractional)
        st.subheader("Simulation Results")
        st.dataframe(result_df.style.format("{:.4f}"))
        st.line_chart(result_df)

        # --- Additional Visualization: Total Portfolio Value Over Time ---
        total_value = result_df.copy()
        for col in total_value.columns:
            try:
                price = get_price(col)
                total_value[col] = total_value[col] * price
            except:
                total_value[col] = 0
        total_value["Total Value"] = total_value.sum(axis=1)
        st.subheader("Projected Portfolio Value Over Time")
        st.line_chart(total_value["Total Value"])

        # --- Optional: Show Cash Overflow ---
        st.subheader("Cash Overflow (Uninvested Dividends)")
        st.info("This version assumes all dividends are reinvested; to track cash overflow, additional brokerage logic is needed.")


        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download CSV", result_df.to_csv().encode(), f"{model_name}_simulation.csv", "text/csv")
        with col2:
            excel_bytes = result_df.to_excel(index=True, engine='openpyxl')
            st.download_button("Download Excel", excel_bytes, f"{model_name}_simulation.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if save_option and model_name:
            model_data = {
                "tickers": tickers,
                "holdings": holdings,
                "periods": periods,
                "reinvest_freq": reinvest_freq,
                "allow_fractional": allow_fractional
            }
            save_user_model(user, model_name, model_data, public=save_option)
            st.success(f"Model '{model_name}' saved.")

with tab2:
    st.subheader("Load Existing Model")
    model_list = get_user_models(user)
    if model_list:
        selected_model = st.selectbox("Choose a model to load", model_list)
        if st.button("Load & Simulate"):
            data = json.load(open(Path("models") / user / f"{selected_model}.json"))
            result_df = simulate_model(
                data["tickers"],
                data["holdings"],
                data["periods"],
                data["reinvest_freq"],
                data["allow_fractional"]
            )
            st.write(f"**Model Name**: {selected_model}")
            st.write(f"**Tickers**: {data['tickers']}")
            st.write(f"**Holdings**: {data['holdings']}")
            st.dataframe(result_df.style.format("{:.4f}"))
            st.line_chart(result_df)

        # --- Additional Visualization: Total Portfolio Value Over Time ---
        total_value = result_df.copy()
        for col in total_value.columns:
            try:
                price = get_price(col)
                total_value[col] = total_value[col] * price
            except:
                total_value[col] = 0
        total_value["Total Value"] = total_value.sum(axis=1)
        st.subheader("Projected Portfolio Value Over Time")
        st.line_chart(total_value["Total Value"])

        # --- Optional: Show Cash Overflow ---
        st.subheader("Cash Overflow (Uninvested Dividends)")
        st.info("This version assumes all dividends are reinvested; to track cash overflow, additional brokerage logic is needed.")

            st.download_button("Download CSV", result_df.to_csv().encode(), f"{selected_model}_simulation.csv", "text/csv")
    else:
        st.info("No saved models found.")


with st.expander("Browse Public Models"):
    public_models = list_public_models()
    if public_models:
        for owner, name, data in public_models:
            st.markdown(f"**{owner}/{name}**")
            st.json(data)
    else:
        st.info("No public models available yet.")
