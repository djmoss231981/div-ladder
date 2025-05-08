import streamlit as st
import yfinance as yf
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Dividend Ladder Simulator", layout="wide")

# --- Helpers ---
def save_model(name, model_data):
    Path("models").mkdir(exist_ok=True)
    with open(f"models/{name}.json", "w") as f:
        json.dump(model_data, f)

def load_model(name):
    with open(f"models/{name}.json", "r") as f:
        return json.load(f)

def list_models():
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
    freq_map = {'Monthly': 12, 'Quarterly': 4, 'Annually': 1}
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

# --- UI ---
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
            tickers.append(ticker.upper())
            holdings.append(shares)

        periods = st.number_input("Number of Years to Simulate", min_value=1, value=5)
        reinvest_freq = st.selectbox("Dividend Reinvestment Frequency", ['Monthly', 'Quarterly', 'Annually'])
        allow_fractional = st.checkbox("Allow Fractional Shares?", value=True)
        save_option = st.checkbox("Save this model")

        submitted = st.form_submit_button("Simulate")

    if submitted:
        result_df = simulate_model(tickers, holdings, periods, reinvest_freq, allow_fractional)
        st.subheader("Simulation Results")
        st.dataframe(result_df.style.format("{:.4f}"))

        st.line_chart(result_df)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download as CSV", result_df.to_csv().encode(), f"{model_name}_simulation.csv", "text/csv")
        with col2:
            excel_bytes = result_df.to_excel(index=True, engine='openpyxl')
            st.download_button("Download as Excel", excel_bytes, f"{model_name}_simulation.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if save_option and model_name:
            model_data = {
                "tickers": tickers,
                "holdings": holdings,
                "periods": periods,
                "reinvest_freq": reinvest_freq,
                "allow_fractional": allow_fractional
            }
            save_model(model_name, model_data)
            st.success(f"Model '{model_name}' saved.")

with tab2:
    st.subheader("Load Existing Model")
    model_list = list_models()
    if model_list:
        selected_model = st.selectbox("Choose a model to load", model_list)
        if st.button("Load & Simulate"):
            data = load_model(selected_model)
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
            st.download_button("Download CSV", result_df.to_csv().encode(), f"{selected_model}_simulation.csv", "text/csv")
    else:
        st.info("No saved models found.")