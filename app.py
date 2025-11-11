import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="SFX Intelligence: Financial, ESG & Geopolitical AI", layout="wide")
st.title("🌍 SFX Intelligence – AI Insights for Finance, ESG, and Geopolitics")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ App Controls")
default_tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
tickers_input = st.sidebar.text_input(
    "Enter company tickers (comma separated):", 
    ", ".join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

max_tokens = st.sidebar.slider("Max tokens for analysis", 50, 300, 150)
temperature = st.sidebar.slider("Temperature", 0.2, 1.0, 0.7)

# --- Load GPT-2 model once ---
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# --- Financial Data Section ---
st.header("📊 Financial Data Overview")
selected_ticker = st.selectbox("Select a company:", tickers)

try:
    stock = yf.Ticker(selected_ticker)
    hist = stock.history(period="1y")
    info = stock.info

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
    col2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
    col3.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")

    st.line_chart(hist["Close"], height=250)
except Exception as e:
    st.warning(f"Could not retrieve data for {selected_ticker}: {e}")

# --- ESG Section (Fixed Version) ---
st.header("🌱 ESG Snapshot")

esg_data = {"Company": [], "ESG Score": [], "Environmental": [], "Social": [], "Governance": []}

for t in tickers:
    try:
        s = yf.Ticker(t).sustainability
        if s is not None and "Value" in s.columns:
            esg_data["Company"].append(t)
            esg_data["ESG Score"].append(s.loc["totalEsg", "Value"] if "totalEsg" in s.index else None)
            esg_data["Environmental"].append(s.loc["environmentScore", "Value"] if "environmentScore" in s.index else None)
            esg_data["Social"].append(s.loc["socialScore", "Value"] if "socialScore" in s.index else None)
            esg_data["Governance"].append(s.loc["governanceScore", "Value"] if "governanceScore" in s.index else None)
        else:
            esg_data["Company"].append(t)
            esg_data["ESG Score"].append(None)
            esg_data["Environmental"].append(None)
            esg_data["Social"].append(None)
            esg_data["Governance"].append(None)
    except Exception as e:
        esg_data["Company"].append(t)
        esg_data["ESG Score"].append(None)
        esg_data["Environmental"].append(None)
        esg_data["Social"].append(None)
        esg_data["Governance"].append(None)

esg_df = pd.DataFrame(esg_data)
st.dataframe(esg_df)

# --- AI Geopolitical + ESG + Finance Analysis ---
st.header("🤖 AI-Generated Insight")

default_prompt = f"Analyse the financial performance, ESG factors, and geopolitical risks for {selected_ticker} in {datetime.now().year}."
prompt = st.text_area("Enter your question or scenario:", value=default_prompt, height=150)

if st.button("Generate Insight"):
    with st.spinner("Generating insight..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### 🧭 AI Analysis")
        st.write(result)

st.markdown("---")
st.caption("Powered by GPT-2 • Yahoo Finance • Streamlit • SFX Intelligence © 2025")
