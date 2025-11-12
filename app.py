import streamlit as st
import pandas as pd
import yfinance as yf
from openai import OpenAI
import world_bank_data as wb

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Stock Analyst", layout="wide")

st.title("🌍 AI-Powered Financial, ESG & Geopolitical Analysis")

# ---- SIDEBAR CONFIG ----
st.sidebar.header("🔧 Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT)", "AAPL")

persona = st.sidebar.selectbox(
    "Select Analyst Persona",
    [
        "Financial Analyst",
        "Sustainability (ESG) Expert",
        "Geopolitical Risk Advisor",
        "Integrated Analyst (All-in-One)"
    ]
)

# ---- LOAD API KEY FROM STREAMLIT SECRETS ----
try:
    openai_api_key = st.secrets["general"]["openai_api_key"]
    client = OpenAI(api_key=openai_api_key)
except Exception:
    st.error("🔑 Please set your OpenAI API key in `.streamlit/secrets.toml` under `[general] openai_api_key`.")
    st.stop()

# ---- FINANCIAL DATA ----
st.subheader(f"📈 Financial Data for {ticker}")

try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info

    if not hist.empty:
        st.line_chart(hist["Close"], use_container_width=True)
    else:
        st.warning("No historical data available.")

    financial_summary = {
        "Market Cap": info.get("marketCap", "N/A"),
        "Revenue (TTM)": info.get("totalRevenue", "N/A"),
        "Gross Margins": info.get("grossMargins", "N/A"),
        "Operating Margins": info.get("operatingMargins", "N/A"),
        "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Country": info.get("country", "United States"),
        "Industry": info.get("industry", "N/A")
    }

    fin_df = pd.DataFrame(financial_summary.items(), columns=["Metric", "Value"])
    st.dataframe(fin_df, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching financial data: {e}")
    st.stop()

# ---- ESG DATA ----
st.subheader("🌱 ESG Data (from Yahoo Finance when available)")

try:
    esg_df = stock.sustainability
    if esg_df is not None:
        st.dataframe(esg_df, use_container_width=True)
        esg_data = esg_df.to_dict()
    else:
        st.warning("No ESG data available for this company — using fallback values.")
        esg_data = {
            "environmentScore": 65,
            "socialScore": 72,
            "governanceScore": 68
        }
except Exception:
    st.warning("ESG data not available — using fallback values.")
    esg_data = {
        "environmentScore": 65,
        "socialScore": 72,
        "governanceScore": 68
    }

# ---- GEOPOLITICAL DATA ----
st.subheader("🗺️ Geopolitical Risk Indicators")

country_name = financial_summary.get("Country", "United States")

# Map common Yahoo country names to World Bank ISO codes
country_map = {
    "United States": "US",
    "United Kingdom": "GB",
    "Germany": "DE",
    "France": "FR",
    "China": "CN",
    "Japan": "JP",
    "Brazil": "BR",
    "India": "IN",
    "Canada": "CA",
    "Australia": "AU",
    "Netherlands": "NL",
}

country_code = country_map.get(country_name, "US")

try:
    gpi = wb.get_series(
        ["PV.PER.RNK", "GE.PER.RNK"],  # Political Stability, Government Effectiveness
        id_or_value="id",
        simplify_index=True,
        country=country_code,
        date="2020:2023"
    ).mean().to_dict()

    gpi_data = {
        "Country": country_name,
        "Political Stability (Rank)": round(gpi.get("PV.PER.RNK", 50), 2),
        "Governance Effectiveness (Rank)": round(gpi.get("GE.PER.RNK", 50), 2)
    }

    geo_df = pd.DataFrame(gpi_data.items(), columns=["Indicator", "Value"])
    st.dataframe(geo_df, use_container_width=True)

except Exception:
    st.warning("Unable to fetch geopolitical data — using average global benchmarks.")
    gpi_data = {
        "Country": country_name,
        "Political Stability (Rank)": 50,
        "Governance Effectiveness (Rank)": 55
    }

# ---- USER PROMPT ----
st.subheader("🧠 AI Analysis")

user_prompt = st.text_area(
    "Enter a custom question or leave blank for automatic analysis:",
    "Provide a holistic assessment of the company's financial performance, ESG profile, and geopolitical risks for 2025."
)

# ---- SYSTEM PROMPT BASED ON PERSONA ----
if persona == "Financial Analyst":
    system_prompt = (
        "You are a senior financial analyst. Focus on profitability, growth, valuation ratios, and macroeconomic factors."
    )
elif persona == "Sustainability (ESG) Expert":
    system_prompt = (
        "You are an ESG and sustainability expert. Assess environmental, social, and governance performance "
        "and its long-term implications for investors."
    )
elif persona == "Geopolitical Risk Advisor":
    system_prompt = (
        "You are a geopolitical risk advisor. Assess how global politics, trade dynamics, and regional risks "
        "affect the company's operations and market outlook."
    )
else:
    system_prompt = (
        "You are an integrated analyst combining financial, ESG, and geopolitical insights. "
        "Provide a balanced strategic assessment with actionable investment implications."
    )

# ---- RUN ANALYSIS ----
if st.button("Run Analysis"):
    with st.spinner("Generating AI analysis..."):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Company: {ticker}\n"
                f"Financials: {financial_summary}\n"
                f"ESG: {esg_data}\n"
                f"Geopolitics: {gpi_data}\n"
                f"Task: {user_prompt}"
            )}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            analysis = response.choices[0].message.content
            st.markdown(f"### 🧩 {persona} Insights")
            st.write(analysis)

        except Exception as e:
            st.error(f"Error generating AI analysis: {e}")
