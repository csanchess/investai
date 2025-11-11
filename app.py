import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# --- Page setup ---
st.set_page_config(page_title="Geopolitical Risk GPT-2", layout="wide")
st.title("🌍 Geopolitical Risk Analysis — GPT-2")

# --- Load model ---
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# --- Sidebar controls ---
st.sidebar.header("⚙️ Model Settings")
max_new_tokens = st.sidebar.slider("Max tokens", 20, 300, 100)
temperature = st.sidebar.slider("Temperature", 0.2, 1.0, 0.7)

# --- User input ---
prompt = st.text_area(
    "Enter a scenario, question, or topic:",
    value="Geopolitical risks in 2025 related to energy and global trade include",
    height=150,
)

# --- Generate text ---
if st.button("Generate Analysis"):
    with st.spinner("Analysing..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### 🧩 Generated Analysis")
        st.write(result)

st.markdown("---")
st.caption("Powered by Hugging Face GPT-2 · Streamlit · Open Source")
