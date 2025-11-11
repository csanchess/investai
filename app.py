import streamlit as st
import gpt_2_simple as gpt2
import os
from build_dataset import build_financial_esg_corpus

st.title("💹 ESG & Financial AI Assistant (GPT-2)")

MODELS_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.sidebar.header("Setup")
tickers_input = st.sidebar.text_input("Enter company tickers (comma-separated):", "AAPL,MSFT,GOOG")
num_days = st.sidebar.slider("Number of days of historical data:", 30, 180, 90)

model_name = "124M"

# Step 1: Build Dataset
if st.sidebar.button("📘 Build ESG-Financial Dataset"):
    tickers = [t.strip() for t in tickers_input.split(",")]
    dataset_path = build_financial_esg_corpus(tickers, num_days, output_file=os.path.join(DATA_DIR, "financial_esg_corpus.txt"))
    st.session_state["dataset_path"] = dataset_path
    st.success(f"✅ Dataset built: {dataset_path}")

# Step 2: Fine-tune GPT-2
if st.sidebar.button("⚙️ Fine-tune GPT-2 on Dataset"):
    if "dataset_path" not in st.session_state:
        st.error("Please build the dataset first.")
    else:
        sess = gpt2.start_tf_sess()
        st.info("Training started... may take a few minutes.")
        gpt2.finetune(sess,
                      dataset=st.session_state["dataset_path"],
                      model_name=model_name,
                      steps=200,
                      restore_from='fresh',
                      run_name='run1',
                      print_every=10,
                      sample_every=50,
                      save_every=100)
        gpt2.save_gpt2(sess)
        st.success("✅ Fine-tuning complete!")
        st.session_state["trained"] = True

# Step 3: Generate Insights
st.markdown("## 💬 Ask a Question or Write a Prompt")
prompt = st.text_area("Example: *Compare Tesla and Microsoft ESG performance.*")

if st.button("🧠 Generate Insight"):
    sess = gpt2.start_tf_sess()
    try:
        if "trained" in st.session_state and st.session_state["trained"]:
            gpt2.load_gpt2(sess, run_name="run1")
        else:
            gpt2.load_gpt2(sess, model_name=model_name)

        output = gpt2.generate(sess,
                               prefix=prompt,
                               length=150,
                               temperature=0.7,
                               return_as_list=True)[0]

        st.markdown("### 🧩 Generated Insight")
        st.write(output)

    except Exception as e:
        st.error(f"Error generating text: {e}")
