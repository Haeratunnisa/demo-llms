import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import torch

# ============================================================
# FUNGSI LOAD MODEL (pakai default, tidak pakai use_fast=False)
# ============================================================
@st.cache_resource
def load_gpt2():
    return pipeline('text-generation', model='distilgpt2')

@st.cache_resource
def load_flan(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_mt5(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate(prompt, tokenizer, model, max_new=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_indonesian(text):
    return any(ord(c) > 127 for c in text)

# ============================================================
# UI SIDEBAR
# ============================================================
st.set_page_config(page_title="Demo LLM + GPT", layout="wide")
st.title("🚀 Demo LLM + GPT")
st.sidebar.title("🧠 Pilih Demo")
demo = st.sidebar.radio("", [
    "1. Base Model (Text Completion)",
    "2. Zero-shot vs Few-shot",
    "3. Instruction Fine-tuning",
    "4. General vs Custom Build"
])

# ============================================================
# DEMO 1
# ============================================================
if demo == "1. Base Model (Text Completion)":
    st.header("📝 Demo 1: Base Model (Pretraining)")
    st.caption("Model dasar hanya **menyelesaikan kalimat**, bukan menjawab pertanyaan.")
    st.info("ℹ️ Model `distilgpt2` dilatih dengan teks **Bahasa Inggris**. Input Indonesia akan menghasilkan output acak.")

    prompt = st.text_area("✏️ Awal kalimat:", value="The future of AI is")
    max_new = st.slider("📏 Jumlah token baru", 5, 50, 16)

    if st.button("🚀 Jalankan", type="primary"):
        with st.spinner("Memuat model..."):
            generator = load_gpt2()
            set_seed(42)
        with st.spinner("Menghasilkan..."):
            result = generator(prompt, max_new_tokens=max_new, do_sample=False)
        st.success("✅ Output:")
        st.write(result[0]['generated_text'])

        if is_indonesian(prompt):
            st.warning("⚠️ Anda memasukkan huruf Indonesia, model ini tidak mendukungnya. Hasil di atas kemungkinan tidak masuk akal.")

# ============================================================
# DEMO 2
# ============================================================
elif demo == "2. Zero-shot vs Few-shot":
    st.header("📊 Demo 2: Zero-shot vs Few-shot")
    st.caption("Klasifikasi sentimen. **Wajib Bahasa Inggris**.")
    st.warning("⚠️ Flan-T5 hanya mendukung Bahasa Inggris.")

    model_choice = st.selectbox("Model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
    zero_prompt = st.text_area("Zero-shot prompt:", value="Classify sentiment: I love this!", height=80)
    few_examples = st.text_area("Contoh few-shot:", 
        value="Input: Bad\nOutput: Negative\nInput: Good\nOutput: Positive\nInput: ", height=150)
    test_text = st.text_input("Teks uji:", value="I absolutely love this!")

    if st.button("🚀 Bandingkan", type="primary"):
        if is_indonesian(zero_prompt) or is_indonesian(test_text):
            st.error("❌ Demo ini hanya untuk BAHASA INGGRIS.")
        else:
            with st.spinner("Memuat model..."):
                tokenizer, model = load_flan(model_choice)
            with st.spinner("Zero-shot..."):
                z_out = generate(zero_prompt, tokenizer, model, 20)
            with st.spinner("Few-shot..."):
                f_out = generate(few_examples + test_text, tokenizer, model, 20)
            
            col1, col2 = st.columns(2)
            col1.metric("Zero-shot", z_out)
            col2.metric("Few-shot", f_out)
            st.info("💡 Few-shot biasanya lebih akurat karena ada contoh.")

# ============================================================
# DEMO 3
# ============================================================
elif demo == "3. Instruction Fine-tuning":
    st.header("📄 Demo 3: Instruction Fine-tuning")
    task = st.selectbox("Tugas:", ["Summarization (Inggris)", "Translation (Inggris → Indonesia)"])

    if task == "Summarization (Inggris)":
        model_choice = st.selectbox("Model", ["google/flan-t5-small", "google/flan-t5-base"])
        text_in = st.text_area("📝 Teks Inggris:", 
            value="Large Language Models are neural networks designed to understand text. They use transformers.")
        if st.button("🔍 Ringkas"):
            tokenizer, model = load_flan(model_choice)
            result = generate(f"Summarize: {text_in}", tokenizer, model, 80)
            st.success(result)

    else:
        text_in = st.text_area("📝 Teks Inggris:", value="The weather is beautiful today.")
        if st.button("🌐 Terjemahkan"):
            with st.spinner("Memuat mT5-small..."):
                tokenizer, model = load_mt5('google/mt5-small')
            result = generate(f"translate English to Indonesian: {text_in}", tokenizer, model, 60)
            st.success(f"🇮🇩 {result}")

# ============================================================
# DEMO 4
# ============================================================
else:
    st.header("⚙️ Demo 4: General vs Custom Build (Simulasi)")
    st.caption("Menggunakan **mT5-small** (multilingual) agar bisa menjawab pertanyaan Indonesia.")

    model_choice = st.selectbox("Model mT5:", ["google/mt5-small", "google/mt5-base"])
    question = st.text_input("❓ Pertanyaan:", value="cara mengatasi hipertensi")

    if st.button("🚀 Bandingkan", type="primary"):
        with st.spinner("Memuat model..."):
            tokenizer, model = load_mt5(model_choice)
        
        general = generate(f"Answer this question: {question}", tokenizer, model, 80)
        custom = generate(f"You are a medical doctor. Answer this medical question: {question}", tokenizer, model, 80)

        col1, col2 = st.columns(2)
        col1.write("**🧑‍💻 General Purpose**")
        col1.write(general)
        col2.write("**🏥 Custom Medical**")
        col2.write(custom)

        st.caption("📌 Custom Build sejati dilatih data privat sehingga lebih akurat dan aman.")
