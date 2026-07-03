import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import torch

# ============================================================
# MODEL LOADER (dengan cache, pakai model yang sudah terbukti stabil)
# ============================================================

@st.cache_resource
def load_gpt2():
    # GPT-2 paling kecil dan stabil
    return pipeline('text-generation', model='distilgpt2')

@st.cache_resource
def load_flan(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_translator():
    # Model terjemahan khusus Inggris-Indonesia (stabil, kecil)
    model_name = "Helsinki-NLP/opus-mt-en-id"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ============================================================
# FUNGSI GENERATE
# ============================================================

def generate(prompt, tokenizer, model, max_new=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate(text):
    tokenizer, model = load_translator()
    # Model opus-mt-en-id langsung menerima teks Inggris
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_indonesian(text):
    return any(ord(c) > 127 for c in text)

# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Demo LLM", layout="wide")
st.title("🚀 Demo LLM + GPT")
st.sidebar.title("🧠 Pilih Demo")

demo = st.sidebar.radio("", [
    "1. Base Model (Text Completion)",
    "2. Zero-shot vs Few-shot",
    "3. Translation (Inggris → Indonesia)",
    "4. General vs Custom Build"
])

# ============================================================
# DEMO 1
# ============================================================

if demo == "1. Base Model (Text Completion)":
    st.header("📝 Demo 1: Base Model (Pretraining)")
    st.caption("Model dasar hanya **menyelesaikan kalimat**, bukan menjawab pertanyaan.")
    st.info("ℹ️ `distilgpt2` dilatih dengan teks Inggris. Input Indonesia hasilnya acak.")

    prompt = st.text_area("✏️ Awal kalimat:", value="The future of AI is")
    max_new = st.slider("📏 Jumlah token baru", 5, 50, 16)

    if st.button("🚀 Jalankan", type="primary"):
        with st.spinner("Memuat..."):
            generator = load_gpt2()
            set_seed(42)
        with st.spinner("Menghasilkan..."):
            result = generator(prompt, max_new_tokens=max_new, do_sample=False)
        st.success("✅ Output:")
        st.write(result[0]['generated_text'])

        if is_indonesian(prompt):
            st.warning("⚠️ Input mengandung huruf Indonesia, model Inggris tidak mendukung.")

# ============================================================
# DEMO 2
# ============================================================

elif demo == "2. Zero-shot vs Few-shot":
    st.header("📊 Demo 2: Zero-shot vs Few-shot")
    st.caption("Klasifikasi sentimen. **Wajib Bahasa Inggris**.")
    st.warning("⚠️ Flan-T5 hanya mendukung Inggris.")

    model_choice = st.selectbox("Model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
    zero_prompt = st.text_area("Zero-shot prompt:", value="Classify sentiment: I love this!", height=80)
    few_examples = st.text_area("Contoh few-shot:", 
        value="Input: Bad\nOutput: Negative\nInput: Good\nOutput: Positive\nInput: ", height=150)
    test_text = st.text_input("Teks uji:", value="I absolutely love this!")

    if st.button("🚀 Bandingkan", type="primary"):
        if is_indonesian(zero_prompt) or is_indonesian(test_text):
            st.error("❌ Hanya untuk BAHASA INGGRIS.")
        else:
            with st.spinner("Memuat..."):
                tokenizer, model = load_flan(model_choice)
            with st.spinner("Zero-shot..."):
                z_out = generate(zero_prompt, tokenizer, model, 20)
            with st.spinner("Few-shot..."):
                f_out = generate(few_examples + test_text, tokenizer, model, 20)
            col1, col2 = st.columns(2)
            col1.metric("Zero-shot", z_out)
            col2.metric("Few-shot", f_out)
            st.info("💡 Few-shot lebih akurat karena ada contoh.")

# ============================================================
# DEMO 3 (TRANSLATION) - PAKAI OPUS-MT YANG STABIL
# ============================================================

elif demo == "3. Translation (Inggris → Indonesia)":
    st.header("🌐 Demo 3: Translation")
    st.caption("Menggunakan **Helsinki-NLP/opus-mt-en-id**, model terjemahan khusus Inggris-Indonesia.")

    text_in = st.text_area("✏️ Teks bahasa Inggris:", value="The weather is beautiful today.")
    if st.button("🌐 Terjemahkan", type="primary"):
        if text_in.strip():
            with st.spinner("Menerjemahkan..."):
                result = translate(text_in)
            st.success("🇮🇩 Hasil:")
            st.write(result)
        else:
            st.warning("Teks kosong.")

# ============================================================
# DEMO 4
# ============================================================

else:
    st.header("⚙️ Demo 4: General vs Custom Build")
    st.caption("Simulasi perbedaan jawaban general vs spesialis medis. Gunakan mT5-small.")

    model_choice = st.selectbox("Model mT5:", ["google/mt5-small", "google/mt5-base"])
    question = st.text_input("❓ Pertanyaan:", value="cara mengatasi hipertensi")

    if st.button("🚀 Bandingkan", type="primary"):
        with st.spinner("Memuat..."):
            tokenizer, model = load_flan(model_choice)  # Flan-T5 tidak bisa Indonesia, tapi mT5 bisa
            # Kita override dengan mT5
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            mt_tokenizer = AutoTokenizer.from_pretrained(model_choice)
            mt_model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)
        general = generate(f"Answer: {question}", mt_tokenizer, mt_model, 80)
        custom = generate(f"You are a medical doctor. Answer: {question}", mt_tokenizer, mt_model, 80)
        col1, col2 = st.columns(2)
        col1.write("**General**")
        col1.write(general)
        col2.write("**Custom Medical**")
        col2.write(custom)
