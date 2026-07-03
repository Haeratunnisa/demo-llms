import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Demo Translation LLM", layout="centered")
st.title("🌐 Demo Terjemahan Inggris → Indonesia")
st.caption("Menggunakan model mT5-small (multilingual)")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_translator():
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate(text):
    tokenizer, model = load_translator()
    prompt = f"translate English to Indonesian: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== UI ====================
english_text = st.text_area(
    "📝 Masukkan teks bahasa Inggris:",
    value="The weather is beautiful today. I love this city."
)

if st.button("🌐 Terjemahkan", type="primary"):
    if not english_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menerjemahkan..."):
            result = translate(english_text)
        st.success("🇮🇩 Hasil terjemahan:")
        st.write(result)
