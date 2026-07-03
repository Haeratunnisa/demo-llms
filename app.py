import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import torch

# ==================== FUNGSI LOAD MODEL (dengan use_fast=False) ====================
@st.cache_resource
def load_gpt2(model_name):
    return pipeline('text-generation', model=model_name, device_map='auto' if torch.cuda.is_available() else None)

@st.cache_resource
def load_flan(model_name='google/flan-t5-base'):
    # use_fast=False agar tidak butuh tiktoken
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto' if torch.cuda.is_available() else None)
    return tokenizer, model

@st.cache_resource
def load_mt5(model_name='google/mt5-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto' if torch.cuda.is_available() else None)
    return tokenizer, model

def generate(prompt, tokenizer, model, max_new_tokens=100, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== CEK BAHASA INDONESIA ====================
def is_indonesian(text):
    return any(ord(c) > 127 for c in text)

# ==================== SIDEBAR ====================
st.set_page_config(page_title="Demo LLM + GPT", layout="wide")
st.title("🚀 Demo LLM + GPT")
st.sidebar.title("🧠 Pilih Demo")
demo = st.sidebar.radio(
    "Demo:",
    [
        "1. Base Model (GPT-2)",
        "2. Zero-shot vs Few-shot",
        "3. Instruction Fine-tuning",
        "4. General vs Custom Build (mT5)"
    ]
)

# ==================== DEMO 1 ====================
if demo == "1. Base Model (GPT-2)":
    st.header("📝 Demo 1: Base Model (Pretraining)")
    st.caption("Model dasar hanya **menyelesaikan kalimat**, bukan menjawab pertanyaan.")
    
    model_choice = st.selectbox("Pilih model:", ["English (gpt2)", "Indonesian (cahya/gpt2-small-indonesian)"])
    model_map = {
        "English (gpt2)": "gpt2",
        "Indonesian (cahya/gpt2-small-indonesian)": "cahya/gpt2-small-indonesian"
    }
    selected = model_map[model_choice]
    default_prompt = "The future of AI is" if "English" in model_choice else "apel adalah"
    prompt = st.text_area("✏️ Awal kalimat:", value=default_prompt)
    max_new = st.slider("📏 Panjang token baru (max_new_tokens)", 5, 50, 16)

    if st.button("🚀 Jalankan", type="primary"):
        with st.spinner("Memuat..."):
            generator = load_gpt2(selected)
            set_seed(42)
        with st.spinner("Menghasilkan..."):
            result = generator(
                prompt,
                max_new_tokens=max_new,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id
            )
        st.success("✅ Output:")
        st.write(result[0]['generated_text'])
        
        if "English" in model_choice and is_indonesian(prompt):
            st.warning("⚠️ Model Inggris + input Indonesia = hasil kacau. Pilih 'Indonesian' untuk hasil lebih baik.")

# ==================== DEMO 2 ====================
elif demo == "2. Zero-shot vs Few-shot":
    st.header("📊 Demo 2: Zero-shot vs Few-shot Learning")
    st.caption("Bandingkan klasifikasi sentimen tanpa contoh vs dengan contoh. **Wajib pakai Bahasa Inggris**.")
    
    model_choice = st.selectbox("Model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
    
    zero_prompt = st.text_area(
        "✏️ Zero-shot prompt (tanpa contoh):",
        value="Classify sentiment of this text: I absolutely love this new phone!",
        height=80
    )
    few_examples = st.text_area(
        "📚 Contoh few-shot:",
        value="Input: The food was terrible and cold.\nOutput: Negative\nInput: The movie was amazing!\nOutput: Positive\nInput: ",
        height=150
    )
    test_text = st.text_input("🧪 Teks uji:", value="I absolutely love this new phone!")
    
    if st.button("🚀 Bandingkan", type="primary"):
        if is_indonesian(zero_prompt) or is_indonesian(test_text):
            st.error("❌ Demo ini hanya mendukung BAHASA INGGRIS. Input Anda mengandung Bahasa Indonesia.")
        else:
            with st.spinner("Memuat Flan-T5..."):
                tokenizer, model = load_flan(model_choice)
            
            with st.spinner("Zero-shot..."):
                z_out = generate(zero_prompt, tokenizer, model, max_new_tokens=20, do_sample=False)
            with st.spinner("Few-shot..."):
                f_prompt = few_examples + test_text
                f_out = generate(f_prompt, tokenizer, model, max_new_tokens=20, do_sample=False)
            
            st.subheader("📋 Hasil")
            colA, colB = st.columns(2)
            colA.metric("Zero-shot", z_out)
            colB.metric("Few-shot", f_out)
            st.info("💡 Few-shot biasanya lebih akurat karena model belajar dari contoh.")

# ==================== DEMO 3 ====================
elif demo == "3. Instruction Fine-tuning":
    st.header("📄 Demo 3: Instruction Fine-tuning")
    
    task = st.selectbox("Pilih tugas:", ["Summarization (Inggris)", "Translation (Inggris → Indonesia)"])
    
    if task == "Summarization (Inggris)":
        model_choice = st.selectbox("Model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
        text_in = st.text_area("📝 Teks Inggris:", 
            value="Large Language Models are neural networks designed to understand and generate text. They use transformers and are trained on massive datasets.")
        if st.button("🔍 Ringkas"):
            tokenizer, model = load_flan(model_choice)
            result = generate(f"Summarize: {text_in}", tokenizer, model, 80)
            st.success(result)
    else:
        text_in = st.text_area("📝 Teks Inggris yang akan diterjemahkan:", 
            value="The weather is beautiful today. I love this city.")
        if st.button("🌐 Terjemahkan"):
            with st.spinner("Memuat mT5 (multilingual)..."):
                tokenizer, model = load_mt5('google/mt5-base')  # use_fast=False otomatis
            result = generate(f"translate English to Indonesian: {text_in}", tokenizer, model, 60)
            st.success(f"🇮🇩 {result}")

# ==================== DEMO 4 (FIX) ====================
else:  # Demo 4
    st.header("⚙️ Demo 4: General Purpose vs Custom Build (Simulasi)")
    st.caption("Menggunakan **mT5-base** (multilingual) sehingga bisa menjawab pertanyaan dalam Bahasa Indonesia.")
    st.info("💡 Perhatikan perbedaan jawaban: General (tanpa konteks) vs Custom (dirole sebagai dokter medis).")
    
    model_choice = st.selectbox("Pilih model mT5:", ["google/mt5-small", "google/mt5-base"])
    
    question = st.text_input(
        "❓ Pertanyaan (bisa Bahasa Indonesia atau Inggris):",
        value="cara mengatasi hipertensi"
    )
    
    if st.button("🚀 Bandingkan", type="primary"):
        with st.spinner(f"Memuat {model_choice}..."):
            tokenizer, model = load_mt5(model_choice)
        
        general_prompt = f"Answer this question: {question}"
        custom_prompt = f"You are a specialized medical doctor. Answer this medical question: {question}"
        
        with st.spinner("General..."):
            general_out = generate(general_prompt, tokenizer, model, max_new_tokens=80, do_sample=False)
        with st.spinner("Custom Medical..."):
            custom_out = generate(custom_prompt, tokenizer, model, max_new_tokens=80, do_sample=False)
        
        st.subheader("📋 Perbandingan Jawaban")
        col1, col2 = st.columns(2)
        col1.write("**🧑‍💻 General Purpose**")
        col1.write(general_out)
        col2.write("**🏥 Custom Medical**")
        col2.write(custom_out)
        
        st.caption(
            "📌 Catatan: Custom Build sejati dilatih dengan data privat (finance/medis) "
            "sehingga akurasi lebih tinggi, privasi terjaga, dan latency lebih rendah."
        )
