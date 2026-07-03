import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

# Cache model loading
@st.cache_resource
def load_gpt2():
    return pipeline('text-generation', model='gpt2', device_map='auto' if torch.cuda.is_available() else None)

@st.cache_resource
def load_flan(model_name='google/flan-t5-small'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto' if torch.cuda.is_available() else None)
    return tokenizer, model

def generate_flan(prompt, tokenizer, model, max_len=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sidebar
st.sidebar.title("Pilih Demo")
demo = st.sidebar.radio(
    "Demo:",
    ("1. Base Model (GPT-2) - Text Completion",
     "2. Zero-shot vs Few-shot Learning",
     "3. Instruction Fine-tuning (Summarization & Translation)",
     "4. General Purpose vs Custom Build (Simulasi)")
)

st.title("Demo LLM + GPT")
st.markdown("Aplikasi ini mendemonstrasikan konsep-konsep dari presentasi tentang Large Language Models.")

if demo == "1. Base Model (GPT-2) - Text Completion":
    st.header("Demo 1: Base Model (Pretraining)")
    st.write("Model GPT-2 (base) hanya dapat menyelesaikan kalimat, bukan menjawab pertanyaan.")
    prompt = st.text_area("Masukkan awal kalimat:", value="The future of artificial intelligence is")
    max_len = st.slider("Panjang maksimum (token)", min_value=10, max_value=100, value=30)
    if st.button("Jalankan"):
        with st.spinner("Menghasilkan..."):
            generator = load_gpt2()
            result = generator(prompt, max_length=max_len, num_return_sequences=1)
            output = result[0]['generated_text']
            st.success("Output:")
            st.write(output)

elif demo == "2. Zero-shot vs Few-shot Learning":
    st.header("Demo 2: Zero-shot vs Few-shot Learning")
    st.write("Bandingkan performa model tanpa contoh (zero-shot) vs dengan contoh (few-shot) untuk klasifikasi sentimen.")
    model_choice = st.selectbox("Pilih model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
    zero_prompt = st.text_area("Zero-shot prompt:", value="Classify the sentiment of this text: I absolutely love this new phone!")
    few_examples = st.text_area("Contoh few-shot (format: Input: ... Output: ...)", 
                                value="Input: The food was terrible and cold.\nOutput: Negative\nInput: The movie was amazing!\nOutput: Positive\nInput: ")
    test_text = st.text_input("Teks yang akan diklasifikasi (akan ditambahkan ke few-shot):", value="I absolutely love this new phone!")
    if st.button("Jalankan Perbandingan"):
        with st.spinner("Memuat model..."):
            tokenizer, model = load_flan(model_choice)
        # Zero-shot
        with st.spinner("Zero-shot..."):
            zero_out = generate_flan(zero_prompt, tokenizer, model, max_len=20)
        # Few-shot: gabungkan contoh + test
        few_prompt = few_examples + test_text
        with st.spinner("Few-shot..."):
            few_out = generate_flan(few_prompt, tokenizer, model, max_len=20)
        st.subheader("Hasil")
        col1, col2 = st.columns(2)
        col1.metric("Zero-shot", zero_out)
        col2.metric("Few-shot", few_out)

elif demo == "3. Instruction Fine-tuning (Summarization & Translation)":
    st.header("Demo 3: Instruction Fine-tuning")
    st.write("Model Flan-T5 di-fine-tune untuk mengikuti instruksi seperti summarization dan translation.")
    model_choice = st.selectbox("Pilih model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
    task = st.selectbox("Pilih tugas", ["Summarization", "Translation (to Indonesian)"])
    if task == "Summarization":
        text_input = st.text_area("Masukkan teks yang akan diringkas:", 
                                  value="Large Language Models are neural networks designed to understand and generate text. They use transformer architectures and are trained on massive datasets. Applications include translation, chatbots, and code generation.")
        if st.button("Ringkas"):
            with st.spinner("Memuat model..."):
                tokenizer, model = load_flan(model_choice)
            prompt = f"Summarize the following text: {text_input}"
            with st.spinner("Merangkum..."):
                result = generate_flan(prompt, tokenizer, model, max_len=80)
            st.success("Ringkasan:")
            st.write(result)
    else:  # Translation
        text_input = st.text_area("Masukkan teks bahasa Inggris yang akan diterjemahkan ke Indonesia:", 
                                  value="The weather is beautiful today.")
        if st.button("Terjemahkan"):
            with st.spinner("Memuat model..."):
                tokenizer, model = load_flan(model_choice)
            prompt = f"Translate to Indonesian: {text_input}"
            with st.spinner("Menerjemahkan..."):
                result = generate_flan(prompt, tokenizer, model, max_len=50)
            st.success("Terjemahan:")
            st.write(result)

else:  # Demo 4
    st.header("Demo 4: General Purpose vs Custom Build (Simulasi)")
    st.write("Bandingkan jawaban model general dengan jawaban yang diarahkan sebagai model spesialis (medis).")
    model_choice = st.selectbox("Pilih model Flan-T5", ["google/flan-t5-small", "google/flan-t5-base"])
    question = st.text_input("Pertanyaan:", value="What is the best treatment for high blood pressure?")
    if st.button("Bandingkan"):
        with st.spinner("Memuat model..."):
            tokenizer, model = load_flan(model_choice)
        general_prompt = f"Answer this: {question}"
        custom_prompt = f"You are a specialized medical AI assistant. Answer the medical question based on standard clinical guidelines.\nQuestion: {question}\nAnswer:"
        with st.spinner("General..."):
            general_out = generate_flan(general_prompt, tokenizer, model, max_len=80)
        with st.spinner("Custom (Medis)..."):
            custom_out = generate_flan(custom_prompt, tokenizer, model, max_len=80)
        st.subheader("Hasil")
        col1, col2 = st.columns(2)
        col1.write("**General Purpose**")
        col1.write(general_out)
        col2.write("**Custom Medical (simulasi)**")
        col2.write(custom_out)
        st.info("Catatan: Ini hanya simulasi. Custom build sejati menggunakan data privat untuk akurasi lebih tinggi dan privasi lebih baik.")
