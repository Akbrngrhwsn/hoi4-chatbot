import os
import requests
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─── 1. KONFIGURASI APLIKASI DAN API KEY ───────────────────────────────────
# Memuat environment variables dari file .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Mengatur konfigurasi halaman Streamlit. Harus menjadi perintah pertama.
st.set_page_config(
    page_title="HOI4 RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. FUNGSI UTAMA & PEMUATAN MODEL ─────────────────────────────────────

@st.cache_resource
def load_models_and_vectorstore():
    """
    Memuat semua model dan vectorstore sekali saja untuk efisiensi.
    Streamlit akan menyimpan resource ini dalam cache.
    """
    st.info("📥 Memuat knowledge base (FAISS) dan model embedding...")
    try:
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vstore_path = "hoi4_vectorstore"
        if not os.path.exists(vstore_path):
            st.error(f"Folder vectorstore '{vstore_path}' tidak ditemukan. Pastikan path sudah benar.")
            st.stop()
        vectorstore = FAISS.load_local(
            vstore_path, embeddings=embedder, allow_dangerous_deserialization=True
        )
        evaluator = SentenceTransformer("all-MiniLM-L6-v2")
        st.success("✅ Knowledge base dan model embedding siap.")
        return vectorstore, evaluator
    except Exception as e:
        st.error(f"Gagal memuat model atau vectorstore: {e}")
        st.stop()

def retrieve_context(query: str, k: int = 4) -> str:
    """
    Mengambil potongan konteks yang relevan dari vectorstore.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)

def groq_chat(model_name: str, question: str, context: str) -> tuple[str, float]:
    """
    Mengirim request ke Groq API dan mengembalikan respons serta waktu eksekusi.
    """
    HEADERS = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Anda adalah seorang ahli strategi dan sejarawan game Hearts of Iron IV. Berikan jawaban yang detail, akurat, dan relevan berdasarkan konteks yang diberikan."},
            {"role": "user", "content": f"Berdasarkan konteks berikut:\n---\n{context}\n---\n\nJawab pertanyaan ini:\n{question}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1024, # Token ditambah untuk jawaban lebih panjang
        "top_p": 0.9,
    }
    
    start_time = time.time()
    try:
        r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=90)
        end_time = time.time()
        response_time = end_time - start_time
        
        if r.status_code != 200:
            error_message = f"❌ Error {r.status_code}: {r.text}"
            return error_message, response_time
            
        response_content = r.json()["choices"][0]["message"]["content"]
        return response_content, response_time
        
    except requests.exceptions.Timeout:
        return "❌ Error: Permintaan timeout setelah 90 detik.", time.time() - start_time
    except Exception as e:
        return f"❌ Error: Terjadi exception - {e}", time.time() - start_time

def evaluate_similarity(question: str, answer: str) -> float:
    """
    Menghitung kemiripan semantik antara pertanyaan dan jawaban.
    """
    if not answer or answer.startswith("❌"):
        return 0.0
    try:
        q_embedding = evaluator.encode([question])
        a_embedding = evaluator.encode([answer])
        return cosine_similarity(q_embedding, a_embedding)[0][0]
    except Exception:
        return 0.0

# ─── 3. STRUKTUR DAN TATA LETAK GUI (STREAMLIT) ─────────────────────────────

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 HoI4 RAG Chatbot")
    st.markdown("---")
    st.header("Navigasi")
    page = st.radio("Pilih Halaman:", ["🏠 Halaman Utama", "📖 Cara Penggunaan", "ℹ️ Tentang & Kredit"])
    st.markdown("---")
    st.info(
        "Prototipe ini membandingkan dua model LLM dari Meta yang di-host di Groq "
        "untuk menjawab pertanyaan seputar game *Hearts of Iron IV*."
    )

# Muat model dan vectorstore
vectorstore, evaluator = load_models_and_vectorstore()

# --- Inisialisasi Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================================================================
# ─── HALAMAN UTAMA (CHATBOT) ──────────────────────────────────────────────────
# ==============================================================================
if page == "🏠 Halaman Utama":
    st.title("🧠 Chatbot Perbandingan Model HoI4")
    st.markdown(
        "Ajukan pertanyaan tentang *Hearts of Iron IV*! Sistem **Retrieval-Augmented Generation (RAG)** akan mencari "
        "informasi relevan dari knowledge base, lalu dua model AI akan memberikan jawaban berdasarkan informasi tersebut."
    )
    # Memperbarui deskripsi model yang dibandingkan
    st.markdown(
        """
        **Model yang Dibandingkan:**
        - **LLaMA 3 8B (Meta):** Model dasar yang cepat dan efisien.
        - **Gemma 2 9B (Google):** Model generasi baru dari Google, dikenal dengan performa tinggi.
        """
    )
    st.markdown("---")

    # Cek API Key di awal
    if not GROQ_API_KEY:
        st.error("Kunci API Groq (GROQ_API_KEY) belum diatur. Mohon atur di file .env Anda.")
        st.stop()

    question = st.text_input(
        "💬 **Ajukan pertanyaan Anda di sini:**",
        placeholder="Contoh: Bagaimana cara kerja sistem naval invasion?",
        key="main_question_input"
    )

    if question:
        with st.spinner("🔍 Mencari konteks dan memproses jawaban dari kedua model..."):
            retrieved_ctx = retrieve_context(question)
            
            # Panggil model LLaMA 3 8B
            ans_8b, time_8b   = groq_chat("llama3-8b-8192", question, retrieved_ctx)
            
            # Panggil model Gemma 2 9B
            ans_gemma, time_gemma = groq_chat("gemma2-9b-it", question, retrieved_ctx)
            
            # Evaluasi similaritas
            sim_8b = evaluate_similarity(question, ans_8b)
            sim_gemma = evaluate_similarity(question, ans_gemma)

            # Simpan hasil ke session state dengan nama variabel baru
            st.session_state.history.insert(0, {
                "question": question,
                "context": retrieved_ctx,
                "ans_8b": ans_8b, "time_8b": time_8b, "sim_8b": sim_8b,
                "ans_gemma": ans_gemma, "time_gemma": time_gemma, "sim_gemma": sim_gemma
            })

    st.markdown("---")
    st.subheader("📜 Riwayat Percakapan & Analisis Perbandingan")

    if not st.session_state.history:
        st.info("Belum ada percakapan. Mulailah dengan mengajukan pertanyaan di atas.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.container(border=True):
                st.markdown(f"#### 💬 Pertanyaan #{len(st.session_state.history) - i}: {entry['question']}")
                
                col1, col2 = st.columns(2)
                
                # --- Kolom Model LLaMA 3 8B ---
                with col1:
                    st.markdown("##### 🦙 LLaMA 3 8B (Meta)")
                    st.success(f"{entry['ans_8b']}", icon="✅")
                    
                    # --- PERUBAHAN TAMPILAN STATISTIK DI SINI ---
                    st.markdown("**Analisis Performa:**")
                    stat_cols_1 = st.columns(3)
                    stat_cols_1[0].metric(label="Waktu Respons", value=f"{entry['time_8b']:.2f} s")
                    stat_cols_1[1].metric(label="Similaritas", value=f"{entry['sim_8b']:.2%}")
                    stat_cols_1[2].metric(label="Jumlah Kata", value=f"{len(entry['ans_8b'].split())}")
                
                # --- Kolom Model Gemma 2 9B ---
                with col2:
                    st.markdown("##### ✨ Gemma 2 9B (Google)")
                    st.info(f"{entry['ans_gemma']}", icon="💡")
                    
                    # --- PERUBAHAN TAMPILAN STATISTIK DI SINI ---
                    st.markdown("**Analisis Performa:**")
                    stat_cols_2 = st.columns(3)
                    stat_cols_2[0].metric(label="Waktu Respons", value=f"{entry['time_gemma']:.2f} s")
                    stat_cols_2[1].metric(label="Similaritas", value=f"{entry['sim_gemma']:.2%}")
                    stat_cols_2[2].metric(label="Jumlah Kata", value=f"{len(entry['ans_gemma'].split())}")
                
                with st.expander("Lihat Konteks yang Digunakan (Dari Knowledge Base)"):
                    st.text(entry['context'])

# ==============================================================================
# ─── HALAMAN CARA PENGGUNAAN ──────────────────────────────────────────────────
# ==============================================================================
elif page == "📖 Cara Penggunaan":
    st.title("📖 Panduan Penggunaan Chatbot")
    st.markdown("---")
    st.info("Halaman ini menjelaskan cara kerja aplikasi ini dan bagaimana cara menggunakannya secara efektif.", icon="💡")

    st.header("1. Konsep Dasar: Retrieval-Augmented Generation (RAG)")
    st.markdown(
        """
        Chatbot ini tidak hanya mengandalkan pengetahuan internal dari model AI. Sebaliknya, ia menggunakan metode **RAG**:
        1.  **Retrieval (Pengambilan):** Saat Anda mengajukan pertanyaan, sistem terlebih dahulu mencari informasi yang paling relevan dari sebuah "knowledge base" khusus (Vectorstore FAISS) yang berisi data tentang *Hearts of Iron IV*.
        2.  **Augmentation (Penambahan):** Informasi yang ditemukan ini kemudian digabungkan dengan pertanyaan Anda sebagai "konteks" tambahan.
        3.  **Generation (Pembuatan):** Pertanyaan asli beserta konteks yang relevan dikirim ke model AI (LLaMA 3 8B dan 70B), yang kemudian menghasilkan jawaban yang jauh lebih akurat dan spesifik berdasarkan konteks tersebut.
        """
    )

    st.header("2. Langkah-langkah Penggunaan")
    st.markdown(
        """
        1.  **Buka Halaman Utama:** Pastikan Anda berada di halaman "🏠 Halaman Utama" yang bisa dipilih dari navigasi di sidebar kiri.
        2.  **Ketik Pertanyaan:** Gunakan kotak input untuk mengetik pertanyaan Anda mengenai mekanik, strategi, atau aspek lain dari game HoI4.
        3.  **Tunggu Jawaban:** Setelah menekan Enter, aplikasi akan memproses permintaan Anda. Ini mungkin memakan waktu beberapa detik karena melibatkan pencarian konteks dan pemanggilan dua model AI.
        4.  **Analisis Hasil:** Dua jawaban akan muncul berdampingan, satu dari model LLaMA 3 8B dan satu dari LLaMA 3 70B.
        """
    )
    
    st.header("3. Memahami Analisis Perbandingan")
    st.markdown(
        """
        Di bawah setiap jawaban, Anda akan menemukan kotak analisis dengan metrik berikut:
        - **Waktu Respons:** Berapa lama (dalam detik) model membutuhkan waktu untuk menghasilkan jawaban. Biasanya, model yang lebih kecil (8B) lebih cepat.
        - **Similaritas Jawaban:** Mengukur seberapa relevan jawaban secara semantik dengan pertanyaan Anda (menggunakan model `all-MiniLM-L6-v2`). Skor yang lebih tinggi (mendekati 100%) menunjukkan jawaban yang lebih relevan.
        - **Jumlah Kata:** Total kata dalam jawaban yang dihasilkan. Ini membantu mengukur keringkasan atau kedalaman jawaban.
        - **Lihat Konteks:** Anda bisa membuka bagian ini untuk melihat teks mentah dari knowledge base yang digunakan oleh AI untuk menyusun jawabannya.
        """
    )

# ==============================================================================
# ─── HALAMAN TENTANG & KREDIT ────────────────────────────────────────────────
# ==============================================================================
elif page == "ℹ️ Tentang & Kredit":
    st.title("ℹ️ Tentang Prototipe dan Kredit")
    st.markdown("---")
    
    st.header("Identitas Prototipe")
    st.markdown(
        """
        - **Nama Aplikasi:** HoI4 RAG Chatbot
        - **Tujuan:** Menjadi asisten cerdas untuk pemain *Hearts of Iron IV* dengan menyediakan jawaban yang akurat dan relevan secara kontekstual. Prototipe ini juga berfungsi sebagai platform untuk membandingkan performa dua model Large Language Model (LLM) yang berbeda dalam tugas yang sama.
        - **Versi:** 2.0 (Revisi dengan GUI dan Analisis Lengkap)
        - **Pengembang Modifikasi:** Gemini (Google AI)
        """
    )

    st.header("Kredit & Teknologi yang Digunakan")
    st.markdown(
        """
        Aplikasi ini tidak akan terwujud tanpa teknologi dan pustaka sumber terbuka yang luar biasa berikut ini:

        - **Framework Aplikasi:**
          - [Streamlit](https.streamlit.io/): Untuk membangun antarmuka pengguna (GUI) yang interaktif dan modern dengan Python.
        
        - **Model Bahasa & Inferensi:**
          - [Groq](https://groq.com/): Menyediakan platform inferensi LLM super cepat.
          - [Meta LLaMA 3 (8B & 70B)](https://ai.meta.com/blog/meta-llama-3/): Model dasar yang digunakan untuk menghasilkan jawaban.
        
        - **Retrieval-Augmented Generation (RAG):**
          - [LangChain](https://www.langchain.com/): Framework untuk mengorkestrasi alur RAG, menghubungkan komponen-komponen seperti vectorstore dan LLM.
          - [FAISS](https://faiss.ai/) (dari Meta AI): Pustaka untuk pencarian kemiripan yang efisien, digunakan sebagai vectorstore.
          - [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers): Untuk model embedding (`all-MiniLM-L6-v2`) yang mengubah teks menjadi vektor numerik untuk pencarian.

        - **Pustaka Python Lainnya:**
          - `requests`: Untuk melakukan panggilan HTTP ke API Groq.
          - `python-dotenv`: Untuk manajemen variabel lingkungan (API Key).
          - `scikit-learn`: Untuk menghitung *cosine similarity*.
        """
    )