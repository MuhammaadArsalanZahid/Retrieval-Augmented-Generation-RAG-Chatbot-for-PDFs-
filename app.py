"""app.py - Streamlit interface (Step 9). Run with: streamlit run app.py"""
import streamlit as st
from loader import load_pdf, chunk_pages
from vector_store import VectorStore, load_embedder
from generator import load_generator, generate_answer, NOT_FOUND

st.set_page_config(page_title="PDF Question Answering", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models (first time can take a few minutes)...")
def get_models():
    embedder = load_embedder()
    tok, model = load_generator()
    return embedder, tok, model


embedder, tokenizer, gen_model = get_models()

# ---- session state: keeps index and chat so the page does not redo work ----
st.session_state.setdefault("store", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc_names", [])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📁 Document")
    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size (characters)", 200, 1000, 500, 50)
    overlap = st.slider("Chunk overlap (characters)", 0, 300, 100, 10)
    top_k = st.slider("Passages to retrieve", 1, 6, 3)
    threshold = st.slider(
    "Relevance threshold",
    0.0, 1.0, 0.55, 0.05,
    help="If the best match scores below this, the app says 'not found'. A higher threshold may reject relevant passages."
)

    if st.button("Process document", type="primary", use_container_width=True):
        if not files:
            st.warning("Please upload at least one PDF first.")
        elif overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            all_pages, bad_files = [], []
            for f in files:
                try:
                    pages = load_pdf(f, f.name)
                except Exception:
                    st.error(f"Could not read '{f.name}'. Is it a valid PDF?")
                    continue
                if not pages:
                    bad_files.append(f.name)
                all_pages.extend(pages)
            for name in bad_files:
                st.warning(f"'{name}' has no extractable text (probably a scanned PDF).")
            if all_pages:
                chunks = chunk_pages(all_pages, chunk_size, overlap)
                bar = st.progress(0.0, text="Creating embeddings...")
                store = VectorStore(embedder)
                store.build(chunks, progress_callback=lambda x: bar.progress(x, text="Creating embeddings..."))
                bar.empty()
                st.session_state.store = store
                st.session_state.doc_names = [f.name for f in files]
                st.session_state.messages = []
                st.success(f"Done! {len(all_pages)} pages, {len(chunks)} chunks.")

    if st.session_state.doc_names:
        st.caption("Loaded: " + ", ".join(st.session_state.doc_names))

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages)
        st.download_button("⬇️ Download chat", text, "chat.txt", use_container_width=True)

# ---------------- Main area ----------------
st.title("📄 PDF Question Answering")
st.write("Upload a PDF, click **Process document**, then ask questions. Answers come only from your document, with page numbers.")


def show_sources(sources):
    with st.expander("Sources"):
        for c, score in sources:
            st.markdown(f"**{c['file']} — page {c['page']}** (score {score:.2f})")
            st.caption(c["text"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a question about your document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        store = st.session_state.store
        if store is None:
            answer, sources = "Please upload and process a PDF first (use the sidebar).", []
            st.markdown(answer)
        elif not question.strip():
            answer, sources = "Please type a question.", []
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                results = store.search(question, k=top_k)
                if not results or results[0][1] < threshold:
                    answer, sources = NOT_FOUND, []
                else:
                    answer = generate_answer(question, [c for c, _ in results], tokenizer, gen_model)
                    sources = [] if answer == NOT_FOUND else results
            st.markdown(answer)
            if sources:
                pages = sorted({c["page"] for c, _ in sources})
                st.caption("Pages: " + ", ".join(map(str, pages)))
                show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
