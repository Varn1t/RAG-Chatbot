import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript
import os

st.set_page_config(page_title="RAG Bot", page_icon="📚")
st.title("📚 RAG Chatbot")
st.caption("Powered by RAG + llama3")

def build_qa_chain(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    try:
        loader = PyPDFLoader(tmp_path)
        return loader.load()
    finally:
        os.unlink(tmp_path)

def load_youtube(url):
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")
    
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    full_text = " ".join([t.text for t in transcript])
    return [Document(page_content=full_text, metadata={"source": url})]

# UI
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
yt_url = st.text_input("Or paste a YouTube URL")

source_key = None
docs = None

if uploaded_file:
    source_key = uploaded_file.name
elif yt_url:
    source_key = yt_url

if source_key and source_key != st.session_state.get("current_source"):
    st.session_state.current_source = source_key
    st.session_state.messages = []
    with st.spinner("Processing document..."):
        try:
            if uploaded_file:
                docs = process_pdf(uploaded_file)
            else:
                docs = load_youtube(yt_url)
            st.session_state.qa_chain = build_qa_chain(docs)
        except Exception as e:
            st.error(f"Error processing: {e}")
            st.session_state.current_source = None

if source_key and "qa_chain" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input(f"Ask a question about {source_key}..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": query})
                    answer = response["result"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error querying model: {e}")
else:
    if not source_key:
        st.info("☝️ Upload a PDF or paste a YouTube URL to get started.")