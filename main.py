import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

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

def load_pdf(path_or_url):
    loader = PyPDFLoader(path_or_url)
    return loader.load()

def main():
    print("Welcome to the CLI RAG Chatbot!")
    print("1. Load a PDF (URL or local path)")
    print("2. Load a YouTube Video URL")
    
    choice = input("Select an option (1 or 2): ").strip()
    
    pages = []
    if choice == "1":
        path = input("Enter PDF URL or local path: ").strip()
        print(f"Loading PDF from {path}...")
        try:
            pages = load_pdf(path)
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return
    elif choice == "2":
        url = input("Enter YouTube Video URL: ").strip()
        print(f"Loading Transcript from {url}...")
        try:
            pages = load_youtube(url)
        except Exception as e:
            print(f"Error loading YouTube transcript: {e}")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    if not pages:
        print("No content could be loaded. Exiting.")
        return

    print(f"Loaded {len(pages)} documents/pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50
    )

    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")
    print("\nSample Chunk: ")
    print(chunks[0].page_content[:200] + "...")

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # embed all chunks and store in FAISS
    print("\nBuilding vector store... (may take a minute)")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("rag_index")
    print("Done! Vector store saved to 'rag_index'")

    # Load the saved vector store
    vectorstore = FAISS.load_local("rag_index", embeddings, allow_dangerous_deserialization=True)
    # Convert to retriever (fetches top 3 relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Hook up llama3
    llm = OllamaLLM(model="llama3")
    
    # Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
    )

    # Ask a question
    print("\nRAG is ready! Ask anything about the document. Type 'exit' to quit \n")
    while True: 
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            print("Thanks for using the RAG model!")
            break
        else:
            response = qa_chain.invoke({"query": query})
            print(f"\nBot:   {response['result']}\n")

if __name__ == "__main__":
    main()