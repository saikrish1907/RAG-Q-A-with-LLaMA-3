import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from llama_runner import generate_response


EMBED_MODEL = SentenceTransformer("Embedding Model Path")

# Load & Split
def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Embed & Store in FAISS
def create_or_update_vectorstore(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = EMBED_MODEL.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts

# Retrieve
def retrieve_relevant_chunks(query, index, texts, k=3):
    query_embedding = EMBED_MODEL.encode([query])
    _, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# RAG Pipeline
def answer_query(query, index, texts):
    context = "\n\n".join(retrieve_relevant_chunks(query, index, texts))
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    return generate_response(prompt)
