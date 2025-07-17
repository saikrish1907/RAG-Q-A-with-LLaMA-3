import os
import streamlit as st
from rag import load_and_split, create_or_update_vectorstore, answer_query

st.title("📄 RAG Q&A with LLaMA 3")
uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])

if uploaded_file:
    # Ensure 'documents' folder exists
    os.makedirs("documents", exist_ok=True)

    file_path = f"documents/{uploaded_file.name}"

    # Save the uploaded PDF to local disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded!")

    with st.spinner("Indexing document..."):
        chunks = load_and_split(file_path)
        index, texts = create_or_update_vectorstore(chunks)
        st.success("Document indexed successfully!")

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Thinking..."):
                answer = answer_query(query, index, texts)
                st.markdown(f"**Answer:** {answer}")
