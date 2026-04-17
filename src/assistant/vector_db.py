import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.documents import Document
import pdfplumber
from langchain_core.documents import Document

VECTOR_DB_PATH = "database"

def get_or_create_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
        print("Loaded existing vector DB")
        return vectorstore   

    docs = []
    folder_path = "./report_structures"

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            try:
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted

                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": file}
                        )
                    )

            except Exception as e:
                print(f"Error loading {file}: {e}")

    if not docs:
        print("❌ No docs found")
        return None

    semantic_text_splitter = SemanticChunker(embeddings)
    documents = semantic_text_splitter.split_documents(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )
    split_documents = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        split_documents,
        embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print(f"Created DB with {len(docs)} docs")

    return vectorstore

def load_excel_data(file_path):
    df = pd.read_excel(file_path)

    docs = []
    for _, row in df.iterrows():
        symptoms = row.get("symptoms", row.iloc[0])
        disease = row.get("disease", row.iloc[1])

        text = f"Symptoms: {symptoms} | Disease: {disease}"
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "symptom2disease"}
            )
        )

    return docs

def add_documents(documents):
    """
    Add new documents to the existing vector store.

    Args:
        documents: List of documents to add to the vector store
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Process the new documents
    semantic_text_splitter = SemanticChunker(embeddings)
    documents = semantic_text_splitter.split_documents(documents)

    # Split resulting documents into smaller chunks SemanticChunker
    # doesn't have a max chunk size parameter, so we use 
    # RecursiveCharacterTextSplitter to avoid having large chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    split_documents = text_splitter.split_documents(documents)

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # Add to existing vector store
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        vectorstore.add_documents(split_documents)
    else:
        # Create new vector store if it doesn't exist
        vectorstore = Chroma.from_documents(
            split_documents,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    return vectorstore