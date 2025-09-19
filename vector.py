import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Debug
print("Working directory:", os.getcwd())

# --- Paths ---
csv_file = r"datapdf\clients.csv"
faq_file = r"datapdf\Cold Call FAQ.pdf"
db_path = r"chroma_db"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"❌ CSV not found at {csv_file}")
if not os.path.exists(faq_file):
    raise FileNotFoundError(f"❌ FAQ PDF not found at {faq_file}")

# --- 1. Load CSV ---
df = pd.read_csv(csv_file)

client_documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=f"Record for {row['Name']}", 
        metadata={
            "Type": "Client",
            "Name": row["Name"],
            "Location": row["Location"],
            "LastService": row["Last Bought Service"],
            "PurchaseDate": row["Purchase Date"],
            "ServiceDetails": row["Service Details"]
        }
    )
    client_documents.append(doc)

# Load FAQ PDF
loader = PyPDFLoader(faq_file)
faq_documents = loader.load()
for doc in faq_documents:
    doc.metadata["Type"] = "FAQ"

# Combine all documents
all_documents = client_documents + faq_documents

# --- 4. Embedding model
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# --- 5. Build Chroma DB ---
vectorstore = Chroma.from_documents(
    all_documents,
    embedding_model,
    persist_directory=db_path
)
vectorstore.persist()
