# rag_chatbot_v2.py — Nivel Intermedio OPTIMIZADO
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore           # ← quita el "as InMemoryStore"
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- RUTAS ---
PROYECTO_ROOT = Path(__file__).parent.parent
RUTA_PDF = PROYECTO_ROOT / "100cosas-es.pdf"
CHROMA_DB_PATH = PROYECTO_ROOT / "chroma_db"

if not RUTA_PDF.exists():
    raise FileNotFoundError(f"PDF no encontrado en {RUTA_PDF}")

print(f"- Cargando PDF desde: {RUTA_PDF}")

# --- FASE 1: CARGA CON METADATOS ---
loader = PyPDFLoader(str(RUTA_PDF))
documentos = loader.load()
print(f"- {len(documentos)} páginas cargadas")

# Enriquece metadatos
for doc in documentos:
    doc.metadata["fuente"] = "100cosas-es"
    doc.metadata["seccion"] = "inicio" if doc.metadata.get("page", 0) < 20 else "cuerpo"

# --- FASE 2: CHUNKING INTELIGENTE ---
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=100)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Verifica si ya existe la DB
if CHROMA_DB_PATH.exists():
    # Carga la DB existente (rápido, evita re-embeddings)
    vectorstore = Chroma(
        collection_name="rag_v2",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_PATH)
    )
    print("- Chroma DB cargada desde disco (rápido)")
    
    # Para parent retriever, necesitamos reconstruir la store en memoria
    # (no se persiste InMemoryStore)
    store = InMemoryStore()
    chunks_planos = child_splitter.split_documents(documentos)
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
else:
    # Primera ejecución: crea todo de cero
    print("- Creando Chroma DB (esto toma ~1 min en primera ejecución)...")
    vectorstore = Chroma(
        collection_name="rag_v2",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_PATH)
    )
    store = InMemoryStore()
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    parent_retriever.add_documents(documentos)
    print("- Chroma DB creada y guardada en disco")

# --- FASE 3: HYBRID SEARCH ---
chunks_planos = child_splitter.split_documents(documentos)
bm25_retriever = BM25Retriever.from_documents(chunks_planos)
bm25_retriever.k = 4

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.35, 0.65]  # 35% BM25 (léxico), 65% vector (semántico)
)
print("- Hybrid retriever listo (BM25 + Vector)")

# --- FASE 4: PIPELINE ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote SOLO en el siguiente contexto.
Si no sabes la respuesta, di "No encontré esa información en el documento."

Contexto: {context}
Pregunta: {question}
""")

def formatear_docs(docs):
    return "\n\n".join(
        f"[Página {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )

cadena = (
    {"context": hybrid_retriever | formatear_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nChatbot v2 listo. Escribe 'salir' para terminar.\n")
while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    respuesta = cadena.invoke(pregunta)
    print(f"\nAsistente: {respuesta}\n")