import langchain
print(langchain.__version__)

# rag_chatbot.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- CONFIGURACIÓN ---
load_dotenv()  # carga la key desde .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró OPENAI_API_KEY en el .env")
print("API key cargada correctamente")

RUTA_PDF = os.path.join(os.path.dirname(__file__), "100cosas-es.pdf")

# --- FASE 1: INDEXACIÓN ---
loader = PyPDFLoader(RUTA_PDF)
documentos = loader.load()
print(f"   → {len(documentos)} páginas cargadas")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documentos)
print(f"   → {len(chunks)} chunks creados")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("   → Vector store listo!")

# --- FASE 2: PIPELINE ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote SOLO en el siguiente contexto.
Si no sabes la respuesta, di "No encontré esa información en el documento."

Contexto: {context}
Pregunta: {question}
""")

def formatear_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

cadena = (
    {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("\n Chatbot listo. Escribe 'salir' para terminar.\n")

# --- CHAT LOOP ---
while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    respuesta = cadena.invoke(pregunta)
    print(f"\nAsistente: {respuesta}\n")