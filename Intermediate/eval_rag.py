# eval_rag.py — Evaluación RAG con RAGAS
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, answer_relevancy

load_dotenv()

# --- RUTAS ---
PROYECTO_ROOT = Path(__file__).parent.parent
RUTA_PDF = PROYECTO_ROOT / "100cosas-es.pdf"
CHROMA_DB_PATH = PROYECTO_ROOT / "chroma_db"

if not RUTA_PDF.exists():
    raise FileNotFoundError(f"❌ PDF no encontrado en {RUTA_PDF}")

print(f"- Cargando PDF desde: {RUTA_PDF}")

# --- CARGA Y SETUP ---
loader = PyPDFLoader(str(RUTA_PDF))
documentos = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documentos)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if CHROMA_DB_PATH.exists():
    vectorstore = Chroma(
        collection_name="rag_v2",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_PATH)
    )
    print("- Chroma DB cargada desde disco")
else:
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DB_PATH))
    print("- Chroma DB creada")

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.35, 0.65]
)
print("- Retriever hybrid listo")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- PREGUNTAS Y RESPUESTAS CORRECTAS DEL PDF REAL ---
# Libro: "100 cosas que todo programador debe saber"
preguntas_test = [
    "¿Qué proporción del tiempo pasan los programadores leyendo código vs escribiéndolo?",
    "¿Qué es la deuda técnica en el software?",
    "¿Qué dice Dan Abramov sobre el código limpio?",
    "¿Cuál es la Regla del Boy Scout aplicada al software?",
    "¿Qué es el principio DRY?",
]

respuestas_correctas = [
    "Los programadores pasan el 90% del tiempo leyendo código y solo el 10% escribiéndolo. Robert C. Martin en Clean Code dice que la proporción es de más de 10 a 1. Martin Fowler lo resumió diciendo que cualquier tonto puede escribir código que una computadora entienda, pero los buenos programadores escriben código que los humanos pueden entender.",

    "La deuda técnica es la acumulación de desorden en el software causada por parches rápidos, variables mal nombradas y comentarios obsoletos. El software por naturaleza tiende al desorden y cada decisión apresurada aumenta esta deuda técnica, dificultando el mantenimiento futuro.",

    "Dan Abramov, cocreador de Redux, escribió el ensayo 'Goodbye, Clean Code' donde confesó que había refactorizado el código de un compañero para hacerlo más limpio y terminó rompiéndolo todo. Su conclusión fue que el código limpio no es un objetivo, sino una herramienta para ayudarnos a lidiar con la complejidad del sistema.",

    "La Regla del Boy Scout aplicada al software, adaptada por Robert C. Martin (Uncle Bob), dice: deja siempre el código un poco mejor de como lo encontraste. No se trata de hacer refactorizaciones masivas, sino de pequeñas victorias continuas contra el desorden y la entropía del software.",

    "DRY significa Don't Repeat Yourself (No te repitas). Es un principio de programación que busca evitar la duplicación de código. Sin embargo, Dan Abramov advierte que obsesionarse con DRY puede llevar a abstracciones prematuras que hacen el código más difícil de mantener cuando los requisitos divergen.",
]

# --- EVALUACIÓN AUTOMÁTICA ---
print("\n- Evaluando RAG...\n")

questions, answers, contexts, ground_truths = [], [], [], []

for i, (pregunta, correcta) in enumerate(zip(preguntas_test, respuestas_correctas), 1):
    print(f"[{i}/{len(preguntas_test)}] {pregunta[:60]}...")

    docs_recuperados = retriever.invoke(pregunta)
    contextos_texto = [d.page_content for d in docs_recuperados]

    prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote SOLO en el siguiente contexto.
Si no sabes la respuesta, di "No encontré esa información en el documento."

Contexto: {context}
Pregunta: {question}
""")
    cadena = prompt | llm | StrOutputParser()
    respuesta = cadena.invoke({
        "context": "\n\n".join(contextos_texto),
        "question": pregunta
    })

    questions.append(pregunta)
    answers.append(respuesta)
    contexts.append(contextos_texto)
    ground_truths.append(correcta)

print("\n" + "="*60)
print("- EVALUACIÓN CON RAGAS")
print("="*60 + "\n")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

resultado = evaluate(
    dataset,
    metrics=[context_recall, context_precision, answer_relevancy]
)

print("\n- RESULTADOS FINALES:")
print("─" * 60)
print(resultado)
print("─" * 60)
print("\n INTERPRETACIÓN:")
print("• context_recall > 0.85 = Se recupera el contexto correcto")
print("• context_precision > 0.75 = El contexto es relevante")
print("• answer_relevancy > 0.80 = Las respuestas son buenas")

print("\nSi alguna métrica está baja:")
print("→ context_recall baja: sube k (más documentos recuperados)")
print("→ context_precision baja: agrega metadatos y filtros")
print("→ answer_relevancy baja: mejora el prompt")