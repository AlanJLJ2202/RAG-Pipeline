# RAG-Pipeline

Pipeline de **Retrieval-Augmented Generation (RAG)** con LangChain, ChromaDB y OpenAI para convertir un PDF en un asistente conversacional con respuestas basadas en contexto real.

Este repositorio incluye:
- una versión **Beginner** (RAG base),
- una versión **Intermediate** (RAG híbrido optimizado),
- y un script de **evaluación automática** con RAGAS.

---

## 📁 Estructura del proyecto

```bash
RAG-Pipeline/
├── 100cosas-es.pdf
├── requirements.txt
├── README.md
├── Beginner/
│   └── rag_test.py
└── Intermediate/
    ├── rag_test_v2.py
    └── eval_rag.py
```

---

## 🚀 Qué hace cada script

### 1) `Beginner/rag_test.py`
Implementa un RAG clásico:
- carga PDF,
- chunking con `RecursiveCharacterTextSplitter`,
- embeddings con `text-embedding-3-small`,
- búsqueda vectorial con Chroma,
- respuesta con `gpt-4o-mini`.

Ideal para entender el flujo base de extremo a extremo.

### 2) `Intermediate/rag_test_v2.py`
Versión optimizada:
- metadatos por documento,
- `ParentDocumentRetriever`,
- búsqueda híbrida con `EnsembleRetriever` (**BM25 + vectorial**),
- persistencia de Chroma en disco (`/chroma_db`) para acelerar ejecuciones posteriores.

### 3) `Intermediate/eval_rag.py`
Evalúa calidad del RAG usando **RAGAS** con métricas:
- `context_recall`
- `context_precision`
- `answer_relevancy`

Incluye preguntas y respuestas de referencia para medir rendimiento del pipeline.

---

## ⚙️ Requisitos

- Python 3.10+ recomendado
- Clave de OpenAI en variable de entorno: `OPENAI_API_KEY`

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## 🔐 Configuración de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

---

## ▶️ Ejecución

Desde la raíz del repositorio:

### RAG básico
```bash
python Beginner/rag_test.py
```

### RAG intermedio (híbrido)
```bash
python Intermediate/rag_test_v2.py
```

### Evaluación con RAGAS
```bash
python Intermediate/eval_rag.py
```

En los scripts de chat puedes salir escribiendo `salir`.

---

## 🧠 Notas importantes

- El PDF objetivo actual es `100cosas-es.pdf` en la raíz del repositorio.
- Si ya existe `chroma_db`, la versión intermedia reutiliza los embeddings guardados.
- La calidad de respuesta depende del chunking, del valor de `k` en retrieval y del prompt.

---

## 📌 Próximos pasos sugeridos

- Añadir soporte para múltiples PDFs.
- Incluir filtros por metadatos (capítulo, sección, páginas).
- Agregar trazabilidad de fuentes en cada respuesta.
- Automatizar evaluación continua de métricas RAGAS.
