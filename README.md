# RAG-Pipeline

Language: **English (default)** | [Español](#español)

Retrieval-Augmented Generation (RAG) pipeline built with LangChain, ChromaDB, and OpenAI to turn a PDF into a grounded conversational assistant.

This repository includes:
- a **Beginner** version (baseline RAG),
- an **Intermediate** version (optimized hybrid RAG),
- and an **automatic evaluation** script with RAGAS.

---

## 📁 Project structure

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

## 🚀 What each script does

### 1) `Beginner/rag_test.py`
Baseline RAG implementation:
- PDF loading,
- chunking with `RecursiveCharacterTextSplitter`,
- embeddings with `text-embedding-3-small`,
- vector retrieval with Chroma,
- answer generation with `gpt-4o-mini`.

### 2) `Intermediate/rag_test_v2.py`
Optimized version:
- enriched document metadata,
- `ParentDocumentRetriever`,
- hybrid retrieval with `EnsembleRetriever` (**BM25 + vector**),
- Chroma persistence in `/chroma_db` for faster subsequent runs.

### 3) `Intermediate/eval_rag.py`
RAG quality evaluation using **RAGAS** metrics:
- `context_recall`
- `context_precision`
- `answer_relevancy`

---

## ⚙️ Requirements

- Recommended Python 3.10+
- OpenAI key in environment variable: `OPENAI_API_KEY`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment setup

Create a `.env` file at the repository root:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run

From the repository root:

### Baseline RAG
```bash
python Beginner/rag_test.py
```

### Intermediate hybrid RAG
```bash
python Intermediate/rag_test_v2.py
```

### RAGAS evaluation
```bash
python Intermediate/eval_rag.py
```

In chat scripts, type `salir` to exit.

---

## 🧠 Important notes

- The target PDF is currently `100cosas-es.pdf` at the repository root.
- If `chroma_db` already exists, the intermediate version reuses stored embeddings.
- Response quality depends on chunking strategy, retrieval `k`, and prompt design.

---

## 📌 Suggested next steps

- Add support for multiple PDFs.
- Add metadata filtering (chapter, section, page ranges).
- Add source traceability in answers.
- Automate continuous RAGAS evaluation.

---

## Español

Si prefieres leer esta guía en español, usa esta sección.

### Resumen
Este repositorio implementa un pipeline RAG con:
- versión básica (`Beginner/rag_test.py`),
- versión intermedia optimizada (`Intermediate/rag_test_v2.py`),
- evaluación automática con RAGAS (`Intermediate/eval_rag.py`).

### Instalación
```bash
pip install -r requirements.txt
```

### Configuración
Crear `.env` en la raíz:
```env
OPENAI_API_KEY=tu_api_key_aqui
```

### Ejecución
```bash
python Beginner/rag_test.py
python Intermediate/rag_test_v2.py
python Intermediate/eval_rag.py
```
