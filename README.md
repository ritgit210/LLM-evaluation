# LLM Traceability & Eval Pipeline with Langfuse

Standalone PDF RAG pipeline built with LangGraph, wired up to a self-hosted
Langfuse for observability — tracing, an analytics dashboard, LLM-as-judge
evaluation, and golden-dataset experiments. Everything runs locally and
independently; the only external dependency is the chat LLM API.

Embeddings use an open-source `sentence-transformers` model (runs locally,
no API key/cost); the chat/generation and judge steps use OpenAI.

## Files

- `rag_graph.py` — PDF ingestion (chunk + embed + FAISS) and the LangGraph
  `retrieve -> generate` RAG graph. Also the CLI entry point.
- `observability.py` — Langfuse client, tracing helper, analytics
  (dashboard summary / daily trends / per-model cost breakdown), and
  LLM-as-judge answer evaluation.
- `docker-compose.langfuse.yml` — self-hosted Langfuse stack (web, worker,
  Postgres, ClickHouse, MinIO, Redis).
- `.env.example` — copy to `.env` and fill in.

## Flows

### 1. Ingest

```
[ PDF file ]
      │
      ▼
[ Chunk — RecursiveCharacterTextSplitter ]
      │
      ▼
[ Embed — HuggingFace sentence-transformers (local, no API key) ]
      │
      ▼
[ FAISS index written to VECTOR_STORE_DIR ]
```

- **Chunk** — `ingest_pdf()` splits the PDF into overlapping text chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP`).
- **Embed** — each chunk vectorized locally via `sentence-transformers/all-MiniLM-L6-v2`.
- **Index** — vectors saved to a local FAISS index for later similarity search.

### 2. Ask (RAG query)

```
[ Question ]
      │
      ▼
[ traced_run() — opens a Langfuse trace + CallbackHandler ]
      │
      ▼
[ retrieve — FAISS similarity_search(question, k=TOP_K) ]
      │
      ▼
[ generate — ChatOpenAI(context + question) via Groq/OpenAI ]
      │
      ▼
[ Answer returned; trace flushed to Langfuse ]
      │
      ▼  (only with --evaluate)
[ evaluate_answer() — LLM judge scores faithfulness/relevance/completeness ]
      │
      ▼
[ create_score() — score attached to this trace in Langfuse ]
```

- **traced_run** — assigns a `trace_id`, tags the run, wires a `CallbackHandler` so every LangGraph step is captured.
- **retrieve** — pulls the top-`K` chunks from the FAISS index for the question.
- **generate** — LLM answers strictly from retrieved context.
- **evaluate_answer** *(optional)* — a second LLM call judges the answer and writes a numeric score back onto the same trace.

### 3. Analytics (read path)

```
[ CLI: dashboard | trends | models ]
      │
      ▼
[ get_langfuse_client() — connect to self-hosted Langfuse ]
      │
      ▼
[ client.api.trace.list() / legacy.observations_v1.get_many() ]
      │
      ▼
[ Aggregate — cost, tokens, latency, per-model totals ]
      │
      ▼
[ JSON printed to stdout ]
```

- **Query** — pulls traces (for cost/latency) and observations (for token/model breakdown) over a date window.
- **Aggregate** — plain-Python summation/grouping, no LLM calls.
- **Output** — one flat JSON object per command, no side effects on Langfuse.

### 4. Golden-dataset evaluation (Datasets + Experiments)

```
[ rag_eval_questions.json ]
      │
      ▼  load-dataset
[ create_dataset_item() — input=question, expected_output=answer ]
      │
      ▼  (stored as a Langfuse Dataset)

[ run-experiment ]
      │
      ▼
[ task() → app.invoke(question) — our RAG graph generates the answer ]
      │
      ▼
[ Langfuse SDK: span.update(output=answer) — links trace to the dataset run ]
      │
      ▼  (async, server-side — no code triggers this)
[ Managed evaluator (Run on: Experiments) auto-fires on the new run ]
      │
      ▼
[ Judge LLM (Langfuse LLM connection) scores output vs expected_output ]
      │
      ▼
[ Score + comment stored on the trace, visible in the dataset run view ]
```

- **load-dataset** — one-time (idempotent) upload of question/ground-truth pairs; no LLM involved.
- **run-experiment** — *we* generate the answer (`app.invoke`); the *SDK* attaches it to the trace and links it to the dataset run.
- **Auto-trigger** — a UI-configured evaluator scoped to this dataset fires automatically on new runs; nothing in this repo calls it.
- **Scoring** — the evaluator's judge LLM compares `answer` vs `expected_output` and writes a Score back to Langfuse.

## Setup

```bash
cd rag_langfuse_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and the LANGFUSE_* secrets
```

The embedding model (`sentence-transformers/all-MiniLM-L6-v2` by default) is
downloaded from Hugging Face and cached locally the first time you run
`ingest` or `ask` — no API key needed for it.

## 1. Start Langfuse

```bash
docker compose -f docker-compose.langfuse.yml up -d
```

UI at http://localhost:3100 — an org/project/admin user are auto-created from
the `LANGFUSE_*` values in `.env` (`LANGFUSE_INIT_*` in the compose file).

> Note: the stack binds local ports 3100, 5433, 8123/9000, 9090/9091 and
> 6379. Make sure nothing else on the machine is already using them.

Verify connectivity:

```bash
python observability.py verify
```

## 2. Ingest a PDF

```bash
python rag_graph.py ingest ./data/sample.pdf
```

Builds a FAISS index at `VECTOR_STORE_DIR` (default `./vector_store`).

## 3. Ask questions

```bash
python rag_graph.py ask "What is the termination clause?"
```

Add `--evaluate` to also run the LLM-as-judge and push an `overall` score
onto the trace in Langfuse:

```bash
python rag_graph.py ask "What is the termination clause?" --evaluate
```

## 4. Analytics

```bash
python observability.py dashboard   # totals: traces, tokens, cost, avg latency
python observability.py trends      # daily usage trend
python observability.py models      # cost/token breakdown per LLM model
```

## 5. Golden-dataset evaluation (Datasets + Experiments)

For a repeatable eval set with known-correct answers (see
`data/rag_eval_questions.json` — question/answer/key_facts triples), load it
into a Langfuse Dataset once, then run it through the RAG graph as a
Langfuse Experiment as often as you like (e.g. after changing the prompt,
chunk size, or embedding model):

```bash
python observability.py load-dataset data/rag_eval_questions.json   # idempotent, matches on question "id"
python observability.py run-experiment vell_harbour_story           # dataset name defaults to the JSON's "corpus" field, minus extension
```

Each item's expected answer is attached to its span as the experiment's
"expected output". In the Langfuse UI, configure a managed evaluator (e.g.
the Ragas **Answer Correctness** template) with **Run on: Experiments**
scoped to this dataset, and map:

- `ground_truth` → Expected Output
- `answer` → Output

No manual metadata wiring needed — Langfuse resolves both directly from the
experiment run. Re-running `run-experiment` creates a new named run each
time, so successive evaluator scores are comparable side-by-side per run in
the UI.
