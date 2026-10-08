# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — all returning honest `not_implemented` placeholders.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Health

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}`.

### List models

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Expected: six lines: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

### Query — each RAG mode (semantic)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

Expected: `"status":"ok"`, up to 3 results in non-increasing score order, populated `trace`. Needs ingested data (Story 2.2). `filters: {"act": ["IPC_1860"], "status": ["in_force"]}` returns `no_results`; `limit: 21` returns 422.

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid`.

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid-reranked`.

### Query — structured

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "structured"}'
```

Expected: `"status":"not_implemented"`, message references `structured`.

### Query — decomposition

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "decomposition"}'
```

Expected: `"status":"not_implemented"`, message references `decomposition`.

### Query — hyde

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hyde"}'
```

Expected: `"status":"not_implemented"`, message references `hyde`.

### Query — empty question (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: 422 validation error (question below min_length 1).

### Chat completions — JSON (non-streaming)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
```

Expected: SSE `data:` frames with `delta` role then content, ending with `data: [DONE]`.

### Chat completions — invalid model (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: 400 with `"type":"invalid_request_error"`, `"code":"model_not_found"`.

### Chat completions — no user message (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid", "messages": [{"role": "system", "content": "You are helpful."}]}'
```

Expected: 400 with `"code":"missing_user_message"`.

## Story 2.1 — Document Ingestion

What it adds: PDF extraction script that produces inspectable JSONL section-level corpora from the BNS and IPC bare act PDFs.

Prerequisite: Story 1.1 complete, `data/raw/` PDFs and `PROVENANCE.md` present, `uv sync` done.

### Run extraction

```bash
uv run python scripts/extract_sections.py
```

Expected: prints parser name/version, record counts (~358 BNS, ~500 IPC), count of `needs_review` flags, count of empty-text records. Both `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist.

### Re-run (skip)

```bash
uv run python scripts/extract_sections.py
```

Expected: prints "BNS corpus up to date — skipping" and "IPC corpus up to date — skipping". No records appended or overwritten.

## Story 2.2 — MongoDB Vector Index

What it adds: loads the corpus into MongoDB (sources, sections, chunks, embeddings) and creates `vector_index`. Needs `MONGODB_URI` and `VOYAGE_API_KEY` in `.env`.

### Run ingestion

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: ends with `=== Ingestion complete ===`; `Collections: sources=2, sections=858, chunks=923, embeddings=923`; `embeddings == chunks: True`; `Index status: READY`; sample vector length 1024. Re-running skips existing records. A free Voyage key takes about 30 minutes on the first run; it is safe to interrupt and re-run.

## Story 2.3 — Semantic Retrieval

What it adds: `pattern: "semantic"` on `/v1/query` returns ranked source passages. Needs the Story 2.2 data.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Success

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

Expected: `"status":"ok"`, `pattern` `semantic`, up to 3 results in non-increasing score order, populated `trace`.

### No results (filters match nothing)

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "theft", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'   | jq '{status, message, results}'
```

Expected: HTTP 200, `"status":"no_results"`, empty `results` (IPC is repealed).

### Invalid filter (failure)

```bash
curl -s -w ' HTTP %{http_code}
' http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "theft", "pattern": "semantic", "filters": {"act": {"$ne": "x"}}}'
```

Expected: HTTP 422. `"limit": 21` also returns 422.

## Story 3.1 — Grounded Answer Generation

What it adds: `generate_answer: true` on semantic `/v1/query` attaches `generation` (outcome, claims, citations). Needs ingested data and the LiteLLM proxy from `.env`.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Answerable

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}'   | jq '{status, g: (.generation | {outcome, model, provider, context_outcome, text: (.text[:300]), claims: [.claims[] | {t: .text[:80], e: .evidence_labels}], citations: [.citations[] | {label, chunk_id, section_id, act, heading}], trace}), ctx: [.results[] | {chunk_id, section_id, act, score}]}'
```

Expected: `outcome: "answered"`, non-empty `text`, claims with labels, citations whose `chunk_id` values appear in `ctx` and `trace.labels`, and `section_id` prefix matching `act` (`bns:` ↔ `BNS_2023`).

### Unsupported question

Same command with `"question": "What is the GST rate on restaurant services?"`.

Expected: `outcome: "insufficient_evidence"`, empty `text`, no claims or citations; `ctx` still lists the retrieved passages.

### Generation unavailable

Start the API with `GENERATION_API_KEY=` empty (or the proxy stopped) and repeat the answerable request.

Expected: HTTP 200, `outcome: "unavailable"`, empty `text`, `results` present.

### No generation requested

Semantic request without `generate_answer`. Expected: `"generation": null`.

## Story 3.2 — Streamed Answers with Confidence

What it adds: `rag-semantic` on chat streams a grounded answer with a confidence footer; `/v1/query` shares the same path. Needs ingested data and the generation proxy.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Streamed answer

```bash
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"rag-semantic","stream":true,"messages":[{"role":"user","content":"What is the punishment for theft under the BNS?"}]}' \
  | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' | head -c 1500
```

Expected: `DRAFT — checking evidence`, answer text with `[E1]` labels, then `Evidence check passed — confidence: high` and `Sources:` lines; the raw stream ends with `data: [DONE]`.

### Unsupported question

Same command with `"content":"What is the GST rate on restaurant services?"`.

Expected: one insufficient-evidence sentence, no confidence line, no `DRAFT`.

### Same operation on /v1/query

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"What is the punishment for theft under the BNS?","pattern":"semantic","generate_answer":true}' \
  | jq '{status, g: (.generation | {outcome, confidence, attempts, issues, citations: [.citations[] | {label, section_id}]})}'
```

Expected: `outcome: "answered"`, `confidence: "high"`, citations matching the chat sources.

### Wrong API key (failure)

With `CAPSTONE_API_KEY` set in `.env`:

```bash
curl -s -w ' HTTP %{http_code}\n' http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer wrong" \
  -H "Content-Type: application/json" -d '{"model":"rag-semantic","messages":[{"role":"user","content":"hi"}]}'
```

Expected: HTTP 401, `"code":"invalid_api_key"`.

