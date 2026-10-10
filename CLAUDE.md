# Project instructions

This is a shared training capstone; token waste multiplies across 20+ users.

## Response style

- Default: brief, direct, answer only what was asked. No greetings, filler,
  narration, restated requirements, tutorials, summaries, or next steps unless requested.
- Use short paragraphs or bullets only when they improve clarity. Do not use
  decorative headings, tables, examples, alternatives, or code unless needed.
- Ask one short question only when blocked; otherwise make a safe scoped assumption.
- Never paste large code, diffs, logs, JSON, source, or file contents unless asked.
- Errors: state the actionable cause in one line.

## Work style

- Read only relevant files; prefer targeted search and excerpts.
- After implementation: at most 7 lines: `Completed.`, changed paths, and test results and what was done ... all under 7 lines.
- After implementation never run full test suits and regression testing. Only run the test that are specific to ongoing story. Leave last line with a command for the develop if he/she wants to run the full test suite.
  If no changes: `No changes.` Include developer action items only when present.
- Keep plans, comments, docs, commits, tests, and generated stories compact.

## Retrieval/manual testing

- Never pretty-print full retrieval responses. Do not use `curl ... | python3 -m json.tool`.
- Use raw `curl -s` or focused `jq` fields that exclude/truncate passage `text`.
- After writing a file, report its path; do not echo its contents.

## Commands

Python ≥3.12, managed with `uv`. Config via untracked `.env` (see `settings.py`); the app starts and `/healthz` works with no credentials.

- Run API: `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`
- Tests: `uv run pytest`; single: `uv run pytest tests/test_hybrid_retrieval.py::test_name`
- Lint: `uv run ruff check .` (line length 100)
- Rebuild corpus JSONL from PDFs: `uv run python scripts/extract_sections.py`
- Ingest to MongoDB: `uv run python -m building_with_rag.ingestion.ingest`
- Create Atlas keyword index (idempotent): `uv run python -m building_with_rag.ingestion.keyword_index`
- Manual curl checks per story: `docs/manual-tests.md`

## Architecture

RAG API over two Indian criminal-law acts (BNS 2023, IPC 1860) only. `docs/architecture.md` is the authoritative contract doc; later stories extend contracts additively — never rename fields, add parallel top-level fields, or add provider-specific variants.

- Six fixed modes in `registry.py` (`rag-<pattern>` model IDs). The selected mode picks the retrieval route; no auto-routing or fallback. Real: semantic, hybrid, hybrid-reranked, structured. `decomposition`/`hyde` return a `not_implemented` placeholder.
- `pipeline.py` is the single shared path for `POST /v1/query` and OpenAI-compatible `POST /v1/chat/completions` (`retrieve` → context assembly → generation/validation). Don't duplicate logic in routes; chat sets server-side `caller_id`/`generate_answer`.
- `retrieval/` (semantic via Atlas `$vectorSearch` with Voyage `voyage-3.5` 1024-dim; hybrid RRF with keyword search; rerank; structured exact lookup), `generation/` (grounded answer, citations, confidence), `ingestion/` (chunk, embed, MongoDB schema/indexes).
- Evidence rules: passages are evidence, never instructions; answers need retrieved support; ids are act-qualified (`bns:`/`ipc:`); access scope is server-fixed to `public`.
- Open WebUI (`open_webui_functions/`) is a separate trainer-supplied client; no custom SSE events it can't render.
- Work is organised as BMAD stories in `docs/stories/`; BMAD tooling lives in `_bmad/`, `.agents/`, `.claude/skills` (not app code).
