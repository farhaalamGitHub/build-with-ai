# Story 5.2 — Query Decomposition and HyDE

Tenth story of the "Building Intelligence with RAG" course. Follows Story 5.1. Makes the last two registered modes real: `decomposition` (one compound comparison question → bounded subquestions → evidence per step) and `hyde` (one hypothetical retrieval text → embedding → real passages). These are small retrieval extensions, not agents, routers, graph construction, or a new application.

## Purpose

- `pattern: "decomposition"` (model `rag-decomposition`): split → retrieve per subquestion → union → `QueryResult`, then the existing grounded-answer path.
- `pattern: "hyde"` (model `rag-hyde`): direct retrieval + hypothetical-text retrieval → dedupe → `QueryResult`, then the existing grounded-answer path.
- The selected mode decides the route; no auto-routing, no fallback to or from other modes, no loops (at most one split call; at most one hypothetical call).

Reuse, do not rename or replace: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, the `Pattern`/`rag-<pattern>` registry, `/v1/models`, `QueryError`, `semantic.effective_filters`, semantic/hybrid retrieval and embedding helpers, `assemble_context`/grounded answer/citations/confidence, streaming framing, and the existing settings (`VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`, `MONGODB_*`). **No new or renamed provider variables, endpoints, dependencies, UI screens, or streaming events.** Semantic, hybrid, hybrid-reranked and structured stay exactly as they are.

## Prerequisites

- Stories 1.1, 2.1–2.3, 3.1, 3.2, 4.1, 4.2, 5.1 complete. Read `docs/architecture.md` first.
- `.env` (untracked) with MongoDB, Voyage and generation settings; ingestion done; `chunk_text_index`/`vector_index` queryable. Never print keys or URIs.

### Findings to respect (verified in the repo)

- **`SubquestionEvidence` is not defined** (not in Story 1.1 text, `contracts.py`, or `docs/`); `QueryResult.subquestions` is currently `list[str]`. Search once more for an uncommitted definition. If none, add `SubquestionEvidence` (Pydantic, `extra="forbid"`: `subquestion: str`, `status: Literal["evidenced","no_evidence"]`, `results: list[RetrievedChunk]`, `reason: str | None`) and change the field type to `list[SubquestionEvidence]` (still empty by default). Safe: no mode populates it yet. No other contract changes.
- **`.env.example` is not in the repo** (settings live in `settings.py`; `.env` is untracked). Use the names above; do not create `.env.example` unless the trainer supplies it.
- `docs/manual-tests.md` already contains a "Story 5.2" section and a line saying `rag-decomposition` is real; it also references trace fields (`path`, `call_count`, `hyde_calls`, `stop_reason`) that no committed code produces. Reconcile it with this story's actual trace (§4) instead of appending a second section.
- `pipeline.REAL_PATTERNS` and `answer_events`/`should_answer` gate generation; chat has no way to show `partial_answer` unless the existing plain-message path is reused (see §3).
- Generation calls go through `generation/answer.py` (`_endpoint`, `_body`, OpenAI-style `/chat/completions` via LiteLLM settings). Reuse these helpers for the split and hypothetical calls; do not add a second client or settings.
- Voyage embedding reuses `semantic._embed` (`voyage-3.5`, 1,024 dims). Query-type embedding is correct for the hypothetical text.

### Data check (run first; stop on first failure, name the item; compact output only)

1. Yes/no only: `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY` set.
2. Counts only: `chunks` and `sections` documents; both indexes queryable (yes/no). No text printed.

## Work to do

### 1. Shared bounded model call (`src/building_with_rag/retrieval/` new small module, e.g. `expansion.py`)

- `_complete_json(system, user) -> str`: one non-streaming call via the existing generation settings, `temperature 0`, short `max_tokens`, request timeout reusing the existing generation timeout. Missing generation settings → 503 `retrieval_not_ready` naming the setting (not its value), checked before any retrieval call. Provider error/timeout → 502 `retrieval_upstream_error` (exception type only logged).
- The user's question is data inside the prompt, never instructions; the prompt states the corpus scope (BNS 2023 and IPC 1860 only).

### 2. Decomposition (`retrieval/decomposition.py`)

- `split(question) -> list[str]`: one call returning JSON `{"subquestions": [...]}`. Accept 1–3 items; each a non-empty string ≤ 300 chars, unique after case/whitespace normalisation, about BNS/IPC criminal law. Empty, non-JSON, wrong shape, >3, duplicates, or out-of-scope text → `status: "clarify"`, empty `results`, no retrieval, message asks for a clearer compound question. No retry, no repair loop.
- For each accepted subquestion, run the **existing hybrid retrieval** (`hybrid` route, same filters/scope validation via `effective_filters`; no reranking) with a small per-step limit. A subquestion with ≥1 real passage → `SubquestionEvidence(status="evidenced", results=…)`; none → `no_evidence` with a `reason`. Every accepted subquestion is required.
- Outcomes (HTTP 200 unless noted): `ok` (all steps evidenced); `partial_answer` (some evidenced, some not: reason names the unsupported subquestion(s); no complete comparison is synthesised); `clarify` (split rejected, or no step has evidence); 503/502 as above.
- `QueryResult.subquestions` = the evidence list; top-level `results` = deduplicated union (by `chunk_id`, first-seen order), every real passage retained. Do not invent passages for an unevidenced step.
- Trace (no secrets): `mode: "decomposition"`, `decomposition` (`accepted` count, per-step `{subquestion, status, result_count}`), `sufficiency` (`complete|partial|none`), `reason`, `filters`, `caller_id`, `result_count`.

### 3. HyDE (`retrieval/hyde.py`)

- `hypothesize(question) -> str`: one call returning a short passage (≤ 800 chars) written in the vocabulary of an Indian criminal-law statute. Empty, whitespace, JSON/markdown-fenced garbage, or over-length → `status: "hyde_unavailable"`, empty `results`, no embedding call, message says no hypothetical text was produced. No fabricated fallback.
- Retrieval: (a) direct — existing semantic search on the raw question → `hyde_direct_candidates`; (b) hypothetical — `semantic._embed(hypothetical)` with the existing model, then existing `vector_search` → `hyde_query_candidates`. Final `results` = deduplicated real passages from both lists (by `chunk_id`, ranked by best score, bounded to the existing semantic limit).
- The hypothetical text appears **only** in `hyde_hypothetical_text_debug` (and is never in `message`, `trace`, `generation`, prompts, citations, chat output, or logs). It is never evidence, never passed to `assemble_context`, never citable.
- Outcomes: `ok`; `no_results` (both lists empty); `hyde_unavailable`; 503/502 as above. Trace: `mode: "hyde"`, `hyde` (`direct_count`, `query_count`, `final_count`, `hypothetical_chars`), `filters`, `caller_id`, `result_count`.

### 4. Routing, answers, streaming

- `pipeline.retrieve`: `DECOMPOSITION` → decomposition, `HYDE` → hyde. Add both to `REAL_PATTERNS`. Registry entries and model IDs unchanged.
- `ok` / `partial_answer`: results pass through the existing `assemble_context` → grounded answer → citations/confidence unchanged. For `partial_answer`, `/v1/query` keeps `trace.reason`; chat prints the unsupported-part message first as plain content, then the grounded answer for the evidenced parts only, with the existing DRAFT/confidence/`Sources:` framing. No complete-comparison claim when a step lacks evidence; any answer text must be limited to evidenced subquestions.
- `clarify`, `hyde_unavailable`, `no_results`: no answer model call; chat yields `result.message` as plain content (same path Story 5.1 uses for non-`ok`).
- No new SSE event types; Open WebUI Pipe unchanged unless `rag-decomposition`/`rag-hyde` are not already offered and sent.

### 5. Docs and tests

- `docs/architecture.md`: add "Decomposition and HyDE (Story 5.2)": split bounds and rejection rules, per-step required evidence, `partial_answer`/`clarify`/`hyde_unavailable` statuses, `SubquestionEvidence` in contracts, HyDE field usage, the **hypothetical-text boundary** (debug only, never evidence), reused settings, trace shapes. Update the modes paragraph (all six real), `QueryResult.status` values, and limitations (max 3 subquestions, no retry, hybrid-only per step, one hypothetical, no reranking, quality depends on the generation model, no memory, no auto-routing). Commands unchanged unless a new one is added.
- `docs/manual-tests.md`: replace the pre-existing Story 5.2 section with one matching this story; update the "still a placeholder" cases (`tests/test_smoke.py`, Story 1.1 query/chat entries for `decomposition`/`hyde`) to expect real behaviour or use a different not-implemented example only if one exists (none will — remove those placeholder cases).
- `tests/test_decomposition_hyde.py` and `tests/test_smoke.py` as in the checks below.

## Completion checks (compact; never print passages, vectors, keys, or whole responses)

Diagnostic forms (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"How does the BNS punishment for murder differ from the IPC punishment for murder?","pattern":"decomposition","generate_answer":true}' \
  | jq '{status, reason: .trace.reason, t: .trace.decomposition, s: [.subquestions[] | {subquestion, status, reason, ids: [.results[].section_id]}], g: (.generation | {outcome, confidence, text: .text[:200]})}'
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"What happens to someone who secretly takes a neighbour'"'"'s bike?","pattern":"hyde","generate_answer":true}' \
  | jq '{status, t: .trace.hyde, direct: [.hyde_direct_candidates[].section_id], hyde: [.hyde_query_candidates[].section_id], final: [.results[] | {section_id, score}], debug_chars: (.hyde_hypothetical_text_debug | length), g: (.generation | {outcome, confidence, text: .text[:200]})}'
```

1. **Decomposition (prepared BNS-vs-IPC comparison):** `status: "ok"`, 2–3 `evidenced` steps with BNS and IPC ids, every id also in top-level `results`; inspect the evidence ids and passages (excerpts only) **before** viewing the chat answer.
2. **Partial/insufficient:** a question whose second half has no corpus evidence → `partial_answer`, reason names that subquestion, step marked `no_evidence`, no passages invented. A vague or out-of-scope question ("Compare it with the other law") → `clarify`, no retrieval.
3. **HyDE (prepared vocabulary-mismatch question):** `ok`; `hyde_direct_candidates` and `hyde_query_candidates` populated; `results` deduplicated; `debug_chars > 0`; hypothetical text absent from `message`, `trace`, `generation`, citations and chat. Inspect final ids/passages before the chat answer. Missing generation configuration → 503 `retrieval_not_ready` (no fabricated text).
4. **Chat (Open WebUI):** `rag-decomposition` and `rag-hyde` stream DRAFT, confidence and `Sources:`; `partial_answer` prints the unsupported-part message first; nothing from the hypothetical text appears.
5. **Six-mode smoke (`/v1/query` and Open WebUI):** `GET /v1/models` lists six IDs; one small question each for `semantic`, `hybrid`, `hybrid-reranked`, `structured` ("What does BNS section 103 say?"), `decomposition`, `hyde` returns its expected status and response shape (`status`, `results`, `trace.mode`); chat streams for all six; no earlier contract changed.
6. **Offline tests** (`tests/test_decomposition_hyde.py`; no network; fake the model call and retrieval; few cases): (a) split accepts 1–3 unique questions; rejects empty, malformed JSON, 4 items, duplicates and out-of-scope → `clarify` with no retrieval call; (b) per-step evidence, union dedupe keeps every passage, one unevidenced step → `partial_answer` naming it, none → `clarify`; (c) hypothetical rejected when empty/malformed → `hyde_unavailable` and no embedding call; (d) HyDE lists, dedupe, and the text appears only in `hyde_hypothetical_text_debug` (not in message, trace, or `assemble_context` input); (e) missing generation setting → 503 before any retrieval; (f) `pipeline.retrieve` routes both modes, and non-`ok` statuses make no answer call.
7. `tests/test_smoke.py`: the two placeholder cases are replaced with offline route/shape checks; `/v1/models` still lists six IDs.
8. `uv run ruff check src/building_with_rag` passes.

Run only this story's checks (the new test file, `tests/test_smoke.py`, ruff, and the live smoke in 1–5). Do not run the full suite. If live prerequisites are unavailable, run only checks 6–8 and name the blocked prerequisite.

## Handover

Record: files created/changed; data-check results (yes/no, counts); commands run; compact diagnostics for checks 1–3; whether `SubquestionEvidence` already existed; which prepared questions were used and why; architecture and manual-tests sections updated; six-mode smoke result (one line per mode); deviations from this story.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`) for the developer to run if wanted.
