# Capstone Architecture — Building Intelligence with RAG

Fixed design choices, contracts, and trust boundaries for the capstone RAG API. Later stories extend these contracts additively; they never replace them with simplified alternatives, rename fields, or add provider-specific variants.

## Scope

One small application: the capstone RAG API. No chat frontend (Open WebUI is a separate trainer-supplied client) and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Act-qualified identifiers (`bns:` / `ipc:` prefixes) avoid confusing the two acts.
- Supplied provenance lives at `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.

## Trust boundaries

Kept light for this course:

- Validate API input.
- Preserve source origin on retrieved passages.
- Do not put secrets in code, responses, or logs.
- No multi-user authorization, no security program, no evaluation harness.

## Fixed embedding choices

- Model: `voyage-3.5`, model version `voyage-3.5`, 1,024 dimensions.
- Used for every document and query embedding.
- Later stories reuse these names and choices without renaming or adding provider-specific alternatives.

## Course modes and shared registry

One shared registry (single source of truth) holds only:

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

`semantic` (Story 2.3), `hybrid` (Story 4.1), `hybrid-reranked` (Story 4.2), and `structured` (Story 5.1) are real on both `POST /v1/query` and `/v1/chat/completions` (answers: Stories 3.1–3.2). The selected mode decides the route; there is no automatic routing or fallback between modes. `decomposition` and `hyde` still return an honest `not_implemented` placeholder until their own stories.

## API contracts

### `QueryRequest` (`POST /v1/query`)

- `question`: string, 1–4,000 characters, required.
- `pattern`: one of the six modes.
- `caller_id`: optional.
- `filters`: optional `SemanticFilters` — `act`, `status`, `access_level`, each a list.
- `limit`: default 5, range 1–20.
- `generate_answer`: default false.
- `required_acts`: optional.
- `chapter`: optional.

The classroom seed may resolve only its fixed local demo caller, but keeps `caller_id` and does not replace it with a custom request shape.

### `QueryResult`

- `pattern`, `status`, `message`, `trace`, `results`.
- Optional `generation`.
- Additive empty-by-default fields later modes use (`omitted_candidates` is used by `hybrid-reranked`): `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the existing hybrid/rerank fields.

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`: real passages for `semantic`, `hybrid`, `hybrid-reranked`, and `structured`, a `run_pattern` placeholder for every other mode.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and shared pipeline (`pipeline.py`: `retrieve` + `answer_events`) as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. No duplicated implementations, no custom SSE events that Open WebUI cannot render.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only normal answer text derived from that same result. Story 3.2 renders `DRAFT` lines, the confidence footer, and sources as labelled text around the streamed answer; the full `GenerationResult` stays in `QueryResult.generation` on `/v1/query`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`.

## Corpus

Section-level JSONL corpus produced from `data/raw/` PDFs by `scripts/extract_sections.py`. One JSON object per line, one record per section. No MongoDB, no embeddings, no vector indexes.

### Parser

- **Library**: `pymupdf` (fitz) 1.28.2 — chosen because it handles both Word-to-PDF (BNS) and Ghostscript-produced (IPC) PDFs, extracts text with layout, and has no system-level dependencies.
- **Extraction command**: `uv run python scripts/extract_sections.py`

### Output format

JSONL files at `data/processed/`:

| File | Records | Sections |
|---|---|---|
| `data/processed/bns_sections.jsonl` | 358 | 1–358 |
| `data/processed/ipc_sections.jsonl` | 500 | 1–511 (11 unextractable) |

Each record has 14 fields: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`.

### Known limitations

- **IPC PDF quality**: 11 sections (4, 5, 18, 34, 40, 75, 161, 162, 163, 164, 165) have no extractable text from the scanned/Ghostscript-produced PDF. Sections 161–165 were repealed by the Prevention of Corruption Act 1988; sections 4, 5, 18, 34, 40, 75 are in portions of the PDF where pymupdf text extraction returns insufficient characters.
- **IPC section headings**: Some IPC sections (e.g. 262, 511) have empty headings due to missing heading text in the extracted text stream.
- **IPC footnotes**: Amendment footnotes and historical annotations are interleaved with section text and may appear as inline artifacts in section `text`.
- **BNS chapter markers**: Chapter boundaries are detected from `CHAPTER <roman>` lines in the body text. The BNS index (pages 2–19) provides section headings; the correspondence table (pages 20–73) is skipped.
- **Source-hash safety rule**: If a source PDF hash changes, the corpus for that act is regenerated as an atomic replacement. Records from different PDF versions are never mixed in one corpus file.

## Semantic retrieval (Story 2.3)

Flow: validate → scope filters → embed query → `$vectorSearch` → resolve chunk/section → `QueryResult`. Code: `src/building_with_rag/retrieval/semantic.py`.

- Validate: `question` trimmed, non-empty. `filters` values must be known strings (unknown fields rejected). `required_acts`, `chapter`, or a `caller_id` other than `WEBUI_DEMO_CALLER_ID` → HTTP 422.
- Scope: server fixes `access_level` to `["public"]`; caller `act`/`status` lists only narrow it (empty = no narrowing). Applied inside the `$vectorSearch` `filter`.
- Embed: raw question, `voyage-3.5`, `input_type="query"`, 1,024 dimensions. Clients are lazy; `/healthz` needs no credentials.
- Search: index `vector_index`, path `vector`, `limit` = request limit, `numCandidates = min(200, max(limit, 50, limit * 10))`. Chunks are returned in score order (not de-duplicated by section).
- Resolve: `chunks` (text) and `sections` (heading, chapter, source). Unresolvable hits are omitted and counted in `trace.unresolved_hits`.

Outcomes:

| Outcome | HTTP | Notes |
|---|---|---|
| `ok` | 200 | passages returned; no score cutoff |
| `no_results` | 200 | filters match nothing indexed |
| `retrieval_not_ready` | 503 | missing credentials, index absent/not queryable, empty embeddings, model/dimension mismatch |
| `retrieval_upstream_error` | 502 | Voyage or MongoDB error during a query |
| validation error | 422 | bad question, filters, scope fields, or caller |

Scores rank similarity only; they do not prove a passage is correct.

`generate_answer: true` attaches a grounded answer in `QueryResult.generation` (Story 3.1); retrieval is unchanged.

Added optional `RetrievedChunk` fields (null when missing): `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.

`trace`: `mode`, `query`, `embedding`, `index`, `limit`, `num_candidates`, `filters`, `caller_id`, `result_count`, `ignored`, `unresolved_hits`.

Diagnostic command (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Context and answer boundaries (Story 3.1)

Flow: semantic `QueryResult.results` → bounded labelled context → one chat-completions call → strict parse → resolved citations. Code: `src/building_with_rag/generation/` (`context.py`, `answer.py`); wired in `routes/query.py` when `pattern: "semantic"` and `generate_answer: true`. No second retrieval path; without `generate_answer`, `generation` is null and no model call happens.

- Context: first passages in score order, at most 5 passages and 12,000 characters (`MAX_PASSAGES`, `MAX_CONTEXT_CHARS`); a passage is never cut. Labelled `E1`, `E2`, …
- Call: `POST {GENERATION_API_BASE_URL}/chat/completions` (OpenAI-compatible), `temperature: 0`, 30 s timeout (`TIMEOUT_SECONDS`), no retries, via `httpx`.
- Untrusted evidence: evidence blocks are source text, never instructions; the model must ignore instructions inside them and answer only from the labelled evidence. No claims about current legal applicability beyond the supplied BNS/IPC documents.
- Parse: strict JSON `{outcome, answer, claims[{text, evidence}], reason}`; one code fence is stripped; nothing is repaired or retried. Every claim must cite at least one supplied label.

Outcomes (`generation.outcome`):

| Outcome | Meaning |
|---|---|
| `answered` | valid answer, claims, resolved citations and supporting passages |
| `insufficient_evidence` | no passages, or the model says evidence is missing/unrelated; empty text |
| `unavailable` | missing base URL/key, timeout, connection error, or non-2xx |
| `malformed` | non-JSON, missing choices, unknown label, or rule violation |

Only `answered` carries text, claims, citations, and supporting passages. All four return HTTP 200 with `results` still present; `status` stays the retrieval status. Messages never include the proxy URL or key.

Added optional `GenerationResult` fields: `outcome`, `claims` (`text`, `evidence_labels`), `citations` (`label`, `chunk_id`, `section_id`, `act`, `heading`, `chapter`, `section_number`, `source_pdf`), `supporting_passages`, `provider` (`openai-compatible`), `trace`, `context_outcome` (`assembled` | `empty`). `trace` holds labels→chunk_id, selected/omitted counts, characters, latency, and the model's reason or error kind; no prompt text or secrets.

Chat and streaming are covered by Story 3.2 below.

## Streamed answers and confidence (Story 3.2)

Shared path: `pipeline.py` — `retrieve(request)` (semantic → `semantic_query`, other modes → placeholder) then `answer_events(question, retrieval)`. `/v1/query` drains the events and attaches the final `GenerationResult`; chat forwards `text`/`notice` events and renders the footer from the same final result. One request is one generation/validation operation, never two independent model calls. Retrieval errors are raised before any streaming starts (OpenAI-style envelope on chat, existing detail on query). Non-semantic modes and `generate_answer=false` keep placeholder text / `generation: null` with no model call.

Event flow (`generation/answer.py`, `stream_answer`):

1. Context assembly (unchanged, once per request). Empty context → `insufficient_evidence`, no model call.
2. Attempt: streamed plain-text answer with inline `[E1]` labels (`stream: true`, `temperature: 0`, 30 s timeout per call, `httpx` only). The first characters are buffered so an `INSUFFICIENT_EVIDENCE: <reason>` reply is never streamed as an answer. Each attempt that produces text starts with the `DRAFT — checking evidence` notice.
3. Checks, recording every failure as an issue: `citation_labels` (every cited label was supplied; nothing is stripped or rewritten), `claim_cited` (every sentence/bullet carries a label; text non-empty), `support` (one non-streamed validator call judges each claim against its cited passage; strict JSON).
4. `MAX_ATTEMPTS = 2`: a retry runs only after a failed attempt, and its prompt carries the plain-language issues. The stream shows `Check failed: <detail>. Retrying (attempt 2 of 2)…` between attempts.

Added optional `GenerationResult` fields: `confidence` (`high` only when the final attempt passes all checks; `low` when it fails a check; `None` when no text could be judged), `issues` (`attempt`, `check`, `detail`; all attempts kept), `attempts` (`attempt`, `status` passed|failed|unjudged, `chars`, `latency_ms`), `draft_answer` (last attempt text when it did not pass), `low_confidence_reason`.

Outcome mapping (existing four values): passed → `answered` (text = validated answer, confidence high); final check failed → `malformed` with empty `text`, `draft_answer`, `issues`, confidence low; provider failure → `unavailable`; validator unreachable → `unavailable` with `draft_answer`, no confidence; validator invalid reply → `malformed`, no confidence; `insufficient_evidence` unchanged. Only `answered` carries claims, citations, and supporting passages.

What Open WebUI shows (text only): `DRAFT — checking evidence` before each attempt's text; on pass, `Evidence check passed — confidence: high` and `Sources:` lines (`E1 · BNS §303 · Theft · bns:303`); on a failed final check, `DRAFT — low confidence, not the final answer.` with the reason and failed checks; for `insufficient_evidence`, one plain sentence with no confidence. Streamed drafts cannot be retracted, so the closing text always says whether the answer was checked.

Failure after text began: one final line `Answer generation unavailable — the text above is an unchecked draft.`, then the stop chunk and `data: [DONE]` (HTTP 200 once streaming has begun). Before any text: `Answer generation unavailable.`

`CAPSTONE_API_KEY`: when non-empty, `/v1/chat/completions` requires `Authorization: Bearer <key>` (constant-time compare; otherwise 401 `invalid_api_key`, before streaming). When empty there is no check. `/v1/query`, `/v1/models`, and `/healthz` are unchanged. Chat also accepts the Pipe's nested `rag_options.filters.{act,status}` in addition to the flat fields.

## Hybrid retrieval (Story 4.1)

`pattern: "hybrid"` (`rag-hybrid`) runs two routes over the same corpus, chunks, and embeddings as semantic and merges them with one fixed fusion rule. Context assembly, citations, confidence, and streaming are the shared path (`pipeline.retrieve` + `answer_events`), unchanged. Code: `src/building_with_rag/retrieval/hybrid.py`.

- Keyword route: Atlas Search (`$search`, `text` operator, BM25 via `{$meta: "searchScore"}`) on `chunks.text`. It is not `$text`, not regex, and not the vector index. `embedding_input` and `heading` are not searched. The question is only the `text.query` value.
- Index `chunk_text_index` on `chunks` (`type: "search"`): `text` as `string` with `lucene.standard`; `act`, `status`, `access_level` as `token`; `dynamic: false`. Defined in `ingestion/mongodb_schema.py` (`KEYWORD_INDEX_DEFINITION`). Create it once with `uv run python -m building_with_rag.ingestion.keyword_index` (idempotent; a different existing definition is reported and never replaced; no Voyage calls, no data writes). `ingest` ignores this index.
- Filters: same effective filters as semantic (server-fixed `access_level=["public"]`; caller `act`/`status` lists only narrow), applied inside `$search.compound.filter` with the `in` operator, and inside `$vectorSearch.filter` for the vector route.
- Fusion: Reciprocal Rank Fusion over ranks only (raw BM25 and cosine scores are not comparable). Each route returns its top `ROUTE_DEPTH = max(limit, min(50, max(20, 4 * limit)))` chunks, ranked from 1. `fused_score = Σ 1 / (RRF_K + rank)` over the routes that returned the chunk, `RRF_K = 60`, equal weights. Order: `fused_score` desc, then `semantic_rank` (missing last), then `chunk_id`. Top `limit` are returned; `fused_rank` is 1-based over the full fused list. Chunks (not sections) are fused by `chunk_id`.
- `RetrievedChunk` gains optional `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank` (all `None` in semantic mode; `None` for a route that did not return the chunk). In hybrid, `score` = `fused_score`.
- `trace`: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic` (`index`, `limit`, `num_candidates`, `hit_count`), `keyword` (`index`, `path`, `operator`, `limit`, `hit_count`), `fusion` (`method`, `k`, `weights`, `route_depth`), `contribution` (`both`, `semantic_only`, `keyword_only` among returned results).
- Outcomes: `ok`; `no_results` (filters match nothing, or both routes empty); 503 `retrieval_not_ready` (credentials missing, `vector_index` or `chunk_text_index` missing/not queryable, empty embeddings; the message names the keyword-index command; never silently degraded to semantic-only); 502 `retrieval_upstream_error` (Voyage or MongoDB error); 422 for unsupported scope fields.
- Limitation: rank-only fusion ignores score magnitude; the `text` operator matches any query term (OR), so long natural-language questions can pull in common words; section numbers match only when they appear inside chunk `text`; no stemming or synonyms beyond the standard analyzer. Atlas (M0 or higher) is required; there is no `$text` or client-side keyword fallback. Scores rank only; they do not prove correctness.

Diagnostic command (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}' \
  | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
```

## Re-ranking (Story 4.2)

`pattern: "hybrid-reranked"` (`rag-hybrid-reranked`) runs existing hybrid retrieval, cuts it to a bounded candidate set, makes **one** provider re-ranking call, and passes only the final selected evidence to the shared path (`pipeline.retrieve` + `answer_events`: context assembly, grounded answer, citations, confidence, footer, streaming, all unchanged). `context` is built from `results` only, never `omitted_candidates`. The selected mode decides the route; there is no automatic mode choice and no fallback. Code: `src/building_with_rag/retrieval/rerank.py`.

Settings (existing names; defaults): `RERANK_API_BASE_URL` (`https://api.voyageai.com/v1`), `RERANK_API_KEY` (empty; required), `RERANK_MODEL_NAME` (`rerank-2.5`), `RERANK_REQUEST_TIMEOUT_SECONDS` (30), `RERANK_CANDIDATE_LIMIT` (20), `RERANK_SEND_LIMIT` (10), `RERANK_RETURN_LIMIT` (5). Valid when `1 ≤ RETURN ≤ SEND ≤ CANDIDATE ≤ 20` and timeout ≥ 1; otherwise 503 `retrieval_not_ready` naming the setting (not its value). `RERANK_API_KEY` never falls back to `VOYAGE_API_KEY`.

Request: one `POST {RERANK_API_BASE_URL}/rerank` via `httpx`, `Authorization: Bearer <RERANK_API_KEY>`, JSON `{model, query, documents}` (no `top_k`, so every sent candidate is scored), document text `"{heading}\n{chunk text}"`, no retries. Reply: `data[]` of `{index, relevance_score}` and optional `usage.total_tokens`. Validation: `data` is a non-empty list; each `index` is an int within the sent range and unique; each `relevance_score` is a finite number; every sent candidate is scored. Anything else is a provider failure; scores are never invented or filled in.

Selection:
1. Candidates = top `RERANK_CANDIDATE_LIMIT` of the hybrid fused list (`fused_rank` = pre-rerank order).
2. Sent = first `RERANK_SEND_LIMIT` by `fused_rank`; the rest are cut before re-ranking (`omitted_reason: "not_sent_to_reranker"`, rerank fields `None`).
3. Sent candidates are ordered by `relevance_score` desc, ties by `fused_rank`; `rerank_rank` is 1-based.
4. Final = first `min(request.limit, RERANK_RETURN_LIMIT)`; the other scored candidates are cut after re-ranking (`omitted_reason: "below_return_limit"`, rerank score/rank kept).

Result shape: `results` = final evidence ordered by `rerank_rank` with `score == rerank_score` (non-increasing); every hybrid field (`semantic_*`, `keyword_*`, `fused_*`) stays as the "before" evidence. `omitted_candidates` = all cut candidates in `fused_rank` order. New optional `RetrievedChunk` fields: `rerank_score`, `rerank_rank`, `omitted_reason` (all `None` in semantic and hybrid).

Trace: `mode`, `query`, `filters`, `caller_id`, `result_count`, `hybrid` (`embedding`, `semantic`, `keyword`, `fusion`, `contribution`, `unresolved_hits`), `rerank` (`model`, `candidate_limit`, `send_limit`, `return_limit`, `candidates`, `sent`, `returned`, `omitted_before`, `omitted_after`, `latency_ms`, `usage_tokens` when reported).

Outcomes: `ok`; `no_results` (hybrid found nothing; no provider call); 503 `retrieval_not_ready` (missing or invalid rerank configuration, checked before any Voyage/MongoDB call); 502 `retrieval_upstream_error` (timeout, connection error, non-2xx, or invalid reply: no results, no fallback to hybrid order); hybrid errors pass through. Messages never include the URL, key, or provider body.

Limitations: candidates outside the top `RERANK_CANDIDATE_LIMIT` are never seen; the re-ranker scores each passage independently against the question; its scores are model-specific, uncalibrated, not comparable with fused scores or across questions, and there is no score cutoff; it adds one provider call (latency, cost, rate limits) per request; no retry and no fallback; the answer context cap (5 passages / 12,000 characters) still applies. Re-ranking scores rank only; they do not prove correctness.

Diagnostic command (text truncated; merges returned and omitted records):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"What is the difference between culpable homicide and murder?","pattern":"hybrid-reranked","limit":5}' \
  | jq '{status, t: .trace.rerank, r: [(.results[]|.+{kept:true}), (.omitted_candidates[]|.+{kept:false})] | map({section_id, kept, fr: .fused_rank, fs: .fused_score, rr: .rerank_rank, rs: .rerank_score, why: .omitted_reason, text: .text[:50]}) | sort_by(.fr)}'
```

## Structured exact retrieval (Story 5.1)

`pattern: "structured"` (`rag-structured`) classifies the question into validated `StructuredSignals`, then makes one read-only `find_one` on the `sections` collection for the exact BNS or IPC section named. It is not LLM extraction, not a semantic-search replacement, and not a router: the selected mode decides the route, with no fallback to or from other modes. No Voyage call, embedding, vector index, or keyword index is used. Code: `src/building_with_rag/retrieval/structured.py`; `StructuredSignals` is in `contracts.py` (`intent`, `act`, `section_number` 1–999, `chapter`, `status` ok|recommendation|clarification_needed, `reason`).

**Exact-input contract.** Only the classifier's validated output — `act`, `section_number`, and an optional validated `chapter` — plus server filters may reach MongoDB. The raw question text never enters a predicate, and no operators come from request fields. The predicate is `{act, section_number, access_level: {$in: ["public"]}}` plus `status: {$in: …}` when `filters.status` is given and `chapter` when given; one document is fetched, without `provenance`. `chapter` comes only from `QueryRequest.chapter` (1–40 characters of letters, digits, spaces, `.` and `-`; otherwise 422 `unsupported_option`), never from the question text. `required_acts` and a foreign `caller_id` are rejected with 422.

Classifier rules, in order (pure, rule-based, no LLM, no I/O):
1. "how many", "count", "total number" → `aggregation`, `recommendation`, no lookup.
2. "list", "all sections", "which sections", "sections in/under …" → `filter`, `recommendation`, no lookup.
3. A section reference (`section N`, `sec. N`, `s. N`, `§N`; integer only) plus exactly one act token (`BNS`, `IPC`, "Bharatiya Nyaya Sanhita", "Indian Penal Code", case-insensitive) and one distinct number → `ok`.
4. A section number with no act, both acts, several numbers, a non-integer (for example `103A`) or an out-of-range number → `clarification_needed`; acts are never guessed. An act filter that excludes the named act also gives `clarification_needed` with no query.
5. Anything else → `recommendation` (use `semantic` or `hybrid` for open questions).

Outcomes (`QueryResult.status`, HTTP 200 unless noted): `ok` (one `RetrievedChunk` with `chunk_id = section_id`, `text` = section text, `score = 1.0` marking an exact match, not a similarity; source fields copied); `not_found` (valid request, no record in this corpus — not a claim that the law has no such section); `clarification_needed`; `recommendation`; 503 `retrieval_not_ready` (missing `MONGODB_URI`, only when a lookup is needed); 502 `retrieval_upstream_error` (MongoDB failure). Trace: `mode`, `signals`, `mongodb_called`, `collection`, `filters`, `caller_id`, `result_count`, and for `ok` a `record` (`section_id`, `status`, `source_status_version`).

**Answer boundary.** Retrieval returns the exact record. An explanation is produced only through the existing grounded-answer path when `generate_answer: true` (chat always requests it), and only after an `ok` lookup: `pipeline.should_answer` and `answer_events` make no model call for `not_found`, `clarification_needed`, or `recommendation`. Chat then streams the result `message` as plain content with the same framing, and `/v1/query` leaves `generation` unset. `status` and `source_status_version` describe the source document; they do not claim current legal applicability.

Limitations: integer section numbers only (no `103A`); no multi-section or cross-act comparison; filter and aggregation questions are recognised but not executed; 11 IPC sections (4, 5, 18, 34, 40, 75, 161–165) are absent from `sections`; rule-based phrasing misses reworded questions; a section longer than the 12,000-character context cap yields `insufficient_evidence` when an answer is requested (direct inspection still returns it).

Diagnostic command (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"What does BNS section 103 say?","pattern":"structured"}' \
  | jq '{status, t: (.trace | {signals, mongodb_called, record}), r: [.results[] | {section_id, act, status, text: .text[:60]}]}'
```
