# AthenAI Backend

This is the backend for the AthenAI study tool. It provides API endpoints for:
- Chat (LLM interaction)
- Bulk upload of transcripts, OCR, and video descriptions
- Search and query within uploaded content
- Quiz generation
- Summarization
- Study guide generation
- Video time-linking

## Running the Backend

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start the server:
   ```sh
   uvicorn main:app --reload
   ```

## Qwen LLM Integration
- The default real model is `Qwen/Qwen3-4B-Instruct-2507`, loaded in FP16 (about 8 GB of VRAM plus KV cache). It is an instruct-only checkpoint with no thinking mode.
- Set `ATHENAI_MOCK_LLM=0` to use the real model.
- Set `ATHENAI_REQUIRE_CUDA=1` to fail fast when no GPU is visible.
- Set `ATHENAI_QWEN_DEVICE=cuda:0` for a local or AWS GPU.
- Optionally set `ATHENAI_QWEN_MODEL` to a local model directory, for example `..\models\Qwen3-4B-Instruct-2507`. If it is not set, Transformers downloads `Qwen/Qwen3-4B-Instruct-2507` from Hugging Face.
- Tune answer depth with `ATHENAI_MAX_NEW_TOKENS` (default `1024`). Longer study artifacts such as quizzes, summaries, study guides, takeaways, and explanations use `ATHENAI_LONG_TASK_MAX_NEW_TOKENS` (default `2048`).
- Chat responses and study artifacts include every uploaded source chunk by default. Set `ATHENAI_RAG_FULL_ARTIFACT_CONTEXT_CHARS` to a positive character limit only if very large uploads exceed your model's context window.
- Tune generation style with `ATHENAI_TEMPERATURE` (default `0.55`), `ATHENAI_TOP_P` (default `0.9`), and `ATHENAI_TOP_K` (default `40`).
- Set `ATHENAI_MEDIASITE_API_KEY` to enable Mediasite API access for preview-stack integrations.
- Set `ATHENAI_MEDIASITE_BASE_URL` to your Mediasite API root, for example `https://mysite.mediasite.com/Mediasite/api/v1`.
- Optional authentication fallbacks:
  - `ATHENAI_MEDIASITE_USERNAME` and `ATHENAI_MEDIASITE_PASSWORD` for Basic auth. The Mediasite REST API demo sends these with `sfapikey`.
  - `ATHENAI_MEDIASITE_APPLICATION_TICKET` for `Mediasite-Application-Ticket`.
  - `ATHENAI_MEDIASITE_AUTHORIZATION_TICKET` for `Authorization: SfAuthTicket ...`.

### Local GPU test

On Windows, run:

```cmd
run_api.cmd
```

The script uses `..\models\Qwen3-4B-Instruct-2507` when that directory exists, otherwise it uses the Hugging Face model id.

### AWS GPU deployment

Use the same environment variables on the instance:

```sh
export ATHENAI_QWEN_MODEL=Qwen/Qwen3-4B-Instruct-2507
export ATHENAI_QWEN_DEVICE=cuda:0
export ATHENAI_MOCK_LLM=0
export ATHENAI_REQUIRE_CUDA=1
export ATHENAI_PRELOAD_MODEL=1
uvicorn main:app --host 0.0.0.0 --port 8001
```

If the instance has no internet access, download or copy the model to disk and set `ATHENAI_QWEN_MODEL` to that local directory.

## API Endpoints
- `/upload/` — Bulk upload content (transcripts, OCR, descriptions)
- `/chat/` — Chat with the LLM
- `/search/` — Search uploaded content
- `/quiz/` — Generate quizzes
- `/summarize/` — Summarize content
- `/video-link/` — Link to video timestamps
- `GET /session/{id}` — Restore a session's chat history and document list
- `DELETE /session/{id}` — Erase a session's uploads, chat history, and saved snapshot immediately
- `/mediasite/status` — Report whether Mediasite connection settings are configured
- `/mediasite/import-presentation/` — Fetch available Mediasite presentation metadata, OCR content, caption text, and slide details into the current RAG session
- `/mediasite/channel-presentations/` — List presentation GUIDs from a Mediasite channel/catalog, showcase channel, or playlist
- `/mediasite/import-channel/` — Poll `MediasiteChannels('id')/Presentations` for presentation GUIDs and import `Presentations('id')/OcrContent`, `Presentations('id')/CaptionContent`, and `Presentations('id')/SlideDetailsContent` for each presentation

Example Mediasite import request:

```json
{
  "session_id": "optional-existing-session",
  "base_url": "https://mysite.mediasite.com/Mediasite/api/v1",
  "api_key": "your-api-key",
  "username": "user",
  "password": "password",
  "presentation_id": "presentation-guid"
}
```

You can omit `base_url`, `api_key`, and auth fields when the matching environment variables are set. The server's
own Mediasite credentials are only ever sent to `ATHENAI_MEDIASITE_BASE_URL`'s origin: a request may name another
`base_url` only if it is `https` and its host is listed in `ATHENAI_MEDIASITE_ALLOWED_HOSTS` (comma-separated), and
then it must bring its own credentials. `impersonate_username` requires a username and password in the request.
Paging links that point at a different host are refused.

Example channel/catalog import request:

```json
{
  "session_id": "optional-existing-session",
  "base_url": "https://mysite.mediasite.com/Mediasite/api/v1",
  "api_key": "your-api-key",
  "username": "user",
  "password": "password",
  "channel_id": "channel-or-catalog-guid",
  "resource_type": "MediasiteChannels",
  "max_pages": 20
}
```

Supported `resource_type` values are `MediasiteChannels` (also used for legacy catalog/channel wording), `ShowcaseChannels`, and `Playlists`.

## Notes
- Uploaded content and chat history are stored in memory and mirrored to short-lived JSON session snapshots so they survive UI/API restarts.
- Session snapshots default to `backend/.athenai_sessions` and expire after 12 hours. Tune with `ATHENAI_SESSION_TTL_HOURS`, change the folder with `ATHENAI_SESSION_STORE_DIR`, or set `ATHENAI_SESSION_PERSISTENCE=0` to keep sessions memory-only.
- Logs default to stdout. Set `ATHENAI_LOG_LEVEL` (`INFO`, `DEBUG`, etc.) and `ATHENAI_LOG_FILE` to write rotating logs; `run_api.cmd` writes to `backend/athenai-api.log`.
- CORS defaults to all origins without credentials. Restrict it with `ATHENAI_CORS_ORIGINS` (comma-separated) in deployment.
- Internet search is not implemented. The `use_internet` request field is accepted but ignored, and responses always report `used_internet: false`.

## Behavior configuration

AthenAI's behavior is split into layers that are maintained and tested separately:

| Layer | File | What it controls | Tests |
| --- | --- | --- | --- |
| Persona, source grounding, safety, and security guidance for the model | `behavior_policy.py` | The system prompt, assembled from `PERSONA_GUIDANCE`, `SOURCE_GROUNDING_RULES`, `CITATION_FORMAT_RULES`, `SAFETY_RULES`, `SECURITY_RULES`, and `HISTORY_RULES` | `tests/test_behavior_policy.py` |
| Prompt screening | `moderation_utils.py` | Whether a prompt reaches the model (`allow`), gets a fixed supportive reply (`respond`: self-harm, other users' data, venting), or is briefly declined (`decline`: weapons, violence, sexual content, harassment, account intrusion). Allowed prompts can carry trusted guidance notes (for example "quoting a slur for history" or "sexual-health question"). Reply wording lives in `RESPONSES`; note wording lives in `GUIDANCE`. | `tests/test_moderation_utils.py` |
| Enforced application controls | `security_utils.py`, `main.py` | Session ownership tokens, argument bounds, upload limits, output sanitizing, Mediasite host allowlist, log redaction | `tests/test_security_utils.py`, `tests/test_main.py` |
| Untrusted-content isolation | `qwen_utils.py` | Wraps the question and sources in untrusted blocks, flags injection-like source text (`PROMPT_INJECTION_PATTERNS`) | `tests/test_qwen_utils.py`, `tests/test_behavior_policy.py` |

Profanity, missing manners, and criticism of AthenAI never cause a refusal. Polite wording never makes a harmful
request acceptable. The screen is deliberately narrow; ambiguous cases go to the model along with the safety rules.

`ATHENAI_GENERAL_KNOWLEDGE` (default `1`) lets the model add brief background knowledge to explain a concept or flag
a likely error, always labeled "Outside your material" and never cited as a source. Set it to `0` to restrict answers
strictly to uploaded material.

## Security controls

These are enforced in code, independent of anything the model writes. Prompt wording alone cannot guarantee
protection against prompt injection, so the controls below assume the model may be manipulated.

- **Session ownership.** Creating a session (by upload, chat, Mediasite import, or `GET /session/{id}` for a new id)
  returns a `session_token` once. Every later request for that session must send it in the
  `X-AthenAI-Session-Token` header, or the API returns 403. Only a SHA-256 hash of the token is stored. Sessions saved
  before this change, which have no owner, are claimed by the next caller. Set `ATHENAI_REQUIRE_SESSION_TOKEN=0` only
  for local debugging.
- **Retrieval stays inside the session.** The model only sees chunks from the caller's own session, and it has no
  tools, credentials, or network access.
- **Output sanitizing.** Model output cannot render images or active HTML, and links survive only when the exact URL
  already appears in the student's message or a source that does not carry injection-like text.
- **Argument limits.** `top_k` is clamped to `ATHENAI_MAX_TOP_K` (default 20). Prompts over `ATHENAI_MAX_PROMPT_CHARS`
  (default 12000) return 413. Files over `ATHENAI_MAX_UPLOAD_BYTES` (default 50 MB) are skipped. Zip archives are
  checked against `ATHENAI_MAX_ZIP_ENTRIES` (500) and `ATHENAI_MAX_ZIP_UNCOMPRESSED_BYTES` (200 MB) before
  decompression. Mediasite `max_pages`, `max_presentations`, and `timeout_seconds` are bounded.
- **Logs.** Prompt text and owner tokens are never logged; moderation logs record only the category. Session ids in
  request paths are redacted.

Not yet covered: user identity. AthenAI has no login, so a session belongs to whichever browser holds its token, and
Mediasite imports run with the server's service account rather than the student's own Mediasite permissions. Binding
sessions and imports to a real user (for example LTI or Mediasite SSO) needs an identity provider integration.

## Behavior evaluation

`evals/behavior_cases.json` holds the evaluation cases. Each case has an input, expected and unacceptable behavior,
an expected route, and heuristic checks for live model output. To add a case, append to that file.

```sh
python -m pytest tests/test_behavior_evals.py        # deterministic layer, no model needed
python -m evals.run_behavior_evals                   # same, with a report in evals/reports/
python -m evals.run_behavior_evals --live --repeat 3 # score the real model (GPU), 3 samples per case
```

The deterministic layer runs every case through the real API with a recording stand-in for the model. It checks
routing (unnecessary refusals and missed harmful requests), guidance notes, injection flagging, history handling,
output sanitizing, and cross-session access. The live layer scores real responses with several independent
signals, not exact wording. Responses that fail a check are marked "Needs review" in `evals/reports/latest.md`,
alongside the case's expected and unacceptable behavior, so a person makes the final call.
