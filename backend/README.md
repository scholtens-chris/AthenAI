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
- The default real model is `Qwen/Qwen3-1.7B`, the official Qwen 1.7B chat/instruction-capable checkpoint. The separate base checkpoint is `Qwen/Qwen3-1.7B-Base`.
- Set `ATHENAI_MOCK_LLM=0` to use the real model.
- Set `ATHENAI_REQUIRE_CUDA=1` to fail fast when no GPU is visible.
- Set `ATHENAI_QWEN_DEVICE=cuda:0` for a local or AWS GPU.
- Optionally set `ATHENAI_QWEN_MODEL` to a local model directory, for example `..\models\Qwen3-1.7B`. If it is not set, Transformers downloads `Qwen/Qwen3-1.7B` from Hugging Face.
- Tune answer depth with `ATHENAI_MAX_NEW_TOKENS` (default `1024`). Longer study artifacts such as quizzes, summaries, study guides, takeaways, and explanations use `ATHENAI_LONG_TASK_MAX_NEW_TOKENS` (default `2048`).
- Chat responses and study artifacts include every uploaded source chunk by default. Set `ATHENAI_RAG_FULL_ARTIFACT_CONTEXT_CHARS` to a positive character limit only if very large uploads exceed your model's context window.
- Tune generation style with `ATHENAI_TEMPERATURE` (default `0.55`), `ATHENAI_TOP_P` (default `0.9`), and `ATHENAI_TOP_K` (default `40`).

### Local GPU test

On Windows, run:

```cmd
run_api.cmd
```

The script uses `..\models\Qwen3-1.7B-Instruct` or `..\models\Qwen3-1.7B` when either directory exists, otherwise it uses the Hugging Face model id.

### AWS GPU deployment

Use the same environment variables on the instance:

```sh
export ATHENAI_QWEN_MODEL=Qwen/Qwen3-1.7B
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

## Notes
- Uploaded content is stored in memory per session (for demo/dev; use Redis for production).
- CORS is enabled for frontend integration.
- Option to use only uploaded content or include external internet sources is supported via API.
