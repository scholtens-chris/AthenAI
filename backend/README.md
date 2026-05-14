# AthenAI Backend

This is the backend for the AthenAI study tool. It provides API endpoints for:
- Chat (LLM interaction)
- Bulk upload of transcripts, OCR, and video descriptions
- Search and query within uploaded content
- Quiz generation
- Summarization
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
- The backend is ready for local Qwen LLM integration. Add the Qwen model and inference logic where indicated in `main.py`.

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
