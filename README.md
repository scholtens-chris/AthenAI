# AthenAI

AthenAI is an AI-powered study agent for students using Mediasite. It helps students study from their course video transcripts and from additional materials they choose to upload, such as notes, Word documents, PDFs, PowerPoint slides, and other class resources.

The agent is designed to stay grounded in the student's provided course content. It uses Mediasite transcripts and uploaded study materials only. It does not answer from internet references or outside web sources.

## What It Does

- Answers questions about uploaded course materials
- Searches across transcripts, notes, documents, and slides
- Generates summaries and key takeaways
- Creates quizzes for active recall
- Builds study guides from the available course content
- Links answers back to video timestamps when transcript timing is available

## Source Policy

AthenAI only uses:

- Mediasite video transcripts
- Study materials uploaded by the student
- Extracted text from supported files, including PDFs, Word documents, and PowerPoint presentations

AthenAI does not use:

- Internet search results
- External articles or websites
- Unprovided reference materials

This keeps answers focused on the material students are actually responsible for studying.

## Project Structure

```text
AthenAI/
  backend/   FastAPI backend for uploads, search, chat, quizzes, summaries, and study guides
  frontend/  React/Vite frontend for the study interface
  models/    Optional local model files
```

## Running Locally

### Backend

```sh
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The backend exposes endpoints for uploading materials, chatting with the study agent, searching content, generating quizzes, summarizing content, and creating study guides.

### Frontend

```sh
cd frontend
npm install
npm run dev
```

The frontend is built with React and Vite.

## Model Configuration

The backend supports local or downloaded Qwen model checkpoints through environment variables. See [backend/README.md](backend/README.md) for backend-specific model setup, GPU options, and API details.

## Intended Use

AthenAI is meant to support studying, review, and comprehension. Students should verify important answers against their course materials, especially when preparing for exams or assignments.
