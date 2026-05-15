import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import Counter

import uuid
import zipfile
import io
import json
import xml.etree.ElementTree as ET
import re

from qwen_utils import QwenLLM

# In-memory session store (replace with Redis for production)
sessions = {}

CHUNK_WORDS = int(os.getenv("ATHENAI_RAG_CHUNK_WORDS", "220"))
CHUNK_OVERLAP = int(os.getenv("ATHENAI_RAG_CHUNK_OVERLAP", "45"))
DEFAULT_RAG_TOP_K = int(os.getenv("ATHENAI_RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("ATHENAI_RAG_MAX_CONTEXT_CHARS", "8000"))
OVERVIEW_CHUNK_LIMIT = int(os.getenv("ATHENAI_RAG_OVERVIEW_CHUNKS", "8"))
FULL_ARTIFACT_CONTEXT_CHARS = int(os.getenv("ATHENAI_RAG_FULL_ARTIFACT_CONTEXT_CHARS", "0"))

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
    "i", "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

STUDY_TASK_PATTERNS = {
    "quiz": (
        r"\bquiz\b",
        r"\btest me\b",
        r"\bpractice question",
        r"\bflashcards?\b",
    ),
    "summary": (
        r"\bsummar(?:y|ize|ise|ise)\b",
        r"\brecap\b",
        r"\bchapter\s+\d+\b",
    ),
    "find": (
        r"\bfind\b",
        r"\bwhere\b",
        r"\bwhen\b",
        r"\btimestamp\b",
        r"\bexplained?\b",
        r"\bmentioned?\b",
    ),
    "takeaways": (
        r"\bkey takeaways?\b",
        r"\bmain points?\b",
        r"\bbig ideas?\b",
        r"\bimportant concepts?\b",
    ),
    "study_guide": (
        r"\bstudy\s+guides?\b",
        r"\bstudyguide\b",
        r"\breview\s+guides?\b",
        r"\blearning\s+objectives?\b",
    ),
    "explain": (
        r"\bexplain\b",
        r"\bteach me\b",
        r"\bwalk me through\b",
        r"\bhelp me understand\b",
        r"\blecture\b",
    ),
}

BROAD_STUDY_PROMPTS = {
    "quiz",
    "quiz me",
    "test me",
    "explain this",
    "explain this lecture",
    "summarize",
    "summarize this",
    "summarize this lecture",
    "what are key takeaways",
    "key takeaways",
    "main points",
    "study guide",
    "make a study guide",
    "create a study guide",
    "review guide",
}

# Initialize Qwen LLM (update model_path/device as needed)
qwen_llm = QwenLLM()

@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_model = os.getenv("ATHENAI_PRELOAD_MODEL", "0") == "1"
    if preload_model and not qwen_llm.use_mock_model:
        qwen_llm._load_model()
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")

def ensure_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {"content": [], "chunks": []}
    sessions[session_id].setdefault("content", [])
    sessions[session_id].setdefault("chunks", [])
    return sessions[session_id]

def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks

def add_document_to_session(session_id: str, filename: str, text: str) -> int:
    session = ensure_session(session_id)
    document_id = str(uuid.uuid4())
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    chunks = chunk_text(cleaned_text)
    if not chunks:
        return 0

    session["content"].append({"id": document_id, "filename": filename, "text": cleaned_text})

    for index, chunk in enumerate(chunks):
        tokens = tokenize(chunk)
        session["chunks"].append({
            "id": f"{document_id}:{index}",
            "document_id": document_id,
            "filename": filename,
            "chunk_index": index,
            "text": chunk,
            "tokens": Counter(tokens),
            "token_count": max(1, len(tokens)),
        })
    return len(chunks)

def retrieve_context(session_id: Optional[str], query: str, top_k: int = DEFAULT_RAG_TOP_K) -> list[dict]:
    if not session_id or session_id not in sessions:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_counts = Counter(query_tokens)
    scored = []
    for chunk in sessions[session_id].get("chunks", []):
        overlap_terms = set(query_counts) & set(chunk["tokens"])
        if not overlap_terms:
            continue
        term_score = sum(min(query_counts[token], chunk["tokens"][token]) for token in overlap_terms)
        coverage_score = len(overlap_terms) / max(1, len(query_counts))
        density_score = term_score / chunk["token_count"]
        score = term_score + (coverage_score * 2.0) + density_score
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    used_chars = 0
    for score, chunk in scored[:max(1, top_k)]:
        text = chunk["text"]
        if used_chars + len(text) > MAX_CONTEXT_CHARS:
            text = text[:max(0, MAX_CONTEXT_CHARS - used_chars)].rstrip()
        if not text:
            break
        used_chars += len(text)
        results.append({
            "id": chunk["id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "score": round(score, 4),
            "text": text,
        })
        if used_chars >= MAX_CONTEXT_CHARS:
            break
    return results

def infer_study_task(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.lower()).strip()
    for task, patterns in STUDY_TASK_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return task
    return "answer"

def is_broad_study_prompt(prompt: str, task: str) -> bool:
    normalized = re.sub(r"\s+", " ", prompt.lower()).strip(" ?.!")
    content_tokens = tokenize(normalized)
    if normalized in BROAD_STUDY_PROMPTS:
        return True
    if task in {"quiz", "takeaways", "study_guide"} and len(content_tokens) <= 3:
        return True
    if task in {"explain", "summary"} and len(content_tokens) <= 4:
        return True
    return False

def retrieve_session_overview(session_id: Optional[str], limit: int = OVERVIEW_CHUNK_LIMIT) -> list[dict]:
    if not session_id or session_id not in sessions:
        return []

    chunks = sessions[session_id].get("chunks", [])
    if not chunks:
        return []

    selected = []
    seen_documents = set()
    for chunk in chunks:
        document_id = chunk["document_id"]
        if document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        selected.append(chunk)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        selected_ids = {chunk["id"] for chunk in selected}
        stride = max(1, len(chunks) // max(1, limit))
        for index in range(0, len(chunks), stride):
            chunk = chunks[index]
            if chunk["id"] not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk["id"])
            if len(selected) >= limit:
                break

    results = []
    used_chars = 0
    for chunk in selected[:limit]:
        text = chunk["text"]
        if used_chars + len(text) > MAX_CONTEXT_CHARS:
            text = text[:max(0, MAX_CONTEXT_CHARS - used_chars)].rstrip()
        if not text:
            break
        used_chars += len(text)
        results.append({
            "id": chunk["id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "score": 0,
            "text": text,
        })
        if used_chars >= MAX_CONTEXT_CHARS:
            break
    return results

def material_context_entry(chunk: dict, score: float, text: str) -> dict:
    return {
        "id": chunk["id"],
        "filename": chunk["filename"],
        "chunk_index": chunk["chunk_index"],
        "score": score,
        "text": text,
    }

def retrieve_full_session_context(session_id: Optional[str], max_chars: int = FULL_ARTIFACT_CONTEXT_CHARS) -> list[dict]:
    if not session_id or session_id not in sessions:
        return []

    chunks = sessions[session_id].get("chunks", [])
    if not chunks:
        return []

    results = []
    used_chars = 0
    for chunk in chunks:
        text = chunk["text"]
        if max_chars > 0 and used_chars + len(text) > max_chars:
            remaining_chars = max(0, max_chars - used_chars)
            text = text[:remaining_chars].rstrip()
        if not text:
            break
        used_chars += len(text)
        results.append(material_context_entry(chunk, 0, text))
        if max_chars > 0 and used_chars >= max_chars:
            break
    return results

def retrieve_study_context(session_id: Optional[str], prompt: str, top_k: int) -> tuple[list[dict], str]:
    task = infer_study_task(prompt)

    return retrieve_full_session_context(session_id), task

def public_source(source: dict) -> dict:
    return {
        "id": source["id"],
        "filename": source["filename"],
        "chunk_index": source["chunk_index"],
        "score": source["score"],
        "preview": source["text"][:360],
    }

@app.get("/llm/status")
async def llm_status():
    """Report the currently configured LLM mode and model."""
    return {
        "model": qwen_llm.model_path,
        "mock": qwen_llm.use_mock_model,
        "loaded": qwen_llm.model is not None,
        "device": str(qwen_llm.device),
        "max_new_tokens": qwen_llm.max_new_tokens,
        "long_task_max_new_tokens": qwen_llm.long_task_max_new_tokens,
        "temperature": qwen_llm.temperature,
        "top_p": qwen_llm.top_p,
        "top_k": qwen_llm.top_k,
    }

def parse_json(content: bytes) -> str:
    try:
        data = json.loads(content.decode("utf-8", errors="ignore"))
        if isinstance(data, dict):
            return json.dumps(data, indent=2)
        elif isinstance(data, list):
            return "\n".join([json.dumps(item) for item in data])
        else:
            return str(data)
    except Exception:
        return "[Invalid JSON]"

def parse_dfxp(content: bytes) -> str:
    try:
        root = ET.fromstring(content.decode("utf-8", errors="ignore"))
        texts = [elem.text for elem in root.iter() if elem.text]
        return " ".join(texts)
    except Exception:
        return "[Invalid DFXP XML]"

def parse_srt(content: bytes) -> str:
    text = content.decode("utf-8", errors="ignore")
    # Remove SRT numbering and timestamps
    text = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", text)
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", "", text)
    return re.sub(r"\n+", " ", text).strip()

def parse_vtt(content: bytes) -> str:
    text = content.decode("utf-8", errors="ignore")
    # Remove WEBVTT header and timestamps
    text = re.sub(r"WEBVTT.*?\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", "", text)
    return re.sub(r"\n+", " ", text).strip()

def parse_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def parse_docx(content: bytes) -> str:
    try:
        from docx import Document

        document = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception:
        return ""

def parse_pptx(content: bytes) -> str:
    try:
        from pptx import Presentation

        presentation = Presentation(io.BytesIO(content))
        texts = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    texts.append(shape.text)
        return "\n".join(texts)
    except Exception:
        return ""

def extract_files_from_zip(zip_bytes: bytes):
    files = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            files.append({"filename": name, "content": z.read(name)})
    return files

def parse_file(filename: str, content: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext in {"txt", "md", "csv"}:
        return parse_txt(content)
    elif ext == "json":
        return parse_json(content)
    elif ext == "dfxp":
        return parse_dfxp(content)
    elif ext == "srt":
        return parse_srt(content)
    elif ext == "vtt":
        return parse_vtt(content)
    elif ext == "pdf":
        return parse_pdf(content)
    elif ext == "docx":
        return parse_docx(content)
    elif ext == "pptx":
        return parse_pptx(content)
    else:
        return content.decode("utf-8", errors="ignore")

@app.post("/upload/")
async def upload_content(session_id: Optional[str] = Form(None), files: List[UploadFile] = File(...)):
    """Bulk upload transcripts, OCR, and descriptions. Every upload is indexed for RAG."""
    if not session_id:
        session_id = str(uuid.uuid4())
    ensure_session(session_id)
    indexed_files = []
    skipped_files = []
    added_chunk_count = 0
    for file in files:
        fname = file.filename
        content = await file.read()
        if fname.lower().endswith(".zip"):
            extracted = extract_files_from_zip(content)
            for f in extracted:
                text = parse_file(f["filename"], f["content"])
                chunks_added = add_document_to_session(session_id, f["filename"], text)
                if chunks_added:
                    indexed_files.append(f["filename"])
                    added_chunk_count += chunks_added
                else:
                    skipped_files.append(f["filename"])
        else:
            text = parse_file(fname, content)
            chunks_added = add_document_to_session(session_id, fname, text)
            if chunks_added:
                indexed_files.append(fname)
                added_chunk_count += chunks_added
            else:
                skipped_files.append(fname)
    return {
        "session_id": session_id,
        "files": [f.filename for f in files],
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
        "added_chunk_count": added_chunk_count,
        "chunk_count": len(sessions[session_id]["chunks"]),
    }


@app.post("/chat/")
async def chat(request: Request):
    """RAG-first chat endpoint for LLM interaction (Qwen)."""
    data = await request.json()
    prompt = data.get("prompt", "")
    session_id = data.get("session_id")
    use_internet = data.get("use_internet", False)
    image = data.get("image")  # Expecting base64 or bytes (optional)

    top_k = int(data.get("top_k", DEFAULT_RAG_TOP_K))
    context, study_task = retrieve_study_context(session_id, prompt, top_k)
    if session_id and session_id in sessions and sessions[session_id].get("chunks") and not context:
        response = "I could not find relevant uploaded material for that question. Try asking with terms from your notes or upload more source files."
        return {
            "response": response,
            "usage": qwen_llm.estimate_usage(prompt, response, context=[], study_task=study_task),
            "used_internet": False,
            "rag_first": True,
            "sources": [],
        }
    if not context:
        response = "Upload study files first so I can answer from your material."
        return {
            "response": response,
            "usage": qwen_llm.estimate_usage(prompt, response, context=[], study_task=study_task),
            "used_internet": False,
            "rag_first": True,
            "sources": [],
        }

    # TODO: handle image input (decode if base64)
    chat_result = qwen_llm.chat_with_usage(
        prompt,
        image=image,
        context=context,
        use_internet=use_internet,
        study_task=study_task,
    )
    return {
        "response": chat_result["response"],
        "usage": chat_result["usage"],
        "used_internet": False,
        "rag_first": True,
        "study_task": study_task,
        "sources": [public_source(source) for source in context],
    }

@app.post("/search/")
async def search(request: Request):
    """Search within uploaded content."""
    data = await request.json()
    session_id = data.get("session_id")
    query = data.get("query") or data.get("prompt") or ""
    top_k = int(data.get("top_k", DEFAULT_RAG_TOP_K))
    return {"results": [public_source(source) for source in retrieve_context(session_id, query, top_k)]}

@app.post("/quiz/")
async def quiz(request: Request):
    """Generate quiz from uploaded content."""
    data = await request.json()
    session_id = data.get("session_id")
    prompt = data.get("prompt") or "Quiz me on the uploaded material."
    context = retrieve_full_session_context(session_id)
    if not context:
        return {"quiz": [], "response": "Upload study files first so I can build a quiz from your material.", "sources": []}
    chat_result = qwen_llm.chat_with_usage(prompt, context=context, study_task="quiz")
    response = chat_result["response"]
    return {"quiz": response, "response": response, "usage": chat_result["usage"], "sources": [public_source(source) for source in context]}

@app.post("/summarize/")
async def summarize(request: Request):
    """Summarize uploaded content or specific sections."""
    data = await request.json()
    session_id = data.get("session_id")
    prompt = data.get("prompt") or "Summarize the uploaded material."
    top_k = int(data.get("top_k", DEFAULT_RAG_TOP_K))
    context, _study_task = retrieve_study_context(session_id, prompt, top_k)
    if not context:
        return {"summary": "Upload study files first so I can summarize your material.", "sources": []}
    chat_result = qwen_llm.chat_with_usage(prompt, context=context, study_task="summary")
    response = chat_result["response"]
    return {"summary": response, "response": response, "usage": chat_result["usage"], "sources": [public_source(source) for source in context]}

@app.post("/video-link/")
async def video_link(request: Request):
    """Find and link to specific times in videos."""
    data = await request.json()
    session_id = data.get("session_id")
    prompt = data.get("prompt") or data.get("query") or ""
    context = retrieve_full_session_context(session_id)
    if not context:
        return {"links": [], "response": "I could not find that topic in the uploaded material.", "sources": []}
    chat_result = qwen_llm.chat_with_usage(prompt, context=context, study_task="find")
    response = chat_result["response"]
    return {"links": [public_source(source) for source in context], "response": response, "usage": chat_result["usage"], "sources": [public_source(source) for source in context]}
