# Add missing import for os
import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from contextlib import asynccontextmanager

import uuid
import zipfile
import io
import json
import xml.etree.ElementTree as ET
import re

from qwen_utils import QwenLLM

# In-memory session store (replace with Redis for production)
sessions = {}

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

@app.get("/llm/status")
async def llm_status():
    """Report the currently configured LLM mode and model."""
    return {
        "model": qwen_llm.model_path,
        "mock": qwen_llm.use_mock_model,
        "loaded": qwen_llm.model is not None,
        "device": str(qwen_llm.device),
        "max_new_tokens": qwen_llm.max_new_tokens,
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
    if ext == "txt":
        return parse_txt(content)
    elif ext == "json":
        return parse_json(content)
    elif ext == "dfxp":
        return parse_dfxp(content)
    elif ext == "srt":
        return parse_srt(content)
    elif ext == "vtt":
        return parse_vtt(content)
    else:
        return content.decode("utf-8", errors="ignore")

@app.post("/upload/")
async def upload_content(session_id: Optional[str] = Form(None), files: List[UploadFile] = File(...)):
    """Bulk upload transcripts, OCR, and descriptions. Accepts txt, json, dfxp, srt, vtt, or zip of these."""
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {"content": []}
    for file in files:
        fname = file.filename
        content = await file.read()
        if fname.lower().endswith(".zip"):
            extracted = extract_files_from_zip(content)
            for f in extracted:
                text = parse_file(f["filename"], f["content"])
                sessions[session_id]["content"].append({"filename": f["filename"], "text": text})
        else:
            text = parse_file(fname, content)
            sessions[session_id]["content"].append({"filename": fname, "text": text})
    return {"session_id": session_id, "files": [f.filename for f in files]}


@app.post("/chat/")
async def chat(request: Request):
    """Chat endpoint for LLM interaction (Qwen)."""
    data = await request.json()
    prompt = data.get("prompt", "")
    session_id = data.get("session_id")
    use_internet = data.get("use_internet", False)
    image = data.get("image")  # Expecting base64 or bytes (optional)
    context = None
    if session_id and session_id in sessions:
        context = [item["text"][:2500] for item in sessions[session_id]["content"][-3:]]
    # TODO: handle image input (decode if base64)
    response = qwen_llm.chat(prompt, image=image, context=context, use_internet=use_internet)
    return {"response": response, "used_internet": use_internet}

@app.post("/search/")
async def search(request: Request):
    """Search within uploaded content."""
    data = await request.json()
    # Placeholder: Implement search logic
    return {"results": []}

@app.post("/quiz/")
async def quiz(request: Request):
    """Generate quiz from uploaded content."""
    data = await request.json()
    # Placeholder: Implement quiz logic
    return {"quiz": []}

@app.post("/summarize/")
async def summarize(request: Request):
    """Summarize uploaded content or specific sections."""
    data = await request.json()
    # Placeholder: Implement summarization logic
    return {"summary": "[Summary placeholder]"}

@app.post("/video-link/")
async def video_link(request: Request):
    """Find and link to specific times in videos."""
    data = await request.json()
    # Placeholder: Implement video time-linking logic
    return {"links": []}
