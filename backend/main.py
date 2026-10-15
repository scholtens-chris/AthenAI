import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import Counter
from pathlib import Path

import uuid
import zipfile
import io
import json
import xml.etree.ElementTree as ET
import re
import time

from qwen_utils import QwenLLM, detect_prompt_injection
from moderation_utils import ModerationDecision, assess_prompt
from security_utils import (
    SESSION_TOKEN_HEADER,
    bounded_int,
    hash_session_token,
    mediasite_host_allowed,
    new_session_token,
    redact_path,
    same_origin,
    sanitize_model_output,
    session_token_matches,
    valid_session_id,
)
from mediasite_utils import (
    MediasiteApiError,
    MediasiteClient,
    extract_presentation_ids,
    extract_presentation_text,
    flatten_odata_collection,
    importable_caption_text,
    importable_ocr_text,
    importable_slide_details_text,
    odata_next_link,
    odata_path,
    presentation_filename,
)
from transcript_utils import (
    build_transcript_units,
    chunk_transcript_units,
    cited_source_label,
    format_timestamp,
    parse_timed_segments,
    render_citations,
    session_has_multiple_titles,
    source_preview,
    title_from_filename,
)


def configure_logging() -> logging.Logger:
    log_level = os.getenv("ATHENAI_LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=log_level, format=log_format)
    else:
        root_logger.setLevel(log_level)

    log_file = os.getenv("ATHENAI_LOG_FILE")
    if log_file and not any(
        isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == str(Path(log_file).resolve())
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    return logging.getLogger("athenai.api")


logger = configure_logging()

# In-memory session store (replace with Redis for production)
sessions = {}

CHUNK_WORDS = int(os.getenv("ATHENAI_RAG_CHUNK_WORDS", "220"))
CHUNK_OVERLAP = int(os.getenv("ATHENAI_RAG_CHUNK_OVERLAP", "45"))
DEFAULT_RAG_TOP_K = int(os.getenv("ATHENAI_RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("ATHENAI_RAG_MAX_CONTEXT_CHARS", "8000"))
OVERVIEW_CHUNK_LIMIT = int(os.getenv("ATHENAI_RAG_OVERVIEW_CHUNKS", "8"))
FULL_ARTIFACT_CONTEXT_CHARS = int(os.getenv("ATHENAI_RAG_FULL_ARTIFACT_CONTEXT_CHARS", "24000"))
CHAT_HISTORY_EXCHANGES = int(os.getenv("ATHENAI_CHAT_HISTORY_EXCHANGES", "2"))
CHAT_HISTORY_MESSAGE_CHARS = int(os.getenv("ATHENAI_CHAT_HISTORY_MESSAGE_CHARS", "1500"))
# The latest answer is what follow-ups ask about most, and a quiz's answer key sits at its end.
CHAT_HISTORY_LATEST_ANSWER_CHARS = int(os.getenv("ATHENAI_CHAT_HISTORY_LATEST_ANSWER_CHARS", "4000"))
SESSION_TTL_SECONDS = int(float(os.getenv("ATHENAI_SESSION_TTL_HOURS", "12")) * 60 * 60)
SESSION_STORE_DIR = Path(os.getenv("ATHENAI_SESSION_STORE_DIR", Path(__file__).parent / ".athenai_sessions"))
SESSION_PERSISTENCE_ENABLED = os.getenv("ATHENAI_SESSION_PERSISTENCE", "1") != "0"
MEDIASITE_API_KEY = os.getenv("ATHENAI_MEDIASITE_API_KEY")
MEDIASITE_BASE_URL = os.getenv("ATHENAI_MEDIASITE_BASE_URL")
MEDIASITE_USERNAME = os.getenv("ATHENAI_MEDIASITE_USERNAME")
MEDIASITE_PASSWORD = os.getenv("ATHENAI_MEDIASITE_PASSWORD")
MEDIASITE_APPLICATION_TICKET = os.getenv("ATHENAI_MEDIASITE_APPLICATION_TICKET")
MEDIASITE_AUTHORIZATION_TICKET = os.getenv("ATHENAI_MEDIASITE_AUTHORIZATION_TICKET")
# Extra hosts a request may name as base_url. Server credentials are still only sent to the configured base URL.
MEDIASITE_ALLOWED_HOSTS = [host for host in os.getenv("ATHENAI_MEDIASITE_ALLOWED_HOSTS", "").split(",") if host.strip()]
# Each session is owned by whoever created it; requests must present its owner token (see security_utils).
SESSION_TOKENS_REQUIRED = os.getenv("ATHENAI_REQUIRE_SESSION_TOKEN", "1") != "0"
CORS_ORIGINS = [origin.strip() for origin in os.getenv("ATHENAI_CORS_ORIGINS", "*").split(",") if origin.strip()]
MAX_PROMPT_CHARS = int(os.getenv("ATHENAI_MAX_PROMPT_CHARS", "12000"))
MAX_TOP_K = int(os.getenv("ATHENAI_MAX_TOP_K", "20"))
MAX_UPLOAD_BYTES = int(os.getenv("ATHENAI_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_ZIP_ENTRIES = int(os.getenv("ATHENAI_MAX_ZIP_ENTRIES", "500"))
MAX_ZIP_UNCOMPRESSED_BYTES = int(os.getenv("ATHENAI_MAX_ZIP_UNCOMPRESSED_BYTES", str(200 * 1024 * 1024)))
MEDIASITE_CHANNEL_RESOURCE_TYPES = {
    "mediasitechannel": "MediasiteChannels",
    "mediasitechannels": "MediasiteChannels",
    "channel": "MediasiteChannels",
    "channels": "MediasiteChannels",
    "catalog": "MediasiteChannels",
    "catalogs": "MediasiteChannels",
    "showcasechannel": "ShowcaseChannels",
    "showcasechannels": "ShowcaseChannels",
    "playlist": "Playlists",
    "playlists": "Playlists",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
    "i", "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

# Checked in order; the first task with a matching pattern wins. Specific study artifacts come
# before the generic find/explain keywords so "make a study guide for when we covered X" is
# still a study guide.
STUDY_TASK_PATTERNS = {
    "flashcards": (
        r"\bflash[\s-]*cards?\b",
        r"\b(?:cue|index|study|revision|memory|review)\s+cards?\b",
        r"\banki\b",
    ),
    "quiz": (
        r"\bquiz(?:z?es|z?ed|z?ing)?\b",
        r"\btest me\b",
        r"\btest (?:my|our) (?:knowledge|understanding|memory|recall)\b",
        r"\b(?:practice|mock|sample|review|exam|test)\s+(?:questions?|tests?|exams?|problems?)\b",
        r"\bmultiple[\s-]*choice\b",
        r"\bmcqs?\b",
        r"\btrue[\s/-]*(?:or[\s-]*)?false\b",
        r"\bask me (?:some |a few |\d+ )?questions\b",
        r"\b(?:drill|question) me\b",
        r"\bcheck my understanding\b",
    ),
    "study_guide": (
        r"\bstudy[\s-]*guides?\b",
        r"\breview\s+guides?\b",
        r"\brevision\s+(?:guides?|notes|sheets?)\b",
        r"\b(?:study|review|cheat|crib)\s+sheets?\b",
        r"\bstudy\s+(?:notes|plan)\b",
        r"\blearning\s+objectives?\b",
        r"\bexam\s+prep\b",
        r"\bprep(?:are)? (?:me )?for (?:the |my |our |an? )?(?:exams?|tests?|midterms?|finals?|quiz(?:zes)?)\b",
        r"\bhelp me (?:study|revise|review)\b",
    ),
    "takeaways": (
        r"\b(?:key\s+)?take[\s-]?aways?\b",
        r"\b(?:main|key)\s+(?:points?|ideas?|concepts?)\b",
        r"\bbig ideas?\b",
        r"\bimportant (?:concepts?|points?|ideas?)\b",
        r"\bmost important (?:things?|parts?)\b",
        r"\bwhat should i (?:remember|know)\b",
    ),
    "summary": (
        r"\bsummar(?:y|ies|i[sz]e[sd]?|i[sz]ing)\b",
        r"\brecap\b",
        r"\boverview\b",
        r"\bsynopsis\b",
        r"\btl;?dr\b",
        r"\bgist\b",
        r"\bin a nutshell\b",
        r"\bchapter\s+\d+\b",
    ),
    "find": (
        r"\bfind\b",
        r"\blocate\b",
        r"\bwhere\b",
        r"\bwhen\b",
        r"\btimestamps?\b",
        r"\bwhat time\b",
        r"\bwhich (?:video|lecture|recording|part|section|slide)\b",
        r"\bexplained\b",
        r"\bmention(?:s|ed)?\b",
    ),
    "explain": (
        r"\bexplain(?:s|ing)?\b",
        r"\bexplanation\b",
        r"\bteach me\b",
        r"\bwalk me through\b",
        r"\bhelp me understand\b",
        r"\bbreak (?:it|this|that) down\b",
        r"\beli5\b",
        r"\blecture\b",
    ),
}

# Phrases that point back at AthenAI's previous reply ("explain that more simply", "turn it into
# flashcards", "what's the answer to question 3?"). A message that matches is a follow-up even when
# it names a study task of its own, and it is answered with the excerpts the previous reply cited.
PREVIOUS_REPLY_NOUNS = r"(?:answer|response|reply|explanation|summary|list|quiz|flash ?cards?|cards|study guide|take ?aways?)\b"
FOLLOW_UP_REFERENCE_PATTERNS = (
    r"\b(?:your|(?:that|this|the) (?:last|previous|earlier)) (?:last |previous |earlier )?" + PREVIOUS_REPLY_NOUNS,
    r"\b(?:that|this) " + PREVIOUS_REPLY_NOUNS,
    r"\b(?:above|previous|last|earlier) (?:one|answer|response|reply|points?|questions?)\b",
    r"(?:\b(?:question|point|step|item|card|number|option)|#) ?\d+\b",
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final) (?:one|point|question|step|item|card|option|bullet|part)\b",
    r"\bwhat (?:do|did) you mean\b",
    r"\bwhy (?:do|did) you (?:say|think|pick|choose)\b",
    r"\b(?:are you sure|is (?:that|this|it) (?:right|correct|true|accurate))\b",
    r"\bi (?:don'?t|do not|didn'?t|did not) (?:understand|get)\b",
    r"\b(?:simpler|more simply|shorter|more detail|less detail|elaborate|expand on|go deeper|clarify|rephrase|reword|simplify|shorten)\b",
    # A bare pronoun object: "explain that", "turn it into flashcards", "quiz me on those again".
    r"\b(?:that|this|it|those|these|them)(?= *(?:$|[?.!,;]|(?:again|into|in|as|more|for|from|to|with|up|down|please|simpler|shorter)\b))",
)

# Questions about AthenAI itself get a fixed reply without touching the uploaded material, so the
# answer never drags in unrelated sources. Each pattern must match the whole (normalized) prompt.
ASSISTANT_NAME_PATTERN = r"(?:athen ?ai|athena)"
ABOUT_ASSISTANT_PATTERNS = (
    r"what(?:'s| is) your name",
    r"what (?:should|do|can) i call you",
    r"(?:who|what) (?:are|r) (?:you|u)",
    r"who(?:'s| is) this",
    r"(?:who|what) is " + ASSISTANT_NAME_PATTERN,
    r"(?:tell me about|introduce|describe) yourself",
    r"what (?:can|do) you do",
    r"what are you (?:for|able to do|capable of)",
    r"how (?:can|do) you help(?: me)?",
    r"how do you work",
    r"who (?:made|built|created|developed|designed|programmed) (?:you|" + ASSISTANT_NAME_PATTERN + r")",
    r"are you (?:an? )?(?:ai|bot|robot|chatbot|human|real person|person)",
    r"what (?:ai |language )?model (?:are you|do you use|is this)",
)
ABOUT_ASSISTANT_PREFIX = r"(?:(?:hi|hello|hey|ok|okay|so)(?: there)?(?: " + ASSISTANT_NAME_PATTERN + r")?,? )?"
ABOUT_ASSISTANT_SUFFIX = r"(?:,? " + ASSISTANT_NAME_PATTERN + r")?(?:,? (?:exactly|again|anyway))?(?:,? please)?"
ABOUT_ASSISTANT_RESPONSE = (
    "I'm AthenAI, an AI study assistant. I answer questions using the study material you upload "
    "or import from Mediasite, and I cite the exact sources and video timestamps I draw from. "
    "I can summarize lectures, explain concepts, pull out key takeaways, build study guides, "
    "flashcards, and quizzes, and help you find where a topic was covered. "
    "I'm not a person and I can make mistakes, so check anything important against your material. "
    "Ask me about your material to get started."
)

# Greetings and good manners get a kind reply without retrieval. Matching runs on the raw prompt
# (case-insensitive) so a courtesy can be cut off the front of a real question before retrieval.
SOCIAL_SEPARATOR = r"[\W_]+"
SOCIAL_NAME = r"(?:" + SOCIAL_SEPARATOR + ASSISTANT_NAME_PATTERN + r")?"
SOCIAL_PHRASES = {
    "greeting": r"(?:hi|hello|hey|hiya|heya|howdy|yo|greetings|good (?:morning|afternoon|evening|day))(?: there| again| everyone| all)?",
    "how_are_you": r"(?:how(?: are|['’]re| r) (?:you|u|ya)(?: doing)?(?: today)?|how(?:['’]s| is) it going|what['’]?s up|sup)",
    "apology": r"(?:sorry(?: about that| for that)?|my bad|oops|apologies)",
    "thanks": (
        r"(?:(?:thanks?(?: you)?|thx|ty|tysm)(?: (?:so|very) much| a (?:lot|bunch|ton))?(?: again)?"
        r"(?: for (?:that|this|the help|your help|helping|everything))?"
        r"|(?:much|really) appreciated|(?:i )?(?:really )?appreciate (?:it|that|you|the help)|cheers)"
    ),
    "compliment": (
        r"(?:(?:that(?:['’]s| is| was)|you(?:['’]re| are)|this is) (?:so |very |really )?"
        r"(?:great|awesome|amazing|helpful|perfect|brilliant|wonderful|fantastic|the best)"
        r"|(?:good|great|nice) (?:job|work|answer)|well done|you rock|that help(?:s|ed))"
    ),
    "acknowledgement": r"(?:ok(?:ay)?|k|cool|great|awesome|nice|perfect|got it|gotcha|sounds good|understood|i see|makes sense|alright|all right|sure)",
    "please": r"(?:please|pls|plz)",
    "farewell": (
        r"(?:(?:good ?)?bye(?: now)?|bye bye|see (?:you|ya)(?: later| soon)?|later|good ?night|take care"
        r"|have a (?:good|great|nice) (?:day|night|one|evening))"
    ),
}
SOCIAL_REGEXES = {
    category: re.compile(r"(?:" + phrase + r")" + SOCIAL_NAME + r"(?=[\W_]|$)", re.IGNORECASE)
    for category, phrase in SOCIAL_PHRASES.items()
}
SOCIAL_SEPARATOR_REGEX = re.compile(SOCIAL_SEPARATOR)
# Only these can be cut from the front of a question without punctuation after them ("hey what is X").
UNAMBIGUOUS_LEADING_SOCIAL = {"greeting", "thanks"}
TRAILING_COURTESY_REGEX = re.compile(
    r"(?:" + SOCIAL_SEPARATOR + r"(?:please|pls|plz|thanks?(?: you)?(?: in advance)?|thx|ty))+[\W_]*$",
    re.IGNORECASE,
)
# Each reply is an opener plus a follow-up; a multi-part message gets every opener in this order and
# the follow-up of its last part ("hi, how are you?" -> greeting + how_are_you openers).
SOCIAL_REPLIES = {
    "greeting": (
        "Hi there!",
        "What would you like to study today? I can summarize, explain, quiz you, or find where a topic was covered in your material.",
    ),
    "how_are_you": ("I'm doing well, thank you for asking!", "What would you like to work on today?"),
    "apology": ("No need to apologize!", "What can I help you with?"),
    "thanks": ("You're very welcome!", "Let me know if there's anything else I can help with."),
    "compliment": ("Thank you, that's very kind of you!", "Let me know if there's anything else I can help with."),
    "acknowledgement": ("Glad to help!", "Let me know if there's anything else you'd like to go over."),
    "please": ("Of course!", "What would you like help with?"),
    "farewell": ("Goodbye, and good luck with your studies!", "Come back anytime."),
}
# Courtesies at the start of a real question that earn a kind opener before the answer.
SOCIAL_PREFIX_CATEGORIES = ("greeting", "how_are_you", "apology", "thanks", "compliment")

BROAD_STUDY_PROMPTS = {
    "quiz",
    "quiz me",
    "test me",
    "flashcards",
    "create flashcards",
    "make flashcards",
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


def session_now() -> int:
    return int(time.time())


def session_expires_at() -> int:
    return session_now() + max(60, SESSION_TTL_SECONDS)


def session_file_path(session_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
    return SESSION_STORE_DIR / f"{safe_id}.json"


def normalize_session(session: Optional[dict] = None) -> dict:
    session = session or {}
    session.setdefault("content", [])
    session.setdefault("chunks", [])
    session.setdefault("chat_history", [])
    session["updated_at"] = int(session.get("updated_at") or session_now())
    session["expires_at"] = int(session.get("expires_at") or session_expires_at())
    for chunk in session.get("chunks", []):
        if not isinstance(chunk.get("tokens"), Counter):
            chunk["tokens"] = Counter(chunk.get("tokens", {}))
    return session


def is_session_expired(session: dict) -> bool:
    return int(session.get("expires_at") or 0) <= session_now()


def touch_session(session_id: str) -> None:
    session = normalize_session(sessions.get(session_id))
    session["updated_at"] = session_now()
    session["expires_at"] = session_expires_at()
    sessions[session_id] = session


def persist_session(session_id: str) -> None:
    if not SESSION_PERSISTENCE_ENABLED:
        logger.debug("session_persist_skipped persistence_disabled session_id=%s", session_id)
        return
    session = sessions.get(session_id)
    if not session:
        logger.debug("session_persist_skipped empty_session session_id=%s", session_id)
        return
    SESSION_STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = session_file_path(session_id)
    temp_path = path.with_suffix(".tmp")
    payload = normalize_session(session).copy()
    try:
        temp_path.write_text(json.dumps(payload), encoding="utf-8")
        temp_path.replace(path)
        logger.debug(
            "session_persisted session_id=%s path=%s chunks=%s messages=%s expires_at=%s",
            session_id,
            path,
            len(payload.get("chunks", [])),
            len(payload.get("chat_history", [])),
            payload.get("expires_at"),
        )
    except OSError:
        logger.exception("session_persist_failed session_id=%s path=%s", session_id, path)


def load_session_from_disk(session_id: str) -> Optional[dict]:
    if not SESSION_PERSISTENCE_ENABLED:
        return None
    path = session_file_path(session_id)
    if not path.exists():
        return None
    try:
        session = normalize_session(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        logger.exception("session_load_failed session_id=%s path=%s", session_id, path)
        return None
    if is_session_expired(session):
        sessions.pop(session_id, None)
        try:
            path.unlink()
        except OSError:
            logger.exception("session_expired_delete_failed session_id=%s path=%s", session_id, path)
        logger.info("session_expired session_id=%s path=%s", session_id, path)
        return None
    sessions[session_id] = session
    logger.info(
        "session_loaded session_id=%s path=%s chunks=%s messages=%s",
        session_id,
        path,
        len(session.get("chunks", [])),
        len(session.get("chat_history", [])),
    )
    return session


def cleanup_expired_sessions() -> None:
    removed_memory = 0
    for session_id, session in list(sessions.items()):
        if is_session_expired(session):
            sessions.pop(session_id, None)
            removed_memory += 1
    if not SESSION_PERSISTENCE_ENABLED or not SESSION_STORE_DIR.exists():
        if removed_memory:
            logger.info("session_cleanup_complete removed_memory=%s removed_disk=0", removed_memory)
        return
    removed_disk = 0
    for path in SESSION_STORE_DIR.glob("*.json"):
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("session_cleanup_read_failed path=%s", path)
            continue
        if is_session_expired(session):
            try:
                path.unlink()
                removed_disk += 1
            except OSError:
                logger.exception("session_cleanup_delete_failed path=%s", path)
    if removed_memory or removed_disk:
        logger.info("session_cleanup_complete removed_memory=%s removed_disk=%s", removed_memory, removed_disk)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "startup_begin mock_llm=%s model=%s persistence=%s session_store=%s ttl_seconds=%s",
        qwen_llm.use_mock_model,
        qwen_llm.model_path,
        SESSION_PERSISTENCE_ENABLED,
        SESSION_STORE_DIR,
        SESSION_TTL_SECONDS,
    )
    cleanup_expired_sessions()
    preload_model = os.getenv("ATHENAI_PRELOAD_MODEL", "0") == "1"
    if preload_model and not qwen_llm.use_mock_model:
        logger.info("model_preload_begin model=%s device=%s", qwen_llm.model_path, qwen_llm.device)
        qwen_llm._load_model()
        logger.info("model_preload_complete model=%s device=%s", qwen_llm.model_path, qwen_llm.device)
    yield
    logger.info("shutdown_complete")

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    log_path = redact_path(request.url.path)
    logger.info("request_begin request_id=%s method=%s path=%s", request_id, request.method, log_path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request_failed request_id=%s method=%s path=%s elapsed_ms=%s",
            request_id,
            request.method,
            log_path,
            elapsed_ms,
        )
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s elapsed_ms=%s",
        request_id,
        request.method,
        log_path,
        response.status_code,
        elapsed_ms,
    )
    return response

# Enable CORS for frontend integration
# Sessions are authorized with a header token, not cookies, so credentialed CORS is unnecessary.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")

def ensure_session(session_id: str):
    if session_id not in sessions:
        load_session_from_disk(session_id)
    if session_id not in sessions:
        sessions[session_id] = normalize_session()
    sessions[session_id] = normalize_session(sessions[session_id])
    if is_session_expired(sessions[session_id]):
        sessions[session_id] = normalize_session()
    return sessions[session_id]

def authorize_session(request: Request, session_id: Optional[str], create: bool = True) -> tuple[Optional[str], Optional[str]]:
    """Resolve the caller's session and enforce ownership; return (session_id, newly issued owner token or None).

    A new session (or one saved before owner tokens existed) is claimed by the caller, who receives a token that
    must accompany every later request for that session. Enforced here in code, independent of the model.
    """
    if not session_id:
        if not create:
            return None, None
        session_id = str(uuid.uuid4())
    elif not isinstance(session_id, str) or not valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id.")
    session = ensure_session(session_id)
    if not SESSION_TOKENS_REQUIRED:
        return session_id, None
    stored_hash = session.get("owner_token_hash")
    if stored_hash:
        if not session_token_matches(stored_hash, request.headers.get(SESSION_TOKEN_HEADER)):
            logger.warning("session_access_denied path=%s", redact_path(request.url.path))
            raise HTTPException(status_code=403, detail="This session is not available. Start a new session.")
        return session_id, None
    token = new_session_token()
    session["owner_token_hash"] = hash_session_token(token)
    if session.get("content") or session.get("chat_history"):
        persist_session(session_id)
    return session_id, token


def with_session_token(payload: dict, token: Optional[str]) -> dict:
    if token:
        payload["session_token"] = token
    return payload


def request_prompt(data: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if value:
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"{key} must be text.")
            if len(value) > MAX_PROMPT_CHARS:
                raise HTTPException(
                    status_code=413,
                    detail=f"That message is too long ({len(value):,} characters). Please keep it under {MAX_PROMPT_CHARS:,}.",
                )
            return value
    return default


def request_int(data: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return bounded_int(data.get(key), default, minimum, maximum, key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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

def add_document_to_session(session_id: str, filename: str, text: str, segments: Optional[list[dict]] = None) -> int:
    session = ensure_session(session_id)
    document_id = str(uuid.uuid4())
    title = title_from_filename(filename)
    if segments:
        # Timed transcripts are chunked on passage boundaries so citations can point to a time.
        chunks = chunk_transcript_units(build_transcript_units(segments), CHUNK_WORDS)
        cleaned_text = "\n".join(chunk["text"] for chunk in chunks)
    else:
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        chunks = [{"text": chunk, "start": None, "end": None} for chunk in chunk_text(cleaned_text)]
    if not chunks:
        logger.warning("document_skipped_no_text session_id=%s filename=%s", session_id, filename)
        return 0

    session["content"].append({
        "id": document_id,
        "filename": filename,
        "title": title,
        "timed": bool(segments),
        "text": cleaned_text,
    })

    for index, chunk in enumerate(chunks):
        tokens = tokenize(chunk["text"])
        session["chunks"].append({
            "id": f"{document_id}:{index}",
            "document_id": document_id,
            "filename": filename,
            "title": title,
            "chunk_index": index,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "tokens": Counter(tokens),
            "token_count": max(1, len(tokens)),
        })
    touch_session(session_id)
    persist_session(session_id)
    logger.info(
        "document_indexed session_id=%s filename=%s chars=%s chunks_added=%s session_chunks=%s",
        session_id,
        filename,
        len(cleaned_text),
        len(chunks),
        len(session["chunks"]),
    )
    return len(chunks)

def retrieve_context(session_id: Optional[str], query: str, top_k: int = DEFAULT_RAG_TOP_K) -> list[dict]:
    if not session_id:
        logger.debug("retrieve_context_skipped no_session query_chars=%s", len(query or ""))
        return []
    session = ensure_session(session_id)
    if not session.get("chunks"):
        logger.info("retrieve_context_empty_session session_id=%s query_chars=%s", session_id, len(query or ""))
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        logger.info("retrieve_context_empty_query session_id=%s query_chars=%s", session_id, len(query or ""))
        return []

    query_counts = Counter(query_tokens)
    scored = []
    for chunk in session.get("chunks", []):
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
        results.append(material_context_entry(chunk, round(score, 4), text))
        if used_chars >= MAX_CONTEXT_CHARS:
            break
    logger.info(
        "retrieve_context_complete session_id=%s query_tokens=%s scored_chunks=%s returned=%s used_chars=%s top_k=%s",
        session_id,
        len(query_tokens),
        len(scored),
        len(results),
        used_chars,
        top_k,
    )
    return results

def is_about_assistant_prompt(prompt: str) -> bool:
    normalized = re.sub(r"\s+", " ", (prompt or "").lower().replace("’", "'")).strip(" ?.!")
    return any(
        re.fullmatch(ABOUT_ASSISTANT_PREFIX + pattern + ABOUT_ASSISTANT_SUFFIX, normalized)
        for pattern in ABOUT_ASSISTANT_PATTERNS
    )

def scan_leading_social(prompt: str, mixed: bool) -> tuple[list[str], int]:
    """Consume courtesy phrases from the start of the prompt; return their categories and the end offset.

    With mixed=True the phrases precede a real question, so ambiguous words ("ok", "later") only count
    when punctuation follows them, and farewells are never taken.
    """
    categories: list[str] = []
    separator = SOCIAL_SEPARATOR_REGEX.match(prompt)
    pos = separator.end() if separator else 0
    while pos < len(prompt):
        best_category, best_end = None, pos
        for category, regex in SOCIAL_REGEXES.items():
            if mixed and category == "farewell":
                continue
            match = regex.match(prompt, pos)
            if match and match.end() > best_end:
                best_category, best_end = category, match.end()
        if best_category is None:
            break
        separator = SOCIAL_SEPARATOR_REGEX.match(prompt, best_end)
        if mixed and best_category not in UNAMBIGUOUS_LEADING_SOCIAL:
            if separator is None or not separator.group().strip():
                break
        categories.append(best_category)
        pos = separator.end() if separator else best_end
    return categories, pos

def split_social_prompt(prompt: str) -> tuple[list[str], str]:
    """Split courtesies off a prompt: (leading courtesy categories, remaining question).

    An empty question means the whole message was social, e.g. "hi!" or "thanks, bye".
    """
    prompt = (prompt or "").strip()
    categories, end = scan_leading_social(prompt, mixed=False)
    if categories and end >= len(prompt):
        return categories, ""
    categories, end = scan_leading_social(prompt, mixed=True)
    question = TRAILING_COURTESY_REGEX.sub("", prompt[end:]).strip()
    return categories, question or prompt[end:].strip()

def social_reply(categories: list[str]) -> str:
    present = [category for category in SOCIAL_REPLIES if category in categories]
    openers = " ".join(SOCIAL_REPLIES[category][0] for category in present)
    return f"{openers} {SOCIAL_REPLIES[present[-1]][1]}"

def social_prefix(categories: list[str]) -> str:
    return " ".join(
        SOCIAL_REPLIES[category][0] for category in SOCIAL_PREFIX_CATEGORIES if category in categories
    )

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
    full_material_terms = {
        "all",
        "everything",
        "entire",
        "full",
        "uploaded",
        "material",
        "materials",
        "notes",
        "lecture",
        "lectures",
        "document",
        "documents",
        "file",
        "files",
    }
    if set(content_tokens) & full_material_terms:
        return True
    if task in {"quiz", "takeaways", "study_guide"} and len(content_tokens) <= 3:
        return True
    return False

def retrieve_session_overview(session_id: Optional[str], limit: int = OVERVIEW_CHUNK_LIMIT) -> list[dict]:
    if not session_id:
        return []

    chunks = ensure_session(session_id).get("chunks", [])
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
        results.append(material_context_entry(chunk, 0, text))
        if used_chars >= MAX_CONTEXT_CHARS:
            break
    return results

def material_context_entry(chunk: dict, score: float, text: str) -> dict:
    return {
        "id": chunk["id"],
        "filename": chunk["filename"],
        "title": chunk.get("title") or title_from_filename(chunk["filename"]),
        "chunk_index": chunk["chunk_index"],
        "start": chunk.get("start"),
        "end": chunk.get("end"),
        "score": score,
        "text": text,
    }

def retrieve_full_session_context(session_id: Optional[str], max_chars: int = FULL_ARTIFACT_CONTEXT_CHARS) -> list[dict]:
    if not session_id:
        logger.debug("retrieve_full_context_skipped no_session")
        return []

    chunks = ensure_session(session_id).get("chunks", [])
    if not chunks:
        logger.info("retrieve_full_context_empty session_id=%s", session_id)
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
    logger.info(
        "retrieve_full_context_complete session_id=%s source_chunks=%s returned=%s used_chars=%s max_chars=%s",
        session_id,
        len(chunks),
        len(results),
        used_chars,
        max_chars,
    )
    return results

def retrieve_study_context(session_id: Optional[str], prompt: str, top_k: int) -> tuple[list[dict], str]:
    task = infer_study_task(prompt)
    if is_broad_study_prompt(prompt, task):
        context = retrieve_full_session_context(session_id)
        logger.info(
            "retrieve_study_context_full session_id=%s task=%s prompt_chars=%s context_chunks=%s max_chars=%s",
            session_id,
            task,
            len(prompt or ""),
            len(context),
            FULL_ARTIFACT_CONTEXT_CHARS,
        )
        return context, task

    context = retrieve_context(session_id, prompt, top_k)
    logger.info(
        "retrieve_study_context_retrieved session_id=%s task=%s prompt_chars=%s context_chunks=%s top_k=%s",
        session_id,
        task,
        len(prompt or ""),
        len(context),
        top_k,
    )
    return context, task

def recent_exchanges(session_id: Optional[str], limit: int = CHAT_HISTORY_EXCHANGES) -> list[tuple[dict, dict]]:
    """The latest (question, answer) pairs answered from the material, oldest first.

    Courtesy replies and "no material" notices carry no sources and give a follow-up nothing to refer back to.
    """
    if not session_id or limit <= 0:
        return []
    history = ensure_session(session_id).get("chat_history", [])
    exchanges = [
        (question, answer)
        for question, answer in zip(history, history[1:])
        if question.get("sender") == "user" and answer.get("sender") == "ai" and answer.get("sources")
    ]
    return exchanges[-limit:]

def clip_turn(text: str, max_chars: int = CHAT_HISTORY_MESSAGE_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."

def conversation_turns(exchanges: list[tuple[dict, dict]]) -> list[dict]:
    turns = []
    for position, (question, answer) in enumerate(exchanges, start=1):
        answer_chars = CHAT_HISTORY_LATEST_ANSWER_CHARS if position == len(exchanges) else CHAT_HISTORY_MESSAGE_CHARS
        turns.append({"role": "user", "content": clip_turn(split_social_prompt(question.get("text", ""))[1])})
        turns.append({"role": "assistant", "content": clip_turn(answer.get("text", ""), answer_chars)})
    return turns

def refers_to_previous_answer(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").lower().replace("’", "'")).strip()
    return any(re.search(pattern, normalized) for pattern in FOLLOW_UP_REFERENCE_PATTERNS)

def follow_up_retrieval_query(question: str, exchanges: list[tuple[dict, dict]]) -> str:
    """Retrieve a follow-up together with the question it follows.

    A message with no study task of its own ("that's wrong, note taking is on the roadmap"), or one that
    points back at the last answer ("explain that more simply"), continues that answer, and its words
    alone rarely match the material that answer came from.
    """
    if not exchanges or (infer_study_task(question) != "answer" and not refers_to_previous_answer(question)):
        return question
    previous = split_social_prompt(exchanges[-1][0].get("text", ""))[1]
    return f"{previous}\n{question}" if previous else question

def follow_up_context(session_id: Optional[str], answer: dict, retrieved: list[dict]) -> list[dict]:
    """The excerpts the previous answer cited, then freshly retrieved ones, within the context budget.

    A question about an answer ("is that right?", "explain point 2") needs the evidence that answer
    was built from, even when its own words match none of it.
    """
    if not session_id:
        return retrieved
    chunks = {chunk["id"]: chunk for chunk in ensure_session(session_id).get("chunks", [])}
    cited = [
        material_context_entry(chunks[source["id"]], source.get("score", 0), chunks[source["id"]]["text"])
        for source in answer.get("sources", [])
        if source.get("id") in chunks
    ]
    results, seen, used_chars = [], set(), 0
    for entry in [*cited, *retrieved]:
        if entry["id"] in seen:
            continue
        text = entry["text"][:max(0, MAX_CONTEXT_CHARS - used_chars)].rstrip()
        if not text:
            break
        seen.add(entry["id"])
        used_chars += len(text)
        results.append({**entry, "text": text})
    return results

def public_source(source: dict, cited_times: Optional[list[float]] = None) -> dict:
    return {
        "id": source["id"],
        "filename": source["filename"],
        "title": source.get("title") or title_from_filename(source["filename"]),
        "chunk_index": source["chunk_index"],
        "start": source.get("start"),
        "end": source.get("end"),
        "start_label": format_timestamp(source.get("start")),
        "cited_times": cited_times or [],
        "label": cited_source_label(source, cited_times),
        "score": source["score"],
        "preview": source_preview(source, cited_times),
    }


def present_response(session_id: Optional[str], response: str, context: list[dict], prompt: str = "") -> tuple[str, list[dict]]:
    """Turn the model's numbered citations into video names and timestamps.

    Returns the rendered text and the sources the answer actually cited, in citation order. If the
    model cited nothing, every retrieved source is returned so the answer is still traceable. Images and
    links to URLs that neither the student nor the sources mentioned are removed first.
    """
    documents = ensure_session(session_id).get("content", []) if session_id else []
    # URLs in sources that carry injection-like text are not trusted; an attacker can plant a URL there.
    trusted_sources = [source.get("text", "") for source in context if not detect_prompt_injection(source.get("text", ""))]
    response = sanitize_model_output(response, [prompt, *trusted_sources])
    cited: dict[int, list[float]] = {}
    text = render_citations(response, context, include_title=session_has_multiple_titles(documents), cited=cited)
    if not cited:
        return text, [public_source(source) for source in context]
    return text, [public_source(context[index], times) for index, times in cited.items()]


def public_chat_message(message: dict) -> dict:
    return {
        key: value
        for key, value in message.items()
        if key in {"sender", "text", "sources", "tokenUsage", "examples"}
    }


def append_chat_message(session_id: Optional[str], message: dict) -> None:
    if not session_id:
        return
    session = ensure_session(session_id)
    session.setdefault("chat_history", []).append(message)
    touch_session(session_id)
    persist_session(session_id)


def mediasite_client_from_payload(data: dict) -> MediasiteClient:
    """Build a Mediasite client, never sending the server's credentials anywhere but the configured origin."""
    requested_base_url = data.get("base_url")
    use_server_credentials = True
    if requested_base_url and requested_base_url.rstrip("/") != (MEDIASITE_BASE_URL or "").rstrip("/"):
        if not mediasite_host_allowed(requested_base_url, MEDIASITE_BASE_URL, MEDIASITE_ALLOWED_HOSTS):
            raise ValueError(
                "base_url must be an https URL on the configured Mediasite host or a host listed in "
                "ATHENAI_MEDIASITE_ALLOWED_HOSTS."
            )
        use_server_credentials = bool(MEDIASITE_BASE_URL) and same_origin(requested_base_url, MEDIASITE_BASE_URL)

    def server(value: Optional[str]) -> Optional[str]:
        return value if use_server_credentials else None

    impersonate_username = data.get("impersonate_username")
    if impersonate_username and not (data.get("username") and data.get("password")):
        # Impersonating with the server's service account would let any caller read as any Mediasite user.
        raise ValueError("impersonate_username requires a username and password supplied with the request.")
    return MediasiteClient(
        base_url=requested_base_url or MEDIASITE_BASE_URL or "",
        api_key=data.get("api_key") or server(MEDIASITE_API_KEY) or "",
        username=data.get("username") or server(MEDIASITE_USERNAME),
        password=data.get("password") or server(MEDIASITE_PASSWORD),
        impersonate_username=impersonate_username,
        authorization_ticket=data.get("authorization_ticket") or server(MEDIASITE_AUTHORIZATION_TICKET),
        application_ticket=data.get("application_ticket") or server(MEDIASITE_APPLICATION_TICKET),
        timeout_seconds=bounded_int(data.get("timeout_seconds"), 20, 1, 60, "timeout_seconds"),
    )


def public_mediasite_config() -> dict:
    return {
        "configured": bool(MEDIASITE_BASE_URL and MEDIASITE_API_KEY),
        "base_url": MEDIASITE_BASE_URL,
        "has_api_key": bool(MEDIASITE_API_KEY),
        "has_username": bool(MEDIASITE_USERNAME),
        "has_password": bool(MEDIASITE_PASSWORD),
        "has_application_ticket": bool(MEDIASITE_APPLICATION_TICKET),
        "has_authorization_ticket": bool(MEDIASITE_AUTHORIZATION_TICKET),
    }


def mediasite_error_detail(exc: MediasiteApiError) -> dict:
    message = str(exc)
    if exc.status_code == 403 and "AuthFailureAsError" in exc.body:
        message = (
            "Mediasite rejected the configured credentials with 403 AuthFailureAsError. "
            "The API key reached the Mediasite API, but it is not authorized to read this resource. "
            "The Mediasite REST API demo sends sfapikey plus Basic username/password auth; "
            "configure ATHENAI_MEDIASITE_USERNAME and ATHENAI_MEDIASITE_PASSWORD, or provide an "
            "application/authorization ticket with access to the presentation."
        )
    elif exc.status_code == 400 and "InvalidKey" in exc.body:
        message = (
            "Mediasite rejected the ID as an invalid OData key. Channel imports require a real "
            "Mediasite channel/catalog GUID, not a friendly channel URL slug such as "
            "'mediasiteadmin-mediasiteadmin'. For one video, paste a watch/play URL or use "
            "presentation:<presentation-guid>."
        )
    return {"message": message, "body": exc.body[:500]}


def mediasite_channel_resource(resource_type: Optional[str]) -> str:
    normalized = re.sub(r"[^a-zA-Z]", "", resource_type or "MediasiteChannels").lower()
    resource = MEDIASITE_CHANNEL_RESOURCE_TYPES.get(normalized)
    if not resource:
        supported = sorted(set(MEDIASITE_CHANNEL_RESOURCE_TYPES.values()))
        raise ValueError(f"Unsupported channel resource_type. Use one of: {', '.join(supported)}.")
    return resource


def fetch_mediasite_collection(client: MediasiteClient, path: str, params: Optional[dict] = None, max_pages: int = 20) -> list[dict]:
    items: list[dict] = []
    next_path: Optional[str] = path
    next_params = params
    pages = 0
    while next_path and pages < max(1, max_pages):
        data = client.get_json(next_path, params=next_params)
        items.extend(item for item in flatten_odata_collection(data) if isinstance(item, dict))
        next_path = odata_next_link(data)
        next_params = None
        pages += 1
    return items


def fetch_channel_presentation_ids(
    client: MediasiteClient,
    channel_id: str,
    resource_type: Optional[str] = None,
    max_pages: int = 20,
) -> list[str]:
    resource = mediasite_channel_resource(resource_type)
    presentations = fetch_mediasite_collection(
        client,
        odata_path(resource, channel_id, "Presentations"),
        params={"$select": "Id,Title,PresentationId,RootId"},
        max_pages=max_pages,
    )
    ids: list[str] = []
    seen: set[str] = set()
    for presentation_id in extract_presentation_ids(presentations):
        if presentation_id not in seen:
            ids.append(presentation_id)
            seen.add(presentation_id)
    return ids


def import_mediasite_presentation_text(client: MediasiteClient, session_id: str, presentation_id: str, filename: Optional[str] = None) -> dict:
    presentation = client.get_json(odata_path("Presentations", presentation_id), params={"$select": "full"})
    ocr_text = importable_ocr_text(client, presentation_id)
    caption_text = importable_caption_text(client, presentation_id)
    slide_details_text = importable_slide_details_text(client, presentation_id)

    text_parts = [
        extract_presentation_text(presentation),
        ocr_text,
        caption_text,
        slide_details_text,
    ]
    text = re.sub(r"\s+", " ", " ".join(part for part in text_parts if part)).strip()
    import_filename = filename or presentation_filename(presentation_id, presentation)
    chunks_added = add_document_to_session(session_id, import_filename, text)
    return {
        "presentation_id": presentation_id,
        "filename": import_filename,
        "chunks_added": chunks_added,
        "has_text": bool(text),
        "has_caption_text": bool(caption_text),
        "has_ocr_text": bool(ocr_text),
        "has_slide_details_text": bool(slide_details_text),
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


@app.get("/mediasite/status")
async def mediasite_status():
    """Report whether Mediasite API connection settings are available."""
    return public_mediasite_config()


@app.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    """Restore short-lived chat/session state after a UI or API restart."""
    session_id, token = authorize_session(request, session_id)
    session = ensure_session(session_id)
    return with_session_token({
        "session_id": session_id,
        "expires_at": session["expires_at"],
        "chat_history": [public_chat_message(message) for message in session.get("chat_history", [])],
        "documents": [
            {"id": item["id"], "filename": item["filename"], "title": item.get("title") or title_from_filename(item["filename"])}
            for item in session.get("content", [])
        ],
        "chunk_count": len(session.get("chunks", [])),
    }, token)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Erase a session's uploads, chat history, and saved snapshot now instead of waiting for it to expire."""
    session_id, _token = authorize_session(request, session_id)
    sessions.pop(session_id, None)
    if SESSION_PERSISTENCE_ENABLED:
        try:
            session_file_path(session_id).unlink(missing_ok=True)
        except OSError:
            logger.exception("session_delete_failed")
            raise HTTPException(status_code=500, detail="Could not delete the saved session.")
    logger.info("session_deleted")
    return {"deleted": True}


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
        logger.exception("parse_json_failed bytes=%s", len(content or b""))
        return "[Invalid JSON]"

def parse_dfxp(content: bytes) -> str:
    try:
        root = ET.fromstring(content.decode("utf-8", errors="ignore"))
        texts = [elem.text for elem in root.iter() if elem.text]
        return " ".join(texts)
    except Exception:
        logger.exception("parse_dfxp_failed bytes=%s", len(content or b""))
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
        logger.exception("parse_pdf_failed bytes=%s", len(content or b""))
        return ""

def parse_docx(content: bytes) -> str:
    try:
        from docx import Document

        document = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception:
        logger.exception("parse_docx_failed bytes=%s", len(content or b""))
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
        logger.exception("parse_pptx_failed bytes=%s", len(content or b""))
        return ""

def extract_files_from_zip(zip_bytes: bytes):
    files = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        entries = [info for info in z.infolist() if not info.filename.endswith("/")]
        # Check declared sizes before decompressing so a small archive cannot expand into gigabytes.
        if len(entries) > MAX_ZIP_ENTRIES or sum(info.file_size for info in entries) > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise zipfile.BadZipFile("Archive exceeds the allowed number of files or uncompressed size.")
        for info in entries:
            files.append({"filename": info.filename, "content": z.read(info)})
    logger.info("zip_extracted files=%s bytes=%s", len(files), len(zip_bytes or b""))
    return files

def parse_file(filename: str, content: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    logger.debug("parse_file_begin filename=%s ext=%s bytes=%s", filename, ext, len(content or b""))
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
async def upload_content(request: Request, session_id: Optional[str] = Form(None), files: List[UploadFile] = File(...)):
    """Bulk upload transcripts, OCR, and descriptions. Every upload is indexed for RAG."""
    session_id, session_token = authorize_session(request, session_id)
    logger.info("upload_begin session_id=%s file_count=%s", session_id, len(files))
    indexed_files = []
    skipped_files = []
    added_chunk_count = 0
    for file in files:
        fname = file.filename
        content = await file.read()
        logger.info("upload_file_read session_id=%s filename=%s bytes=%s", session_id, fname, len(content or b""))
        if len(content or b"") > MAX_UPLOAD_BYTES:
            logger.warning("upload_file_too_large session_id=%s filename=%s bytes=%s", session_id, fname, len(content))
            skipped_files.append(fname)
            continue
        if fname.lower().endswith(".zip"):
            try:
                extracted = extract_files_from_zip(content)
            except zipfile.BadZipFile:
                logger.exception("upload_zip_invalid session_id=%s filename=%s", session_id, fname)
                skipped_files.append(fname)
                continue
            for f in extracted:
                text = parse_file(f["filename"], f["content"])
                segments = parse_timed_segments(f["filename"], f["content"])
                chunks_added = add_document_to_session(session_id, f["filename"], text, segments)
                if chunks_added:
                    indexed_files.append(f["filename"])
                    added_chunk_count += chunks_added
                else:
                    skipped_files.append(f["filename"])
        else:
            text = parse_file(fname, content)
            segments = parse_timed_segments(fname, content)
            chunks_added = add_document_to_session(session_id, fname, text, segments)
            if chunks_added:
                indexed_files.append(fname)
                added_chunk_count += chunks_added
            else:
                skipped_files.append(fname)
    touch_session(session_id)
    persist_session(session_id)
    logger.info(
        "upload_complete session_id=%s uploaded_files=%s indexed=%s skipped=%s added_chunks=%s total_chunks=%s",
        session_id,
        len(files),
        len(indexed_files),
        len(skipped_files),
        added_chunk_count,
        len(sessions[session_id]["chunks"]),
    )
    return with_session_token({
        "session_id": session_id,
        "files": [f.filename for f in files],
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
        "added_chunk_count": added_chunk_count,
        "chunk_count": len(sessions[session_id]["chunks"]),
    }, session_token)


@app.post("/mediasite/import-presentation/")
async def import_mediasite_presentation(request: Request):
    """Fetch a Mediasite presentation's available text and index it for study."""
    data = await request.json()
    presentation_id = data.get("presentation_id")
    if not presentation_id:
        raise HTTPException(status_code=400, detail="presentation_id is required.")
    session_id, session_token = authorize_session(request, data.get("session_id"))
    try:
        client = mediasite_client_from_payload(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "mediasite_import_begin session_id=%s presentation_id=%s base_url=%s",
        session_id,
        presentation_id,
        client.base_url,
    )
    try:
        import_result = import_mediasite_presentation_text(
            client,
            session_id,
            presentation_id,
            filename=data.get("filename"),
        )
    except MediasiteApiError as exc:
        logger.warning(
            "mediasite_import_failed session_id=%s presentation_id=%s status=%s",
            session_id,
            presentation_id,
            exc.status_code,
        )
        raise HTTPException(
            status_code=exc.status_code or 502,
            detail=mediasite_error_detail(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not import_result["chunks_added"]:
        logger.warning("mediasite_import_no_text session_id=%s presentation_id=%s", session_id, presentation_id)
        raise HTTPException(
            status_code=422,
            detail=(
                "No importable text was found for that presentation. The API returned metadata, "
                "but no caption text, slide OCR text, or downloadable caption content."
            ),
        )

    logger.info(
        "mediasite_import_complete session_id=%s presentation_id=%s filename=%s chunks_added=%s total_chunks=%s",
        session_id,
        presentation_id,
        import_result["filename"],
        import_result["chunks_added"],
        len(sessions[session_id]["chunks"]),
    )
    return with_session_token({
        "session_id": session_id,
        "presentation_id": presentation_id,
        "filename": import_result["filename"],
        "indexed_files": [import_result["filename"]],
        "added_chunk_count": import_result["chunks_added"],
        "has_caption_text": import_result["has_caption_text"],
        "has_ocr_text": import_result["has_ocr_text"],
        "has_slide_details_text": import_result["has_slide_details_text"],
        "chunk_count": len(sessions[session_id]["chunks"]),
    }, session_token)


@app.post("/mediasite/channel-presentations/")
async def list_mediasite_channel_presentations(request: Request):
    """List presentation GUIDs from a Mediasite channel, catalog, showcase channel, or playlist."""
    data = await request.json()
    channel_id = data.get("channel_id") or data.get("catalog_id") or data.get("playlist_id")
    if not channel_id:
        raise HTTPException(status_code=400, detail="channel_id, catalog_id, or playlist_id is required.")
    try:
        client = mediasite_client_from_payload(data)
        presentation_ids = fetch_channel_presentation_ids(
            client,
            channel_id,
            resource_type=data.get("resource_type"),
            max_pages=bounded_int(data.get("max_pages"), 20, 1, 50, "max_pages"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MediasiteApiError as exc:
        logger.warning(
            "mediasite_channel_presentations_failed channel_id=%s status=%s",
            channel_id,
            exc.status_code,
        )
        raise HTTPException(
            status_code=exc.status_code or 502,
            detail=mediasite_error_detail(exc),
        ) from exc

    logger.info(
        "mediasite_channel_presentations_complete channel_id=%s resource_type=%s count=%s",
        channel_id,
        data.get("resource_type") or "MediasiteChannels",
        len(presentation_ids),
    )
    return {
        "channel_id": channel_id,
        "resource_type": mediasite_channel_resource(data.get("resource_type")),
        "presentation_ids": presentation_ids,
        "count": len(presentation_ids),
    }


@app.post("/mediasite/import-channel/")
async def import_mediasite_channel(request: Request):
    """Import captions and slide OCR for every presentation in a Mediasite channel/catalog."""
    data = await request.json()
    channel_id = data.get("channel_id") or data.get("catalog_id") or data.get("playlist_id")
    if not channel_id:
        raise HTTPException(status_code=400, detail="channel_id, catalog_id, or playlist_id is required.")
    session_id, session_token = authorize_session(request, data.get("session_id"))
    try:
        client = mediasite_client_from_payload(data)
        presentation_ids = data.get("presentation_ids") or fetch_channel_presentation_ids(
            client,
            channel_id,
            resource_type=data.get("resource_type"),
            max_pages=bounded_int(data.get("max_pages"), 20, 1, 50, "max_pages"),
        )
        if not isinstance(presentation_ids, list) or not all(isinstance(item, str) for item in presentation_ids):
            raise ValueError("presentation_ids must be a list of IDs.")
        max_presentations = bounded_int(data.get("max_presentations"), len(presentation_ids), 0, 200, "max_presentations")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MediasiteApiError as exc:
        raise HTTPException(
            status_code=exc.status_code or 502,
            detail=mediasite_error_detail(exc),
        ) from exc

    presentation_ids = presentation_ids[:max_presentations]
    imported = []
    skipped = []
    failed = []
    added_chunk_count = 0

    logger.info(
        "mediasite_channel_import_begin session_id=%s channel_id=%s presentation_count=%s",
        session_id,
        channel_id,
        len(presentation_ids),
    )
    for presentation_id in presentation_ids:
        try:
            result = import_mediasite_presentation_text(client, session_id, presentation_id)
        except MediasiteApiError as exc:
            failed.append({
                "presentation_id": presentation_id,
                "status": exc.status_code,
                "message": str(exc),
            })
            continue
        except ValueError as exc:
            failed.append({"presentation_id": presentation_id, "message": str(exc)})
            continue

        if result["chunks_added"]:
            imported.append(result)
            added_chunk_count += result["chunks_added"]
        else:
            skipped.append(result)

    logger.info(
        "mediasite_channel_import_complete session_id=%s channel_id=%s imported=%s skipped=%s failed=%s added_chunks=%s total_chunks=%s",
        session_id,
        channel_id,
        len(imported),
        len(skipped),
        len(failed),
        added_chunk_count,
        len(sessions[session_id]["chunks"]),
    )
    return with_session_token({
        "session_id": session_id,
        "channel_id": channel_id,
        "resource_type": mediasite_channel_resource(data.get("resource_type")),
        "presentation_ids": presentation_ids,
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "indexed_files": [item["filename"] for item in imported],
        "added_chunk_count": added_chunk_count,
        "chunk_count": len(sessions[session_id]["chunks"]),
    }, session_token)


@app.post("/chat/")
async def chat(request: Request):
    """RAG-first chat endpoint for LLM interaction (Qwen)."""
    data = await request.json()
    prompt = request_prompt(data, "prompt")
    use_internet = data.get("use_internet", False)
    image = data.get("image")  # Expecting base64 or bytes (optional)
    top_k = request_int(data, "top_k", DEFAULT_RAG_TOP_K, 1, MAX_TOP_K)
    session_id, session_token = authorize_session(request, data.get("session_id"))
    logger.info(
        "chat_begin session_id=%s prompt_chars=%s use_internet=%s has_image=%s",
        session_id,
        len(prompt or ""),
        use_internet,
        image is not None,
    )

    decision = assess_prompt(prompt)
    blocked = decision.action != "allow"
    social_categories, question = ([], prompt) if blocked else split_social_prompt(prompt)
    courtesy = social_prefix(social_categories)

    def kindly(text: str) -> str:
        return f"{courtesy}\n\n{text}" if courtesy else text

    def respond(payload: dict) -> dict:
        return with_session_token({"session_id": session_id, **payload}, session_token)

    def fixed_reply(response: str, study_task: str) -> dict:
        append_chat_message(session_id, {"sender": "user", "text": prompt})
        append_chat_message(session_id, {"sender": "ai", "text": response, "sources": []})
        logger.info("chat_fixed_reply session_id=%s study_task=%s", session_id, study_task)
        return respond({
            "response": response,
            "usage": None,
            "used_internet": False,
            "rag_first": False,
            "study_task": study_task,
            "sources": [],
        })

    if blocked:
        # Only the category is logged, never the prompt text.
        logger.warning("chat_moderated session_id=%s action=%s category=%s", session_id, decision.action, decision.category)
        return fixed_reply(decision.response, "moderated")
    if social_categories and not question:
        return fixed_reply(social_reply(social_categories), "social")
    if is_about_assistant_prompt(question):
        return fixed_reply(kindly(ABOUT_ASSISTANT_RESPONSE), "about_assistant")

    exchanges = recent_exchanges(session_id)
    retrieval_query = follow_up_retrieval_query(question, exchanges)
    is_follow_up = retrieval_query != question
    refers_back = bool(exchanges) and refers_to_previous_answer(question)
    if refers_back:
        # "Turn that into flashcards" keeps its own task but works from the answer it points at,
        # not from a broad sweep of the whole material.
        study_task = infer_study_task(question)
        context = follow_up_context(session_id, exchanges[-1][1], retrieve_context(session_id, retrieval_query, top_k))
    else:
        context, study_task = retrieve_study_context(session_id, retrieval_query, top_k)
        if is_follow_up:
            # The earlier question's task (e.g. "summarize") shapes retrieval, not the reply to the follow-up.
            study_task = "answer"
    logger.info(
        "chat_context session_id=%s study_task=%s context_chunks=%s top_k=%s history_exchanges=%s follow_up=%s refers_back=%s guidance_notes=%s",
        session_id,
        study_task,
        len(context),
        top_k,
        len(exchanges),
        is_follow_up,
        refers_back,
        len(decision.guidance),
    )
    append_chat_message(session_id, {"sender": "user", "text": prompt})
    if ensure_session(session_id).get("chunks") and not context:
        response = kindly("I could not find relevant uploaded material for that question. Try asking with terms from your notes or upload more source files.")
        append_chat_message(session_id, {"sender": "ai", "text": response, "sources": []})
        logger.warning("chat_no_relevant_context session_id=%s study_task=%s", session_id, study_task)
        return respond({
            "response": response,
            "usage": qwen_llm.estimate_usage(prompt, response, context=[], study_task=study_task),
            "used_internet": False,
            "rag_first": True,
            "sources": [],
        })
    if not context:
        response = kindly("Upload study files first so I can answer from your material.")
        append_chat_message(session_id, {"sender": "ai", "text": response, "sources": []})
        logger.warning("chat_no_uploaded_context session_id=%s study_task=%s", session_id, study_task)
        return respond({
            "response": response,
            "usage": qwen_llm.estimate_usage(prompt, response, context=[], study_task=study_task),
            "used_internet": False,
            "rag_first": True,
            "sources": [],
        })

    # TODO: handle image input (decode if base64)
    try:
        chat_result = qwen_llm.chat_with_usage(
            question,
            image=image,
            context=context,
            use_internet=use_internet,
            study_task=study_task,
            history=conversation_turns(exchanges),
            guidance=list(decision.guidance),
        )
    except Exception:
        logger.exception(
            "chat_llm_failed session_id=%s study_task=%s context_chunks=%s",
            session_id,
            study_task,
            len(context),
        )
        raise
    chat_result["response"], sources = present_response(session_id, chat_result["response"], context, prompt)
    chat_result["response"] = kindly(chat_result["response"])
    append_chat_message(
        session_id,
        {
            "sender": "ai",
            "text": chat_result["response"],
            "sources": sources,
            "tokenUsage": {
                "label": "Answer",
                "tokens": chat_result["usage"].get("completion_tokens"),
                "total": chat_result["usage"].get("total_tokens"),
                "maxNewTokens": chat_result["usage"].get("max_new_tokens"),
                "retriedForQuality": chat_result["usage"].get("retried_for_quality"),
                "hitTokenLimit": chat_result["usage"].get("hit_token_limit"),
                "estimated": chat_result["usage"].get("estimated"),
            },
        },
    )
    usage = chat_result.get("usage", {})
    logger.info(
        "chat_complete session_id=%s study_task=%s response_chars=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s hit_token_limit=%s retried_for_quality=%s",
        session_id,
        study_task,
        len(chat_result.get("response", "")),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
        usage.get("hit_token_limit"),
        usage.get("retried_for_quality"),
    )
    return respond({
        "response": chat_result["response"],
        "usage": chat_result["usage"],
        "used_internet": False,
        "rag_first": True,
        "study_task": study_task,
        "sources": sources,
    })


def screen_tool_prompt(endpoint: str, session_id: Optional[str], prompt: str) -> ModerationDecision:
    decision = assess_prompt(prompt)
    if decision.action != "allow":
        logger.warning("%s_moderated session_id=%s action=%s category=%s", endpoint, session_id, decision.action, decision.category)
    return decision


@app.post("/search/")
async def search(request: Request):
    """Search within uploaded content."""
    data = await request.json()
    session_id, _token = authorize_session(request, data.get("session_id"), create=False)
    query = request_prompt(data, "query", "prompt")
    top_k = request_int(data, "top_k", DEFAULT_RAG_TOP_K, 1, MAX_TOP_K)
    results = [public_source(source) for source in retrieve_context(session_id, query, top_k)]
    logger.info("search_complete session_id=%s query_chars=%s top_k=%s results=%s", session_id, len(query), top_k, len(results))
    return {"results": results}

@app.post("/quiz/")
async def quiz(request: Request):
    """Generate quiz from uploaded content."""
    data = await request.json()
    session_id, _token = authorize_session(request, data.get("session_id"), create=False)
    prompt = request_prompt(data, "prompt", default="Quiz me on the uploaded material.")
    decision = screen_tool_prompt("quiz", session_id, prompt)
    if decision.action != "allow":
        return {"quiz": [], "response": decision.response, "sources": []}
    context = retrieve_full_session_context(session_id)
    if not context:
        logger.warning("quiz_no_context session_id=%s", session_id)
        return {"quiz": [], "response": "Upload study files first so I can build a quiz from your material.", "sources": []}
    logger.info("quiz_begin session_id=%s prompt_chars=%s context_chunks=%s", session_id, len(prompt), len(context))
    try:
        chat_result = qwen_llm.chat_with_usage(prompt, context=context, study_task="quiz", guidance=list(decision.guidance))
    except Exception:
        logger.exception("quiz_llm_failed session_id=%s context_chunks=%s", session_id, len(context))
        raise
    response, sources = present_response(session_id, chat_result["response"], context, prompt)
    logger.info("quiz_complete session_id=%s response_chars=%s", session_id, len(response))
    return {"quiz": response, "response": response, "usage": chat_result["usage"], "sources": sources}

@app.post("/summarize/")
async def summarize(request: Request):
    """Summarize uploaded content or specific sections."""
    data = await request.json()
    session_id, _token = authorize_session(request, data.get("session_id"), create=False)
    prompt = request_prompt(data, "prompt", default="Summarize the uploaded material.")
    decision = screen_tool_prompt("summarize", session_id, prompt)
    if decision.action != "allow":
        return {"summary": decision.response, "response": decision.response, "sources": []}
    top_k = request_int(data, "top_k", DEFAULT_RAG_TOP_K, 1, MAX_TOP_K)
    context, _study_task = retrieve_study_context(session_id, prompt, top_k)
    if not context:
        logger.warning("summarize_no_context session_id=%s prompt_chars=%s", session_id, len(prompt))
        return {"summary": "Upload study files first so I can summarize your material.", "sources": []}
    logger.info("summarize_begin session_id=%s prompt_chars=%s context_chunks=%s", session_id, len(prompt), len(context))
    try:
        chat_result = qwen_llm.chat_with_usage(prompt, context=context, study_task="summary", guidance=list(decision.guidance))
    except Exception:
        logger.exception("summarize_llm_failed session_id=%s context_chunks=%s", session_id, len(context))
        raise
    response, sources = present_response(session_id, chat_result["response"], context, prompt)
    logger.info("summarize_complete session_id=%s response_chars=%s", session_id, len(response))
    return {"summary": response, "response": response, "usage": chat_result["usage"], "sources": sources}

@app.post("/video-link/")
async def video_link(request: Request):
    """Find and link to specific times in videos."""
    data = await request.json()
    session_id, _token = authorize_session(request, data.get("session_id"), create=False)
    prompt = request_prompt(data, "prompt", "query")
    decision = screen_tool_prompt("video_link", session_id, prompt)
    if decision.action != "allow":
        return {"links": [], "response": decision.response, "sources": []}
    context = retrieve_full_session_context(session_id)
    if not context:
        logger.warning("video_link_no_context session_id=%s prompt_chars=%s", session_id, len(prompt))
        return {"links": [], "response": "I could not find that topic in the uploaded material.", "sources": []}
    logger.info("video_link_begin session_id=%s prompt_chars=%s context_chunks=%s", session_id, len(prompt), len(context))
    try:
        chat_result = qwen_llm.chat_with_usage(prompt, context=context, study_task="find", guidance=list(decision.guidance))
    except Exception:
        logger.exception("video_link_llm_failed session_id=%s context_chunks=%s", session_id, len(context))
        raise
    response, sources = present_response(session_id, chat_result["response"], context, prompt)
    logger.info("video_link_complete session_id=%s response_chars=%s", session_id, len(response))
    return {"links": sources, "response": response, "usage": chat_result["usage"], "sources": sources}
