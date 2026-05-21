import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


class MediasiteApiError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class MediasiteClient:
    base_url: str
    api_key: str
    username: Optional[str] = None
    password: Optional[str] = None
    impersonate_username: Optional[str] = None
    authorization_ticket: Optional[str] = None
    application_ticket: Optional[str] = None
    timeout_seconds: int = 20

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("Mediasite base URL is required.")
        if not self.api_key:
            raise ValueError("Mediasite API key is required.")

    @property
    def api_root(self) -> str:
        if self.base_url.lower().endswith("/api/v1"):
            return self.base_url
        return f"{self.base_url}/api/v1"

    def headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/plain, */*",
            "sfapikey": self.api_key,
        }
        if self.application_ticket:
            headers["Mediasite-Application-Ticket"] = self.application_ticket
        if self.authorization_ticket:
            headers["Authorization"] = f"SfAuthTicket {self.authorization_ticket}"
        elif self.username and self.password and self.impersonate_username:
            encoded = base64.b64encode(
                f"{self.username}:{self.password}:{self.impersonate_username}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"SfIdentTicket {encoded}"
        elif self.username and self.password:
            encoded = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        return headers

    def resource_url(self, path: str, params: Optional[dict[str, Any]] = None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{self.api_root}/{path.lstrip('/')}"
        if params:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urllib.parse.urlencode(params)}"
        return url

    def request(self, path: str, params: Optional[dict[str, Any]] = None) -> tuple[bytes, str]:
        url = self.resource_url(path, params=params)
        request = urllib.request.Request(url, headers=self.headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("content-type", "")
                return response.read(), content_type
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise MediasiteApiError(
                f"Mediasite API returned HTTP {exc.code} for {url}.",
                status_code=exc.code,
                body=body,
            ) from exc
        except urllib.error.URLError as exc:
            raise MediasiteApiError(f"Could not reach Mediasite API at {url}: {exc.reason}") from exc

    def get_json(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        body, _content_type = self.request(path, params=params)
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            preview = body[:300].decode("utf-8", errors="replace")
            raise MediasiteApiError(f"Mediasite response was not valid JSON: {preview}") from exc

    def get_text(self, path: str, params: Optional[dict[str, Any]] = None) -> str:
        body, _content_type = self.request(path, params=params)
        return body.decode("utf-8", errors="replace")


def odata_id(value: str) -> str:
    cleaned = value.strip().strip("'")
    if not re.fullmatch(r"[A-Za-z0-9-]+", cleaned):
        raise ValueError("Mediasite IDs may only contain letters, numbers, or dashes.")
    return cleaned


def odata_path(resource: str, mediasite_id: str, child: Optional[str] = None) -> str:
    path = f"{resource}('{odata_id(mediasite_id)}')"
    if child:
        path = f"{path}/{child.strip('/')}"
    return path


def flatten_odata_collection(data: Any) -> list[Any]:
    if isinstance(data, dict):
        for key in ("value", "Items", "Results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return data if isinstance(data, list) else [data]


def odata_next_link(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in ("@odata.nextLink", "odata.nextLink", "NextPageLink", "nextLink"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def extract_presentation_ids(data: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in flatten_odata_collection(data):
        if not isinstance(item, dict):
            continue
        for key in ("Id", "PresentationId", "MediasiteId", "RootId"):
            value = item.get(key)
            if isinstance(value, str) and value and value not in seen:
                ids.append(value)
                seen.add(value)
                break
    return ids


def _collect_text_values(value: Any, keys: set[str], fragments: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in keys and isinstance(nested, str) and nested.strip():
                fragments.append(nested.strip())
            else:
                _collect_text_values(nested, keys, fragments)
    elif isinstance(value, list):
        for item in value:
            _collect_text_values(item, keys, fragments)


def extract_presentation_text(
    presentation: Any,
    ocr_content: Any = None,
    caption_content: Any = None,
    slide_details_content: Any = None,
) -> str:
    fragments: list[str] = []
    _collect_text_values(
        presentation,
        {"Title", "Description", "PrimaryPresenter", "TagList"},
        fragments,
    )
    _collect_text_values(
        ocr_content,
        {"Title", "Content", "OcrText", "Text"},
        fragments,
    )
    _collect_text_values(
        caption_content,
        {"CaptionText", "Text", "ContentExcerpt"},
        fragments,
    )
    _collect_text_values(
        slide_details_content,
        {"Title", "Content", "OcrText", "Text"},
        fragments,
    )
    return re.sub(r"\s+", " ", " ".join(str(fragment) for fragment in fragments)).strip()


def presentation_filename(presentation_id: str, presentation: Any) -> str:
    title = presentation.get("Title") if isinstance(presentation, dict) else None
    safe_title = re.sub(r"[^A-Za-z0-9._ -]+", "", title or "").strip().replace(" ", "_")
    safe_title = safe_title[:80] if safe_title else "mediasite_presentation"
    return f"{safe_title}_{odata_id(presentation_id)}.txt"


def _importable_content_text(client: MediasiteClient, content: Any, text_keys: set[str]) -> str:
    fragments: list[str] = []
    _collect_text_values(content, text_keys, fragments)
    for item in flatten_odata_collection(content):
        if not isinstance(item, dict):
            continue
        for key in ("DownloadUrl", "ThumbnailUrl", "Url", "ContentUrl", "Href"):
            url = item.get(key)
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                try:
                    fragments.append(client.get_text(url))
                except MediasiteApiError:
                    continue
    return re.sub(r"\s+", " ", " ".join(fragments)).strip()


def importable_ocr_text(client: MediasiteClient, presentation_id: str) -> str:
    try:
        ocr_content = client.get_json(odata_path("Presentations", presentation_id, "OcrContent"))
    except MediasiteApiError as exc:
        if exc.status_code == 404:
            return ""
        raise
    return _importable_content_text(client, ocr_content, {"Title", "Content", "OcrText", "Text"})


def importable_slide_details_text(client: MediasiteClient, presentation_id: str) -> str:
    try:
        slide_details_content = client.get_json(odata_path("Presentations", presentation_id, "SlideDetailsContent"))
    except MediasiteApiError as exc:
        if exc.status_code == 404:
            return ""
        raise
    return _importable_content_text(
        client,
        slide_details_content,
        {"Title", "Content", "OcrText", "Text"},
    )


def importable_caption_text(client: MediasiteClient, presentation_id: str) -> str:
    try:
        caption_content = client.get_json(odata_path("Presentations", presentation_id, "CaptionContent"))
    except MediasiteApiError as exc:
        if exc.status_code == 404:
            return ""
        raise
    return _importable_content_text(client, caption_content, {"CaptionText", "Text", "ContentExcerpt"})
