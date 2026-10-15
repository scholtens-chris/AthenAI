import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import PurePath
from typing import Optional

# Descriptive transcripts mix speech with visual descriptions and on-screen text. On-screen text
# captured from UI screenshots is long and mostly menus, URLs, and garbled characters, so it is
# dropped; short on-screen text (name captions, stat cards, titles) is kept.
MAX_ONSCREEN_TEXT_CHARS = 300
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|edu|net|io)\b")

# Consecutive speech fragments are merged into units of roughly this many words so each
# timestamp marker covers a readable passage instead of a few words.
SPEECH_UNIT_MIN_WORDS = 35
SPEECH_UNIT_MAX_WORDS = 60
SPEECH_UNIT_MAX_GAP_SECONDS = 10

TRANSCRIPT_NAME_SUFFIXES = re.compile(
    r"(?i)[\s_-]*(?:timed[\s_-]*)?(?:descriptive[\s_-]*)?(?:transcript|captions?|subtitles?)$"
)
SRT_TIME = r"(\d{1,2}:)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
CUE_PATTERN = re.compile(rf"{SRT_TIME}\s*-->\s*{SRT_TIME}")
DESCRIPTIVE_SEGMENT_PATTERN = re.compile(
    r"\[(\d{1,2}):(\d{2}):(\d{2})(?:[,.](\d{1,3}))?\]\s*\[([^\]]+)\]\s*(.*?)(?=\[\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?\]\s*\[|\Z)",
    re.S,
)
TIME_MARKER_PATTERN = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]")
CITATION_PATTERN = re.compile(
    r"\[\s*(?:(?:sources?|excerpts?|chunks?)\s*:?\s*#?)?(\d+(?:\s*(?:,|and|&)\s*\d+)*)\s*\]",
    re.I,
)
PROSE_SOURCE_PATTERN = re.compile(r"\b(?:(?:source|excerpt|chunk)\s+)+#?(\d+)\b", re.I)
# The model sometimes copies an excerpt heading's time range, e.g. [2:30-3:14].
TIME_RANGE_PATTERN = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\s*[-–]\s*\d+:\d{2}(?::\d{2})?\]")
WORD_PATTERN = re.compile(r"[a-z0-9']+")
DESCRIPTION_PREFIX = re.compile(r"\((?:Visual|On screen|Sound)\)")
PREVIEW_CHARS = 360


def title_from_filename(filename: str) -> str:
    stem = PurePath(filename or "source").stem
    stem = TRANSCRIPT_NAME_SUFFIXES.sub("", stem) or stem
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip() or "source"


def format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_timestamp_label(label: str) -> float:
    seconds = 0
    for part in label.split(":"):
        seconds = seconds * 60 + int(part)
    return float(seconds)


def _clock_seconds(hours, minutes, seconds, millis) -> float:
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int((millis or "0").ljust(3, "0")) / 1000


def _segment_kind(tag: str) -> str:
    tag = tag.strip().lower()
    if tag.startswith("speech"):
        return "speech"
    if tag.startswith("visual"):
        return "visual"
    if "screen" in tag or tag.startswith("text"):
        return "onscreen"
    return "sound"


def _parse_cues(text: str) -> list[dict]:
    segments = []
    lines = text.replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(lines):
        match = CUE_PATTERN.search(lines[index])
        if not match:
            index += 1
            continue
        g = match.groups()
        start = _clock_seconds((g[0] or "").rstrip(":"), g[1], g[2], g[3])
        end = _clock_seconds((g[4] or "").rstrip(":"), g[5], g[6], g[7])
        index += 1
        cue_lines = []
        while index < len(lines) and lines[index].strip() and not CUE_PATTERN.search(lines[index]):
            cue_lines.append(lines[index].strip())
            index += 1
        # SRT cue numbers sit on the line before the next cue's timing line.
        if cue_lines and index < len(lines) and CUE_PATTERN.search(lines[index]) and cue_lines[-1].isdigit():
            cue_lines.pop()
        cue_text = re.sub(r"<[^>]+>", "", " ".join(cue_lines)).strip()
        if cue_text:
            kind = "sound" if re.fullmatch(r"[\[(][^\])]+[\])]", cue_text) else "speech"
            segments.append({"start": start, "end": end, "kind": kind, "text": cue_text})
    return segments


def _parse_descriptive(text: str) -> list[dict]:
    segments = []
    for hours, minutes, seconds, millis, tag, body in DESCRIPTIVE_SEGMENT_PATTERN.findall(text):
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            start = _clock_seconds(hours, minutes, seconds, millis)
            segments.append({"start": start, "end": start, "kind": _segment_kind(tag), "text": body})
    return segments


def _dfxp_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    clock = re.fullmatch(r"(\d+):(\d{2}):(\d{2})(?:[.,](\d+))?", value)
    if clock:
        return _clock_seconds(clock.group(1), clock.group(2), clock.group(3), (clock.group(4) or "0")[:3])
    offset = re.fullmatch(r"([\d.]+)(h|m|s|ms|t)", value)
    if offset:
        amount, unit = float(offset.group(1)), offset.group(2)
        return {"h": amount * 3600, "m": amount * 60, "s": amount, "ms": amount / 1000, "t": amount / 10_000_000}[unit]
    return None


def _parse_dfxp(text: str) -> list[dict]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    segments = []
    for element in root.iter():
        start = _dfxp_seconds(element.get("begin"))
        if start is None:
            continue
        cue_text = re.sub(r"\s+", " ", " ".join(element.itertext())).strip()
        if cue_text:
            end = _dfxp_seconds(element.get("end"))
            segments.append({"start": start, "end": end if end is not None else start, "kind": "speech", "text": cue_text})
    return segments


def parse_timed_segments(filename: str, content: bytes) -> Optional[list[dict]]:
    """Return timed transcript segments, or None when the file carries no timing."""
    ext = (filename or "").lower().rsplit(".", 1)[-1]
    if ext not in {"srt", "vtt", "txt", "dfxp", "xml"}:
        return None
    text = content.decode("utf-8-sig", errors="ignore")
    if ext in {"dfxp", "xml"}:
        segments = _parse_dfxp(text)
    elif len(DESCRIPTIVE_SEGMENT_PATTERN.findall(text[:5000])) >= 2:
        segments = _parse_descriptive(text)
    else:
        segments = _parse_cues(text)
    return segments if len(segments) >= 2 else None


def is_noisy_onscreen_text(text: str) -> bool:
    return len(text) > MAX_ONSCREEN_TEXT_CHARS or bool(URL_PATTERN.search(text))


def build_transcript_units(segments: list[dict]) -> list[dict]:
    """Clean timed segments and merge speech fragments into timestamped passages."""
    units = []
    seen = set()
    speech = None

    def flush_speech():
        nonlocal speech
        if speech:
            units.append(speech)
            speech = None

    for segment in sorted(segments, key=lambda item: item["start"]):
        text = segment["text"].strip()
        key = (round(segment["start"], 1), segment["kind"], text)
        if not text or key in seen:
            continue
        seen.add(key)
        kind = segment["kind"]
        if kind == "onscreen" and is_noisy_onscreen_text(text):
            continue
        if kind != "speech":
            flush_speech()
            if kind == "onscreen":
                text = re.sub(r"\s*\|\s*", "; ", text).strip(" ;")
            label = {"visual": "Visual", "onscreen": "On screen", "sound": "Sound"}[kind]
            units.append({"start": segment["start"], "end": segment["end"], "text": f"({label}) {text}"})
            continue
        if speech and segment["start"] - speech["end"] > SPEECH_UNIT_MAX_GAP_SECONDS:
            flush_speech()
        if speech is None:
            speech = {"start": segment["start"], "end": segment["end"], "text": text}
        else:
            speech["text"] = f"{speech['text']} {text}"
            speech["end"] = max(speech["end"], segment["end"])
        word_count = len(speech["text"].split())
        ends_sentence = bool(re.search(r"[.!?][\"')\]]*$", speech["text"]))
        if word_count >= SPEECH_UNIT_MAX_WORDS or (word_count >= SPEECH_UNIT_MIN_WORDS and ends_sentence):
            flush_speech()
    flush_speech()
    return units


def render_unit(unit: dict) -> str:
    return f"[{format_timestamp(unit['start'])}] {unit['text']}"


def chunk_transcript_units(units: list[dict], chunk_words: int) -> list[dict]:
    """Group units into chunks on unit boundaries, so every chunk starts at a known time."""
    chunks = []
    current = []
    current_words = 0
    for unit in units:
        unit_words = len(unit["text"].split())
        if current and current_words + unit_words > chunk_words:
            chunks.append(current)
            current, current_words = [], 0
        current.append(unit)
        current_words += unit_words
    if current:
        chunks.append(current)
    return [
        {
            "text": "\n".join(render_unit(unit) for unit in group),
            "start": group[0]["start"],
            "end": max(unit["end"] for unit in group),
        }
        for group in chunks
    ]


def source_heading(source: dict) -> str:
    title = source.get("title") or title_from_filename(source.get("filename", "source"))
    if source.get("start") is None:
        return title
    return f"{title}, {format_timestamp(source['start'])}-{format_timestamp(source.get('end'))}"


def _words(text: str) -> set[str]:
    return {word for word in WORD_PATTERN.findall(text.lower()) if len(word) > 2}


def best_timestamp(source: dict, claim: str) -> Optional[float]:
    """Pick the timestamp of the passage in a source that best matches the cited claim."""
    if source.get("start") is None:
        return None
    text = source.get("text", "")
    markers = list(TIME_MARKER_PATTERN.finditer(text))
    if not markers:
        return source["start"]
    claim_words = _words(claim)
    best_time, best_score = source["start"], 0
    for index, marker in enumerate(markers):
        passage_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        score = len(claim_words & _words(text[marker.end():passage_end]))
        if score > best_score:
            best_time, best_score = parse_timestamp_label(marker.group(1)), score
    return best_time


def citation_label(source: dict, timestamp: Optional[float], include_title: bool) -> str:
    title = source.get("title") or title_from_filename(source.get("filename", "source"))
    if timestamp is None:
        return title
    if include_title:
        return f"{title}, {format_timestamp(timestamp)}"
    return format_timestamp(timestamp)


def _claim_before(text: str, position: int) -> str:
    start = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position))
    return text[start + 1:position]


def render_citations(
    response: str, sources: list[dict], include_title: bool, cited: Optional[dict] = None
) -> str:
    """Replace model citations like [2] or [Source: 2] with video names and timestamps.

    When `cited` is given, it is filled with {source index: [cited timestamps]} in citation order.
    """
    if not sources:
        return response

    def record(index: int, timestamp: Optional[float]):
        if cited is not None:
            times = cited.setdefault(index, [])
            if timestamp is not None and timestamp not in times:
                times.append(timestamp)

    def bracket_citation(match: re.Match) -> str:
        numbers = [int(number) for number in re.findall(r"\d+", match.group(1))]
        claim = _claim_before(response, match.start())
        labels = []
        for number in numbers:
            if 1 <= number <= len(sources):
                timestamp = best_timestamp(sources[number - 1], claim)
                record(number - 1, timestamp)
                label = citation_label(sources[number - 1], timestamp, include_title)
                if label not in labels:
                    labels.append(label)
        return f"({'; '.join(labels)})" if labels else ""

    def prose_citation(match: re.Match) -> str:
        number = int(match.group(1))
        if not 1 <= number <= len(sources):
            return match.group(0)
        source = sources[number - 1]
        claim = response[match.end():match.end() + 200]
        title = source.get("title") or title_from_filename(source.get("filename", "source"))
        timestamp = best_timestamp(source, claim)
        record(number - 1, timestamp)
        if timestamp is None:
            return title
        place = title if include_title else "the video"
        return f"{place} at {format_timestamp(timestamp)}"

    rendered = CITATION_PATTERN.sub(bracket_citation, response)
    rendered = PROSE_SOURCE_PATTERN.sub(prose_citation, rendered)
    rendered = TIME_RANGE_PATTERN.sub(r"(\1)", rendered)
    return re.sub(r"[ \t]+([.,;:])", r"\1", rendered)


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _timed_passages(text: str) -> list[tuple[float, str]]:
    markers = list(TIME_MARKER_PATTERN.finditer(text))
    passages = []
    for index, marker in enumerate(markers):
        passage_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[marker.end():passage_end].strip()
        if body:
            passages.append((parse_timestamp_label(marker.group(1)), body))
    return passages


def _passage_from(passages: list[tuple[float, str]], timestamp: Optional[float], limit: int) -> str:
    """Read forward from the cited passage, skipping visual/on-screen descriptions after it."""
    first = 0
    if timestamp is not None:
        first = next((index for index, (time, _) in enumerate(passages) if time >= timestamp), 0)
    parts = []
    for offset, (_, body) in enumerate(passages[first:]):
        if offset and DESCRIPTION_PREFIX.match(body):
            continue
        parts.append(body)
        if sum(len(part) + 1 for part in parts) >= limit:
            break
    return _clip(" ".join(parts), limit)


def source_preview(source: dict, timestamps: Optional[list[float]] = None, limit: int = PREVIEW_CHARS) -> str:
    """Show the passage a citation points to, rather than the start of the retrieved chunk."""
    text = source.get("text", "")
    passages = _timed_passages(text) if source.get("start") is not None else []
    if not passages:
        return _clip(text, limit)
    times = list(dict.fromkeys(timestamps or [])) or [None]
    if len(times) == 1:
        return _passage_from(passages, times[0], limit)
    per_time = max(limit // len(times), 160)
    return "\n".join(f"[{format_timestamp(time)}] {_passage_from(passages, time, per_time)}" for time in times)


def cited_source_label(source: dict, timestamps: Optional[list[float]] = None) -> str:
    if not timestamps or source.get("start") is None:
        return source_heading(source)
    title = source.get("title") or title_from_filename(source.get("filename", "source"))
    return f"{title}, {', '.join(format_timestamp(time) for time in sorted(set(timestamps)))}"


def session_has_multiple_titles(documents: list[dict]) -> bool:
    titles = Counter(item.get("title") or title_from_filename(item.get("filename", "")) for item in documents)
    return len(titles) > 1
