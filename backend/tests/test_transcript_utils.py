from transcript_utils import (
    build_transcript_units,
    chunk_transcript_units,
    format_timestamp,
    parse_timed_segments,
    render_citations,
    session_has_multiple_titles,
    source_heading,
    title_from_filename,
)

DESCRIPTIVE = (
    "Timed Descriptive Transcript "
    "[00:00:00,000] [Visual] A campus with students walking. "
    "[00:00:01,000] [Speech] Every day institutions create video. "
    "[00:00:04,000] [Speech] Captions help everyone follow along. "
    "[00:00:04,000] [Speech] Captions help everyone follow along. "
    "[00:00:05,000] [On-screen Text] Jane Doe | Director | Example University "
    "[00:01:05,000] [On-screen Text] Browse | Home | Search | preview.example.com/Manage | " + "Menu | " * 80 + " "
    "[00:01:10,000] [Speech] The dashboard measures accessibility progress."
).encode()


def test_title_from_filename_strips_transcript_suffixes():
    assert title_from_filename("Anthem Award_descriptive_transcript.txt") == "Anthem Award"
    assert title_from_filename("biology_139.srt") == "biology 139"
    assert title_from_filename("Cell_Lecture_captions.vtt") == "Cell Lecture"
    assert title_from_filename("notes.pdf") == "notes"


def test_format_timestamp():
    assert format_timestamp(None) == ""
    assert format_timestamp(65.5) == "1:05"
    assert format_timestamp(3725) == "1:02:05"


def test_descriptive_transcript_drops_noisy_onscreen_text_and_duplicates():
    segments = parse_timed_segments("talk_descriptive_transcript.txt", DESCRIPTIVE)
    units = build_transcript_units(segments)
    text = "\n".join(unit["text"] for unit in units)

    assert "(Visual) A campus with students walking." in text
    assert "(On screen) Jane Doe; Director; Example University" in text
    assert "preview.example.com" not in text
    assert "Menu" not in text
    assert text.count("Captions help everyone follow along.") == 1
    assert units[-1]["start"] == 70


def test_srt_and_vtt_keep_timing():
    srt = b"1\n00:00:01,000 --> 00:00:04,000\nHello there.\n\n2\n00:01:05,500 --> 00:01:08,000\n[Music]\n\n3\n00:01:09,000 --> 00:01:12,000\nCells divide.\n"
    segments = parse_timed_segments("bio.srt", srt)
    assert [(segment["start"], segment["kind"], segment["text"]) for segment in segments] == [
        (1.0, "speech", "Hello there."),
        (65.5, "sound", "[Music]"),
        (69.0, "speech", "Cells divide."),
    ]

    vtt = b"WEBVTT\n\n00:05.000 --> 00:07.000\n<v Speaker>First line.\n\n00:08.000 --> 00:09.000\nSecond line.\n"
    assert [segment["text"] for segment in parse_timed_segments("bio.vtt", vtt)] == ["First line.", "Second line."]


def test_dfxp_keeps_timing():
    dfxp = b'<tt><body><div><p begin="00:00:02.000" end="00:00:03.000">Alpha</p><p begin="4.5s">Beta</p></div></body></tt>'
    assert [(segment["start"], segment["text"]) for segment in parse_timed_segments("bio.dfxp", dfxp)] == [
        (2.0, "Alpha"),
        (4.5, "Beta"),
    ]


def test_untimed_files_return_none():
    assert parse_timed_segments("notes.txt", b"Plain notes about mitosis.") is None
    assert parse_timed_segments("slides.pdf", b"%PDF") is None


def test_chunks_start_on_passage_boundaries():
    units = [
        {"start": 0, "end": 5, "text": "one two three"},
        {"start": 10, "end": 15, "text": "four five six"},
        {"start": 20, "end": 25, "text": "seven eight"},
    ]
    chunks = chunk_transcript_units(units, chunk_words=6)
    assert [(chunk["start"], chunk["end"]) for chunk in chunks] == [(0, 15), (20, 25)]
    assert chunks[0]["text"] == "[0:00] one two three\n[0:10] four five six"


def test_source_heading_uses_title_and_time_range():
    assert source_heading({"filename": "Talk_transcript.srt", "start": 62, "end": 104}) == "Talk, 1:02-1:44"
    assert source_heading({"filename": "notes.pdf", "start": None}) == "notes"


def test_render_citations_uses_best_matching_timestamp():
    sources = [
        {
            "filename": "Talk_transcript.srt",
            "title": "Talk",
            "start": 60,
            "text": "[1:00] Welcome to the session.\n[1:20] The dashboard measures accessibility progress.",
        },
        {"filename": "notes.pdf", "title": "notes", "start": None, "text": "Mitosis notes"},
    ]
    response = "The dashboard measures progress [1]. Notes cover mitosis [Source: 2]. Ignore [9]."
    assert render_citations(response, sources, include_title=False) == (
        "The dashboard measures progress (1:20). Notes cover mitosis (notes). Ignore."
    )
    assert render_citations("It measures progress [1, 2].", sources, include_title=True) == (
        "It measures progress (Talk, 1:20; notes)."
    )
    assert render_citations("According to Source 1, the dashboard measures progress.", sources, False) == (
        "According to the video at 1:20, the dashboard measures progress."
    )
    assert render_citations("It is in Source excerpt 1 [1:00-1:44] where progress is measured.", sources, False) == (
        "It is in the video at 1:20 (1:00) where progress is measured."
    )


def test_session_has_multiple_titles():
    assert not session_has_multiple_titles([{"filename": "a.srt"}])
    assert session_has_multiple_titles([{"filename": "a.srt"}, {"filename": "b.srt"}])
