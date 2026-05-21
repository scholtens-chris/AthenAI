import io
import json
import sys
import types
import zipfile

import pytest
from fastapi.testclient import TestClient

import main
from mediasite_utils import MediasiteClient, extract_presentation_text, odata_path


class FakeLLM:
    model_path = "fake/model"
    use_mock_model = True
    model = None
    device = "cpu"
    max_new_tokens = 128
    long_task_max_new_tokens = 256
    temperature = 0.5
    top_p = 0.9
    top_k = 40

    def __init__(self):
        self.calls = []

    def estimate_usage(self, prompt, response, context=None, study_task="answer"):
        return {
            "prompt_tokens": len(prompt.split()) or 1,
            "completion_tokens": len(response.split()) or 1,
            "total_tokens": (len(prompt.split()) or 1) + (len(response.split()) or 1),
            "estimated": True,
            "study_task": study_task,
        }

    def chat_with_usage(self, prompt, image=None, context=None, use_internet=False, study_task="answer"):
        self.calls.append(
            {
                "prompt": prompt,
                "image": image,
                "context": context,
                "use_internet": use_internet,
                "study_task": study_task,
            }
        )
        return {
            "response": f"{study_task}: {prompt}",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "max_new_tokens": 128,
                "estimated": False,
            },
        }


def zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolate_session_store(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "SESSION_STORE_DIR", tmp_path / "sessions")
    monkeypatch.setattr(main, "SESSION_PERSISTENCE_ENABLED", True)
    monkeypatch.setattr(main, "SESSION_TTL_SECONDS", 60 * 60)
    main.sessions.clear()
    yield
    main.sessions.clear()


def test_parse_helpers_cover_supported_text_formats():
    assert main.parse_txt(b"hello") == "hello"
    assert main.parse_json(json.dumps({"a": 1}).encode()) == '{\n  "a": 1\n}'
    assert main.parse_json(b'[{"a": 1}, {"b": 2}]') == '{"a": 1}\n{"b": 2}'
    assert main.parse_json(b'"plain"') == "plain"
    assert main.parse_json(b"{broken") == "[Invalid JSON]"
    assert main.parse_dfxp(b"<tt><body><p>Hello</p><p>world</p></body></tt>") == "Hello world"
    assert main.parse_dfxp(b"<tt>") == "[Invalid DFXP XML]"
    assert main.parse_srt(b"1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\nWorld") == "Hello 2 World"
    assert main.parse_vtt(b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\nWorld") == "Hello World"
    assert main.parse_file("notes.md", b"# Title") == "# Title"
    assert main.parse_file("data.json", b'{"x": 1}') == '{\n  "x": 1\n}'
    assert main.parse_file("captions.dfxp", b"<tt><p>line</p></tt>") == "line"
    assert main.parse_file("captions.srt", b"00:00:01.000 --> 00:00:02.000\nline") == "line"
    assert main.parse_file("captions.vtt", b"WEBVTT\nline") == "line"
    assert main.parse_file("unknown.bin", b"\xffhello") == "hello"


def test_document_indexing_retrieval_and_context_limits(monkeypatch):
    main.sessions.clear()
    monkeypatch.setattr(main, "CHUNK_WORDS", 4)
    monkeypatch.setattr(main, "CHUNK_OVERLAP", 1)
    monkeypatch.setattr(main, "MAX_CONTEXT_CHARS", 20)

    session = main.ensure_session("s1")
    assert session["content"] == []
    assert session["chunks"] == []
    assert session["chat_history"] == []
    assert main.tokenize("The Cells, and cells divide!") == ["cells", "cells", "divide"]
    assert main.chunk_text("", chunk_words=3, overlap=1) == []
    assert main.chunk_text("one two three", chunk_words=5, overlap=1) == ["one two three"]
    assert main.chunk_text("one two three four five six", chunk_words=3, overlap=1) == [
        "one two three",
        "three four five",
        "five six",
    ]

    assert main.add_document_to_session("s1", "bio.txt", "Mitosis makes cells divide quickly") == 1
    assert main.add_document_to_session("s1", "empty.txt", "   ") == 0

    results = main.retrieve_context("s1", "where do cells divide", top_k=2)
    assert results[0]["filename"] == "bio.txt"
    assert results[0]["preview"] if "preview" in results[0] else results[0]["text"]
    assert len(results[0]["text"]) <= 20
    assert main.retrieve_context(None, "cells") == []
    assert main.retrieve_context("missing", "cells") == []
    assert main.retrieve_context("s1", "the and") == []
    assert main.retrieve_context("s1", "photosynthesis") == []

    monkeypatch.setattr(main, "MAX_CONTEXT_CHARS", 0)
    assert main.retrieve_context("s1", "cells", top_k=1) == []


def test_study_task_inference_overview_full_context_and_public_source(monkeypatch):
    main.sessions.clear()
    monkeypatch.setattr(main, "MAX_CONTEXT_CHARS", 10)
    main.add_document_to_session("s2", "first.txt", "alpha beta gamma delta")
    main.add_document_to_session("s2", "second.txt", "diagram concept review")

    assert main.infer_study_task("Quiz me on this") == "quiz"
    assert main.infer_study_task("Create flashcards") == "flashcards"
    assert main.infer_study_task("Summarize chapter 3") == "summary"
    assert main.infer_study_task("Find where mitosis was mentioned") == "find"
    assert main.infer_study_task("What are the key takeaways?") == "takeaways"
    assert main.infer_study_task("Make a study guide") == "study_guide"
    assert main.infer_study_task("Explain photosynthesis") == "explain"
    assert main.infer_study_task("Answer this") == "answer"
    assert main.is_broad_study_prompt("quiz me", "quiz")
    assert main.is_broad_study_prompt("create flashcards", "flashcards")
    assert main.is_broad_study_prompt("main points", "takeaways")
    assert main.is_broad_study_prompt("explain this", "explain")
    assert not main.is_broad_study_prompt("explain the exact role of ATP in this pathway", "explain")
    assert not main.is_broad_study_prompt("explain photosynthesis", "explain")
    assert main.is_broad_study_prompt("quiz alpha beta", "quiz")
    assert main.is_broad_study_prompt("summarize all uploaded notes", "summary")
    assert not main.is_broad_study_prompt("summarize alpha beta", "summary")

    overview = main.retrieve_session_overview("s2", limit=2)
    assert [item["filename"] for item in overview] == ["first.txt"]
    assert main.retrieve_session_overview(None) == []
    main.sessions["empty"] = {"content": [], "chunks": []}
    assert main.retrieve_session_overview("empty") == []
    assert main.retrieve_full_session_context("empty") == []
    full = main.retrieve_full_session_context("s2", max_chars=12)
    assert full[0]["text"] == "alpha beta g"
    assert main.retrieve_full_session_context("s2", max_chars=0)
    assert main.retrieve_full_session_context(None) == []
    broad_context, broad_task = main.retrieve_study_context("s2", "Quiz me", 4)
    assert broad_task == "quiz"
    assert [item["filename"] for item in broad_context] == ["first.txt", "second.txt"]
    targeted_context, targeted_task = main.retrieve_study_context("s2", "Where is diagram explained?", 1)
    assert targeted_task == "find"
    assert len(targeted_context) == 1
    assert targeted_context[0]["filename"] == "second.txt"

    source = main.public_source(full[0])
    assert source["preview"] == full[0]["text"]


def test_binary_document_parsers_return_empty_on_invalid_content():
    assert main.parse_pdf(b"not a pdf") == ""
    assert main.parse_docx(b"not a docx") == ""
    assert main.parse_pptx(b"not a pptx") == ""


def test_binary_document_parsers_success_paths_with_fake_modules(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePdfReader:
        def __init__(self, _stream):
            self.pages = [FakePage("page one"), FakePage(None)]

    class FakeParagraph:
        def __init__(self, text):
            self.text = text

    class FakeDocument:
        def __init__(self, _stream):
            self.paragraphs = [FakeParagraph("first"), FakeParagraph("second")]

    class FakeShape:
        def __init__(self, text=None):
            if text is not None:
                self.text = text

    class FakeSlide:
        shapes = [FakeShape("title"), FakeShape("")]

    class FakePresentation:
        def __init__(self, _stream):
            self.slides = [FakeSlide()]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakePdfReader))
    monkeypatch.setitem(sys.modules, "docx", types.SimpleNamespace(Document=FakeDocument))
    monkeypatch.setitem(sys.modules, "pptx", types.SimpleNamespace(Presentation=FakePresentation))

    assert main.parse_pdf(b"pdf") == "page one\n"
    assert main.parse_docx(b"docx") == "first\nsecond"
    assert main.parse_pptx(b"pptx") == "title"
    assert main.parse_file("file.pdf", b"pdf") == "page one\n"
    assert main.parse_file("file.docx", b"docx") == "first\nsecond"
    assert main.parse_file("file.pptx", b"pptx") == "title"


def test_extract_files_from_zip_skips_directories():
    files = main.extract_files_from_zip(zip_bytes({"folder/file.txt": "hello", "root.md": "world"}))
    assert {file["filename"] for file in files} == {"folder/file.txt", "root.md"}


def test_api_endpoints_use_uploaded_context_and_llm(monkeypatch):
    main.sessions.clear()
    fake_llm = FakeLLM()
    monkeypatch.setattr(main, "qwen_llm", fake_llm)
    client = TestClient(main.app)

    status = client.get("/llm/status")
    assert status.status_code == 200
    assert status.json()["model"] == "fake/model"

    upload = client.post(
        "/upload/",
        files=[
            ("files", ("notes.txt", b"Cells divide during mitosis and produce identical cells.", "text/plain")),
            ("files", ("pack.zip", zip_bytes({"slides.md": "Mitosis has phases.", "empty.txt": ""}), "application/zip")),
        ],
    )
    body = upload.json()
    assert upload.status_code == 200
    assert body["session_id"]
    assert body["added_chunk_count"] == 2
    assert body["indexed_files"] == ["notes.txt", "slides.md"]
    assert body["skipped_files"] == ["empty.txt"]

    session_id = body["session_id"]
    search = client.post("/search/", json={"session_id": session_id, "query": "mitosis", "top_k": 1})
    assert search.json()["results"][0]["filename"] in {"notes.txt", "slides.md"}

    chat = client.post(
        "/chat/",
        json={"session_id": session_id, "prompt": "Explain mitosis", "use_internet": True, "image": "abc"},
    )
    assert chat.status_code == 200
    assert chat.json()["response"] == "explain: Explain mitosis"
    assert chat.json()["rag_first"] is True
    assert chat.json()["study_task"] == "explain"
    assert chat.json()["sources"]
    assert fake_llm.calls[-1]["use_internet"] is True
    assert fake_llm.calls[-1]["image"] == "abc"

    quiz = client.post("/quiz/", json={"session_id": session_id})
    assert quiz.json()["response"].startswith("quiz:")
    summary = client.post("/summarize/", json={"session_id": session_id, "prompt": "Summarize"})
    assert summary.json()["summary"].startswith("summary:")
    video = client.post("/video-link/", json={"session_id": session_id, "query": "mitosis timestamp"})
    assert video.json()["response"].startswith("find:")


def test_mediasite_client_headers_paths_and_text_extraction():
    client = MediasiteClient(
        base_url="https://example.edu/Mediasite",
        api_key="key-1",
        username="user",
        password="pass",
    )

    assert client.api_root == "https://example.edu/Mediasite/api/v1"
    assert client.resource_url("Presentations('abc')", {"$select": "full"}).endswith(
        "/api/v1/Presentations('abc')?%24select=full"
    )
    assert client.headers()["sfapikey"] == "key-1"
    assert client.headers()["Authorization"].startswith("Basic ")
    assert odata_path("Presentations", "abc-123", "CaptionContent") == "Presentations('abc-123')/CaptionContent"

    identity_client = MediasiteClient(
        base_url="https://example.edu/Mediasite/api/v1",
        api_key="key-1",
        username="admin",
        password="secret",
        impersonate_username="student",
    )
    assert identity_client.headers()["Authorization"].startswith("SfIdentTicket ")

    text = extract_presentation_text(
        {"Title": "Lecture One", "Description": "Intro"},
        ocr_content={"value": [{"Title": "Slide", "OcrText": "Cell division"}]},
        caption_content={"value": [{"CaptionText": "Mitosis begins"}]},
        slide_details_content={"SlideDetails": [{"Content": "Slide detail notes"}]},
    )
    assert text == "Lecture One Intro Slide Cell division Mitosis begins Slide detail notes"


def test_mediasite_import_presentation_indexes_available_text(monkeypatch):
    main.sessions.clear()

    class FakeMediasiteClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]

        def get_json(self, path, params=None):
            if path == "Presentations('abc123')":
                assert params == {"$select": "full"}
                return {"Title": "Cell Lecture", "Description": "Course intro"}
            if path == "Presentations('abc123')/OcrContent":
                return {"value": [{"Title": "Mitosis", "OcrText": "Cells divide"}]}
            if path == "Presentations('abc123')/CaptionContent":
                return {"value": [{"CaptionText": "The nucleus prepares for division."}]}
            if path == "Presentations('abc123')/SlideDetailsContent":
                return {"SlideDetails": [{"Content": "Slide detail line"}]}
            raise AssertionError(path)

        def get_text(self, _path, params=None):
            return ""

    monkeypatch.setattr(main, "MEDIASITE_BASE_URL", "https://example.edu/Mediasite")
    monkeypatch.setattr(main, "MEDIASITE_API_KEY", "key-1")
    monkeypatch.setattr(main, "MediasiteClient", FakeMediasiteClient)

    client = TestClient(main.app)
    response = client.post("/mediasite/import-presentation/", json={"presentation_id": "abc123"})
    body = response.json()

    assert response.status_code == 200
    assert body["session_id"]
    assert body["filename"] == "Cell_Lecture_abc123.txt"
    assert body["added_chunk_count"] == 1
    assert body["has_slide_details_text"] is True
    assert main.retrieve_context(body["session_id"], "nucleus division")[0]["filename"] == body["filename"]


def test_mediasite_channel_listing_and_bulk_import(monkeypatch):
    main.sessions.clear()

    class FakeMediasiteClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]

        def get_json(self, path, params=None):
            if path == "MediasiteChannels('channel1')/Presentations":
                return {"value": [{"Id": "p1", "Title": "One"}, {"PresentationId": "p2", "Title": "Two"}]}
            if path == "Presentations('p1')":
                return {"Title": "First Video", "Description": "Overview"}
            if path == "Presentations('p1')/OcrContent":
                return {"value": [{"OcrText": "slide OCR alpha"}]}
            if path == "Presentations('p1')/CaptionContent":
                return {"value": [{"CaptionText": "caption alpha"}]}
            if path == "Presentations('p1')/SlideDetailsContent":
                return {"SlideDetails": [{"Content": "slide details alpha"}]}
            if path == "Presentations('p2')":
                return {"Title": "Second Video"}
            if path == "Presentations('p2')/OcrContent":
                return {"value": [{"OcrText": "slide OCR beta"}]}
            if path == "Presentations('p2')/CaptionContent":
                return {"value": [{"CaptionText": "caption beta"}]}
            if path == "Presentations('p2')/SlideDetailsContent":
                return {"SlideDetails": [{"Content": "slide details beta"}]}
            raise AssertionError(path)

        def get_text(self, _path, params=None):
            return ""

    monkeypatch.setattr(main, "MEDIASITE_BASE_URL", "https://example.edu/Mediasite")
    monkeypatch.setattr(main, "MEDIASITE_API_KEY", "key-1")
    monkeypatch.setattr(main, "MediasiteClient", FakeMediasiteClient)
    client = TestClient(main.app)

    listing = client.post("/mediasite/channel-presentations/", json={"channel_id": "channel1"})
    assert listing.status_code == 200
    assert listing.json()["presentation_ids"] == ["p1", "p2"]

    imported = client.post("/mediasite/import-channel/", json={"channel_id": "channel1"})
    body = imported.json()
    assert imported.status_code == 200
    assert body["presentation_ids"] == ["p1", "p2"]
    assert [item["presentation_id"] for item in body["imported"]] == ["p1", "p2"]
    assert body["skipped"] == []
    assert body["failed"] == []
    assert body["added_chunk_count"] == 2
    assert body["imported"][0]["has_slide_details_text"] is True
    assert main.retrieve_context(body["session_id"], "caption beta")[0]["filename"] == "Second_Video_p2.txt"


def test_mediasite_invalid_channel_key_explains_guid_requirement(monkeypatch):
    class FakeMediasiteClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]

        def get_json(self, path, params=None):
            raise main.MediasiteApiError(
                "Mediasite API returned HTTP 400.",
                status_code=400,
                body='{"odata.error":{"code":"InvalidKey"}}',
            )

    monkeypatch.setattr(main, "MEDIASITE_BASE_URL", "https://example.edu/Mediasite/api/v1")
    monkeypatch.setattr(main, "MEDIASITE_API_KEY", "key-1")
    monkeypatch.setattr(main, "MediasiteClient", FakeMediasiteClient)

    client = TestClient(main.app)
    response = client.post("/mediasite/import-channel/", json={"channel_id": "mediasiteadmin-mediasiteadmin"})

    assert response.status_code == 400
    assert "Channel imports require a real Mediasite channel/catalog GUID" in response.json()["detail"]["message"]


def test_api_empty_context_messages(monkeypatch):
    main.sessions.clear()
    monkeypatch.setattr(main, "qwen_llm", FakeLLM())
    client = TestClient(main.app)

    assert client.post("/chat/", json={"prompt": "hello"}).json()["response"] == (
        "Upload study files first so I can answer from your material."
    )
    assert client.post("/quiz/", json={}).json()["quiz"] == []
    assert client.post("/summarize/", json={}).json()["summary"] == (
        "Upload study files first so I can summarize your material."
    )
    assert client.post("/video-link/", json={}).json()["links"] == []


def test_upload_skips_empty_plain_file_and_chat_handles_empty_retrieval(monkeypatch):
    main.sessions.clear()
    monkeypatch.setattr(main, "qwen_llm", FakeLLM())
    client = TestClient(main.app)

    upload = client.post(
        "/upload/",
        data={"session_id": "existing"},
        files=[("files", ("empty.txt", b"   ", "text/plain"))],
    )
    assert upload.json()["session_id"] == "existing"
    assert upload.json()["skipped_files"] == ["empty.txt"]

    main.sessions["with-chunks"] = {
        "content": [],
        "chunks": [{"id": "1", "filename": "x", "chunk_index": 0, "text": "x"}],
    }
    monkeypatch.setattr(main, "retrieve_study_context", lambda *_args: ([], "answer"))
    chat = client.post("/chat/", json={"session_id": "with-chunks", "prompt": "missing"})
    assert chat.json()["response"].startswith("I could not find relevant uploaded material")


def test_sessions_persist_to_short_lived_disk_store(monkeypatch):
    monkeypatch.setattr(main, "qwen_llm", FakeLLM())
    client = TestClient(main.app)

    upload = client.post(
        "/upload/",
        files=[("files", ("notes.txt", b"Cells divide during mitosis.", "text/plain"))],
    )
    session_id = upload.json()["session_id"]
    chat = client.post("/chat/", json={"session_id": session_id, "prompt": "Explain mitosis"})
    assert chat.json()["session_id"] == session_id
    assert (main.SESSION_STORE_DIR / f"{session_id}.json").exists()

    main.sessions.clear()
    restored = client.get(f"/session/{session_id}").json()
    assert restored["documents"][0]["filename"] == "notes.txt"
    assert [message["sender"] for message in restored["chat_history"]] == ["user", "ai"]
    assert client.post("/search/", json={"session_id": session_id, "query": "mitosis"}).json()["results"]

    main.sessions[session_id]["expires_at"] = main.session_now() - 1
    main.persist_session(session_id)
    main.sessions.clear()
    expired = client.get(f"/session/{session_id}").json()
    assert expired["documents"] == []
    assert expired["chat_history"] == []


@pytest.mark.anyio
async def test_lifespan_preloads_real_model_when_requested(monkeypatch):
    class LoadingLLM(FakeLLM):
        use_mock_model = False

        def __init__(self):
            super().__init__()
            self.loaded = False

        def _load_model(self):
            self.loaded = True

    fake_llm = LoadingLLM()
    monkeypatch.setattr(main, "qwen_llm", fake_llm)
    monkeypatch.setenv("ATHENAI_PRELOAD_MODEL", "1")

    async with main.lifespan(main.app):
        assert fake_llm.loaded is True
