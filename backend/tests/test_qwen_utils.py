import types

import pytest

import qwen_utils
from qwen_utils import QwenLLM


class FakeOutputRow(list):
    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(item, slice):
            return FakeOutputRow(value)
        return value

    def tolist(self):
        return list(self)


class FakeInputs(dict):
    def __init__(self):
        self.input_ids = [[1, 2, 3]]
        super().__init__({"input_ids": self.input_ids})

    def to(self, _device):
        return self


class FakeTokenizer:
    eos_token_id = 99
    decode_responses = ["<think>ignore</think> Clean answer"]
    decoded = []
    templates = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_args = (args, kwargs)
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        self.__class__.templates.append((messages, kwargs))
        return "templated chat"

    def __call__(self, texts, return_tensors):
        self.last_call = (texts, return_tensors)
        return FakeInputs()

    def decode(self, output_ids, skip_special_tokens=True):
        self.__class__.decoded.append((output_ids, skip_special_tokens))
        return self.__class__.decode_responses.pop(0)


class FakeModel:
    generated = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_args = (args, kwargs)
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True

    def generate(self, **kwargs):
        self.__class__.generated.append(kwargs)
        return [FakeOutputRow([1, 2, 3, 7, 8])]


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def install_fake_generation_stack(monkeypatch, cuda_available=True):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        float16="float16",
        no_grad=lambda: FakeNoGrad(),
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeModel,
        AutoTokenizer=FakeTokenizer,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)


def test_environment_configuration_and_generation_settings(monkeypatch):
    monkeypatch.setenv("ATHENAI_QWEN_MODEL", "local-model")
    monkeypatch.setenv("ATHENAI_QWEN_DEVICE", "cpu")
    monkeypatch.setenv("ATHENAI_MAX_NEW_TOKENS", "33")
    monkeypatch.setenv("ATHENAI_LONG_TASK_MAX_NEW_TOKENS", "44")
    monkeypatch.setenv("ATHENAI_TEMPERATURE", "0.8")
    monkeypatch.setenv("ATHENAI_TOP_P", "0.95")
    monkeypatch.setenv("ATHENAI_TOP_K", "12")
    monkeypatch.setenv("ATHENAI_MOCK_LLM", "1")
    monkeypatch.setenv("ATHENAI_REQUIRE_CUDA", "0")

    llm = QwenLLM()

    assert llm.model_path == "local-model"
    assert llm.device == "cpu"
    assert llm.use_mock_model is True
    assert llm.require_cuda is False
    assert llm._estimate_tokens("") == 1
    assert llm._estimate_tokens("one two") == 2
    assert llm._max_new_tokens_for_task("answer") == 33
    assert llm._max_new_tokens_for_task("quiz") == 44
    assert llm._generation_settings_for_task("quiz") == {
        "do_sample": True,
        "temperature": 0.35,
        "top_p": 0.85,
        "top_k": 12,
    }
    assert llm._generation_settings_for_task("answer")["temperature"] == 0.8


@pytest.mark.parametrize(
    ("response", "needs_retry"),
    [
        ("", True),
        ("Question 1\nA) one\nB) two\nC) three\nD) four", True),
        ("1. What?\nA) one\nAnswer key\n1. A", True),
        (
            "\n".join(
                [
                    "Question 1: What is alpha?",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Question 2: True or False: Beta follows alpha.",
                    "True) True",
                    "False) False",
                    "Question 3: Write an answer explaining gamma.",
                    "Answer key",
                    "1. A) a because [1]",
                    "2. True because [1]",
                    "3. Gamma is expected because [1]",
                ]
            ),
            False,
        ),
        (
            "\n".join(
                [
                    "Question 1: True or False: Beta follows alpha.",
                    "True) True",
                    "Answer key",
                    "1. True because [1]",
                ]
            ),
            True,
        ),
        (
            "\n".join(
                [
                    "Question 1: What is alpha?",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Answer Key:",
                    "Question 1: What is alpha?",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Answer: A) a",
                    "Explanation: Alpha is supported. [1]",
                    "Question 2: What is beta?",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                ]
            ),
            True,
        ),
        (
            "\n".join(
                [
                    "Question 1: What is alpha?",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Answer Key",
                    "1. A) a - Alpha is supported. [1]",
                    "Answer Key Summary:",
                    "Question 1: What is alpha?",
                    "Answer: A) a",
                ]
            ),
            True,
        ),
        (
            "\n".join(
                [
                    "Question 1: What is alpha?",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Answer key",
                    "1. A and B because [1]",
                ]
            ),
            True,
        ),
        (
            "\n".join(
                [
                    "Question 1",
                    "A) a",
                    "B) b",
                    "Question 2",
                    "A) a",
                    "B) b",
                    "Question 3",
                    "A) a",
                    "B) b",
                    "Question 4",
                    "A) a",
                    "B) b",
                    "Answer key",
                    "1. A",
                    "2. B",
                    "3. C",
                    "4. D",
                ]
            ),
            True,
        ),
        (
            "\n".join(
                [
                    "Question 1",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Question 2",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Question 3",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Question 4",
                    "A) a",
                    "B) b",
                    "C) c",
                    "D) d",
                    "Answer key",
                    "1. A",
                    "2. A",
                    "3. A",
                    "4. A",
                ]
            ),
            True,
        ),
    ],
)
def test_quiz_retry_detection(response, needs_retry):
    assert QwenLLM()._quiz_response_needs_retry(response) is needs_retry


def test_build_messages_formats_context_and_quiz_requirements(monkeypatch):
    monkeypatch.setenv("ATHENAI_MOCK_LLM", "1")
    llm = QwenLLM()
    context = [
        {"filename": "notes.txt", "chunk_index": 2, "text": "cell division"},
        "loose source",
    ]

    messages = llm._build_messages("Quiz me", context, "quiz")
    assert messages[0]["role"] == "system"
    user_prompt = messages[1]["content"]
    assert "[1] notes.txt chunk 2" in user_prompt
    assert "[2] source 2" in user_prompt
    assert "Quiz quality requirements" in user_prompt
    assert "Do not include an answer key after each question" in user_prompt
    assert "must not repeat question text or list A-D choices" in user_prompt
    assert "Question: Quiz me" in user_prompt

    retry = llm._build_quiz_retry_messages("Quiz me", context, "bad draft")
    assert retry[-2] == {"role": "assistant", "content": "bad draft"}
    assert "Repair the quiz" in retry[-1]["content"]
    assert "must not repeat full question text or A-D answer choices" in retry[-1]["content"]


def test_build_messages_formats_flashcard_requirements(monkeypatch):
    monkeypatch.setenv("ATHENAI_MOCK_LLM", "1")
    llm = QwenLLM()

    messages = llm._build_messages(
        "Create flashcards",
        [{"filename": "notes.txt", "chunk_index": 0, "text": "Mitosis divides cells."}],
        "flashcards",
    )

    user_prompt = messages[1]["content"]
    assert "Create a flashcard set from the excerpts" in user_prompt
    assert "Card N" in user_prompt
    assert "Front: <question or term>" in user_prompt
    assert "Back: <concise answer, definition, or explanation> [source]" in user_prompt
    assert "Question: Create flashcards" in user_prompt


def test_mock_chat_and_estimated_usage(monkeypatch):
    monkeypatch.setenv("ATHENAI_MOCK_LLM", "1")
    monkeypatch.setenv("ATHENAI_LONG_TASK_MAX_NEW_TOKENS", "99")
    llm = QwenLLM()

    context = [{"filename": "b.txt", "text": "beta"}, {"filename": "a.txt", "text": "alpha"}]
    result = llm.chat_with_usage("Summarize", context=context, study_task="summary")

    assert "2 source chunk(s) from a.txt, b.txt" in result["response"]
    assert result["usage"]["estimated"] is True
    assert result["usage"]["max_new_tokens"] == 99
    assert result["usage"]["hit_token_limit"] is False
    assert llm.chat("No files", context=[], study_task="answer").startswith("Mock response")
    assert llm.estimate_usage("one two", "three")["total_tokens"] >= 3


def test_load_model_requires_cuda_when_unavailable(monkeypatch):
    monkeypatch.setenv("ATHENAI_REQUIRE_CUDA", "1")

    install_fake_generation_stack(monkeypatch, cuda_available=False)
    llm = QwenLLM()

    with pytest.raises(RuntimeError, match="CUDA is required"):
        llm._load_model()


def test_real_generation_branch_loads_model_tracks_usage_and_repairs_quiz(monkeypatch):
    monkeypatch.delenv("ATHENAI_MOCK_LLM", raising=False)
    monkeypatch.delenv("ATHENAI_USE_REAL_QWEN", raising=False)
    monkeypatch.setenv("ATHENAI_REQUIRE_CUDA", "0")
    monkeypatch.setenv("ATHENAI_MAX_NEW_TOKENS", "2")
    monkeypatch.setenv("ATHENAI_LONG_TASK_MAX_NEW_TOKENS", "2")
    install_fake_generation_stack(monkeypatch)
    FakeTokenizer.decode_responses = [
        "<think>ignore</think> Clean answer",
        "Question 1\nA) one",
        "Question 1\nA) one\nB) two\nC) three\nD) four\nAnswer key\n1. B two because [1]",
    ]
    FakeTokenizer.decoded = []
    FakeTokenizer.templates = []
    FakeModel.generated = []

    llm = QwenLLM(model_path="remote/model")
    answer = llm.chat_with_usage("Explain", context=[{"filename": "notes.txt", "text": "facts"}])

    assert answer["response"] == "ignore Clean answer"
    assert answer["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "max_new_tokens": 2,
        "hit_token_limit": True,
        "retried_for_quality": False,
        "estimated": False,
    }
    assert llm.model is not None
    assert llm.tokenizer is not None
    assert FakeModel.from_pretrained_args[1]["torch_dtype"] == "float16"
    assert FakeTokenizer.from_pretrained_args[1]["local_files_only"] is False

    quiz = llm.chat_with_usage("Quiz me", context=[{"filename": "notes.txt", "text": "facts"}], study_task="quiz")

    assert quiz["response"].startswith("Question 1")
    assert quiz["usage"]["retried_for_quality"] is True
    assert len(FakeModel.generated) == 3
    assert "Repair the quiz" in FakeTokenizer.templates[-1][0][-1]["content"]
