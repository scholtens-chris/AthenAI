# Qwen LLM integration utilities
# This is a placeholder for Qwen model loading and inference logic.
# Update with actual Qwen model code as needed.

import os
import re
import threading
from typing import Optional

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_LONG_TASK_MAX_NEW_TOKENS = 2048

TASK_INSTRUCTIONS = {
    "quiz": (
        "Create a multiple-choice study quiz from the excerpts. Include 6-8 questions when enough "
        "material exists. Every question must have exactly four answer options labeled A), B), C), "
        "and D). Only one option may be correct. Distractors must be plausible but clearly wrong "
        "based on the excerpts. Put a complete answer key after the questions with the correct "
        "letter, the option text, a one-sentence explanation, and source citations. Do not make "
        "every answer the same letter."
    ),
    "summary": (
        "Create a structured summary artifact. If the user names a chapter, section, or topic, "
        "focus there. Use short headings, 4-8 substantive bullets, and a final 'What to remember' "
        "line. Cite the supporting source numbers."
    ),
    "find": (
        "Help the student locate the explanation. Lead with the most likely source filename and "
        "chunk/source number, include the exact topic wording found in the excerpts, and explain "
        "why that source matches. If timestamps or chapter markers appear in the excerpt, preserve them."
    ),
    "takeaways": (
        "Extract key takeaways as an artifact. Provide 4-6 developed bullets, each with a title, "
        "a plain-English explanation, and why it matters for studying. Cite source numbers."
    ),
    "study_guide": (
        "Create a study guide using this exact section structure: "
        "Main Ideas, Key Concepts, or Learning Objectives; Core Content; Important Diagrams; "
        "Things to Review; Extra Learning Tools. Under Main Ideas, Key Concepts, or Learning "
        "Objectives, list Key Concept 1, Key Concept 2, Key Concept 3, and add more key concepts "
        "when the excerpts support them. Under Core Content, for each key concept, include a "
        "student self-recall prompt ('Write down everything you remember about the concept before "
        "looking at your notes.'), then source-grounded notes the student should add in a different "
        "color, and repeat for each learning objective covered. Under Important Diagrams, name or "
        "describe diagrams that would help, explain which key concepts each diagram illustrates, "
        "and repeat for all important diagrams supported by class material. Under Things to Review, "
        "make a checklist of confusing or high-priority concepts to revisit. Under Extra Learning "
        "Tools, include Flashcards, Mind maps, and Practice quiz questions. Cite source numbers for "
        "source-grounded notes."
    ),
    "explain": (
        "Explain the material like a tutor. Start with the big idea, then break it into steps or "
        "concepts, define important terms, connect causes and effects, and end with a quick check "
        "for understanding. Cite source numbers for claims."
    ),
    "answer": (
        "Answer the student's request directly. Adapt the format to the request, and cite source "
        "numbers for claims supported by the excerpts."
    ),
}


class QwenLLM:
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or os.getenv("ATHENAI_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        self.device = device or os.getenv("ATHENAI_QWEN_DEVICE")
        self.max_new_tokens = int(os.getenv("ATHENAI_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
        self.long_task_max_new_tokens = int(
            os.getenv(
                "ATHENAI_LONG_TASK_MAX_NEW_TOKENS",
                str(max(self.max_new_tokens, DEFAULT_LONG_TASK_MAX_NEW_TOKENS)),
            )
        )
        self.temperature = float(os.getenv("ATHENAI_TEMPERATURE", "0.55"))
        self.top_p = float(os.getenv("ATHENAI_TOP_P", "0.9"))
        self.top_k = int(os.getenv("ATHENAI_TOP_K", "40"))
        self.use_mock_model = os.getenv("ATHENAI_MOCK_LLM") == "1" or os.getenv("ATHENAI_USE_REAL_QWEN") == "0"
        self.require_cuda = os.getenv("ATHENAI_REQUIRE_CUDA", "1") == "1"
        self.model = None
        self.tokenizer = None
        self._load_lock = threading.Lock()

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(re.findall(r"\S+", text or "")))

    def _max_new_tokens_for_task(self, study_task: str) -> int:
        if study_task in {"quiz", "summary", "takeaways", "study_guide", "explain"}:
            return self.long_task_max_new_tokens
        return self.max_new_tokens

    def _generation_settings_for_task(self, study_task: str) -> dict:
        if study_task == "quiz":
            return {
                "do_sample": True,
                "temperature": min(self.temperature, 0.35),
                "top_p": min(self.top_p, 0.85),
                "top_k": self.top_k,
            }
        return {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def _quiz_response_needs_retry(self, response: str) -> bool:
        if not response:
            return True

        question_count = len(re.findall(r"(?im)^\s*(?:question\s*)?\d+[\).:]", response))
        if question_count == 0:
            question_count = len(re.findall(r"(?im)^\s*question\s+\d+", response))

        option_count = len(re.findall(r"(?im)^\s*[A-D][\).]\s+\S+", response))
        if question_count >= 4 and option_count < question_count * 4:
            return True

        answer_key_match = re.search(r"(?is)\banswer\s+key\b(.+)$", response)
        if not answer_key_match:
            return True

        key_letters = re.findall(r"(?im)^\s*(?:question\s*)?\d+[\).:\s-]+([A-D])\b", answer_key_match.group(1))
        if len(key_letters) >= 4 and len(set(key_letters)) == 1:
            return True

        return False

    def _build_messages(self, prompt: str, context: Optional[list], study_task: str) -> list[dict]:
        context_text = ""
        if context:
            formatted_context = []
            for index, item in enumerate(context, start=1):
                if isinstance(item, dict):
                    label = f"{item.get('filename', 'source')} chunk {item.get('chunk_index', index - 1)}"
                    text = item.get("text", "")
                else:
                    label = f"source {index}"
                    text = str(item)
                formatted_context.append(f"[{index}] {label}\n{text}")
            context_text = "\n\n".join(formatted_context)

        task_instruction = TASK_INSTRUCTIONS.get(study_task, TASK_INSTRUCTIONS["answer"])

        system_prompt = (
            "You are AthenAI, a careful RAG-first study assistant. Use only the retrieved "
            "source excerpts provided by the backend, and cite source numbers like [1] for "
            "specific claims. Synthesize across excerpts instead of merely copying them. "
            "When the material supports it, give a substantive answer with reasoning, nuance, "
            "relationships between ideas, and practical implications for studying. You can "
            "generate useful study artifacts such as quizzes, study guides, summaries, answer "
            "keys, location notes, and key-takeaway lists when the user asks for them. If the "
            "excerpts do not contain enough evidence, say what is missing from the uploaded "
            "material instead of guessing. Do not mention backend implementation details."
        )
        user_prompt = (
            "Retrieved source excerpts:\n"
            f"{context_text}\n\n"
            "Answer expectations:\n"
            "- Start with the direct answer.\n"
            f"- Task-specific format: {task_instruction}\n"
            "- For broad requests, create a complete, useful study artifact rather than asking the user to narrow the prompt.\n"
            "- Use clear headings and readable spacing when the answer is more than one paragraph.\n"
            "- Each bullet should include a brief explanation, not just a headline.\n"
            "- Use citations where they support the point.\n"
            "- End with a short synthesis when useful.\n\n"
            f"Question: {prompt}"
        )
        if study_task == "quiz":
            user_prompt += (
                "\n\nQuiz quality requirements:\n"
                "- Format each question exactly as 'Question N' followed by the question text.\n"
                "- Under every question, include four options on separate lines: A), B), C), and D).\n"
                "- Do not omit choices. Do not write short-answer-only quiz questions.\n"
                "- In the answer key, include each question number, the correct letter, the full option text, "
                "a brief explanation, and the supporting source citation.\n"
                "- Before finalizing, verify that every question has A-D choices and that the answer key "
                "matches those choices."
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_quiz_retry_messages(self, prompt: str, context: Optional[list], previous_response: str) -> list[dict]:
        messages = self._build_messages(prompt, context, "quiz")
        messages.append({
            "role": "assistant",
            "content": previous_response,
        })
        messages.append({
            "role": "user",
            "content": (
                "Repair the quiz. The previous draft was invalid because it was missing complete A-D "
                "options and/or had a broken answer key. Return only the corrected quiz. Every question "
                "must have exactly A), B), C), and D) choices. The answer key must not use the same "
                "letter for every question unless the source material truly forces that, and each key "
                "entry must include the full correct option text, explanation, and source citation."
            ),
        })
        return messages

    def _mock_chat(self, prompt: str, context: Optional[list], study_task: str) -> dict:
        context_count = len(context or [])
        if context_count:
            filenames = sorted({item.get("filename", "uploaded source") for item in context if isinstance(item, dict)})
            source_text = ", ".join(filenames[:3]) if filenames else "uploaded material"
            task_label = study_task if study_task in TASK_INSTRUCTIONS else "answer"
            response = (
                f"Mock response: I found {context_count} source chunk(s) from {source_text} "
                f"and would generate a {task_label} artifact grounded only in that material."
            )
        else:
            response = "Mock response: upload study files first so retrieval can ground the answer."

        messages = self._build_messages(prompt, context, study_task)
        prompt_tokens = sum(self._estimate_tokens(message["content"]) for message in messages)
        completion_tokens = self._estimate_tokens(response)
        return {
            "response": response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
            },
        }

    def estimate_usage(self, prompt: str, response: str, context: Optional[list] = None, study_task: str = "answer") -> dict:
        messages = self._build_messages(prompt, context, study_task)
        prompt_tokens = sum(self._estimate_tokens(message["content"]) for message in messages)
        completion_tokens = self._estimate_tokens(response)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        }

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self._load_lock:
            if self.model is not None and self.tokenizer is not None:
                return

            if self.require_cuda and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required for AthenAI real-model mode, but PyTorch cannot see a CUDA GPU."
                )

            self.device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=os.path.isdir(self.model_path),
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if str(self.device).startswith("cuda") else "auto",
                low_cpu_mem_usage=True,
                local_files_only=os.path.isdir(self.model_path),
            ).to(self.device)
            self.model.eval()

    def chat(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        context: Optional[list] = None,
        use_internet: bool = False,
        study_task: str = "answer",
    ) -> str:
        return self.chat_with_usage(
            prompt,
            image=image,
            context=context,
            use_internet=use_internet,
            study_task=study_task,
        )["response"]

    def chat_with_usage(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        context: Optional[list] = None,
        use_internet: bool = False,
        study_task: str = "answer",
    ) -> dict:
        import torch

        if self.use_mock_model:
            result = self._mock_chat(prompt, context, study_task)
            result["usage"]["max_new_tokens"] = self._max_new_tokens_for_task(study_task)
            result["usage"]["hit_token_limit"] = False
            return result

        if self.model is None or self.tokenizer is None:
            self._load_model()

        # For now, only text input is supported. Image input and internet search are placeholders.
        messages = self._build_messages(prompt, context, study_task)
        retry_messages = None
        if study_task == "quiz":
            retry_messages = self._build_quiz_retry_messages
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.device)
        prompt_tokens = len(inputs.input_ids[0])
        max_new_tokens = self._max_new_tokens_for_task(study_task)
        generation_settings = self._generation_settings_for_task(study_task)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generation_settings,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )
        response = response.replace("<think>", "").replace("</think>", "").strip() or "[No response]"
        completion_tokens = len(output_ids)
        hit_token_limit = completion_tokens >= max_new_tokens and (
            not output_ids or output_ids[-1] != self.tokenizer.eos_token_id
        )
        retried_for_quality = False

        if retry_messages and self._quiz_response_needs_retry(response):
            messages = retry_messages(prompt, context, response)
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.device)
            prompt_tokens = len(inputs.input_ids[0])
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generation_settings,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
            response = response.replace("<think>", "").replace("</think>", "").strip() or "[No response]"
            completion_tokens = len(output_ids)
            hit_token_limit = completion_tokens >= max_new_tokens and (
                not output_ids or output_ids[-1] != self.tokenizer.eos_token_id
            )
            retried_for_quality = True

        return {
            "response": response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "max_new_tokens": max_new_tokens,
                "hit_token_limit": hit_token_limit,
                "retried_for_quality": retried_for_quality,
                "estimated": False,
            },
        }
