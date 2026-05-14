# Qwen LLM integration utilities
# This is a placeholder for Qwen model loading and inference logic.
# Update with actual Qwen model code as needed.

import os
import threading
from typing import Optional

class QwenLLM:
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or os.getenv("ATHENAI_QWEN_MODEL", "Qwen/Qwen3-0.6B")
        self.device = device
        self.max_new_tokens = int(os.getenv("ATHENAI_MAX_NEW_TOKENS", "96"))
        self.use_mock_model = os.getenv("ATHENAI_MOCK_LLM") == "1" or os.getenv("ATHENAI_USE_REAL_QWEN") == "0"
        self.require_cuda = os.getenv("ATHENAI_REQUIRE_CUDA", "1") == "1"
        self.model = None
        self.tokenizer = None
        self._load_lock = threading.Lock()

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
                dtype=torch.float16 if str(self.device).startswith("cuda") else "auto",
                low_cpu_mem_usage=True,
                local_files_only=os.path.isdir(self.model_path),
            ).to(self.device)
            self.model.eval()

    def chat(self, prompt: str, image: Optional[bytes] = None, context: Optional[list] = None, use_internet: bool = False) -> str:
        import torch

        if self.use_mock_model:
            context_count = len(context or [])
            if context_count:
                return f"Mock response: I received your question and found {context_count} uploaded content item(s) available for context."
            return "Mock response: the API is running. Upload study files to add context, or enable the real Qwen model later."

        if self.model is None or self.tokenizer is None:
            self._load_model()

        # For now, only text input is supported. Image input and internet search are placeholders.
        full_prompt = prompt
        if context:
            # Optionally prepend context to the prompt
            context_text = "\n".join(context[-10:])  # Use last 10 items for context
            full_prompt = f"Context:\n{context_text}\n\nUser: {prompt}"
        messages = [{"role": "user", "content": full_prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )
        return response.replace("<think>", "").replace("</think>", "").strip() or "[No response]"
