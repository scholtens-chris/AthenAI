@echo off
set ATHENAI_QWEN_MODEL=C:\Users\brasi\AthenAI\models\Qwen3-0.6B
set ATHENAI_MOCK_LLM=0
set ATHENAI_REQUIRE_CUDA=1
set ATHENAI_MAX_NEW_TOKENS=32
py -m uvicorn main:app --host 0.0.0.0 --port 8001
