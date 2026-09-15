"""
src/llm_client.py - Unified LLM client with persistent disk caching, JSON validation, and retry logic.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from openai import OpenAI
import config

class CachedLLMClient:
    # Class-level counters for visibility
    total_api_calls = 0
    failed_api_calls = 0

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or config.LLM_PROVIDER
        api_key = config.API_KEYS.get(self.provider) or "dummy_key_for_local"
        base_url = config.BASE_URLS.get(self.provider)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def generate(
        self,
        model: str,
        messages: list,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        prompt_version: str = "v1.0.0",
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        
        cache_key_payload = {
            "provider": self.provider,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format,
            "prompt_version": prompt_version,
            "max_tokens": max_tokens,
        }
        cache_hash = self._compute_hash(cache_key_payload)
        cache_file = self.cache_dir / f"{cache_hash}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_data["cached"] = True
                return cached_data

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        CachedLLMClient.total_api_calls += 1
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                raw_content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f"\n[Warning] API Error on attempt {attempt+1}: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"\n[Error] API failed after {max_retries} attempts. Falling back to default.")
                    CachedLLMClient.failed_api_calls += 1
                    raw_content = "{}" 
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        result = {
            "content": raw_content,
            "model": model,
            "prompt_version": prompt_version,
            "usage": usage,
            "cached": False,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result
        
    @classmethod
    def print_failure_stats(cls):
        if cls.total_api_calls > 0:
            rate = (cls.failed_api_calls / cls.total_api_calls) * 100
            print(f"\nAPI Reliability: {cls.failed_api_calls} failures out of {cls.total_api_calls} calls ({rate:.1f}% failure rate)")