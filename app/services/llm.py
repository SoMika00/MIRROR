"""
LLM service using xAI Grok API (OpenAI-compatible).

Budget: $0.50/day
Pricing (grok-2):
  - Input:  $2  / 1M tokens
  - Output: $10 / 1M tokens

Daily limits (for $0.50/day):
  - Max input tokens:  100,000  ($0.20)
  - Max output tokens:  30,000  ($0.30)
  - Total budget:                $0.50
"""

import json
import time
import logging
import threading
import os
from datetime import date
from typing import Optional, Generator
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

# --- Token budget tracking ---
DAILY_BUDGET_FILE = Path("./data/token_budget.json")
# Pricing per token
INPUT_COST_PER_TOKEN = 2.0 / 1_000_000    # $2/1M
OUTPUT_COST_PER_TOKEN = 10.0 / 1_000_000   # $10/1M
DAILY_BUDGET_USD = float(os.environ.get("GROK_DAILY_BUDGET", "0.50"))
# Hard limits (safety net)
MAX_INPUT_TOKENS_DAY = int(DAILY_BUDGET_USD / INPUT_COST_PER_TOKEN * 0.4)   # ~100K
MAX_OUTPUT_TOKENS_DAY = int(DAILY_BUDGET_USD / OUTPUT_COST_PER_TOKEN * 0.6)  # ~30K


def _load_budget() -> dict:
    """Load today's token usage from disk."""
    today = date.today().isoformat()
    try:
        if DAILY_BUDGET_FILE.exists():
            data = json.loads(DAILY_BUDGET_FILE.read_text())
            if data.get("date") == today:
                return data
    except Exception:
        pass
    return {"date": today, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "requests": 0}


def _save_budget(budget: dict):
    """Persist token usage to disk."""
    try:
        DAILY_BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAILY_BUDGET_FILE.write_text(json.dumps(budget, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save budget: {e}")


class LLMService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._client = None
        self._model = os.environ.get("GROK_MODEL", "grok-2")
        self._api_key = os.environ.get("GROK_API_KEY", "")
        self._base_url = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
        self._gen_lock = threading.Lock()
        self._budget_lock = threading.Lock()
        self._current_n_ctx = 131072  # Grok-2 supports 131K context
        self._loaded = False
        if self._api_key:
            self._init_client()

    def _init_client(self):
        """Initialize the OpenAI-compatible client for xAI."""
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self._loaded = True
        logger.info(f"Grok API client initialized (model: {self._model})")

    def _check_budget(self, estimated_input: int = 0) -> bool:
        """Check if we're within daily budget. Raises if over limit."""
        with self._budget_lock:
            budget = _load_budget()
            if budget["input_tokens"] + estimated_input > MAX_INPUT_TOKENS_DAY:
                raise RuntimeError(
                    f"Daily input token limit reached ({budget['input_tokens']}/{MAX_INPUT_TOKENS_DAY}). "
                    f"Budget: ${budget['cost_usd']:.3f}/${DAILY_BUDGET_USD}. Resets tomorrow."
                )
            if budget["output_tokens"] > MAX_OUTPUT_TOKENS_DAY:
                raise RuntimeError(
                    f"Daily output token limit reached ({budget['output_tokens']}/{MAX_OUTPUT_TOKENS_DAY}). "
                    f"Budget: ${budget['cost_usd']:.3f}/${DAILY_BUDGET_USD}. Resets tomorrow."
                )
            if budget["cost_usd"] >= DAILY_BUDGET_USD:
                raise RuntimeError(
                    f"Daily budget exhausted: ${budget['cost_usd']:.3f}/${DAILY_BUDGET_USD}. Resets tomorrow."
                )
        return True

    def _record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage and cost."""
        with self._budget_lock:
            budget = _load_budget()
            budget["input_tokens"] += input_tokens
            budget["output_tokens"] += output_tokens
            cost = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)
            budget["cost_usd"] = round(budget["cost_usd"] + cost, 6)
            budget["requests"] += 1
            _save_budget(budget)
            logger.info(
                f"Tokens: +{input_tokens}in/+{output_tokens}out | "
                f"Day total: {budget['input_tokens']}in/{budget['output_tokens']}out | "
                f"Cost: ${budget['cost_usd']:.4f}/${DAILY_BUDGET_USD}"
            )

    def load(self, **kwargs):
        """No-op for API mode. Kept for interface compatibility."""
        if not self._api_key:
            self._api_key = os.environ.get("GROK_API_KEY", "")
        if self._api_key and not self._client:
            self._init_client()
        if not self._api_key:
            raise RuntimeError("GROK_API_KEY not set. Add it to .env file.")

    def is_loaded(self) -> bool:
        return self._loaded and bool(self._api_key)

    def unload(self):
        """No-op for API mode."""
        pass

    def generate(self, prompt: str = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stream: bool = False,
                 messages: Optional[list] = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Grok API not configured. Set GROK_API_KEY in .env.")

        chat_messages = messages if messages else [{"role": "user", "content": prompt}]

        # Estimate input tokens (~3 chars per token)
        input_chars = sum(len(m.get("content", "")) for m in chat_messages)
        estimated_input = input_chars // 3
        self._check_budget(estimated_input)

        effective_max = min(max_tokens or 1024, 2048)

        with self._gen_lock:
            start = time.time()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=chat_messages,
                max_tokens=effective_max,
                temperature=temperature or 0.7,
                stream=False,
            )
            elapsed = time.time() - start

            text = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else estimated_input
            output_tokens = usage.completion_tokens if usage else len(text) // 4

            self._record_usage(input_tokens, output_tokens)
            logger.info(f"Generated {output_tokens} tokens in {elapsed:.1f}s")

            return text

    def generate_stream(self, prompt: str = None, max_tokens: Optional[int] = None,
                        temperature: Optional[float] = None,
                        messages: Optional[list] = None) -> Generator[str, None, None]:
        if not self.is_loaded():
            raise RuntimeError("Grok API not configured. Set GROK_API_KEY in .env.")

        chat_messages = messages if messages else [{"role": "user", "content": prompt}]

        input_chars = sum(len(m.get("content", "")) for m in chat_messages)
        estimated_input = input_chars // 3
        self._check_budget(estimated_input)

        effective_max = min(max_tokens or 1024, 2048)

        with self._gen_lock:
            start = time.time()
            token_count = 0

            stream = self._client.chat.completions.create(
                model=self._model,
                messages=chat_messages,
                max_tokens=effective_max,
                temperature=temperature or 0.7,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content if delta else None
                if content:
                    token_count += 1
                    yield content

            elapsed = time.time() - start
            # Record usage (estimate input since streaming doesn't return usage)
            self._record_usage(estimated_input, token_count)
            if token_count > 0:
                logger.info(f"Streamed {token_count} tokens in {elapsed:.1f}s")

    def get_info(self) -> dict:
        budget = _load_budget()
        return {
            "provider": "xAI Grok API",
            "model": self._model,
            "loaded": self.is_loaded(),
            "n_ctx": self._current_n_ctx,
            "daily_budget": {
                "limit_usd": DAILY_BUDGET_USD,
                "spent_usd": budget["cost_usd"],
                "remaining_usd": round(DAILY_BUDGET_USD - budget["cost_usd"], 4),
                "input_tokens": budget["input_tokens"],
                "output_tokens": budget["output_tokens"],
                "requests_today": budget["requests"],
                "max_input_tokens": MAX_INPUT_TOKENS_DAY,
                "max_output_tokens": MAX_OUTPUT_TOKENS_DAY,
            },
        }

    def get_budget(self) -> dict:
        """Public method to get current budget status."""
        return _load_budget()


llm_service = LLMService()
