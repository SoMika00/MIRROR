"""
LLM service - xAI Grok API. No model weights run on this machine.

Cost control: a persisted daily budget (default $0.50/day) caps spend.
Pricing is configurable via env (GROK_INPUT_PRICE / GROK_OUTPUT_PRICE,
USD per 1M tokens) so a model change never silently breaks accounting.
"""

import json
import logging
import threading
import time
from datetime import date
from pathlib import Path
from typing import Generator, Optional

from openai import OpenAI

from app.config import grok_cfg

logger = logging.getLogger(__name__)

DAILY_BUDGET_FILE = Path("./data/token_budget.json")

INPUT_COST_PER_TOKEN = grok_cfg.input_price_per_m / 1_000_000
OUTPUT_COST_PER_TOKEN = grok_cfg.output_price_per_m / 1_000_000
DAILY_BUDGET_USD = grok_cfg.daily_budget_usd


def _load_budget() -> dict:
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
        self._model = grok_cfg.model
        self._api_key = grok_cfg.api_key
        self._base_url = grok_cfg.base_url
        self._budget_lock = threading.Lock()
        self._loaded = False
        if self._api_key:
            self._init_client()

    def _init_client(self):
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self._loaded = True
        logger.info(f"Grok API client initialized (model: {self._model})")

    def _check_budget(self, estimated_input: int = 0):
        with self._budget_lock:
            budget = _load_budget()
            projected = budget["cost_usd"] + estimated_input * INPUT_COST_PER_TOKEN
            if projected >= DAILY_BUDGET_USD:
                raise RuntimeError(
                    f"Daily budget exhausted (${budget['cost_usd']:.3f}/${DAILY_BUDGET_USD}). "
                    "The assistant will be back tomorrow."
                )

    def _record_usage(self, input_tokens: int, output_tokens: int):
        with self._budget_lock:
            budget = _load_budget()
            budget["input_tokens"] += input_tokens
            budget["output_tokens"] += output_tokens
            cost = input_tokens * INPUT_COST_PER_TOKEN + output_tokens * OUTPUT_COST_PER_TOKEN
            budget["cost_usd"] = round(budget["cost_usd"] + cost, 6)
            budget["requests"] += 1
            _save_budget(budget)
            logger.info(
                f"Tokens: +{input_tokens}in/+{output_tokens}out | "
                f"Day: {budget['input_tokens']}in/{budget['output_tokens']}out | "
                f"${budget['cost_usd']:.4f}/${DAILY_BUDGET_USD}"
            )

    def load(self, **kwargs):
        if not self._api_key:
            raise RuntimeError("GROK_API_KEY not set. Add it to the .env file.")
        if not self._client:
            self._init_client()

    def is_loaded(self) -> bool:
        return self._loaded and bool(self._api_key)

    def generate(self, prompt: str = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 messages: Optional[list] = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Grok API not configured. Set GROK_API_KEY in .env.")

        chat_messages = messages if messages else [{"role": "user", "content": prompt}]
        estimated_input = sum(len(m.get("content", "")) for m in chat_messages) // 3
        self._check_budget(estimated_input)

        start = time.time()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=chat_messages,
            max_tokens=min(max_tokens or grok_cfg.max_tokens, 2048),
            temperature=temperature if temperature is not None else grok_cfg.temperature,
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
        estimated_input = sum(len(m.get("content", "")) for m in chat_messages) // 3
        self._check_budget(estimated_input)

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=chat_messages,
            max_tokens=min(max_tokens or grok_cfg.max_tokens, 2048),
            temperature=temperature if temperature is not None else grok_cfg.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_count = 0
        usage = None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content if delta else None
                if content:
                    chunk_count += 1
                    yield content

        input_tokens = usage.prompt_tokens if usage else estimated_input
        output_tokens = usage.completion_tokens if usage else chunk_count
        self._record_usage(input_tokens, output_tokens)

    def get_info(self) -> dict:
        budget = _load_budget()
        return {
            "provider": "xAI Grok API",
            "model": self._model,
            "loaded": self.is_loaded(),
            "daily_budget": {
                "limit_usd": DAILY_BUDGET_USD,
                "spent_usd": budget["cost_usd"],
                "remaining_usd": round(max(DAILY_BUDGET_USD - budget["cost_usd"], 0), 4),
                "input_tokens": budget["input_tokens"],
                "output_tokens": budget["output_tokens"],
                "requests_today": budget["requests"],
            },
        }

    def get_budget(self) -> dict:
        return _load_budget()


llm_service = LLMService()
