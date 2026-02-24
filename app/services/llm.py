"""
LLM service using llama-cpp-python for CPU inference.

Phi-4 14B Q4_K_M chosen for:
  - Best quality/size ratio for 14B-class models on CPU
  - Q4_K_M quantization: ~9 GB RAM, good quality retention (perplexity < +0.5 vs FP16)
  - 12 threads → ~5-10 t/s generation on modern x86_64
  - 64 GB RAM leaves ~50 GB headroom after model + KV cache
  - Microsoft MIT license
"""

import gc
import time
import logging
import threading
import os
from typing import Optional, Generator

from app.config import llm_cfg, MODEL_REGISTRY, get_model_by_id, get_default_model

logger = logging.getLogger(__name__)


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
        self.model = None
        self.model_path = llm_cfg.model_path
        self._loaded = False
        self._gen_lock = threading.Lock()
        self._current_model_id = None
        self._current_n_ctx = llm_cfg.n_ctx

    def load(self, model_path: Optional[str] = None, model_id: Optional[str] = None):
        # Resolve model from registry if model_id is given
        if model_id:
            entry = get_model_by_id(model_id)
            if entry:
                model_path = f"./models/{entry['filename']}"
                self._current_model_id = model_id
                self._current_n_ctx = entry.get("n_ctx", llm_cfg.n_ctx)
            else:
                raise ValueError(f"Unknown model_id: {model_id}")
        elif model_path:
            self.model_path = model_path
        else:
            # No model_path and no model_id: use current model_path
            pass

        if model_path:
            self.model_path = model_path

        # Always try to match registry entry by filename
        fname = os.path.basename(self.model_path)
        matched = False
        for m in MODEL_REGISTRY:
            if m["filename"] == fname:
                self._current_model_id = m["id"]
                self._current_n_ctx = m.get("n_ctx", llm_cfg.n_ctx)
                matched = True
                break
        if not matched and not model_id:
            self._current_model_id = None
            self._current_n_ctx = llm_cfg.n_ctx

        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model not found at {self.model_path}. Download a GGUF model first.")
        logger.info(f"Loading LLM: {self.model_path}")
        start = time.time()
        from llama_cpp import Llama
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self._current_n_ctx,
            n_threads=llm_cfg.n_threads,
            n_batch=llm_cfg.n_batch,
            verbose=False,
        )
        elapsed = time.time() - start
        logger.info(f"LLM loaded in {elapsed:.1f}s")
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            self._loaded = False
            self._current_model_id = None
            gc.collect()
            logger.info("LLM unloaded, memory freed")

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stream: bool = False) -> str:
        if not self._loaded:
            raise RuntimeError("LLM not loaded. Load a model first via /api/models/load.")
        with self._gen_lock:
            start = time.time()
            response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or llm_cfg.max_tokens,
                temperature=temperature or llm_cfg.temperature,
                top_p=llm_cfg.top_p,
                repeat_penalty=llm_cfg.repeat_penalty,
                stream=False,
            )
            elapsed = time.time() - start
            text = response["choices"][0]["message"]["content"]
            tokens_used = response.get("usage", {}).get("completion_tokens", 0)
            if tokens_used and elapsed > 0:
                logger.info(f"Generated {tokens_used} tokens in {elapsed:.1f}s ({tokens_used/elapsed:.1f} t/s)")
            return text

    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                        temperature: Optional[float] = None) -> Generator[str, None, None]:
        if not self._loaded:
            raise RuntimeError("LLM not loaded.")
        with self._gen_lock:
            stream = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or llm_cfg.max_tokens,
                temperature=temperature or llm_cfg.temperature,
                top_p=llm_cfg.top_p,
                repeat_penalty=llm_cfg.repeat_penalty,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    def get_info(self) -> dict:
        info = {
            "model_path": self.model_path,
            "loaded": self._loaded,
            "n_ctx": self._current_n_ctx if hasattr(self, '_current_n_ctx') else llm_cfg.n_ctx,
            "n_threads": llm_cfg.n_threads,
            "model_id": getattr(self, '_current_model_id', None),
        }
        if self._loaded and self.model:
            info["n_vocab"] = self.model.n_vocab()
        # Enrich with registry info
        mid = info.get("model_id")
        if mid:
            entry = get_model_by_id(mid)
            if entry:
                info["model_name"] = entry["name"]
                info["ram_gb"] = entry["ram_gb"]
                info["speed_estimate"] = entry["speed_estimate"]
        return info

    def list_available_models(self) -> list:
        models_dir = "./models"
        if not os.path.exists(models_dir):
            return []
        return [f for f in os.listdir(models_dir) if f.endswith(".gguf")]

    def list_registry_models(self) -> list:
        """Return full registry with download status for each model."""
        local_files = set(self.list_available_models())
        result = []
        for m in MODEL_REGISTRY:
            entry = dict(m)
            entry["downloaded"] = m["filename"] in local_files
            entry["active"] = (
                self._loaded
                and getattr(self, '_current_model_id', None) == m["id"]
            )
            result.append(entry)
        return result


llm_service = LLMService()
