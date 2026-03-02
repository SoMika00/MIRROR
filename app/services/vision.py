"""
Vision model service for PDF image understanding.

Uses MiniCPM-V 2.6 INT4 quantized - a compact 8B VLM that can:
  - Understand PDF page layouts, charts, tables, diagrams
  - Extract text from images with OCR-like accuracy
  - Answer questions about visual content

On CPU with INT4 quantization:
  - ~4 GB RAM footprint
  - ~30-60s per image analysis (acceptable for async PDF processing)
  - Loaded on-demand, unloaded when not needed

This service is optional and activated via VISION_ENABLED=1.
"""

import logging
import threading
import time
from typing import Optional

from app.config import vision_cfg

logger = logging.getLogger(__name__)


class VisionService:
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
        self.tokenizer = None
        self.processor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not vision_cfg.enabled:
            logger.info("Vision model disabled (VISION_ENABLED=0)")
            return

        logger.info(f"Loading vision model: {vision_cfg.model_name}")
        start = time.time()
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                vision_cfg.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                vision_cfg.model_name,
                trust_remote_code=True,
            ).eval()

            elapsed = time.time() - start
            logger.info(f"Vision model loaded in {elapsed:.1f}s")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            logger.info("Vision model unloaded")

    def analyze_image(self, image_path: str, question: Optional[str] = None) -> str:
        """
        Analyze an image (e.g., a PDF page rendered as image) and return description.
        """
        if not self._loaded:
            self.load()
        if not self._loaded:
            return ""

        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        prompt = question or "Describe the content of this document page in detail, including any tables, charts, text, and layout structure."

        try:
            msgs = [{"role": "user", "content": [image, prompt]}]
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
            )
            return response
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return ""

    def analyze_pdf_page(self, pdf_path: str, page_num: int = 0) -> str:
        """
        Render a PDF page to image and analyze it with the vision model.
        """
        import fitz
        import tempfile
        import os

        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            return ""

        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = self.analyze_image(tmp_path)
        finally:
            os.unlink(tmp_path)
            doc.close()

        return result

    def get_info(self) -> dict:
        return {
            "model": vision_cfg.model_name,
            "loaded": self._loaded,
            "enabled": vision_cfg.enabled,
            "device": vision_cfg.device,
        }


vision_service = VisionService()
