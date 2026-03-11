"""Image Integrity Agent — LangGraph node.

Performs two checks on images found in the article:
1. EXIF metadata scan for editing software (Photoshop / GIMP indicators)
2. Sarvam AI OCR to extract embedded text

If text is found in images the pipeline loops back to claim_extraction
(via the ``is_second_pass`` flag) to re-run fact-checking on the
enriched content.

Pipeline position:  agent0_post → image_integrity → END  (or → claim_extraction on second pass)
"""

from __future__ import annotations

import base64
import logging
import os
import re
from typing import TYPE_CHECKING, Any

import requests
from dotenv import load_dotenv

if TYPE_CHECKING:
    from agents.claim_extraction import AgentState

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from utils.helpers import read_env_var as _read_env_var
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from utils.helpers import read_env_var as _read_env_var

try:
    from core.config import pipeline_config
except ModuleNotFoundError:
    import sys as _sys2
    from pathlib import Path as _Path2
    _sys2.path.insert(0, str(_Path2(__file__).resolve().parent.parent))
    from core.config import pipeline_config

_img_cfg: dict = pipeline_config.get("image_integrity", {})

SARVAM_BASE_URL: str = os.getenv("SARVAM_API_BASE", "https://api.sarvam.ai")

# Editing-software strings to flag in EXIF (case-insensitive)
_EDIT_SOFTWARE_PATTERNS = [
    "photoshop", "gimp", "lightroom", "affinity", "capture one",
    "darktable", "rawtherapee", "pixelmator", "snapseed", "facetune",
]

# Markdown image pattern:  ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[.*?\]\((https?://[^)]+)\)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# EXIF helper
# ---------------------------------------------------------------------------

def _check_exif(image_bytes: bytes) -> dict[str, Any]:
    """
    Inspect image EXIF data for editing-software indicators.

    Returns a dict with:
      exif_tamper_flag: bool   — True if editing software detected
      exif_software:   str    — raw Software tag value (or '')
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        import io

        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if not exif_data:
            return {"exif_tamper_flag": False, "exif_software": ""}

        software_tag = next(
            (TAGS[k] for k in exif_data if TAGS.get(k) == "Software"),
            None,
        )
        software_value = ""
        if software_tag:
            # software_tag is the tag name; find the value
            for k, v in exif_data.items():
                if TAGS.get(k) == "Software":
                    software_value = str(v)
                    break

        flag = any(
            pat in software_value.lower() for pat in _EDIT_SOFTWARE_PATTERNS
        )
        return {"exif_tamper_flag": flag, "exif_software": software_value}
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ImageIntegrity] EXIF parse failed: %s", exc)
        return {"exif_tamper_flag": False, "exif_software": ""}


# ---------------------------------------------------------------------------
# Sarvam OCR
# ---------------------------------------------------------------------------

def _run_sarvam_ocr(image_bytes: bytes) -> str:
    """
    Send *image_bytes* to Sarvam AI OCR endpoint and return extracted text.

    Returns '' on any failure — safe fallback.

    POST {SARVAM_BASE_URL}/ocr
    Headers: { "api-subscription-key": SARVAM_API_KEY }
    Body:    { "image_url": "data:image/jpeg;base64,..." }
    Response:{ "result": "..." }
    """
    api_key = _read_env_var("SARVAM_API_KEY", "sarvam_api_key")
    if not api_key:
        logger.warning("[ImageIntegrity] SARVAM_API_KEY not set — skipping OCR")
        return ""

    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        image_data_url = f"data:image/jpeg;base64,{b64}"

        response = requests.post(
            f"{SARVAM_BASE_URL}/ocr",
            headers={"api-subscription-key": api_key},
            json={"image_url": image_data_url},
            timeout=30,
        )
        response.raise_for_status()
        return str(response.json().get("result", "")).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ImageIntegrity] Sarvam OCR failed: %s", exc)
        return ""


def _translate_ocr_to_english(text: str) -> str:
    """
    Translate OCR text to English via Agent 0 helpers (if needed).

    Returns the original text on any failure.
    """
    if not text:
        return text
    try:
        try:
            from agents.agent0_multilingual import _detect_language, _translate_to_english
        except ImportError:
            from agent0_multilingual import _detect_language, _translate_to_english  # type: ignore

        lang_code, confidence = _detect_language(text)
        if lang_code.startswith("en") or confidence < 0.7:
            return text
        return _translate_to_english(text, lang_code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ImageIntegrity] OCR translation failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# URL fetcher for remote images
# ---------------------------------------------------------------------------

def _fetch_image(url: str) -> bytes:
    """Download remote image bytes (up to 10 MB)."""
    try:
        jina_api_key = _read_env_var("JINA_READER_API_KEY", "jina_reader_api_key")
        headers: dict[str, str] = {}
        if jina_api_key:
            headers["Authorization"] = f"Bearer {jina_api_key}"

        r = requests.get(url, headers=headers, timeout=20, stream=True)
        r.raise_for_status()
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > 10 * 1024 * 1024:  # 10 MB cap
                    break
        return b"".join(chunks)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ImageIntegrity] Failed to fetch image %s: %s", url, exc)
        return b""


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def image_integrity_node(state: "AgentState") -> dict[str, Any]:
    """
    Scan images found in the article markdown for EXIF tampering and OCR text.

    On first pass (``is_second_pass`` is False/absent):
    - Extracts image URLs from ``raw_markdown``
    - Checks EXIF on first image found and runs Sarvam OCR
    - If meaningful OCR text is found, sets ``is_second_pass=True`` and
      enriches ``user_input`` with the OCR text so claim_extraction re-runs.
    - Sets ``images_processed=True`` to prevent infinite loops.

    On second pass (``is_second_pass`` is True):
    - Resets ``is_second_pass=False`` to signal routing back to END.

    Always safe — never raises; falls back to pass-through on any error.
    """
    try:
        # Guard: prevent infinite loop — only run OCR once
        if state.get("images_processed"):
            return {"is_second_pass": False}

        # Second-pass reset → route to END
        if state.get("is_second_pass"):
            return {"is_second_pass": False}

        raw_markdown: str = state.get("raw_markdown", "") or ""
        image_urls: list[str] = _MD_IMAGE_RE.findall(raw_markdown)

        if not image_urls:
            return {
                "image_urls": [],
                "media_verdicts": [],
                "ocr_text": "",
                "media_risk_level": "LOW",
                "images_processed": True,
                "is_second_pass": False,
            }

        print(f"--- IMAGE INTEGRITY: checking {len(image_urls)} image(s) ---")

        # Process the first image only (most representative / performance cap)
        target_url = image_urls[0]
        image_bytes = _fetch_image(target_url)

        exif_result: dict[str, Any] = {}
        ocr_text = ""

        if image_bytes:
            exif_result = _check_exif(image_bytes)
            ocr_text = _run_sarvam_ocr(image_bytes)
            if ocr_text:
                ocr_text = _translate_ocr_to_english(ocr_text)

        exif_flag = exif_result.get("exif_tamper_flag", False)
        exif_software = exif_result.get("exif_software", "")

        media_verdict = {
            "url": target_url,
            "exif_tamper_flag": exif_flag,
            "exif_software": exif_software,
            "ocr_text": ocr_text,
        }

        # Only loop back if substantial OCR text was found
        has_ocr_text = bool(ocr_text and len(ocr_text.strip()) >= 20)
        media_risk = "HIGH" if (exif_flag or has_ocr_text) else "LOW"

        if has_ocr_text:
            print(
                f"--- IMAGE INTEGRITY: OCR text found ({len(ocr_text)} chars) "
                "— triggering second-pass verification ---"
            )
            # Enrich user_input so claim_extraction re-runs on OCR content
            original_input = state.get("user_input", "")
            enriched_input = (
                f"{original_input}\n\n[Image Text Extracted by OCR]:\n{ocr_text}"
            ).strip()
            return {
                "image_urls": image_urls,
                "media_verdicts": [media_verdict],
                "ocr_text": ocr_text,
                "media_risk_level": media_risk,
                "images_processed": True,
                "is_second_pass": True,
                "user_input": enriched_input,
            }

        print(
            f"--- IMAGE INTEGRITY: {'EXIF flag' if exif_flag else 'no issues'} "
            f"— no OCR text, pipeline continues ---"
        )
        return {
            "image_urls": image_urls,
            "media_verdicts": [media_verdict],
            "ocr_text": "",
            "media_risk_level": media_risk,
            "images_processed": True,
            "is_second_pass": False,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("[ImageIntegrity] Unexpected error — pipeline continues: %s", exc)
        return {
            "image_urls": [],
            "media_verdicts": [],
            "ocr_text": "",
            "media_risk_level": "LOW",
            "images_processed": True,
            "is_second_pass": False,
        }
