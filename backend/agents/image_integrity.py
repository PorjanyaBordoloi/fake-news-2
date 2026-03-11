"""Agent 6 — Image Integrity Node.

Extracts images from the article markdown, fetches them, inspects EXIF
metadata for signs of editing software (Photoshop / GIMP), and runs
Indic OCR via the Sarvam AI vision API. Any text found inside images is
translated to English then appended to ``state["user_input"]`` so the
pipeline re-runs with the richer input.

Loop control
------------
- First pass  (``images_processed`` absent / False):
    Process images. If OCR text found → set ``is_second_pass = True``
    and the routing function sends the graph back to claim_extraction.
- Second pass (``images_processed`` is True):
    Return ``{"is_second_pass": False}`` immediately — routing function
    sees False and exits to END.

Pipeline position:
    agent0_post → image_integrity ─┬→ re_verify: claim_extraction
                                    └→ end:       END
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

if TYPE_CHECKING:
    from agents.claim_extraction import AgentState

load_dotenv()

try:
    from core.config import pipeline_config
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from core.config import pipeline_config

_img_cfg: dict = pipeline_config.get("image_integrity", {})

try:
    from utils.helpers import read_env_var as _read_env_var
except ModuleNotFoundError:
    import sys as _sys2
    from pathlib import Path as _Path2
    _sys2.path.insert(0, str(_Path2(__file__).resolve().parent.parent))
    from utils.helpers import read_env_var as _read_env_var

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (read once from config — not per call)
# ---------------------------------------------------------------------------

_MAX_IMAGES: int = int(_img_cfg.get("max_images", 2))
_TRANSLATION_MODEL: str = str(_img_cfg.get("translation_model", "llama-3.3-70b-versatile"))
_TRANSLATION_TEMPERATURE: float = float(_img_cfg.get("translation_temperature", 0))

# EXIF tag 305 = Software field (reveals editing application)
_EXIF_SOFTWARE_TAG: int = 305
_TAMPER_KEYWORDS: tuple[str, ...] = ("photoshop", "gimp", "affinity", "lightroom")

# Regex: find all markdown-embedded HTTPS image URLs — ![alt](url)
_IMAGE_URL_RE = re.compile(r'!\[.*?\]\((https?://[^\s)]+)\)')

# Single reusable prompt template for OCR → English translation
_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        "Translate the following text (which may be Hinglish, Assamese, Hindi, "
        "Bengali, or mixed-script) into clear English. "
        "If it is already in English, return it unchanged. "
        "Output ONLY the translated text, nothing else.\n\n{text}",
    ),
])


# ---------------------------------------------------------------------------
# Helper: fetch image bytes
# ---------------------------------------------------------------------------

def _fetch_image_bytes(url: str) -> bytes | None:
    """Download image bytes from a URL. Returns None on any network failure."""
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Agent6] Failed to fetch image %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Helper: EXIF inspection
# ---------------------------------------------------------------------------

def _check_exif(image_bytes: bytes) -> dict[str, Any]:
    """
    Open image with Pillow and check the Software EXIF tag.

    Returns a dict with:
    - ``exif_software``: the raw Software string, or None if absent
    - ``exif_tamper_flag``: True if the software name matches _TAMPER_KEYWORDS
    """
    result: dict[str, Any] = {"exif_software": None, "exif_tamper_flag": False}
    try:
        from PIL import Image  # lazy import — optional dependency
        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        software: str = str(exif.get(_EXIF_SOFTWARE_TAG, "") or "").strip()
        if software:
            result["exif_software"] = software
            result["exif_tamper_flag"] = any(
                kw in software.lower() for kw in _TAMPER_KEYWORDS
            )
    except Exception as exc:  # noqa: BLE001
        # PIL not installed, unsupported format, or no EXIF — not fatal
        logger.debug("[Agent6] EXIF extraction skipped: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Helper: Sarvam document_intelligence OCR (async job pipeline)
# ---------------------------------------------------------------------------

def _run_sarvam_ocr(image_bytes: bytes) -> str:
    """
    Run OCR via Sarvam AI document_intelligence async job pipeline.

    Flow:
    1. create_job()         — obtain a DocumentIntelligenceJob
    2. job.upload_file()    — upload image from a temp file path
    3. job.start()          — kick off processing
    4. job.wait_until_complete(timeout=120) — block until done
    5. job.download_output() — download results to a temp dir
    6. Read first .md file from the output dir

    Returns the extracted text string, or "" on all failures.
    """
    api_key = _read_env_var("SARVAM_API_KEY", "sarvam_api_key")
    if not api_key:
        logger.warning("[Agent6] SARVAM_API_KEY not set — skipping OCR")
        return ""

    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=api_key)
        di = client.document_intelligence

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Detect image format from magic bytes to use correct extension
            ext = ".jpg"
            if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                ext = ".png"
            elif image_bytes[:4] == b"%PDF":
                ext = ".pdf"

            img_path = os.path.join(tmp_dir, f"page{ext}")
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            job = di.create_job(language="hi-IN", output_format="md")
            logger.info("[Agent6] Sarvam DI job created: %s", job.job_id)

            job.upload_file(img_path)
            logger.info("[Agent6] Sarvam DI image uploaded")

            # Step 3: start processing
            job.start()
            logger.info("[Agent6] Sarvam DI job started")

            # Step 4: wait (blocks the thread; called via asyncio.to_thread in main.py)
            status_resp = job.wait_until_complete(poll_interval=3.0, timeout=120.0)
            job_state = getattr(status_resp, "job_state", "") or ""
            logger.info("[Agent6] Sarvam DI final state: %s", job_state)
            if job_state.lower() not in ("completed", "success"):
                logger.warning("[Agent6] Sarvam DI job did not complete: %s", job_state)
                return ""

            # Step 5: download output ZIP file, then extract .md files
            zip_path = os.path.join(tmp_dir, "output.zip")
            job.download_output(zip_path)

            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                md_files = [n for n in zf.namelist() if n.endswith(".md")]
                if not md_files:
                    logger.warning("[Agent6] Sarvam DI ZIP has no .md files")
                    return ""
                md_texts = [
                    zf.read(n).decode("utf-8", errors="replace").strip()
                    for n in md_files
                ]

            text = "\n\n".join(md_texts).strip()
            logger.info("[Agent6] Sarvam DI OCR extracted %d chars", len(text))
            return text

    except ImportError:
        logger.debug("[Agent6] sarvamai SDK not installed — skipping OCR")
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Agent6] Sarvam DI OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Helper: translate OCR text to English via ChatGroq
# ---------------------------------------------------------------------------

def _translate_ocr_to_english(text: str) -> str:
    """
    Translate raw OCR text (potentially Indic / mixed-script) to English.

    Uses the same ChatGroq + ChatPromptTemplate pattern as other agents.
    Returns the original text unchanged on any failure so the pipeline
    always has something to work with.
    """
    if not text:
        return ""

    # Truncate to ~6000 chars (~1500 tokens) before sending to LLM to avoid 413 errors
    _limit = int(_img_cfg.get("ocr_translate_char_limit", 6000))
    if len(text) > _limit:
        logger.info("[Agent6] Truncating OCR text from %d to %d chars before translation", len(text), _limit)
        text = text[:_limit]

    groq_api_key = _read_env_var("GROQ_API_KEY", "groq_api_key")
    if not groq_api_key:
        logger.warning("[Agent6] GROQ_API_KEY not set — returning raw OCR text")
        return text

    try:
        llm = ChatGroq(
            model=_TRANSLATION_MODEL,
            temperature=_TRANSLATION_TEMPERATURE,
            api_key=groq_api_key,
        )
        chain = _TRANSLATE_PROMPT | llm
        response = chain.invoke({"text": text})
        translated = (
            response.content if hasattr(response, "content") else str(response)
        ).strip()
        return translated if translated else text
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Agent6] OCR translation failed: %s — returning raw text", exc)
        return text


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def image_integrity_node(state: "AgentState") -> dict:
    """
    Image Integrity node — Agent 6.

    Reads  ``state["raw_markdown"]``
    Writes ``state["image_urls"]``, ``state["media_verdicts"]``,
           ``state["ocr_text"]``, ``state["media_risk_level"]``,
           ``state["images_processed"]``, optionally ``state["user_input"]``
           and ``state["is_second_pass"]``.

    Returns {} on any top-level exception — NEVER breaks the pipeline.
    """
    try:
        # ── Guard: already ran on first pass — skip on second pass ───────────
        if state.get("images_processed"):
            logger.info("[Agent6] images_processed=True — second pass, skipping")
            return {"is_second_pass": False}

        raw_markdown: str = state.get("raw_markdown", "") or ""
        image_urls: list[str] = _IMAGE_URL_RE.findall(raw_markdown)[:_MAX_IMAGES]

        print(f"--- AGENT6 IMAGE INTEGRITY: found {len(image_urls)} image(s) ---")

        # ── No images in article ─────────────────────────────────────────────
        if not image_urls:
            return {
                "images_processed": True,
                "image_urls": [],
                "media_verdicts": [],
                "ocr_text": "",
                "media_risk_level": "LOW",
            }

        # ── Process each image ───────────────────────────────────────────────
        media_verdicts: list[dict[str, Any]] = []
        all_ocr_parts: list[str] = []

        for url in image_urls:
            print(f"--- AGENT6 PROCESSING: {url[:80]} ---")

            image_bytes = _fetch_image_bytes(url)
            if image_bytes is None:
                media_verdicts.append({
                    "url": url,
                    "exif_software": None,
                    "exif_tamper_flag": False,
                    "ocr_found": False,
                    "fetch_error": True,
                })
                continue

            exif_info = _check_exif(image_bytes)
            raw_ocr = _run_sarvam_ocr(image_bytes)
            ocr_found = bool(raw_ocr)

            if exif_info.get("exif_tamper_flag"):
                print(f"--- AGENT6 TAMPER FLAG: {exif_info['exif_software']} ---")
            if ocr_found:
                print(f"--- AGENT6 OCR FOUND ({len(raw_ocr)} chars): {raw_ocr[:80]} ---")
                all_ocr_parts.append(raw_ocr)

            media_verdicts.append({
                "url": url,
                "exif_software": exif_info["exif_software"],
                "exif_tamper_flag": exif_info["exif_tamper_flag"],
                "ocr_found": ocr_found,
                "fetch_error": False,
            })

        # ── Translate combined OCR text to English ───────────────────────────
        combined_raw_ocr = " ".join(all_ocr_parts).strip()
        english_ocr_text = (
            _translate_ocr_to_english(combined_raw_ocr) if combined_raw_ocr else ""
        )

        result: dict[str, Any] = {
            "images_processed": True,
            "image_urls": image_urls,
            "media_verdicts": media_verdicts,
            "ocr_text": english_ocr_text,
            "media_risk_level": "HIGH" if english_ocr_text else "LOW",
        }

        # ── If OCR text found, enrich user_input for second-pass loop ────────
        if english_ocr_text:
            print("--- AGENT6 APPENDING OCR TEXT TO PIPELINE INPUT ---")
            current_input: str = state.get("user_input", "") or ""
            ocr_snippet = english_ocr_text[:3000]  # cap pipeline input enrichment
            result["user_input"] = (
                current_input
                + f"\n\n[TEXT FOUND IN ARTICLE IMAGES]: {ocr_snippet}"
            )
            result["is_second_pass"] = True

        print(
            f"--- AGENT6 COMPLETE | risk={result['media_risk_level']} | "
            f"ocr={'yes' if english_ocr_text else 'no'} "
            f"| tamper={any(v.get('exif_tamper_flag') for v in media_verdicts)} ---"
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Agent6] Image integrity check failed: %s — pipeline continues", exc
        )
        return {}
