"""Agent 0 — Multilingual Language Bridge using Sarvam AI.

Pre-processing:  Detects non-English input → translates to English
                 before claim_extraction runs.
Post-processing: Translates verdicts/explanations back to source
                 language after explanation_generator completes.

Zero changes to Agents 1–5. Purely additive middleware.

Pipeline position:
  Agent0_pre → claim_extraction → ... → explanation_generator → Agent0_post
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

import requests
from dotenv import load_dotenv

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

_ml_cfg: dict = pipeline_config.get("multilingual", {})

try:
    from utils.helpers import read_env_var as _read_env_var
except ModuleNotFoundError:
    import sys as _sys2
    from pathlib import Path as _Path2
    _sys2.path.insert(0, str(_Path2(__file__).resolve().parent.parent))
    from utils.helpers import read_env_var as _read_env_var

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "as-IN":  "Assamese",
    "mni-IN": "Manipuri",
    "brx-IN": "Bodo",
    "hi-IN":  "Hindi",
    "bn-IN":  "Bengali",
    "te-IN":  "Telugu",
    "ta-IN":  "Tamil",
    "kn-IN":  "Kannada",
    "or-IN":  "Odia",
    "gu-IN":  "Gujarati",
    "mr-IN":  "Marathi",
    "pa-IN":  "Punjabi",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SARVAM_BASE_URL: str = os.getenv("SARVAM_API_BASE", "https://api.sarvam.ai")

_MULTILINGUAL_ENABLED: str = _read_env_var("MULTILINGUAL_ENABLED") or str(
    _ml_cfg.get("enabled", "true")
)
_MIN_CONFIDENCE: float = float(
    _read_env_var("MULTILINGUAL_MIN_CONFIDENCE")
    or str(_ml_cfg.get("min_detection_confidence", 0.7))
)

# ---------------------------------------------------------------------------
# Entity protection helpers
# Wraps multi-word capitalized proper noun phrases in <ENTITY>...</ENTITY>
# tags before sending to Sarvam API, then restores them after translation
# to prevent names of people, places, and organizations from being mangled.
# ---------------------------------------------------------------------------

# Match runs of 2+ consecutive capitalized words (e.g. "New Delhi", "Narendra Modi")
_ENTITY_RE = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b')


def _protect_entities(text: str) -> str:
    """Wrap multi-word proper noun phrases in <ENTITY>...</ENTITY> tags."""
    return _ENTITY_RE.sub(lambda m: f"<ENTITY>{m.group(0)}</ENTITY>", text)


def _restore_entities(translated: str) -> str:
    """
    Extract content inside <ENTITY>...</ENTITY> tags in the translated text.
    If the API mangled the tags, attempts a lenient cleanup pass.
    """
    # Happy path: tags survived intact — strip the wrapper, keep the content
    restored = re.sub(r'<ENTITY>(.*?)</ENTITY>', lambda m: m.group(1), translated)
    # Cleanup any remaining broken tag fragments
    restored = re.sub(r'</?ENTITY[^>]*>', '', restored)
    return restored.strip()


# ---------------------------------------------------------------------------
# Unicode-script heuristic — fallback when Sarvam text-lid is unreachable.
# Maps Unicode block ranges to the most common Sarvam language code for that
# script.  Only used when the API call fails or returns low confidence.
# ---------------------------------------------------------------------------

_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0980, 0x09FF, "bn-IN"),   # Bengali / Assamese (same Unicode block)
    (0x0900, 0x097F, "hi-IN"),   # Devanagari — Hindi (also Marathi/Bodo)
    (0x0C00, 0x0C7F, "te-IN"),   # Telugu
    (0x0B80, 0x0BFF, "ta-IN"),   # Tamil
    (0x0C80, 0x0CFF, "kn-IN"),   # Kannada
    (0x0B00, 0x0B7F, "or-IN"),   # Odia
    (0x0A80, 0x0AFF, "gu-IN"),   # Gujarati
    (0x0A00, 0x0A7F, "pa-IN"),   # Gurmukhi (Punjabi)
]


def _heuristic_detect(text: str) -> tuple[str, float]:
    """Identify script block from the dominant Unicode range.

    Returns (language_code, 0.8) so it passes the min-confidence gate.
    Falls back to ("en-IN", 1.0) if text is ASCII.
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return "en-IN", 1.0
    best_lang = max(counts, key=lambda k: counts[k])
    return best_lang, 0.8


# ---------------------------------------------------------------------------
# Internal API key accessor
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    return _read_env_var("SARVAM_API_KEY", "sarvam_api_key")


# ---------------------------------------------------------------------------
# Sarvam API helpers
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of input text using Sarvam AI /text-lid endpoint.

    Returns (language_code, confidence) e.g. ("bn-IN", 0.97).
    Falls back to ("en-IN", 1.0) on any API failure — safe English fallback
    ensures the pipeline continues uninterrupted.

    POST {SARVAM_BASE_URL}/text-lid
    Headers: { "api-subscription-key": SARVAM_API_KEY }
    Body:    { "input": text[:500] }
    Response:{ "language_code": "bn-IN", "confidence": 0.97 }
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("[Agent0] SARVAM_API_KEY not set — skipping language detection")
        return "en-IN", 1.0

    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/text-lid",
            headers={"api-subscription-key": api_key},
            json={"input": text[:500]},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        lang_code = str(data.get("language_code", "en-IN"))
        confidence = float(data.get("confidence", 0.0))
        return lang_code, confidence
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Agent0] Language detection API failed: %s — using script heuristic", exc)
        return _heuristic_detect(text)


def _translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from source_lang to English ("en-IN") using Sarvam AI.

    Wraps multi-word proper noun phrases in <ENTITY>...</ENTITY> before the
    API call and restores them after, preserving names of people, places, and
    organisations that should not be translated.

    Returns the original text on any API failure — safe fallback.

    POST {SARVAM_BASE_URL}/translate
    Body: { input, source_language_code, target_language_code: "en-IN",
            speaker_gender: "Male", mode: "formal", enable_preprocessing: true }
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("[Agent0] SARVAM_API_KEY not set — skipping translation to English")
        return text

    protected = _protect_entities(text)

    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/translate",
            headers={"api-subscription-key": api_key},
            json={
                "input": protected,
                "source_language_code": source_lang,
                "target_language_code": "en-IN",
                "speaker_gender": "Male",
                "mode": "formal",
                "enable_preprocessing": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        translated_raw = response.json().get("translated_text", "")
        if not translated_raw:
            return text
        return _restore_entities(translated_raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Agent0] Translation to English failed: %s", exc)
        return text


def _translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate text from English ("en-IN") back to target_lang using Sarvam AI.

    Returns the original English text on any API failure — safe fallback so
    the frontend can always display at minimum the English version.

    POST {SARVAM_BASE_URL}/translate
    Body: { input, source_language_code: "en-IN", target_language_code,
            speaker_gender: "Male", mode: "formal", enable_preprocessing: true }
    """
    if not text:
        return text

    api_key = _get_api_key()
    if not api_key:
        logger.warning("[Agent0] SARVAM_API_KEY not set — skipping reverse translation")
        return text

    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/translate",
            headers={"api-subscription-key": api_key},
            json={
                "input": text,
                "source_language_code": "en-IN",
                "target_language_code": target_lang,
                "speaker_gender": "Male",
                "mode": "formal",
                "enable_preprocessing": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        translated = response.json().get("translated_text", "")
        return translated if translated else text
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Agent0] Translation from English failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def agent0_pre_node(state: "AgentState") -> dict:
    """
    Pre-processing node — runs BEFORE claim_extraction.

    Detects non-English input text and translates it to English so that
    Agents 1-5 always operate on English. Overwrites state["user_input"]
    with the English translation and stores the original in state["original_text"].

    Skips translation for:
    - Feature flag MULTILINGUAL_ENABLED=false
    - URL inputs  (Jina Reader handles them in English already)
    - English text (lang code starts with "en")
    - Low-confidence detections (< MULTILINGUAL_MIN_CONFIDENCE threshold)
    - Unsupported languages (not in SUPPORTED_LANGUAGES)

    Returns {} on ANY exception — NEVER breaks the pipeline.
    LangGraph treats an empty-dict return as a no-op state update.
    """
    try:
        # ── Feature flag ────────────────────────────────────────────────────
        if str(_MULTILINGUAL_ENABLED).lower() == "false":
            return {}

        user_input: str = str(state.get("user_input") or "").strip()
        if not user_input:
            return {}

        # ── URL check — import locally to avoid circular import ─────────────
        # (claim_extraction.py imports this module at the top level; importing
        # back at module level would create a circular dependency)
        try:
            from agents.claim_extraction import is_url
        except (ImportError, ModuleNotFoundError):
            from claim_extraction import is_url  # type: ignore[no-redef]

        if is_url(user_input):
            # URLs are scraped by Jina Reader which always returns English markdown
            return {}

        # ── Language detection ───────────────────────────────────────────────
        lang_code, confidence = _detect_language(user_input)
        print(
            f"--- AGENT0 DETECTED: {lang_code} "
            f"(confidence={confidence:.2f}) ---"
        )

        # Low-confidence — treat input as English
        if confidence < _MIN_CONFIDENCE:
            return {}

        # Already English — nothing to do
        if lang_code.startswith("en"):
            return {}

        # Unsupported language — pipeline continues on original input
        if lang_code not in SUPPORTED_LANGUAGES:
            logger.warning(
                "[Agent0] Unsupported language '%s' — pipeline runs on original input",
                lang_code,
            )
            return {}

        # ── Translate to English ─────────────────────────────────────────────
        lang_name = SUPPORTED_LANGUAGES[lang_code]
        print(f"--- AGENT0 TRANSLATING FROM {lang_name} TO ENGLISH ---")

        english_text = _translate_to_english(user_input, lang_code)

        # Translation sanity check
        if not english_text or english_text.strip() == user_input.strip():
            logger.warning(
                "[Agent0] Translation returned empty/unchanged result — "
                "falling back to original input"
            )
            return {}

        print(f"--- AGENT0 TRANSLATION COMPLETE ({len(english_text)} chars) ---")

        return {
            "original_text": user_input,
            "source_language": lang_code,
            "is_translated": True,
            "translated_input": english_text,
            "user_input": english_text,  # overwrite so Agent 1 sees English
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Agent0] Pre-processing failed: %s — pipeline continues on original input",
            exc,
        )
        return {}


def agent0_post_node(state: "AgentState") -> dict:
    """
    Post-processing node — runs AFTER explanation_generator.

    Translates plain-English verdict explanations back to the source language
    that was detected in agent0_pre. Results are stored in state["localized_output"]
    so BOTH the English ("explanations") and localized ("localized_output")
    versions coexist in state — the frontend decides which to render.

    Translated fields per claim (from VerdictExplanation):
    - plain_english
    - confidence_statement
    - source_quality_note
    - reader_advisory        (only if not None)
    - evidence_gaps_plain    (only if not None)

    NOT translated: claim_text (referential), verdict labels, truth_score, citations.

    Returns {} on ANY exception — NEVER breaks the pipeline.
    """
    try:
        # Only run if Agent 0 pre-processing translated something
        if not state.get("is_translated"):
            return {}

        target_lang: str = str(state.get("source_language") or "")
        if not target_lang or target_lang not in SUPPORTED_LANGUAGES:
            return {}

        lang_name = SUPPORTED_LANGUAGES[target_lang]
        explanations: dict[str, Any] = state.get("explanations") or {}
        by_claim_en: dict[str, Any] = explanations.get("by_claim") or {}

        if not by_claim_en:
            return {}

        print(f"--- AGENT0 LOCALIZING OUTPUT TO {lang_name} ---")

        # ── Translate each claim's explanation fields ────────────────────────
        localized_by_claim: dict[str, dict[str, Any]] = {}

        for claim_text, ve in by_claim_en.items():
            localized_fields: dict[str, Any] = {}

            # Always-present string fields
            for field in ("plain_english", "confidence_statement", "source_quality_note"):
                value = ve.get(field)
                localized_fields[field] = (
                    _translate_from_english(str(value), target_lang) if value else value
                )

            # Optional fields — only translate when not None
            for field in ("reader_advisory", "evidence_gaps_plain"):
                value = ve.get(field)
                localized_fields[field] = (
                    _translate_from_english(str(value), target_lang) if value else None
                )

            localized_by_claim[claim_text] = localized_fields

        # ── Translate bottom_line ────────────────────────────────────────────
        bottom_line_en: str = str(explanations.get("bottom_line") or "")
        localized_bottom_line = (
            _translate_from_english(bottom_line_en, target_lang)
            if bottom_line_en
            else ""
        )

        localized_output: dict[str, Any] = {
            "language": target_lang,
            "language_name": lang_name,
            "by_claim": localized_by_claim,
            "bottom_line": localized_bottom_line,
        }

        print("--- AGENT0 LOCALIZATION COMPLETE ---")
        return {"localized_output": localized_output}

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Agent0] Post-processing failed: %s — pipeline completes with English output",
            exc,
        )
        return {}
