"""SRT translation with pluggable providers (OpenAI, DeepL).

Preserves timestamps and subtitle structure. Writes ``{base}_ES.srt`` next
to the source file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Protocol

import httpx

from .srt_io import parse_srt, serialize_srt

log = logging.getLogger("subtitler")

DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Language code helpers
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    "EN": "English",
    "ES": "Spanish",
    "PT": "Portuguese",
    "FR": "French",
    "IT": "Italian",
    "DE": "German",
}


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code.upper(), code)


# ---------------------------------------------------------------------------
# Protocol + base
# ---------------------------------------------------------------------------

class Translator(Protocol):
    """Minimum contract a translation provider must fulfil."""

    def translate_texts(self, texts: list[str]) -> list[str]: ...

    def translate_srt(self, src_path: Path, dst_path: Path | None = None) -> Path: ...


class _BaseTranslator:
    """Shared SRT read/write logic. Subclasses implement ``translate_texts``."""

    def __init__(self, source_lang: str = "EN", target_lang: str = "ES") -> None:
        self.source_lang = source_lang.upper()
        self.target_lang = target_lang.upper()

    def translate_texts(self, texts: list[str]) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def translate_srt(
        self,
        src_path: Path,
        dst_path: Path | None = None,
    ) -> Path:
        """Translate ``src_path`` SRT, writing ``dst_path`` (default ``*_ES.srt``)."""
        src_path = Path(src_path)
        if dst_path is None:
            dst_path = src_path.with_name(f"{src_path.stem}.es.srt")
        else:
            dst_path = Path(dst_path)

        subs = parse_srt(src_path)
        if not subs:
            log.warning("No subtitles found in %s", src_path)
            dst_path.write_text("", encoding="utf-8")
            return dst_path

        texts = [s["text"] for s in subs]
        translated = self.translate_texts(texts)
        if len(translated) != len(texts):
            raise RuntimeError(
                f"Provider returned {len(translated)} items, expected {len(texts)}"
            )

        for sub, new_text in zip(subs, translated):
            sub["text"] = new_text

        dst_path.write_text(serialize_srt(subs), encoding="utf-8")
        log.info("Wrote %d translated subtitles to %s", len(subs), dst_path.name)
        return dst_path


# ---------------------------------------------------------------------------
# DeepL provider
# ---------------------------------------------------------------------------

class DeepLTranslator(_BaseTranslator):
    """Translate SRT files using the DeepL REST API."""

    _BATCH_SIZE = 40

    def __init__(
        self,
        api_key: str | None = None,
        source_lang: str = "EN",
        target_lang: str = "ES",
        formality: str | None = None,
        pro: bool | None = None,
    ) -> None:
        super().__init__(source_lang, target_lang)
        key = api_key or os.environ.get("DEEPL_API_KEY")
        if not key:
            raise ValueError("DEEPL_API_KEY not provided (env or constructor)")
        self.api_key = key
        self.formality = formality
        if pro is None:
            pro = not key.endswith(":fx")
        self.url = DEEPL_PRO_URL if pro else DEEPL_FREE_URL

    def translate_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        out: list[str] = []
        for chunk in _chunks(texts, self._BATCH_SIZE):
            out.extend(self._post(chunk))
        return out

    def _post(self, texts: list[str]) -> list[str]:
        payload: dict[str, str | list[str]] = {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "preserve_formatting": "1",
            "split_sentences": "0",
            "text": texts,
        }
        if self.formality:
            payload["formality"] = self.formality

        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(self.url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"DeepL error {r.status_code}: {r.text[:300]}")
        return [t["text"] for t in r.json().get("translations", [])]


# ---------------------------------------------------------------------------
# OpenAI provider (BJJ-aware, keeps technique names in English)
# ---------------------------------------------------------------------------

_BJJ_SYSTEM_PROMPT = """You translate Brazilian Jiu-Jitsu instructional subtitles from {src_name} to {tgt_name}.

Rules, non-negotiable:
1. Keep BJJ technique names and positions in English (examples: guard, half-guard, mount, side control, armbar, kimura, triangle, heel hook, sweep, pass, tripod pass, underhook, overhook, kimura grip, gable grip, knee cut, smash pass, leg drag, berimbolo, de la riva, x-guard, butterfly guard, closed guard, open guard, etc.). Do not translate them.
2. Keep common grappling English terms and actions as-is too (grip, frame, framing, post, base, hook, lapel, sleeve, collar, gi, no-gi, tap, pin, pinning, pummel, pummeling, sprawl, turtle, turtling, scramble, roll, rolling, drill, drilling, crossface, whizzer, backstep, reset, setup, entry, transition, top, bottom, pressure, stack, stacking).
3. Translate ordinary narration, explanations, transitions and body descriptions naturally into neutral, informal {tgt_name} as spoken by a coach.
4. Preserve meaning; do NOT add, shorten or merge content. One input item = one output item, same order.
5. Preserve line breaks inside a subtitle block exactly.
6. Output MUST be a JSON object of the exact shape: {{"t": ["translated item 1", "translated item 2", ...]}} with the same number of items as the input, in the same order.
"""


# Dubbing mode: industry-standard iso-synchronous translation.
# The translator MUST keep each line within a character budget so the TTS
# comes out close to the original slot duration without needing audio stretch.
# This avoids the classic "ES is 40% longer than EN → robotic acceleration".
_BJJ_DUBBING_SYSTEM_PROMPT = """You adapt Brazilian Jiu-Jitsu instructional subtitles from {src_name} to {tgt_name} FOR DUBBING (voice-over).

Priorities in order (industry dubbing standard):
1. PRESERVE CONTENT — every technical concept, BJJ term, instruction and explanation must survive. Losing a technique name or a step is a critical failure. If budget is tight, it is better to slightly overflow than to drop content.
2. TARGET BUDGET — each item has a "max_chars" budget (~{cps} chars/second of {tgt_name} speech). Aim for it. Soft overflow up to ~20% is acceptable when needed to keep the full meaning; audio stretch will absorb it.
3. NATURAL PHRASING — drop pure filler ("you know", "alright", "so", "basically"), compact verbose connectors ("vamos a hacer" → "hacemos", "es importante que" → "es clave"), but never sacrifice technical information for brevity.

Rules, non-negotiable:
1. Keep BJJ technique names and grappling English terms in English (guard, half-guard, armbar, kimura, triangle, heel hook, sweep, pass, underhook, overhook, grip, frame, base, hook, lapel, sleeve, mount, side control, knee cut, smash pass, leg drag, berimbolo, de la riva, x-guard, butterfly guard, closed guard, open guard, crossface, whizzer, tap, pummel, sprawl, scramble, drill, setup, entry, transition, top, bottom, pressure).
2. Use neutral informal {tgt_name} as spoken by a coach. Second person singular ("tú"/"you"-style) unless the original uses plural.
3. One input item = one output item, same order. Never merge, split, or drop items. If an item is short, keep the translation short too — do not pad to fill the budget.
4. Output MUST be a JSON object: {{"t": ["adapted item 1", ...]}} with the same number of items as the input, in the same order.
"""


# Fill-budget variant — used when the source timing is speech-anchored
# (nivel 3 dub track). Here each slot equals real speaker talk time, so
# the ES adaptation must roughly match that duration; otherwise the TTS
# finishes early and leaves audible silence in the middle of the speech.
_BJJ_DUBBING_FILL_SYSTEM_PROMPT = """You adapt Brazilian Jiu-Jitsu instructional subtitles from {src_name} to {tgt_name} FOR DUBBING (voice-over).

This adaptation is time-bound: each item represents a real stretch of
speaker talk time. The TTS engine speaks at a constant rate, so the
number of characters you output directly determines how long the TTS
audio will be. If you write too few characters, the dubbed track will
have silence in the middle of the speaker's speech — the worst failure
mode for a dubbing.

Priorities in order:
1. FILL THE SLOT — target 95-108% of `target_chars`. Undershooting
   leaves audible silence in the middle of the speaker's continuous
   speech (WORST failure mode for dubbing). A slight overshoot is
   absorbed by gentle audio stretch without perceptible artifacts.
2. HARD CEILING 110% — do NOT exceed 110% of target_chars; beyond
   that TTS must speed up too aggressively and accents turn robotic.
3. PRESERVE CONTENT — every technical concept, BJJ term, step and
   explanation must survive. Do not invent new technical content. If
   the literal translation is below 90%, add a short natural
   connective ("fíjate bien", "en este caso", "la idea es",
   "básicamente", "como ves", "observa cómo", "de esta forma") to
   reach the target range. Do NOT pad aggressively or duplicate ideas.
3. NATURAL PHRASING — coach register, second person singular ("tú"),
   informal neutral {tgt_name}. Avoid filler that sounds robotic; prefer
   genuine discourse markers that a real dub actor would say.

Rules, non-negotiable:
1. Keep BJJ technique names and grappling English terms in English
   (guard, half-guard, armbar, kimura, triangle, heel hook, sweep, pass,
   underhook, overhook, grip, frame, base, hook, lapel, sleeve, mount,
   side control, knee cut, smash pass, leg drag, berimbolo, de la riva,
   x-guard, butterfly guard, closed guard, open guard, crossface,
   whizzer, tap, pummel, sprawl, scramble, drill, setup, entry,
   transition, top, bottom, pressure).
2. One input item = one output item, same order. Never merge, split, or
   drop items.
3. Output MUST be a JSON object: {{"t": ["adapted item 1", ...]}} with
   the same number of items as the input, in the same order.
"""


class OpenAITranslator(_BaseTranslator):
    """Translate SRT files via OpenAI Chat Completions (gpt-4o-mini default).

    Batches subtitle items into a single JSON request for coherence and cost.
    Uses ``response_format=json_object`` for reliable parsing.
    """

    # Number of subtitle items per request. 40 is a sweet spot:
    # - keeps latency low
    # - keeps context coherent
    # - well under any token limit even for long lines
    _BATCH_SIZE = 40

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        source_lang: str = "EN",
        target_lang: str = "ES",
        temperature: float = 0.2,
        base_url: str | None = None,
    ) -> None:
        super().__init__(source_lang, target_lang)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not provided (env or constructor)")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.url = (base_url or OPENAI_URL).rstrip("/")
        # If base_url was passed without the /chat/completions suffix, fix it.
        if not self.url.endswith("/chat/completions"):
            if self.url.endswith("/v1"):
                self.url = f"{self.url}/chat/completions"

    def translate_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        out: list[str] = []
        for chunk in _chunks(texts, self._BATCH_SIZE):
            try:
                out.extend(self._translate_batch(chunk))
            except RuntimeError:
                # Batch failed after retries — fall back to one-by-one
                log.warning("Batch of %d failed, translating one-by-one", len(chunk))
                for item in chunk:
                    out.extend(self._translate_batch([item]))
        return out

    def translate_for_dubbing(
        self,
        items: list[dict],
        cps: float = 17.0,
        fill_budget: bool = False,
    ) -> list[str]:
        """Translate ES-dub style: enforces char budget per item from slot duration.

        Each ``item`` must be ``{"text": str, "duration_ms": int}``.
        Budget per item = (duration_ms / 1000) * cps, clamped to [12, 220].

        ``fill_budget=True`` switches to the speech-anchored mode (nivel 3):
        the prompt asks the model to actively reach ~85-105% of the budget
        instead of treating it as a ceiling. Use it when the input slots
        represent real speaker talk time — otherwise undershooting produces
        audible silence in the middle of the dub.
        """
        if not items:
            return []
        # Fill-budget prompt is stricter (per-item targets, upper/lower bounds)
        # and the model occasionally merges/splits items when the batch is
        # large. Smaller batches give cleaner 1:1 counts.
        batch_size = 15 if fill_budget else self._BATCH_SIZE
        out: list[str] = []
        for chunk in _chunks(items, batch_size):
            try:
                out.extend(self._translate_dubbing_batch(chunk, cps, fill_budget))
            except RuntimeError:
                log.warning(
                    "Dubbing batch of %d failed, falling back one-by-one",
                    len(chunk),
                )
                for item in chunk:
                    out.extend(self._translate_dubbing_batch([item], cps, fill_budget))
        return out

    _MAX_RETRIES = 2

    def _translate_batch(self, texts: list[str]) -> list[str]:
        system = _BJJ_SYSTEM_PROMPT.format(
            src_name=_lang_name(self.source_lang),
            tgt_name=_lang_name(self.target_lang),
        )
        user_payload = {"items": texts}
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "Translate each item in `items`. Return JSON "
                        '{"t": [...]} with the same number of items in the same order.\n'
                        + json.dumps(user_payload, ensure_ascii=False)
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            with httpx.Client(timeout=_TIMEOUT) as client:
                r = client.post(self.url, headers=headers, json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:300]}")

            data = r.json()
            try:
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"OpenAI response parse failed: {exc}") from exc

            items = parsed.get("t")
            if not isinstance(items, list):
                items = parsed.get("items")
            if isinstance(items, list) and len(items) == len(texts):
                return [str(x) for x in items]

            last_err = RuntimeError(
                f"OpenAI returned {len(items) if isinstance(items, list) else 'non-list'} items, "
                f"expected {len(texts)}"
            )
            log.warning("OpenAI count mismatch (attempt %d/%d), retrying…",
                        attempt + 1, 1 + self._MAX_RETRIES)

        raise last_err  # type: ignore[misc]

    def _translate_dubbing_batch(
        self, items: list[dict], cps: float, fill_budget: bool = False,
    ) -> list[str]:
        """Iso-synchronous batch: sends items with max_chars budget.

        Retries items that exceed budget by asking the model to shorten them.
        Budget floor = 12 chars (tiny slots still need something sayable),
        ceiling = 220 (Chatterbox chunk limit).

        ``fill_budget=True`` uses the speech-anchored prompt and enforces a
        LOWER bound (output must reach ≥80% of budget) in addition to the
        usual upper bound; retries items that undershoot.
        """
        def _budget(d_ms: int) -> int:
            raw = int((d_ms / 1000.0) * cps)
            return max(15, min(260, raw))

        budget_key = "target_chars" if fill_budget else "max_chars"
        payload_items = [
            {
                "text": it.get("text", ""),
                budget_key: _budget(int(it.get("duration_ms", 1500))),
            }
            for it in items
        ]
        system_tpl = _BJJ_DUBBING_FILL_SYSTEM_PROMPT if fill_budget else _BJJ_DUBBING_SYSTEM_PROMPT
        system = system_tpl.format(
            src_name=_lang_name(self.source_lang),
            tgt_name=_lang_name(self.target_lang),
            cps=f"{cps:.0f}",
        )
        user_instruction = (
            "Translate each item in `items` into an ADAPTED dubbing line "
            "that LANDS BETWEEN 95% AND 108% of its `target_chars`. "
            "Undershooting produces silence gaps in the dub (worst failure). "
            "Prefer a slight overshoot (≤108%) over any undershoot. "
            'Return JSON {"t": [...]} with the same number of items in the '
            "same order.\n"
            if fill_budget else
            "Translate each item in `items` into an ADAPTED dubbing line "
            "that fits within its `max_chars` budget. "
            'Return JSON {"t": [...]} with the same number of items in the '
            "same order.\n"
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        user_instruction
                        + json.dumps({"items": payload_items}, ensure_ascii=False)
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            with httpx.Client(timeout=_TIMEOUT) as client:
                r = client.post(self.url, headers=headers, json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:300]}")
            try:
                content = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"OpenAI response parse failed: {exc}") from exc

            result = parsed.get("t")
            if not isinstance(result, list):
                result = parsed.get("items")
            if not (isinstance(result, list) and len(result) == len(items)):
                got = len(result) if isinstance(result, list) else "non-list"
                last_err = RuntimeError(
                    f"OpenAI returned non-matching count "
                    f"({got} vs {len(items)})"
                )
                # Feedback retry: tell the model the count is wrong so it
                # stops merging/splitting items. Without this the retry
                # loop just hits the same mistake 3 times.
                body["messages"].append(
                    {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
                )
                body["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            f"Your response had {got} items but the input had "
                            f"{len(items)}. You MUST output EXACTLY "
                            f"{len(items)} items in the same order as the "
                            "input. Never merge, split, or drop items — a "
                            "short input item gets a short translation, not "
                            "a merge with its neighbour. Re-translate now "
                            'and return JSON {"t": [...]} with exactly '
                            f"{len(items)} strings."
                        ),
                    },
                )
                continue

            result = [str(x) for x in result]

            budgets = [p[budget_key] for p in payload_items]
            # Upper bound: fill_budget allows +10% overshoot (TTS stretch
            # absorbs softly). Reader mode is lenient (1.25x).
            upper_ratio = 1.10 if fill_budget else 1.25
            over_idx: list[int] = [
                i for i, (txt, b) in enumerate(zip(result, budgets))
                if len(txt) > int(b * upper_ratio)
            ]
            # Undershoot check (only in fill_budget mode): fail below 0.90x.
            # Anything shorter leaves audible silence gaps in the dub.
            under_idx: list[int] = []
            if fill_budget:
                under_idx = [
                    i for i, (txt, b) in enumerate(zip(result, budgets))
                    if len(txt) < int(b * 0.90)
                ]
            offenders = sorted(set(over_idx + under_idx))

            if not offenders or attempt == self._MAX_RETRIES:
                if offenders:
                    log.warning(
                        "Budget mismatch on %d items after retries "
                        "(over=%d, under=%d); accepting",
                        len(offenders), len(over_idx), len(under_idx),
                    )
                return result

            # Feedback loop: tell the model which items missed budget and
            # in which direction, then request a fresh pass.
            feedback = [
                {
                    "index": i,
                    "current_len": len(result[i]),
                    "budget": payload_items[i][budget_key],
                    "direction": (
                        "TOO LONG — shorten"
                        if i in over_idx else
                        "TOO SHORT — expand without changing meaning"
                    ),
                    "current": result[i],
                    "source": payload_items[i]["text"],
                }
                for i in offenders
            ]
            retry_msg = (
                "These items did not land within budget. Rewrite ALL items "
                "(not only offenders). Items marked TOO SHORT must be "
                "expanded with natural connective phrasing so the dub "
                "covers the speaker's talk time without silence. "
                'Return JSON {"t": [...]} with every item rewritten.\n'
                "Details: "
                if fill_budget else
                "These items exceeded their budget. "
                "Produce a shorter adaptation for ALL items again. "
                'Return JSON {"t": [...]} with every item rewritten '
                "to respect budgets.\nOverflow details: "
            )
            body["messages"].append(
                {"role": "assistant", "content": json.dumps({"t": result}, ensure_ascii=False)},
            )
            body["messages"].append(
                {
                    "role": "user",
                    "content": retry_msg + json.dumps(feedback, ensure_ascii=False),
                },
            )

        if last_err:
            raise last_err
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_translator(
    provider: str,
    *,
    api_key: str | None = None,
    source_lang: str = "EN",
    target_lang: str = "ES",
    model: str | None = None,
    formality: str | None = None,
) -> _BaseTranslator:
    """Build a translator for the requested provider name."""
    p = (provider or "").lower().strip()
    if p in ("openai", "gpt", "chatgpt"):
        return OpenAITranslator(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            source_lang=source_lang,
            target_lang=target_lang,
        )
    if p in ("deepl",):
        return DeepLTranslator(
            api_key=api_key,
            source_lang=source_lang,
            target_lang=target_lang,
            formality=formality,
        )
    raise ValueError(f"Unknown translation provider: {provider!r}")


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
