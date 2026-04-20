"""Mix background audio with TTS segments using professional ducking."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


@dataclass
class TtsSegment:
    """A single TTS audio chunk placed on the timeline."""
    audio: AudioSegment
    start_ms: int
    end_ms: int


class AudioMixer:
    """Mix background + TTS voice with ducking.

    During TTS voice playback the background is reduced to
    ``ducking_bg_volume`` with fade transitions of ``ducking_fade_ms``.
    The TTS is boosted by ``ducking_fg_volume`` so it dominates clearly.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mix(
        self,
        background: AudioSegment,
        tts_segments: list[TtsSegment],
    ) -> AudioSegment:
        """Return the final mixed AudioSegment (background + ducked TTS)."""

        if not tts_segments:
            return background

        fg_db = _vol_to_db(self.cfg.ducking_fg_volume)
        ducking_db = _vol_to_db(self.cfg.ducking_bg_volume)
        fade_ms = self.cfg.ducking_fade_ms

        allow_tail = getattr(self.cfg, "allow_video_tail_extension", False)
        tail_max = getattr(self.cfg, "video_tail_extension_max_ms", 8000)

        bg_len = len(background)
        last_tts_end = max(
            (seg.start_ms + len(seg.audio) for seg in tts_segments), default=0,
        )

        if allow_tail and last_tts_end > bg_len:
            # Extend background with silence so trailing ES phrases remain
            # fully audible. Cap extension to avoid runaway (bad SRT → huge pad).
            pad_ms = min(last_tts_end - bg_len, tail_max)
            if last_tts_end - bg_len > tail_max:
                logger.warning(
                    "TTS exceeds video by %d ms but tail cap is %d ms; "
                    "trailing speech will be clipped",
                    last_tts_end - bg_len, tail_max,
                )
            background = background + AudioSegment.silent(
                duration=pad_ms, frame_rate=background.frame_rate,
            )
            bg_len = len(background)
            logger.info("Extended audio track by %d ms to fit ES tail", pad_ms)

        # Clip only phrases that still fall outside the (possibly extended) bg.
        clipped_segments: list[TtsSegment] = []
        for seg in tts_segments:
            if seg.start_ms >= bg_len:
                logger.warning(
                    "Dropping TTS starting at %d ms (past audio end %d ms)",
                    seg.start_ms, bg_len,
                )
                continue
            audio = seg.audio
            end_ms = seg.start_ms + len(audio)
            if end_ms > bg_len:
                trim_to = max(0, bg_len - seg.start_ms)
                if trim_to <= 0:
                    continue
                fade = min(60, trim_to // 4)
                audio = audio[:trim_to].fade_out(fade) if fade > 0 else audio[:trim_to]
            clipped_segments.append(TtsSegment(
                audio=audio, start_ms=seg.start_ms, end_ms=seg.start_ms + len(audio),
            ))

        if not clipped_segments:
            return background

        # Build ducked background (concatenation approach — preserves length)
        ducked_bg = self._apply_ducking(background, clipped_segments, ducking_db, fade_ms)

        # Overlay TTS on top of ducked background
        result = ducked_bg
        for seg in clipped_segments:
            if len(seg.audio) == 0:
                continue
            tts_boosted = seg.audio.apply_gain(fg_db)
            result = result.overlay(tts_boosted, position=seg.start_ms)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_ducking(
        self,
        bg: AudioSegment,
        segments: list[TtsSegment],
        ducking_db: float,
        fade_ms: int,
    ) -> AudioSegment:
        """Reduce background during TTS regions.

        Uses concatenation to preserve exact timeline length.
        The duck regions use apply_gain so the total length is unchanged.
        """
        sorted_segs = sorted(segments, key=lambda s: s.start_ms)

        # Merge overlapping/adjacent TTS regions (with fade padding)
        regions: list[tuple[int, int]] = []
        for seg in sorted_segs:
            r_start = max(0, seg.start_ms - fade_ms)
            r_end = min(len(bg), seg.start_ms + len(seg.audio) + fade_ms)
            if r_end <= r_start:
                continue
            if regions and r_start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(regions[-1][1], r_end))
            else:
                regions.append((r_start, r_end))

        if not regions:
            return bg

        # Assemble ducked background by splicing
        result = AudioSegment.empty()
        prev_end = 0

        for r_start, r_end in regions:
            # Unchanged gap before this region
            if r_start > prev_end:
                result += bg[prev_end:r_start]

            # Duck this region
            chunk = bg[r_start:r_end]
            chunk_len = len(chunk)

            # Fade from full → ducked at entry
            if fade_ms > 0 and chunk_len > fade_ms * 2:
                chunk = chunk.fade(
                    from_gain=0, to_gain=ducking_db,
                    start=0, duration=fade_ms,
                )
                # Fade from ducked → full at exit
                chunk = chunk.fade(
                    from_gain=ducking_db, to_gain=0,
                    start=chunk_len - fade_ms, duration=fade_ms,
                )
            else:
                chunk = chunk.apply_gain(ducking_db)

            result += chunk
            prev_end = r_end

        # Remainder after last ducked region
        if prev_end < len(bg):
            result += bg[prev_end:]

        return result


def _vol_to_db(ratio: float) -> float:
    if ratio <= 0:
        return -120.0
    return 20.0 * math.log10(ratio)
