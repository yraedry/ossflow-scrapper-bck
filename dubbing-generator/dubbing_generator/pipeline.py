"""Dubbing pipeline orchestrator."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from pydub import AudioSegment

from .config import DubbingConfig
from .audio.mixer import AudioMixer, TtsSegment
from .audio.separator import AudioSeparator
from .audio.stretcher import stretch_audio
from .sync.aligner import SrtBlock, SyncAligner
from .sync.drift_corrector import DriftCorrector
from .tts.synthesizer import Synthesizer
from .tts.voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


# ======================================================================
# SRT parsing helpers
# ======================================================================

def _parse_time(time_str: str) -> int:
    """Parse ``HH:MM:SS,mmm`` to milliseconds."""
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def parse_srt(srt_path: Path) -> list[SrtBlock]:
    """Parse an SRT file into a list of :class:`SrtBlock`."""
    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d+)\n"
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.*?)(?=\n\n|\n$|\Z)",
        re.DOTALL,
    )
    blocks: list[SrtBlock] = []
    for m in pattern.finditer(content):
        text = m.group(4).replace("\n", " ").strip()
        text = re.sub(r"\((.*?)\)", r"\1", text)
        blocks.append(SrtBlock(
            index=int(m.group(1)),
            start_ms=_parse_time(m.group(2)),
            end_ms=_parse_time(m.group(3)),
            text=text,
        ))
    return blocks


# ======================================================================
# Pipeline
# ======================================================================

class DubbingPipeline:
    """Orchestrate the full dubbing workflow for a single video."""

    def __init__(
        self,
        config: DubbingConfig,
        progress_cb: ProgressCallback = None,
    ) -> None:
        self.cfg = config
        self._progress_cb = progress_cb

        self.separator = AudioSeparator(config)
        self.voice_cloner = VoiceCloner(config)
        self.synthesizer = Synthesizer(config)
        self.aligner = SyncAligner(config)
        self.drift = DriftCorrector(config)
        self.mixer = AudioMixer(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(
        self,
        video_path: Path,
        srt_path: Path,
        voice_ref: Optional[Path] = None,
    ) -> Path:
        """Run the full dubbing pipeline on one video."""
        base_name = video_path.with_suffix("")
        output_video = base_name.parent / f"{base_name.name}_DOBLADO.mkv"
        output_audio = base_name.parent / f"{base_name.name}_AUDIO_ESP.wav"

        if output_video.exists():
            logger.info("Output already exists, skipping: %s", output_video)
            return output_video

        # 1. Separate background audio
        self._report(0, 6, "Separating background audio...")
        background_path = self.separator.separate(video_path)

        # 2. Voice reference from clean vocals stem (best for XTTS cloning)
        self._report(1, 6, "Extracting voice reference...")
        vocals_stem = video_path.with_name(f"{video_path.stem}_VOCALS.wav")
        effective_voice_ref = voice_ref or (vocals_stem if vocals_stem.exists() else None)
        ref_wav = self.voice_cloner.get_reference(video_path, effective_voice_ref)

        # 3. Parse SRT, plan alignment (pass video duration so the last phrase
        #    can borrow the tail gap instead of spilling past end of video).
        self._report(2, 6, "Planning phrase alignment...")
        blocks = parse_srt(srt_path)
        if getattr(self.cfg, "merge_consecutive_blocks", False):
            blocks = self._merge_consecutive_blocks(
                blocks,
                self.cfg.merge_max_gap_ms,
                getattr(self.cfg, "merge_max_chars", 200),
            )
        video_duration_ms = self._probe_video_duration_ms(video_path)
        planned = self.aligner.plan(blocks, video_duration_ms=video_duration_ms)

        # 4. Synthesize all phrases
        self._report(3, 6, "Synthesizing speech...")
        tts_segments = self._synthesize_all(
            planned, ref_wav, video_duration_ms=video_duration_ms,
        )

        # 5. Mix background + TTS with ducking
        self._report(4, 6, "Mixing audio with ducking...")
        background = AudioSegment.from_wav(str(background_path))
        # Asegura que el background cubre toda la duración del vídeo. Si
        # Demucs entrega un stem más corto, el mux con -shortest truncaría
        # imagen + audio al mínimo → output corrupto antes del final real.
        if video_duration_ms and len(background) < video_duration_ms:
            pad_ms = video_duration_ms - len(background)
            background = background + AudioSegment.silent(
                duration=pad_ms, frame_rate=background.frame_rate,
            )
        mixed = self.mixer.mix(background, tts_segments)
        # Red de seguridad: mixed debe durar al menos el vídeo completo.
        if video_duration_ms and len(mixed) < video_duration_ms:
            pad_ms = video_duration_ms - len(mixed)
            mixed = mixed + AudioSegment.silent(
                duration=pad_ms, frame_rate=mixed.frame_rate,
            )
        mixed.export(str(output_audio), format="wav")

        # 6. Mux into video
        self._report(5, 6, "Muxing final video...")
        self._mux_video(video_path, output_audio, output_video)

        # Cleanup (vocals stem only if auto-generated)
        vocals_stem_to_clean = vocals_stem if (not voice_ref and vocals_stem.exists()) else None
        self._cleanup(output_audio, background_path, *(p for p in [vocals_stem_to_clean] if p))

        self._report(6, 6, f"Done: {output_video.name}")
        logger.info("Dubbed video saved: %s", output_video)
        return output_video

    def process_directory(self, root_dir: Path) -> list[Path]:
        """Process all videos in *root_dir* that have matching SRT files."""
        results: list[Path] = []

        for dirpath, _dirs, files in os.walk(root_dir):
            videos = sorted(
                f for f in files
                if f.lower().endswith(self.cfg.extensions)
                and "_DOBLADO" not in f
            )
            for video_name in videos:
                video_path = Path(dirpath) / video_name
                base = video_path.with_suffix("")

                srt_path = None
                # Híbrido: .es.srt literal (alineado con vídeo) preferido.
                # Merge consecutivo + stretch absorbe la compresión ES. El
                # .dub.es.srt (nivel 3) queda como fallback si no hay literal.
                for _sfx in (".es.srt", ".ES.srt", "_ES.srt", "_ESP.srt", ".dub.es.srt"):
                    _candidate = base.parent / f"{base.name}{_sfx}"
                    if _candidate.exists():
                        srt_path = _candidate
                        break
                if srt_path is None:
                    logger.warning("No SRT found for %s, skipping", video_name)
                    continue

                logger.info("Using ES SRT: %s", srt_path.name)

                try:
                    out = self.process_file(video_path, srt_path)
                    results.append(out)
                except Exception:
                    logger.exception("Error processing %s", video_name)

        return results

    # ------------------------------------------------------------------
    # Synthesis loop — anchor-based sync (each phrase anchored to SRT start)
    # ------------------------------------------------------------------

    def _synthesize_all(
        self,
        planned: list,
        ref_wav: Path,
        video_duration_ms: int | None = None,
    ) -> list[TtsSegment]:
        """Synthesize TTS anchored to SRT timestamps.

        Strategy:
          - Each phrase placed at its SRT start_ms (NO accumulated drift).
          - Fitted to allocated slot via pitch-preserving stretch.
          - Small overflow allowed (max_overflow_ms) to avoid over-compression.
          - Drift corrector adjusts TTS speed based on text-length pressure,
            not positional drift (anchor-based = no positional drift).
        """
        self.drift.reset()
        segments: list[TtsSegment] = []

        total = len(planned)
        max_overflow = self.cfg.max_overflow_ms
        lead_silence = self.cfg.tts_lead_silence_ms

        # Constant-speed synthesis: every phrase at the same natural pace
        # (config.speed_base). No density-driven acceleration. This trades
        # perfect sync for prosodic consistency — the user prefers ES to
        # sound uniform even if it drifts slightly past SRT cues.
        use_constant_speed = getattr(self.cfg, "constant_speed", False)
        allow_tail = getattr(self.cfg, "allow_video_tail_extension", False)

        for i, block in enumerate(planned):
            if not block.text or len(block.text) < 2:
                continue

            if use_constant_speed:
                speed = self.cfg.speed_base
            else:
                density = len(block.text) / max(block.allocated_ms, 1)
                speed = self.drift.check_density(i, density)

            try:
                raw_audio = self.synthesizer.generate(
                    block.text, ref_wav, speed=speed,
                )

                # Gentle rescue stretch (max 1.10x) only if TTS overflows
                # the slot + generous overflow tolerance. We intentionally
                # avoid aggressive compression so delivery stays natural.
                target_ms = block.allocated_ms + max_overflow

                # When video tail extension is enabled we don't cap against
                # end-of-video here: the mixer/muxer will pad the video to
                # fit the final TTS instead of truncating a phrase.
                if not allow_tail and video_duration_ms is not None:
                    room_to_end = video_duration_ms - block.target_start_ms
                    if room_to_end > 0:
                        target_ms = min(target_ms, room_to_end)
                    else:
                        logger.warning(
                            "Phrase %d starts past video end; skipping", i,
                        )
                        continue

                fitted = stretch_audio(
                    raw_audio,
                    target_duration_ms=target_ms,
                    max_ratio=self.cfg.max_compression_ratio,
                    min_ratio=self.cfg.min_compression_ratio,
                )

                # When tail extension is disabled, truncate only as last resort.
                if not allow_tail and video_duration_ms is not None:
                    room_to_end = video_duration_ms - block.target_start_ms
                    if room_to_end > 0 and len(fitted) > room_to_end:
                        fade_ms = min(60, room_to_end // 4)
                        fitted = fitted[:room_to_end].fade_out(fade_ms)

                # Prepend micro-silence for natural attack
                if lead_silence > 0:
                    pad = AudioSegment.silent(
                        duration=lead_silence,
                        frame_rate=fitted.frame_rate,
                    )
                    fitted = pad + fitted

                # Anchor at SRT start_ms (no accumulated drift)
                segments.append(TtsSegment(
                    audio=fitted,
                    start_ms=block.target_start_ms,
                    end_ms=block.target_start_ms + len(fitted),
                ))

            except Exception:
                logger.exception("Error synthesizing phrase %d", i)

            if i % 10 == 0:
                logger.info("Progress: %d / %d phrases", i, total)
                self._report(3, 6, f"Synthesizing: {i}/{total}")

        # Resolve overlaps: if phrase N ends after N+1 starts, nudge N+1 right
        # within the following gap (capped so we never push past the next slot
        # or the end of the video).
        self._resolve_overlaps(segments, planned, video_duration_ms)

        # Speech-anchored (nivel 3): close trailing silence by pulling the
        # next phrase earlier when the current TTS finishes well before the
        # planned slot ends. Keeps natural inter-phrase pad.
        if getattr(self.cfg, "compact_trailing_silence", False):
            self._compact_trailing_silence(segments)

        return segments

    def _resolve_overlaps(
        self,
        segments: list[TtsSegment],
        planned: list,
        video_duration_ms: int | None = None,
    ) -> None:
        """Push later segments forward whenever previous overflows.

        Cascading push: when phrase N overflows into N+1, shift N+1 fully, and
        the shift propagates naturally to N+2, N+3... on the next iterations.
        This avoids audible cuts at the cost of progressive drift from SRT.
        Final drift is absorbed by video tail extension (or clipping if the
        feature is off).
        """
        allow_tail = getattr(self.cfg, "allow_video_tail_extension", False)

        for i in range(len(segments) - 1):
            cur, nxt = segments[i], segments[i + 1]
            if cur.end_ms <= nxt.start_ms:
                continue

            overlap = cur.end_ms - nxt.start_ms

            # When tail extension is allowed we shift fully; the muxer pads
            # the video to fit. Otherwise cap so we don't push past end-of-video.
            if allow_tail or video_duration_ms is None:
                shift = overlap
            else:
                room = max(0, video_duration_ms - nxt.end_ms)
                shift = min(overlap, room)

            if shift > 0:
                nxt.start_ms += shift
                nxt.end_ms += shift

    def _compact_trailing_silence(self, segments: list) -> None:
        """Pull later phrases earlier to close silence left by short TTS.

        Context: with speech-anchored slots (nivel 3 dub track) the slot
        duration = real speaker talk time. The ES TTS often finishes before
        the slot is over — if we leave the next phrase anchored to its
        original start, the dub has audible silence inside the speaker's
        continuous speech. We shift later phrases backwards while keeping
        a minimum inter-phrase gap so the delivery still sounds natural.

        Only runs when ``compact_trailing_silence`` is on. Threshold and
        min gap come from config. Cascades — if phrase i+1 is pulled back,
        phrase i+2 recomputes relative to the new i+1 end.
        """
        threshold = self.cfg.compact_trailing_silence_threshold_ms
        min_gap = self.cfg.compact_min_gap_ms

        for i in range(len(segments) - 1):
            cur, nxt = segments[i], segments[i + 1]
            gap = nxt.start_ms - cur.end_ms
            if gap <= threshold:
                continue
            # Pull nxt back but preserve min_gap for breath.
            shift_back = gap - min_gap
            if shift_back <= 0:
                continue
            nxt.start_ms -= shift_back
            nxt.end_ms -= shift_back

    # ------------------------------------------------------------------
    # FFmpeg muxing
    # ------------------------------------------------------------------

    @staticmethod
    def _mux_video(
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        """Mux the original video + new ES audio + any SRTs.

        Result: one MKV with only the ES dub audio track (EN discarded — the
        original file keeps the EN track). Both subtitle tracks (EN + ES)
        are kept if present so the user can still switch subs in the player.
        """
        base = video_path.with_suffix("")
        # ES subtitle track for the MKV: prefer the literal ES srt (good for
        # reading), never the dubbing-adapted one (.dub.es.srt), which is
        # shortened for iso-synchrony and reads awkwardly as subtitles.
        srt_es = None
        for _sfx in (".es.srt", ".ES.srt", "_ES.srt", "_ESP.srt", "_ESP_DUB.srt"):
            candidate = base.parent / f"{base.name}{_sfx}"
            if candidate.exists():
                srt_es = candidate
                break
        srt_en = None
        for _sfx in (".en.srt", ".EN.srt", "_EN.srt", ".srt"):
            candidate = base.parent / f"{base.name}{_sfx}"
            if candidate.exists() and candidate != srt_es:
                srt_en = candidate
                break

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(audio_path),
        ]
        if srt_en:
            cmd += ["-i", str(srt_en)]
        if srt_es:
            cmd += ["-i", str(srt_es)]

        # Map: video + ONLY the new ES audio (EN original audio discarded by
        # design — the dubbed MKV is meant to be watched in Spanish; the
        # original EN track lives in the source file).
        cmd += [
            "-map", "0:v",
            "-map", "1:a",
        ]
        sub_inputs = []
        if srt_en:
            sub_inputs.append(("2", "eng", "English"))
        if srt_es:
            idx = "3" if srt_en else "2"
            sub_inputs.append((idx, "spa", "Español"))
        for idx, _lang, _title in sub_inputs:
            cmd += ["-map", f"{idx}:0"]

        # Stream copy; only new ES audio is re-encoded. Sin -shortest: el
        # mixer empareja el audio a la duración exacta del vídeo; añadir
        # -shortest truncaba a 1:14 cuando Demucs entregaba un background
        # ligeramente más corto (imagen + audio cortados abruptamente).
        cmd += [
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-c:s", "srt",
            "-metadata:s:a:0", "language=spa",
            "-metadata:s:a:0", "title=Doblaje (ES)",
            "-disposition:a:0", "default",
        ]
        for sub_i, (_idx, lang, title) in enumerate(sub_inputs):
            cmd += [
                f"-metadata:s:s:{sub_i}", f"language={lang}",
                f"-metadata:s:s:{sub_i}", f"title={title}",
            ]

        cmd += [str(output_path)]
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_consecutive_blocks(
        blocks: list[SrtBlock], max_gap_ms: int, max_chars: int = 200,
    ) -> list[SrtBlock]:
        """Group adjacent SRT blocks whose gap is below ``max_gap_ms``.

        Keeps start from the first block and end from the last so the merged
        block stays anchored to the original video timestamps. Produces a
        single TTS call per merged block → continuous prosody, no mid-speech
        resets. Blocks separated by a real pause (> max_gap_ms) stay split.
        Bounded by ``max_chars`` so the merged text never forces Chatterbox
        to split internally (which resets prosody and risks truncation).
        """
        if not blocks:
            return []
        merged: list[SrtBlock] = []
        cur = SrtBlock(
            index=blocks[0].index,
            start_ms=blocks[0].start_ms,
            end_ms=blocks[0].end_ms,
            text=blocks[0].text,
        )
        for nxt in blocks[1:]:
            gap = nxt.start_ms - cur.end_ms
            combined_len = len(cur.text) + 1 + len(nxt.text)
            if gap <= max_gap_ms and combined_len <= max_chars:
                cur = SrtBlock(
                    index=cur.index,
                    start_ms=cur.start_ms,
                    end_ms=nxt.end_ms,
                    text=f"{cur.text} {nxt.text}".strip(),
                )
            else:
                merged.append(cur)
                cur = SrtBlock(
                    index=nxt.index,
                    start_ms=nxt.start_ms,
                    end_ms=nxt.end_ms,
                    text=nxt.text,
                )
        merged.append(cur)
        return merged

    @staticmethod
    def _probe_video_duration_ms(video_path: Path) -> int | None:
        """Return video duration in ms via ffprobe, or None on failure."""
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True, text=True, timeout=10, check=True,
            )
            return int(float(out.stdout.strip()) * 1000)
        except Exception as exc:
            logger.warning("ffprobe failed for %s: %s", video_path, exc)
            return None

    def _report(self, step: int, total: int, message: str) -> None:
        logger.info("[%d/%d] %s", step, total, message)
        if self._progress_cb:
            self._progress_cb(step, total, message)

    @staticmethod
    def _cleanup(*paths: Path) -> None:
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
