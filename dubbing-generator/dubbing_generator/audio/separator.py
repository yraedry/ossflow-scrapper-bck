"""Demucs wrapper for background / vocal separation."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


class AudioSeparator:
    """Separate background and vocal tracks using Demucs (htdemucs)."""

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config

    def separate(self, video_path: Path) -> Path:
        """Separate background audio from *video_path*.

        Returns the path to the background WAV file.
        If the background file already exists it is reused.
        """
        base_folder = video_path.parent
        name_no_ext = video_path.stem
        final_bg_path = base_folder / f"{name_no_ext}_BACKGROUND.wav"

        if final_bg_path.exists():
            logger.info("Background file already exists: %s", final_bg_path)
            return final_bg_path

        logger.info("Separating audio with Demucs (htdemucs two-stems=vocals)...")

        temp_audio_name = "temp_demucs_input"
        temp_audio_wav = base_folder / f"{temp_audio_name}.wav"
        separated_dir = base_folder / "separated"

        try:
            # Extract raw audio from the video
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le",
                    str(temp_audio_wav),
                ],
                check=True,
            )

            # Run Demucs
            result = subprocess.run(
                [
                    "demucs", "-n", "htdemucs",
                    "--two-stems=vocals",
                    "-o", str(separated_dir),
                    str(temp_audio_wav),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Demucs stderr:\n%s", result.stderr)
                logger.error("Demucs stdout:\n%s", result.stdout)
                raise subprocess.CalledProcessError(result.returncode, "demucs", result.stderr)


            # Demucs outputs no_vocals.wav for the background stem
            demucs_out = separated_dir / "htdemucs" / temp_audio_name / "no_vocals.wav"
            vocals_out = separated_dir / "htdemucs" / temp_audio_name / "vocals.wav"

            if not demucs_out.exists():
                raise FileNotFoundError(
                    f"Demucs did not produce expected output at {demucs_out}"
                )

            shutil.move(str(demucs_out), str(final_bg_path))

            # Save vocals stem for XTTS voice reference (clean speech, no music)
            vocals_dest = base_folder / f"{name_no_ext}_VOCALS.wav"
            if vocals_out.exists():
                shutil.move(str(vocals_out), str(vocals_dest))
                logger.info("Vocals stem saved to %s", vocals_dest)

            logger.info("Background saved to %s", final_bg_path)
            return final_bg_path

        finally:
            # Cleanup temporary files
            if temp_audio_wav.exists():
                try:
                    os.remove(temp_audio_wav)
                except OSError:
                    pass
            if separated_dir.exists():
                shutil.rmtree(separated_dir, ignore_errors=True)
