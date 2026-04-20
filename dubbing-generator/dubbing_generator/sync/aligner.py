"""Phrase synchronization with gap-aware time allocation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


@dataclass
class SrtBlock:
    """Parsed SRT subtitle block."""
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class PlannedBlock:
    """A block with its allocated time budget."""
    text: str
    target_start_ms: int
    target_end_ms: int
    allocated_ms: int


class SyncAligner:
    """Plan timing for TTS phrases respecting SRT timestamps.

    Each phrase gets its subtitle slot PLUS a share of the gap before the next
    subtitle (when the gap is > inter_phrase_pad_ms). This way Spanish text
    (usually longer) has room without aggressive time compression.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config

    def plan(
        self,
        blocks: list[SrtBlock],
        video_duration_ms: int | None = None,
    ) -> list[PlannedBlock]:
        """Return PlannedBlocks with slot + borrowed gap time.

        If ``video_duration_ms`` is given, the last phrase borrows the gap up
        to the end of the video (minus a small tail pad), so the final ES line
        has room without spilling past the video end.
        """
        if not blocks:
            return []

        planned: list[PlannedBlock] = []
        n = len(blocks)
        pad = self.cfg.inter_phrase_pad_ms

        for i, block in enumerate(blocks):
            # Borrow from gap to next subtitle (keep small pad as silence)
            if i < n - 1:
                gap_ms = blocks[i + 1].start_ms - block.end_ms
                borrow_ms = max(0, gap_ms - pad)
            else:
                # Last block: borrow whatever remains until the end of the
                # video so a long ES line has room instead of spilling over.
                if video_duration_ms is not None:
                    tail_gap = video_duration_ms - block.end_ms
                    borrow_ms = max(0, tail_gap - pad)
                else:
                    borrow_ms = 0

            slot_ms = max(
                block.duration_ms + borrow_ms,
                self.cfg.min_phrase_duration_ms,
            )

            planned.append(PlannedBlock(
                text=block.text,
                target_start_ms=block.start_ms,
                target_end_ms=block.start_ms + slot_ms,
                allocated_ms=slot_ms,
            ))

        return planned
