"""Default BJJ English terms that must be pronounced with English phonology.

All terms are lowercase. Multi-word terms are stored with single spaces.
The splitter matches case-insensitively on word boundaries; callers may
extend the set via DubbingConfig.xtts_en_terms_extra.
"""

from __future__ import annotations

DEFAULT_BJJ_EN_TERMS: frozenset[str] = frozenset({
    # Grips and arm controls
    "underhook", "underhooks",
    "overhook", "overhooks",
    "two on one",
    "wrist control",
    "collar tie",
    "russian tie",
    "lapel", "lapels",
    "sleeve",
    "grip",
    # Positions
    "guard",
    "closed guard",
    "open guard",
    "half guard",
    "full mount",
    "mount",
    "side control",
    "back control",
    "north south",
    "turtle",
    # Submissions
    "armbar",
    "triangle",
    "kimura",
    "americana",
    "heel hook",
    "rear naked choke",
    "guillotine",
    "bow and arrow",
    # Actions
    "sweep",
    "pass",
    "takedown",
    "shoot",
    "sprawl",
})
