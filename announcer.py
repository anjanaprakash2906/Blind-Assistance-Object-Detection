"""
Announcement manager: decides WHICH detections are actually worth speaking
out loud, and WHEN, so the user isn't bombarded with the same "chair, center,
nearby" message twenty times a second.
"""

import time
from typing import List, Dict


class AnnouncementManager:
    def __init__(self, cooldown_seconds: float, max_per_cycle: int, prioritize_closest: bool):
        self.cooldown_seconds = cooldown_seconds
        self.max_per_cycle = max_per_cycle
        self.prioritize_closest = prioritize_closest
        self._last_spoken: Dict[str, float] = {}

    def _key(self, label: str, direction: str) -> str:
        # Same object type in the same direction is treated as "the same thing"
        # to avoid re-announcing every frame; a new direction is worth a new callout.
        return f"{label}:{direction}"

    def _is_on_cooldown(self, key: str) -> bool:
        last_time = self._last_spoken.get(key)
        if last_time is None:
            return False
        return (time.time() - last_time) < self.cooldown_seconds

    def _mark_spoken(self, key: str):
        self._last_spoken[key] = time.time()

    def build_phrases(self, detections: List[dict]) -> List[str]:
        """
        detections: list of dicts with keys: label, direction, proximity, area_ratio
        Returns a list of natural-language phrases ready for TTS, already
        filtered by cooldown and capped at max_per_cycle.
        """
        if self.prioritize_closest:
            detections = sorted(detections, key=lambda d: d["area_ratio"], reverse=True)

        phrases = []
        for det in detections:
            if len(phrases) >= self.max_per_cycle:
                break

            key = self._key(det["label"], det["direction"])
            if self._is_on_cooldown(key):
                continue

            phrase = self._format_phrase(det)
            phrases.append(phrase)
            self._mark_spoken(key)

        return phrases

    @staticmethod
    def _format_phrase(det: dict) -> str:
        label = det["label"]
        direction = det["direction"]
        proximity = det["proximity"]

        if proximity == "very close":
            return f"{label} very close, {direction}."
        elif proximity == "nearby":
            return f"{label} nearby, {direction}."
        else:
            return f"{label} ahead, {direction}."
