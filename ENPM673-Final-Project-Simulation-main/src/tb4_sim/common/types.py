"""Dataclasses shared by the task implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Simple axis-aligned bounding box."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @staticmethod
    def from_points(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> "BoundingBox":
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x = int(min(xs))
        min_y = int(min(ys))
        max_x = int(max(xs))
        max_y = int(max(ys))
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max(1, max_x - min_x),
            height=max(1, max_y - min_y),
        )


@dataclass
class DetectionResult:
    """Generic detection payload for overlays and control decisions."""

    label: str
    bbox: BoundingBox
    score: float = 0.0
    quad: list[tuple[int, int]] | None = None
    direction: str | None = None
    distance_m: float | None = None
    ttc_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
