from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    point_id: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    point_id: str
    score: float
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

