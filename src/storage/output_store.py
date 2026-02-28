import json
from pathlib import Path
from typing import Any, Protocol


class OutputStore(Protocol):
    def save(self, payload: Any, out_path: str) -> None:
        ...


class LocalOutputStore:
    """File-system output store; can be replaced by blob-backed storage later."""

    def save(self, payload: Any, out_path: str) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
            return

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
