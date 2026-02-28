from typing import Any

from ..services.document_intelligence.extractor import ContentFormat, analyze_any
from ..storage.output_store import LocalOutputStore


def save_output(payload: Any, out_path: str) -> None:
    LocalOutputStore().save(payload, out_path)


__all__ = ["ContentFormat", "analyze_any", "save_output"]
