from typing import Any, Dict

from ..services.document_intelligence import analyze_any
from .types import DirectPipelineOptions


class DirectPipeline:
    """Current extraction flow backed by direct Document Intelligence SDK calls."""

    def run(self, options: DirectPipelineOptions) -> Dict[str, Any]:
        return analyze_any(
            src=options.src,
            model_id=options.model_id,
            content_format=options.content_format,
        )
