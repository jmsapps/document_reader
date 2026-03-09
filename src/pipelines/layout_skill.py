from typing import Any, Dict

from ..services.document_layout_skill import DocumentLayoutSkillService
from .types import LayoutSkillPipelineOptions


class LayoutSkillPipeline:
    """Extraction flow backed by Azure AI Search Document Layout skill."""

    def run(self, options: LayoutSkillPipelineOptions) -> Dict[str, Any]:
        service = DocumentLayoutSkillService()
        return service.run(
            src=options.src,
            input_container=options.input_container,
            name_prefix=options.name_prefix,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
            hard_refresh=options.hard_refresh,
        )
