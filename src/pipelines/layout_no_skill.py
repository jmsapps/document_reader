from typing import Any, Dict

from ..services.document_layout_no_skill import DocumentLayoutNoSkillService
from .types import LayoutNoSkillPipelineOptions


class LayoutNoSkillPipeline:
    """Proof-of-concept layout flow targeting one final index."""

    def run(self, options: LayoutNoSkillPipelineOptions) -> Dict[str, Any]:
        service = DocumentLayoutNoSkillService()
        if options.demo:
            return service.run_demo(
                chunk_container=options.chunk_container,
                name_prefix=options.name_prefix,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                hard_refresh=options.hard_refresh,
            )

        if options.src:
            return service.run(
                src=options.src,
                chunk_container=options.chunk_container,
                name_prefix=options.name_prefix,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                hard_refresh=options.hard_refresh,
            )

        raise ValueError("Missing --src for layout-no-skill when not running --demo.")
