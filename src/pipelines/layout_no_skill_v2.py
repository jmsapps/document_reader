from typing import Any, Dict

from ..services.document_layout_no_skill_v2 import DocumentLayoutNoSkillV2Service
from .types import LayoutNoSkillV2PipelineOptions


class LayoutNoSkillV2Pipeline:
    """Sibling no-skill layout flow with grounded semantic figure retrieval."""

    def run(self, options: LayoutNoSkillV2PipelineOptions) -> Dict[str, Any]:
        service = DocumentLayoutNoSkillV2Service()
        if options.demo:
            return service.run_demo(
                chunk_container=options.chunk_container,
                name_prefix=options.name_prefix,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                content_format=options.content_format,
                hard_refresh=options.hard_refresh,
            )

        if options.src:
            return service.run(
                src=options.src,
                chunk_container=options.chunk_container,
                name_prefix=options.name_prefix,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                content_format=options.content_format,
                hard_refresh=options.hard_refresh,
            )

        raise ValueError("Missing --src for layout-no-skill-v2 when not running --demo.")
