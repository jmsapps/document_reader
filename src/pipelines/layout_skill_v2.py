from typing import Any, Dict

from ..services.document_layout_skill_v2 import DocumentLayoutSkillV2Service
from .types import LayoutSkillV2PipelineOptions


class LayoutSkillV2Pipeline:
    """Proof-of-concept layout flow targeting one final index."""

    def run(self, options: LayoutSkillV2PipelineOptions) -> Dict[str, Any]:
        service = DocumentLayoutSkillV2Service()
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

        raise ValueError("Missing --src for layout-skill-v2 when not running --demo.")
