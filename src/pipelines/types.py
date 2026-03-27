from dataclasses import dataclass
from typing import Literal

from ..services.document_intelligence.extractor import ContentFormat

PipelineName = Literal["direct", "layout-skill", "layout-skill-v2"]


@dataclass(frozen=True)
class DirectPipelineOptions:
    src: str
    model_id: str
    content_format: ContentFormat


@dataclass(frozen=True)
class LayoutSkillPipelineOptions:
    src: str
    input_container: str
    name_prefix: str
    chunk_size: int
    chunk_overlap: int
    hard_refresh: bool


@dataclass(frozen=True)
class LayoutSkillV2PipelineOptions:
    src: str | None
    demo: bool
    chunk_container: str
    name_prefix: str
    chunk_size: int
    chunk_overlap: int
    hard_refresh: bool
