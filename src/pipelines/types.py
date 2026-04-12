from dataclasses import dataclass
from typing import Literal

from ..services.document_intelligence.extractor import ContentFormat

PipelineName = Literal["direct", "layout-skill", "layout-no-skill", "layout-no-skill-v2"]


@dataclass(frozen=True)
class DirectPipelineOptions:
    src: str
    model_id: str
    content_format: ContentFormat


@dataclass(frozen=True)
class LayoutSkillPipelineOptions:
    src: str
    demo: bool
    input_container: str
    name_prefix: str
    chunk_size: int
    chunk_overlap: int
    hard_refresh: bool


@dataclass(frozen=True)
class LayoutNoSkillPipelineOptions:
    src: str | None
    demo: bool
    chunk_container: str
    name_prefix: str
    chunk_size: int
    chunk_overlap: int
    hard_refresh: bool


@dataclass(frozen=True)
class LayoutNoSkillV2PipelineOptions:
    src: str | None
    demo: bool
    chunk_container: str
    name_prefix: str
    chunk_size: int
    chunk_overlap: int
    content_format: Literal["text", "markdown"]
    hard_refresh: bool
