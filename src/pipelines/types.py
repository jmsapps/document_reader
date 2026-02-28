from dataclasses import dataclass
from typing import Literal

from ..services.document_intelligence.extractor import ContentFormat

PipelineName = Literal["direct"]


@dataclass(frozen=True)
class DirectPipelineOptions:
    src: str
    model_id: str = "prebuilt-layout"
    content_format: ContentFormat = "text"
