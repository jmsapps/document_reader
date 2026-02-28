import io
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from ..auth.iam import IAM
from ..conf.conf import get_config


class DocumentIntelligenceService:
    def __init__(self):
        config = get_config()
        credential = self._build_credential(config["api_key"])
        self.client = DocumentIntelligenceClient(
            endpoint=config["endpoint"],
            credential=credential,
        )

    @staticmethod
    def _build_credential(api_key: str | None):
        if api_key:
            return AzureKeyCredential(api_key)

        return IAM().get_credential()

    def analyze_url(
        self,
        url: str,
        model_id: str = "prebuilt-layout",
        content_format: DocumentContentFormat = DocumentContentFormat.TEXT,
    ) -> Any:
        poller = self.client.begin_analyze_document(
            model_id=model_id,
            body={"urlSource": url},
            output_content_format=content_format,
        )
        return poller.result()

    def analyze_bytes(
        self,
        data: bytes,
        model_id: str = "prebuilt-layout",
        content_format: DocumentContentFormat = DocumentContentFormat.TEXT
    ) -> Any:
        poller = self.client.begin_analyze_document(
            model_id=model_id,
            body=io.BytesIO(data),
            output_content_format=content_format
        )

        return poller.result()

    def analyze_file(
        self,
        path: Path,
        model_id: str = "prebuilt-layout",
        content_format: DocumentContentFormat = DocumentContentFormat.TEXT
    ) -> Any:
        with path.open("rb") as f:
            data = f.read()

        return self.analyze_bytes(data=data, model_id=model_id, content_format=content_format)
