from typing import Any

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from src.auth.iam import IAM
from src.conf.conf import get_config


class AISearchService:
    """Basic Azure AI Search service client (API key or IAM)."""

    def __init__(self) -> None:
        config = get_config()
        self.endpoint = config["ai_search_endpoint"]
        self.credential = self._build_credential(config["ai_search_api_key"])

    @staticmethod
    def _build_credential(api_key: str | None) -> AzureKeyCredential | TokenCredential:
        if api_key:
            return AzureKeyCredential(api_key)
        return IAM().get_credential()

    def get_search_client(self, index_name: str) -> SearchClient:
        if not self.endpoint:
            raise ValueError("Endpoint required for getting ai search client.")

        return SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=self.credential,
        )

    def get_search_index_client(self) -> SearchIndexClient:
        if not self.endpoint:
            raise ValueError("Endpoint required for getting ai search client.")

        return SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential,
        )

    def test_connection(self) -> dict[str, Any]:
        search_index_client = self.get_search_index_client()

        first = next(search_index_client.list_index_names(), None)

        return {"ok": True, "first_index": first}

    def list_indexes(self) -> list[str]:
        search_index_client = self.get_search_index_client()

        return list(search_index_client.list_index_names())

    def search(self, index_name: str, query: str = "*", top: int = 5) -> dict[str, Any]:
        client = self.get_search_client(index_name=index_name)
        results = client.search(search_text=query, top=top)
        docs = [dict(doc) for doc in results]

        return {
            "count": len(docs),
            "value": docs,
        }
