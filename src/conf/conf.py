from dotenv import load_dotenv
from typing import TypedDict
import os


class AppConfig(TypedDict):
    document_intelligence_endpoint: str
    document_intelligence_api_key: str | None
    storage_blob_endpoint: str | None
    storage_blob_api_key: str | None
    ai_search_endpoint: str | None
    ai_search_api_key: str | None


def get_config() -> AppConfig:
    load_dotenv()

    document_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

    if not document_intelligence_endpoint:
        raise ValueError("Missing endpoint. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT.")

    document_intelligence_api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")

    blob_endpoint = (os.getenv("AZURE_STORAGE_BLOB_ENDPOINT") or "").strip() or None
    blob_api_key = (os.getenv("AZURE_STORAGE_BLOB_API_KEY") or "").strip() or None
    ai_search_endpoint = (os.getenv("AZURE_AI_SEARCH_ENDPOINT") or "").strip() or None
    ai_search_api_key = (os.getenv("AZURE_AI_SEARCH_API_KEY") or "").strip() or None

    return {
        "document_intelligence_endpoint": document_intelligence_endpoint.rstrip("/"),
        "document_intelligence_api_key": document_intelligence_api_key,
        "storage_blob_endpoint": blob_endpoint.rstrip("/") if blob_endpoint else None,
        "storage_blob_api_key": blob_api_key,
        "ai_search_endpoint": ai_search_endpoint.rstrip("/") if ai_search_endpoint else None,
        "ai_search_api_key": ai_search_api_key,
    }
