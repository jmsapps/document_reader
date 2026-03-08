from dotenv import load_dotenv
from typing import TypedDict
import os


class AppConfig(TypedDict):
    document_intelligence_endpoint: str
    document_intelligence_api_key: str | None
    storage_blob_endpoint: str | None
    storage_blob_api_key: str | None


def get_config() -> AppConfig:
    load_dotenv()

    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")

    if not endpoint:
        raise ValueError("Missing endpoint. Set DOCUMENT_INTELLIGENCE_ENDPOINT.")

    api_key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")

    blob_endpoint = (os.getenv("AZURE_STORAGE_BLOB_ENDPOINT") or "").strip() or None
    blob_api_key = (os.getenv("AZURE_STORAGE_BLOB_API_KEY") or "").strip() or None

    return {
        "document_intelligence_endpoint": endpoint.rstrip("/"),
        "document_intelligence_api_key": api_key,
        "storage_blob_endpoint": blob_endpoint.rstrip("/") if blob_endpoint else None,
        "storage_blob_api_key": blob_api_key,
    }
