from dotenv import load_dotenv
import os
from typing import TypedDict


class AppConfig(TypedDict):
    document_intelligence_endpoint: str
    document_intelligence_api_key: str | None
    storage_blob_endpoint: str | None
    storage_blob_api_key: str | None
    ai_search_endpoint: str | None
    ai_search_api_key: str | None
    embedding_provider: str | None
    foundry_endpoint: str | None
    foundry_api_key: str | None
    openai_endpoint: str | None
    openai_api_key: str | None
    openai_api_version: str | None
    openai_embedding_deployment: str | None
    openai_embedding_dimensions: int | None
    ai_vision_endpoint: str | None
    ai_vision_api_key: str | None
    ai_vision_model_version: str | None
    ai_vision_embedding_dimensions: int | None
    ai_vision_timeout_seconds: int | None


def get_config() -> AppConfig:
    load_dotenv(dotenv_path=".env", override=True)

    document_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

    if not document_intelligence_endpoint:
        raise ValueError("Missing endpoint. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT.")

    document_intelligence_api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")

    blob_endpoint = (os.getenv("AZURE_STORAGE_BLOB_ENDPOINT") or "").strip() or None
    blob_api_key = (os.getenv("AZURE_STORAGE_BLOB_API_KEY") or "").strip() or None
    ai_search_endpoint = (os.getenv("AZURE_AI_SEARCH_ENDPOINT") or "").strip() or None
    ai_search_api_key = (os.getenv("AZURE_AI_SEARCH_API_KEY") or "").strip() or None
    embedding_provider = (os.getenv("AZURE_EMBEDDING_PROVIDER") or "").strip() or None
    foundry_endpoint = (os.getenv("AZURE_FOUNDRY_ENDPOINT") or "").strip() or None
    foundry_api_key = (os.getenv("AZURE_FOUNDRY_API_KEY") or "").strip() or None
    openai_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip() or None
    openai_api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip() or None
    openai_api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "").strip() or None
    openai_embedding_deployment = (
        (os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "").strip() or None
    )
    openai_embedding_dimensions_raw = (os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS") or "").strip()
    openai_embedding_dimensions = (
        int(openai_embedding_dimensions_raw) if openai_embedding_dimensions_raw else None
    )
    ai_vision_endpoint = (os.getenv("AZURE_AI_VISION_ENDPOINT") or "").strip() or None
    ai_vision_api_key = (os.getenv("AZURE_AI_VISION_API_KEY") or "").strip() or None
    ai_vision_model_version = (os.getenv("AZURE_AI_VISION_MODEL_VERSION") or "").strip() or None
    ai_vision_embedding_dimensions_raw = (os.getenv("AZURE_AI_VISION_EMBEDDING_DIMENSIONS") or "").strip()
    ai_vision_embedding_dimensions = (
        int(ai_vision_embedding_dimensions_raw) if ai_vision_embedding_dimensions_raw else None
    )
    ai_vision_timeout_seconds_raw = (os.getenv("AZURE_AI_VISION_TIMEOUT_SECONDS") or "").strip()
    ai_vision_timeout_seconds = (
        int(ai_vision_timeout_seconds_raw) if ai_vision_timeout_seconds_raw else None
    )

    return {
        "document_intelligence_endpoint": document_intelligence_endpoint.rstrip("/"),
        "document_intelligence_api_key": document_intelligence_api_key,
        "storage_blob_endpoint": blob_endpoint.rstrip("/") if blob_endpoint else None,
        "storage_blob_api_key": blob_api_key,
        "ai_search_endpoint": ai_search_endpoint.rstrip("/") if ai_search_endpoint else None,
        "ai_search_api_key": ai_search_api_key,
        "embedding_provider": embedding_provider,
        "foundry_endpoint": foundry_endpoint.rstrip("/") if foundry_endpoint else None,
        "foundry_api_key": foundry_api_key,
        "openai_endpoint": openai_endpoint.rstrip("/") if openai_endpoint else None,
        "openai_api_key": openai_api_key,
        "openai_api_version": openai_api_version,
        "openai_embedding_deployment": openai_embedding_deployment,
        "openai_embedding_dimensions": openai_embedding_dimensions,
        "ai_vision_endpoint": ai_vision_endpoint.rstrip("/") if ai_vision_endpoint else None,
        "ai_vision_api_key": ai_vision_api_key,
        "ai_vision_model_version": ai_vision_model_version,
        "ai_vision_embedding_dimensions": ai_vision_embedding_dimensions,
        "ai_vision_timeout_seconds": ai_vision_timeout_seconds,
    }
