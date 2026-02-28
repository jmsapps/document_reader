from dotenv import load_dotenv
from typing import TypedDict
import os


class AppConfig(TypedDict):
    endpoint: str
    api_key: str | None


def get_config() -> AppConfig:
    load_dotenv()

    endpoint = (
        os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
        or os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
        or os.getenv("DI_ENDPOINT")
    )
    if not endpoint:
        raise ValueError(
            "Missing endpoint. Set DOCUMENTINTELLIGENCE_ENDPOINT (or DOCUMENT_INTELLIGENCE_ENDPOINT, DI_ENDPOINT)."
        )

    api_key = (
        os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
        or os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
        or os.getenv("DI_API_KEY")
    )

    return {
        "endpoint": endpoint.rstrip("/"),
        "api_key": api_key,
    }
