import os
from pathlib import Path
from typing import TypedDict


class AppConfig(TypedDict):
    endpoint: str
    api_key: str | None


def _load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_config() -> AppConfig:
    _load_dotenv()

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
