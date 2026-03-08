import argparse
from pathlib import Path
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError

from src.conf import get_config
from src.pipelines import DirectPipeline, DirectPipelineOptions
from src.services.storage_account import AzureStorageAccountService
from src.storage import LocalOutputStore


def _default_output_path(src: str, content_format: str) -> str:
    folder = content_format
    prefix = f"{content_format}_"
    suffix = ".json"

    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        stem = Path(parsed.path).stem or "document"
    else:
        stem = Path(src).stem or "document"

    return f"{folder}/{prefix}{stem}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze documents with Azure Document Intelligence and save JSON.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--src",
        "-s",
        required=True,
        help="Local path or URL. Supports pdf/images/docx/pptx/xlsx/html and markdown.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="prebuilt-layout",
        help="Document Intelligence model id. Default: prebuilt-layout",
    )
    parser.add_argument(
        "--content-format",
        "-f",
        choices=["text", "markdown", "html"],
        default="text",
        help="Content format for `content` field. text|markdown from DI, html rendered from raw layout.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output path. If omitted, defaults to data/<content_format>/<content_format>_<input_name>.json.",
    )
    args = parser.parse_args()

    pipeline = DirectPipeline()
    options = DirectPipelineOptions(
        src=args.src,
        model_id=args.model,
        content_format=args.content_format,
    )

    try:
        payload = pipeline.run(options)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")

        return 2
    except ValueError as exc:
        print(f"Error: {exc}")

        return 2
    except HttpResponseError as exc:
        print(
            f"Document Intelligence request failed. status={exc.status_code} message={exc.message}"
        )

        return 3

    container = "data"
    filename = args.out or _default_output_path(args.src, args.content_format)
    config = get_config()
    storage_blob_endpoint = config.get("storage_blob_endpoint")
    storage_blob_api_key = config.get("storage_blob_api_key")

    if storage_blob_endpoint:
        storage_service = AzureStorageAccountService(
            endpoint=storage_blob_endpoint,
            api_key=storage_blob_api_key,
        )

        blob_name = filename.lstrip("/").replace("\\", "/")
        if isinstance(payload, str):
            saved_to = storage_service.upload_text(
                container_name=container,
                blob_name=blob_name,
                text=payload,
            )
        else:
            saved_to = storage_service.upload_json(
                container_name=container,
                blob_name=blob_name,
                payload=payload,
            )
    else:
        out_path = f"{container}/{filename}"
        store = LocalOutputStore()
        store.save(payload, out_path)
        saved_to = out_path

    print(f"Saved {saved_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
