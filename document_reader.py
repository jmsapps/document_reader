import argparse
from pathlib import Path
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError

from src.conf import get_config
from src.pipelines import (
    DirectPipeline,
    DirectPipelineOptions,
    LayoutSkillPipeline,
    LayoutSkillPipelineOptions,
    LayoutSkillV2Pipeline,
    LayoutSkillV2PipelineOptions,
    PipelineName,
)
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


def _default_layout_output_path(src: str) -> str:
    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        stem = Path(parsed.path).stem or "document"
    else:
        stem = Path(src).stem or "document"

    return f"layout-skill/layout-skill_{stem}.json"


def _default_layout_v2_output_path(src: str | None, demo: bool) -> str:
    if demo:
        return "layout-skill-v2/layout-skill-v2_demo.json"

    if not src:
        stem = "document"
    else:
        parsed = urlparse(src)
        if parsed.scheme in ("http", "https"):
            stem = Path(parsed.path).stem or "document"
        else:
            stem = Path(src).stem or "document"

    return f"layout-skill-v2/layout-skill-v2_{stem}.json"


def main() -> int:
    base_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    base_parser.add_argument(
        "--pipeline",
        "-p",
        choices=["direct", "layout-skill", "layout-skill-v2"],
        default="direct",
        help="Pipeline to run. direct uses Document Intelligence SDK, layout-skill uses Azure AI Search skillset, layout-skill-v2 is the one-index proof of concept.",
    )
    known_args, _ = base_parser.parse_known_args()
    pipeline_name: PipelineName = known_args.pipeline

    parser = argparse.ArgumentParser(
        description="Analyze documents with Azure Document Intelligence and save JSON.",
        parents=[base_parser],
        allow_abbrev=False,
    )
    parser.add_argument(
        "--src",
        "-s",
        required=False,
        help="Local path or URL. Supports pdf/images/docx/pptx/xlsx/html and markdown.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output path. If omitted, defaults to data/<content_format>/<content_format>_<input_name>.json.",
    )

    if pipeline_name == "layout-skill":
        parser.add_argument(
            "--input-container",
            "-ic",
            default="layout-input",
            help="Blob container used as the layout-skill input source.",
        )
        parser.add_argument(
            "--name-prefix",
            "-np",
            default="document-layout",
            help="Prefix for Azure AI Search objects created by layout-skill.",
        )
        parser.add_argument(
            "--chunk-size",
            "-cs",
            type=int,
            default=2000,
            help="Max characters per text section for layout-skill chunking.",
        )
        parser.add_argument(
            "--chunk-overlap",
            "-co",
            type=int,
            default=200,
            help="Overlap in characters between adjacent layout-skill chunks.",
        )
        parser.add_argument(
            "--hard-refresh",
            "-hr",
            dest="hard_refresh",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Delete existing Search objects and reset the input container before running.",
        )
    elif pipeline_name == "layout-skill-v2":
        parser.add_argument(
            "--chunk-container",
            "-cc",
            default="chunk-container",
            help="Blob container used to store derived per-source JSON artifacts for v2.",
        )
        parser.add_argument(
            "--name-prefix",
            "-np",
            default="document-layout-v2",
            help="Prefix for the Azure AI Search target index created by layout-skill-v2.",
        )
        parser.add_argument(
            "--chunk-size",
            "-cs",
            type=int,
            default=500,
            help="Max characters per normalized chunk for layout-skill-v2.",
        )
        parser.add_argument(
            "--chunk-overlap",
            "-co",
            type=int,
            default=50,
            help="Overlap in characters between adjacent layout-skill-v2 chunks.",
        )
        parser.add_argument(
            "--hard-refresh",
            "-hr",
            dest="hard_refresh",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Delete and recreate the layout-skill-v2 target index before loading records.",
        )
        parser.add_argument(
            "--demo",
            "-d",
            action="store_true",
            help="Run the layout-skill-v2 demo over the files in the demo folder.",
        )
    else:
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
    args = parser.parse_args()

    if pipeline_name != "layout-skill-v2" and not args.src:
        parser.error("--src is required unless --pipeline layout-skill-v2 --demo is used.")

    if pipeline_name == "layout-skill-v2" and not args.demo and not args.src:
        parser.error("--src is required for layout-skill-v2 when not running --demo.")

    try:
        if pipeline_name == "layout-skill":
            payload = LayoutSkillPipeline().run(
                LayoutSkillPipelineOptions(
                    src=args.src,
                    input_container=args.input_container,
                    name_prefix=args.name_prefix,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    hard_refresh=args.hard_refresh,
                )
            )
        elif pipeline_name == "layout-skill-v2":
            payload = LayoutSkillV2Pipeline().run(
                LayoutSkillV2PipelineOptions(
                    src=args.src,
                    demo=args.demo,
                    chunk_container=args.chunk_container,
                    name_prefix=args.name_prefix,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    hard_refresh=args.hard_refresh,
                )
            )
        else:
            payload = DirectPipeline().run(
                DirectPipelineOptions(
                    src=args.src,
                    model_id=args.model,
                    content_format=args.content_format,
                )
            )
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

    if args.out:
        filename = args.out
    elif pipeline_name == "layout-skill":
        filename = _default_layout_output_path(args.src)
    elif pipeline_name == "layout-skill-v2":
        filename = _default_layout_v2_output_path(args.src, args.demo)
    else:
        filename = _default_output_path(args.src, args.content_format)

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
