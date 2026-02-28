import argparse
from pathlib import Path
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError

from src.document_intelligence.reader import analyze_any, save_output


def _default_output_path(src: str) -> str:
    folder = "raw"
    prefix = "raw_"
    suffix = ".json"

    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        stem = Path(parsed.path).stem or "document"
    else:
        stem = Path(src).stem or "document"

    out_dir = Path("data") / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{prefix}{stem}{suffix}")


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
        help="Output path. If omitted, defaults to data/raw.",
    )
    args = parser.parse_args()

    try:
        payload = analyze_any(args.src, model_id=args.model, content_format=args.content_format)
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

    out_path = args.out or _default_output_path(args.src)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_output(payload, out_path)

    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
