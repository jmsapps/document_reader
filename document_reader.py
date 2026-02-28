import argparse
from pathlib import Path
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError

from src.document_intelligence.reader import analyze_any, save_output


def _default_output_path(src: str, mode: str) -> str:
    if mode == "raw":
        folder = "raw"
        prefix = "raw_"
        suffix = ".json"
    elif mode == "normalized":
        folder = "normalized"
        prefix = "normalized_"
        suffix = ".json"
    else:
        folder = "html"
        prefix = "html_"
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
        description="Analyze documents with Azure Document Intelligence and save JSON."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Local path or URL. Supports pdf/images/docx/pptx/xlsx/html and markdown.",
    )
    parser.add_argument(
        "--model",
        default="prebuilt-layout",
        help="Document Intelligence model id. Default: prebuilt-layout",
    )
    parser.add_argument(
        "--mode",
        default="raw",
        choices=["raw", "normalized", "html"],
        help="Output mode. raw=DI JSON, normalized=canonical schema, html=structured HTML.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. If omitted, defaults to data/raw, data/normalized, or data/html based on mode.",
    )
    parser.add_argument(
        "--content-format",
        choices=["text", "markdown"],
        default="text",
        help="Content format of DI output. 'text' or 'markdown'. Default 'text'.",
    )
    args = parser.parse_args()

    try:
        payload = analyze_any(args.src, model_id=args.model, output_mode=args.mode, content_format=args.content_format)
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

    out_path = args.out or _default_output_path(args.src, args.mode)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_output(payload, out_path)

    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
