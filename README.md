# Document Reader (Azure Document Intelligence)

Basic Python document reader:

- Accepts local files or URLs
- Supports `pdf/images/docx/pptx/xlsx/html` directly
- Supports `markdown` via lightweight MD->HTML conversion
- Content formats (`--content-format`):
  - `text`: DI content as plain text
  - `markdown`: DI content as markdown
  - `html`: app-generated HTML written into the JSON `content` field (third mode)
- Auth:
  - uses API key if present
  - otherwise uses IAM (`DefaultAzureCredential`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set .env vars:

```bash
DOCUMENTINTELLIGENCE_ENDPOINT="https://<your-di-name>.cognitiveservices.azure.com/"
DOCUMENTINTELLIGENCE_API_KEY="<your-api-key>"
```

**Note: omitting `DOCUMENTINTELLIGENCE_API_KEY` falls back to managed identity credentials.**

## Run

Basic output:

```bash
python document_reader.py --src ./sample.pdf
```

Analyze URL source:

```bash
python document_reader.py --src "https://<storage-url-or-public-url>"
```

Use `prebuilt-read` if you only need text:

```bash
python document_reader.py --src ./sample.pdf --model prebuilt-read
```

### Content format modes

`--content-format` supports three modes:

- `text` (default): DI plain text output in `content`
- `markdown`: DI markdown output in `content`
- `html`: HTML parsed output in `content`

## Default output paths

If `--out` is omitted:

- `data/<content_format>/<content_format>_<input_name>.json`

You can override with:

```bash
python document_reader.py --src ./sample.pdf --out ./any/path/output.json
```
