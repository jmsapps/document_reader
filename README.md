# Document Reader (Azure Document Intelligence)

Basic Python document reader:

- Accepts local files or URLs
- Supports `pdf/images/docx/pptx/xlsx/html` directly
- Supports `markdown` via lightweight MD->HTML conversion
- Output modes:
  - `raw`: Azure Document Intelligence raw JSON
  - `normalized`: stable structured JSON
  - `html`: JSON payload with key fields + `content_html` string
- Auth:
  - uses API key if present
  - otherwise uses IAM (`DefaultAzureCredential`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set endpoint:

```bash
export DOCUMENTINTELLIGENCE_ENDPOINT="https://<your-di-name>.cognitiveservices.azure.com/"
```

You can also place these values in a local `.env` file; config is loaded automatically.

Choose one auth method:

```bash
# Option A: API key (preferred first)
export DOCUMENTINTELLIGENCE_API_KEY="<your-api-key>"

# Option B: IAM (DefaultAzureCredential)
# e.g., az login (local dev) or managed identity in Azure
```

## Run

Raw output:

```bash
python document_reader.py --src ./sample.pdf --mode raw
```

Normalized JSON:

```bash
python document_reader.py --src ./sample.docx --mode normalized
```

HTML payload JSON:

```bash
python document_reader.py --src ./sample.pdf --mode html
```

Analyze URL source:

```bash
python document_reader.py --src "https://<storage-url-or-public-url>" --mode raw
```

Use `prebuilt-read` if you only need text:

```bash
python document_reader.py --src ./sample.pdf --model prebuilt-read
```

## Default output paths

If `--out` is omitted:

- `--mode raw` -> `data/raw/raw_<input_name>.json`
- `--mode normalized` -> `data/normalized/normalized_<input_name>.json`
- `--mode html` -> `data/html/html_<input_name>.json`

You can override with:

```bash
python document_reader.py --src ./sample.pdf --mode raw --out ./any/path/output.json
```
