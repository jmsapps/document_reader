# Document Reader (Azure Document Intelligence)

Basic Python document reader:

- Accepts local files or URLs
- Supports `pdf/images/docx/pptx/xlsx/html` directly
- Supports `markdown` via lightweight MD->HTML conversion
- Content formats (`--content-format`):
  - `text`: DI content as plain text
  - `markdown`: DI content as markdown
  - `html`: app-generated HTML written into raw JSON `content` field (third mode)
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

Raw output (default text content):

```bash
python document_reader.py --src ./sample.pdf
```

Raw output with markdown content:

```bash
python document_reader.py --src ./sample.docx --content-format markdown
```

Raw output with HTML content:

```bash
python document_reader.py --src ./sample.pdf --content-format html
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

- `data/raw/raw_<input_name>.json`

You can override with:

```bash
python document_reader.py --src ./sample.pdf --out ./any/path/output.json
```
