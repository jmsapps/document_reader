# Document Reader

Python CLI for four document-processing flows:

- `direct`: call Azure Document Intelligence directly and save structured output
- `layout-skill`: run an Azure AI Search pull indexer pipeline using `DocumentIntelligenceLayoutSkill` plus multimodal vectorization for text and images
- `layout-no-skill`: proof-of-concept one-index flow that derives chunk JSON per source and uploads records directly into one Azure AI Search target index
- `layout-no-skill-v2`: sibling one-index flow that converts figures into grounded semantic markdown and embeds final text with Azure OpenAI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipelines

### `direct`

Uses the Document Intelligence SDK against a local file or URL and writes JSON output.

Supported content formats:

- `text`
- `markdown`
- `html`

### `layout-skill`

Uses Azure AI Search with a pull-based pipeline:

- Blob Storage -> Data Source -> Skillset -> Indexer -> Indexes

Current skillset shape:

- `DocumentIntelligenceLayoutSkill` for chunking and image extraction
- `#Microsoft.Skills.Vision.VectorizeSkill` for text chunk vectors
- `#Microsoft.Skills.Vision.VectorizeSkill` for extracted image vectors

Outputs created by the pipeline:

- text chunk index with `chunk_vector`
- image index with `image_vector`
- cropped images projected to storage through the knowledge store

This path uses a Foundry-backed multimodal configuration and the Azure AI Search preview API.

### `layout-no-skill`

Uses an upstream-enrichment proof-of-concept flow:

- local demo files are processed source by source
- per-source derived JSON artifacts are written into `chunk-container`
- JSON input is chunked to simulate the current ingestion style
- PDF input is analyzed with Document Intelligence for text and figure metadata
- extracted figures are enriched with OCR plus Azure AI Vision image-analysis descriptions/tags
- Azure AI Vision multimodal vectors are added during enrichment
- all normalized records are uploaded into one final Azure AI Search index
- the shared one-index schema uses `content` and `contentVector`

### `layout-no-skill-v2`

Uses the same JSON-first no-skill ingestion style with a different figure pipeline:

- local demo files are processed source by source
- per-source derived JSON artifacts are written into `chunk-container`
- PDF input is analyzed with Document Intelligence for text and figure metadata
- each extracted figure produces a deterministic figure-analysis payload
- Azure OpenAI performs grounded multimodal interpretation over image plus labeled evidence
- Azure OpenAI verbalizes the grounded interpretation into semantic markdown
- all normalized records are uploaded into one final Azure AI Search index
- `contentVector` is generated from Azure OpenAI text embeddings over final text content

## Environment

### Required for `direct`

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<doc-intelligence>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<doc-intelligence-key>
```

If `AZURE_STORAGE_BLOB_ENDPOINT` is set, output is uploaded to blob storage. Otherwise output is written locally under `data/`.

### Required for `layout-skill`

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<doc-intelligence>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<doc-intelligence-key>

AZURE_STORAGE_BLOB_ENDPOINT=https://<storage-account>.blob.core.windows.net
AZURE_STORAGE_BLOB_API_KEY=<storage-account-key>

AZURE_AI_SEARCH_ENDPOINT=https://<search-service>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<search-admin-key>

AZURE_FOUNDRY_ENDPOINT=https://<foundry-resource>.services.ai.azure.com
AZURE_FOUNDRY_API_KEY=<foundry-key>
AZURE_AI_VISION_MODEL_VERSION=2023-04-15
AZURE_AI_VISION_EMBEDDING_DIMENSIONS=1024
```

Notes:

- `AZURE_FOUNDRY_ENDPOINT` can be the base Foundry endpoint or a project URL. The code normalizes `/api/projects/...` off the value before attaching it to the skillset.
- `AZURE_AI_VISION_MODEL_VERSION` defaults to `2023-04-15`.
- `AZURE_AI_VISION_EMBEDDING_DIMENSIONS` defaults to `1024`.
- The multimodal path is billable.

### Required for `layout-no-skill`

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<doc-intelligence>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<doc-intelligence-key>

AZURE_AI_SEARCH_ENDPOINT=https://<search-service>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<search-admin-key>

AZURE_STORAGE_BLOB_ENDPOINT=https://<storage-account>.blob.core.windows.net
AZURE_STORAGE_BLOB_API_KEY=<storage-account-key>
```

Notes:

- If blob storage is configured, derived artifacts are written to `chunk-container`.
- If blob storage is not configured, derived artifacts fall back to `local_documents/chunk-container/`.
- PDF and image processing in `layout-no-skill` requires blob storage so image/source URLs stay navigable.
- `layout-no-skill` expects real embedding configuration through `AZURE_EMBEDDING_PROVIDER` and currently supports `azure_ai_vision`.

### Required for `layout-no-skill-v2`

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<doc-intelligence>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<doc-intelligence-key>

AZURE_AI_SEARCH_ENDPOINT=https://<search-service>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<search-admin-key>

AZURE_STORAGE_BLOB_ENDPOINT=https://<storage-account>.blob.core.windows.net
AZURE_STORAGE_BLOB_API_KEY=<storage-account-key>

AZURE_OPENAI_ENDPOINT=https://<azure-openai-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<azure-openai-key>
AZURE_OPENAI_API_VERSION=v1
AZURE_OPENAI_INTERPRET_DEPLOYMENT=<chat-or-multimodal-deployment>
AZURE_OPENAI_VERBALIZATION_DEPLOYMENT=<chat-deployment>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment>
AZURE_OPENAI_EMBEDDING_DIMENSIONS=1536
```

Notes:

- `layout-no-skill-v2` uses Azure OpenAI `v1` API semantics.
- `AZURE_OPENAI_CHAT_DEPLOYMENT` remains a supported fallback for interpretation and verbalization when the purpose-specific deployment names are not set.
- The `v2` path does not require per-deployment model-version env vars.

## Run

### Direct pipeline

Analyze a local file:

```bash
python document_reader.py --pipeline direct --src ./sample.pdf
```

Analyze a URL:

```bash
python document_reader.py --pipeline direct --src "https://<public-or-storage-url>"
```

Use a different DI model:

```bash
python document_reader.py --pipeline direct --src ./sample.pdf --model prebuilt-read
```

Request markdown output:

```bash
python document_reader.py --pipeline direct --src ./sample.pdf --content-format markdown
```

### Layout-skill pipeline

Run the multimodal indexing pipeline:

```bash
python document_reader.py \
  --pipeline layout-skill \
  --src ./sample.pdf \
  --hard-refresh
```

Useful options:

```bash
python document_reader.py \
  --pipeline layout-skill \
  --src ./sample.pdf \
  --input-container layout-input \
  --name-prefix document-layout \
  --chunk-size 450 \
  --chunk-overlap 50 \
  --hard-refresh
```

Notes:

- Use `--hard-refresh` when changing index schema or skillset shape.
- The layout-skill path now caps chunk size for multimodal text vectorization safety. If you pass a larger `--chunk-size`, the service logs that it is reducing it.

### Layout-no-skill pipeline

Run the demo flow across the current demo assets:

```bash
python document_reader.py \
  --pipeline layout-no-skill \
  --demo \
  --hard-refresh
```

### Layout-no-skill-v2 pipeline

Run the grounded semantic demo flow across the current demo assets:

```bash
python document_reader.py \
  --pipeline layout-no-skill-v2 \
  --demo \
  --hard-refresh
```

Run `layout-no-skill-v2` against a single source:

```bash
python document_reader.py \
  --pipeline layout-no-skill-v2 \
  --src ./documents/demo_files/upload_file.json \
  --hard-refresh
```

Useful options:

```bash
python document_reader.py \
  --pipeline layout-no-skill-v2 \
  --demo \
  --chunk-container chunk-container \
  --name-prefix document-layout-no-skill-v2 \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --hard-refresh
```

Notes:

- `layout-no-skill-v2` persists figure-analysis, grounded interpretation, and final markdown support artifacts independently of the indexed contract.
- Figure-derived records in `v2` use text embeddings over semantic markdown, not image-byte embeddings.

Run `layout-no-skill` against a single source:

```bash
python document_reader.py \
  --pipeline layout-no-skill \
  --src ./documents/demo_files/upload_file.json \
  --hard-refresh
```

Useful options:

```bash
python document_reader.py \
  --pipeline layout-no-skill \
  --demo \
  --chunk-container chunk-container \
  --name-prefix document-layout-no-skill \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --hard-refresh
```


## Progress Logs

The `layout-skill` pipeline prints progress to stdout, including:

- storage container setup
- source upload
- datasource/index/skillset/indexer creation
- indexer start and polling status
- chunk and image fetch counts

Example:

```text
[document-layout-skill] Creating or updating skillset 'document-layout-skillset'
[document-layout-skill] Starting indexer 'document-layout-indexer'
[document-layout-skill] Indexer status: execution=inProgress result=pending
[document-layout-skill] Run finished with 12 chunks and 3 images
```

## Output

If `AZURE_STORAGE_BLOB_ENDPOINT` is configured, command output is uploaded to the `data` container.

Default object names:

- `direct`: `data/<content_format>/<content_format>_<input_name>.json`
- `layout-skill`: `data/layout-skill/layout-skill_<input_name>.json`
- `layout-no-skill`: `data/layout-no-skill/layout-no-skill_<input_name>.json` or `data/layout-no-skill/layout-no-skill_demo.json`
- `layout-no-skill-v2`: `data/layout-no-skill-v2/layout-no-skill-v2_<input_name>.json` or `data/layout-no-skill-v2/layout-no-skill-v2_demo.json`

You can override the output path with:

```bash
python document_reader.py --src ./sample.pdf --out custom/output.json
```

## Querying Search Results

After a successful `layout-skill` run, the saved JSON includes:

- created Search object names
- indexer status
- retrieved chunk documents
- retrieved image documents

The chunk results include `chunk_vector`.

The image results include `image_vector` and `image_path`.

`image_path` identifies the cropped image projection stored in blob storage. The index stores metadata and vectors, not the image binary itself.
