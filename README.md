# Document Reader

Python CLI for two document-processing flows:

- `direct`: call Azure Document Intelligence directly and save structured output
- `layout-skill`: run an Azure AI Search pull indexer pipeline using `DocumentIntelligenceLayoutSkill` plus multimodal vectorization for text and images

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
