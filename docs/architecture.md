<!-- @format -->

# Audio Annotation Pipeline

Updated architecture notes for the TeatsingClient20 rollout. Treat this as the system contract; dev-specific walk-through lives in `docs/developer_onboarding.md`.

## System Overview

```
CSV -> Preprocessing -> Pre-annotation -> ASR/Enrichment -> Label Studio -> Post-processing
```

- **CSV ingestion**: `data/user_responses.csv` or any user-provided sheet feeds URLs + metadata.
- **Preprocessing** (`backend/preprocessing`):
  - Validation (duration/size/sample rate) before download.
  - Single ffmpeg pass converts every asset to mono 16 kHz WAV for consistent AssemblyAI input.
  - Uploads artifacts to S3 layout: `/raw`, `/preprocessed`, `/ml-ready`.
- **Annotation module** (`backend/annotation`):
  - `PreAnnotationRunner` attaches schema metadata for Label Studio.
  - `TranscriptionRunner` wraps AssemblyAI (transcription, diarization, optional sentiment analysis).
- **Human loop**: `backend/integrations/labelstudio.py` seeds/updates the project defined in `labelStudio.projectNamePrefix`, reusing existing projects automatically.
- **Post-processing** (`backend/postprocessing`): optional packaging/export with audio segment extraction.

## Updated Config Contract

`config/modules.yaml` now packages the client-facing knobs under:

```yaml
clientConfig:
  clientId: "TeatsingClient20"
  workflow:
    preprocessing: ...
    annotation:
      preAnnotation:
        labelSchema.fields: [start, end, label, intent]
      labelConfig: |
        <View>
          <Labels .../>                  # multi-label regions + intents
          <Audio .../>
          <TextArea name="auto_transcript" ... readonly="true"/>
          <Choices name="intent" .../>
        </View>
```

`devConfig` stores ingestion fallbacks (CSV path, S3 prefixes, runtime checks) so CI/local overrides stay out of client configs.

## Runtime Flow (scripts/run_pipeline.py)

1. **Load + merge config** via `PipelineConfig.load`.
2. **Preprocess** (`PipelineService.preprocess_inputs`):
   - Each audio URL downloads, validates, and runs the clear-audio stack.
   - Artifacts land on S3 based on `preprocessing.store_processed_artifacts` & `store_ml_ready_artifacts`.
3. **Annotation chain**:
   - Pre-annotation metadata injection.
   - AssemblyAI transcription and diarization if `transcription.enabled`.
   - Enrichment (keywords/tagging) if requested.
4. **Label Studio orchestration**:
   - Reuse project ID when found; otherwise create.
   - Push tasks, configure storage/webhooks/ML backend as required.
5. **Post-processing/export** (optional):
   - Merge strategy + final CSV/JSON/SRT outputs.
   - Optional webhook when delivery completes.

## Storage & Data Layout

```
s3://<bucket>/<prefix>/
  raw/
  preprocessed/
  ml-ready/
  annotations/
  reports/ (optional post-processing)
```

Each key carries metadata (row index, channel) to trace origin. Post-processing can additionally call `AudioSegmenter` to slice transcript-aligned WAVs for QA.

## Label Studio Template

The default view includes:

- Region-level labels with speaker diarization from AssemblyAI.
- Transcription text from AssemblyAI.
- Annotation schema exports `[start, end, label, intent]`.

## Feature Flags

- `transcription.sentiment.enabled`: enables AssemblyAI sentiment analysis.
- `clearAudio.enabled`: transcodes to mono 16 kHz WAV for AssemblyAI.
- `postProcessing.extractAudioSegments`: toggles QC snippet extraction.

## Dev Entry Points

- `python scripts/run_pipeline.py --config config/modules.yaml` – main orchestration.
- `python scripts/export_deliverables.py --config ...` – re-export tasks without reprocessing audio.
- `uvicorn backend.app:app --port 9090` – ML backend for Label Studio.

See `docs/developer_onboarding.md` for local setup.

## Troubleshooting

- **Label Studio auth issues** → confirm `LABEL_STUDIO_API_KEY` and `LABEL_STUDIO_BASE_URL`.
- **S3 upload failures** → check AWS creds + bucket permissions.
- **AssemblyAI errors** → inspect `backend/integrations/assemblyai.py` logs; most failures stem from invalid API keys or unsupported languages.

## Repo Map

```
backend/
  annotation/    # Pre-annotation, transcription, enrichment
  pipeline/      # Config models + async service orchestrator
  preprocessing/ # CSV/audio ingestion and clear-audio stack
  integrations/  # AssemblyAI, Label Studio, storage adapters
  postprocessing/# Delivery + export helpers
config/
  modules.yaml           # Active client config
  modules.example.yaml   # Reference template
scripts/
  run_pipeline.py
  export_deliverables.py
```
