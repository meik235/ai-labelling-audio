<!-- @format -->

# modules.yaml Deep Dive (TeatsingClient20 baseline)

The pipeline now uses the split `clientConfig` / `devConfig` format. This document explains what every major block controls and how the backend consumes it. Use it as a reference when onboarding new clients or when you need to chase a config-driven behavior.

---

## 1. Top-Level Layout

```yaml
clientConfig:
  clientId: "TeatsingClient20"
  workflow: ...
  dataSource: ...
  preprocessing: ...
  annotation: ...
  postProcessing: ...

devConfig:
  ingestion: ...
  labelStudio: ...
  postProcessing: ...
  webhooks: ...
  rejectionHandler: ...
```

- `clientConfig` holds knobs that account managers tweak per customer.
- `devConfig` is for engineers/operators (local paths, runtime hooks, per-env overrides).

---

## 2. `clientConfig`

### 2.1 Identity + Workflow

- `clientId`: included in S3 prefixes and logging (see `backend/pipeline/service.py`).
- `workflow.steps`: declarative list of active modules; UI tooling can render this sequence.
- Individual `workflow.*.enabled/description` toggles feed dashboards/documentation but the backend still follows the concrete blocks below (preprocessing/annotation/postProcessing).

### 2.2 Data Source

Controls where tasks originate:

| Field                     | Purpose                                          | Consumed by                                         |
| ------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| `dataSource.type`         | Switches between CSV, DynamoDB, or API ingestion | `backend/pipeline/data_source.py`                   |
| `csv.path`                | Relative path to the CSV when `type=csv`         | `scripts/run_pipeline.py` bootstrap                 |
| `audioLocationPreference` | Hints for download order (S3 vs GDrive)          | `backend/pipeline/service.py` → `preprocess_inputs` |

### 2.3 Preprocessing

Maps 1:1 to `PreprocessingConfig` in `backend/pipeline/config.py`.

- `validation`: duration/size/format checks. Disabled by default but ready to toggle.
- `clearAudio`: runs a single ffmpeg pass to transcode every file to mono, 16 kHz WAV for consistent AssemblyAI input.
  - `metadata.includeFields` decides what metadata we surface to Label Studio downstream.

### 2.4 Annotation

| Block           | Highlights                                                                                                                          | Runtime effect                                                                    |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `labelConfig`   | Inline Label Studio XML with segments + intent choices + read-only ASR text                                                         | Passed directly into `LabelStudioClient.create_or_update_project`                 |
| `preAnnotation` | Controls label schema metadata for Label Studio (`fields: [start, end, label, intent]`)                                    | `PreAnnotationRunner` records schema metadata |
| `transcription` | AssemblyAI config (model/lang/task/timestamps/output format). Optional `sentiment.enabled` toggles AssemblyAI’s sentiment analysis. | `TranscriptionRunner`                                                             |
| `template`      | Placeholder mapping rules for post-processing transforms                                                                            | `backend/postprocessing/workflow.py`                                              |

Exported annotations emit `[start, end, label, intent]`.

### 2.5 Post-processing

The client currently sets `postProcessing.enabled=false` but specifies defaults:

- `finalReportFormat`, `extractAudioSegments`, and `mergeStrategy` flow into `PostProcessingConfig`. Once the flag flips to true, the system knows how to export without revisiting config.

---

## 3. `devConfig`

### 3.1 Ingestion

Provides engineer-only fallbacks and guardrails:

- `csv.localFallbackPath`: used when a client forgets to supply `clientConfig.preprocessing.csv_path`.
- `s3.*`: bucket/prefix pair for raw + processed artifacts (pre-populated for Teatsing).
- `runtimeChecks`: toggles (e.g., `requireKnownSource`) that gate downloads in `preprocess_inputs`.

### 3.2 Label Studio

Overrides operational details without touching `clientConfig`:

| Field                         | Notes                                                                                                 |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| `projectNamePrefix`           | Used verbatim, so re-running the pipeline reuses the same LS project (`AudioProjectTeatsingClient20`) |
| `mlBackend.enabled/url/token` | Hook into FastAPI or Lambda ML backend                                                                |
| `storage.enabled`             | Connects S3 as both source and target (raw uploads + annotation exports)                              |
| `webhook.url`                 | Points to local unified webhook server                                                                |
| `runPreAnnotate`              | If true, we include pre-annotation predictions when uploading tasks                                   |

### 3.3 Engineering Post-processing

Separate from client toggles; lets ops test the export workflow locally (e.g., write to `./.tmp/final_report.csv`, send completion webhooks).

### 3.4 Global Webhooks + Rejection Handler

`devConfig.webhooks` centralizes local webhook endpoints and secrets, while `rejectionHandler.lambda` is ready for AWS automation (function name + payload fields).

---

## 4. Cross-References

| Module                                | Key config fields                                                  |
| ------------------------------------- | ------------------------------------------------------------------ |
| `backend/preprocessing/pipeline.py`   | `clientConfig.preprocessing.*`, `devConfig.ingestion.*`            |
| `backend/annotation/preannotation.py` | `clientConfig.annotation.preAnnotation.*`                          |
| `backend/annotation/transcription.py` | `clientConfig.annotation.transcription.*`                          |
| `backend/pipeline/service.py`         | Entire config; prints segmentation enabled/strategy for visibility |
| `backend/postprocessing/workflow.py`  | `clientConfig.postProcessing.*`, `devConfig.postProcessing.*`      |
| `scripts/run_pipeline.py`             | Entry point that loads the YAML with `PipelineConfig.load`         |

---

## 5. Practical Tips

- **Repoint storage**: update `devConfig.ingestion.s3.*` for new buckets; no code changes needed.
- **Multiple clients**: duplicate `config/modules.yaml`, adjust `clientId`, `projectNamePrefix`, and label schema fields.

For an operational walk-through, see `docs/architecture.md`; for local setup, check `docs/developer_onboarding.md`.
