<!-- @format -->

# Developer Onboarding

Bring a new engineer up to speed in ~30 minutes. Everything here assumes macOS/Linux; adapt commands for Windows as needed.

## 1. Environment

```bash
git clone <repo-url> audio-annotation
cd audio-annotation
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/modules.example.yaml config/modules.yaml  # one-time bootstrap
```

Create `.env` (or export directly):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
LABEL_STUDIO_API_KEY=...
LABEL_STUDIO_BASE_URL=http://localhost:8080
ASSEMBLYAI_API_KEY=...
```

## 2. Config Model

- `config/modules.yaml` = client-facing knobs (preprocessing, annotation, post-processing).
- `devConfig` = engineer overrides (CSV paths, S3 prefixes, runtime checks).
- Pre-annotation schema emits `[start, end, label, intent]`. Update downstream consumers whenever you edit `labelSchema.fields`.

## 3. Core Commands

| Goal                   | Command                                                              | Notes                                             |
| ---------------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| Run full pipeline      | `python scripts/run_pipeline.py --config config/modules.yaml`        | Streams progress + config summary                 |
| Re-export deliverables | `python scripts/export_deliverables.py --config config/modules.yaml` | Uses existing LS project/tasks                    |
| Launch ML backend      | `uvicorn backend.app:app --host 0.0.0.0 --port 9090`                 | Needed only if Label Studio ML backend is enabled |
| Debug Label Studio     | `python scripts/ls_debug.py --project-prefix ...`                    | Lists projects/tasks, dumps annotations           |

## 4. Pipeline Anatomy

1. **Preprocess** (`backend/preprocessing`):
   - Downloads audio, validates, and converts every file to mono 16 kHz WAV (ffmpeg) for consistent ASR/QC playback.
   - Stores artifacts under `s3://<bucket>/<prefix>/{raw,preprocessed,ml-ready}`.
2. **Pre-annotation** (`backend/annotation/preannotation.py`):
   - Attaches schema metadata so Label Studio knows to expect `[start, end, label, intent]`.
3. **Transcription** (`backend/annotation/transcription_runner.py`):
   - AssemblyAI by default; swap via config.
4. **Enrichment** (`backend/annotation/enrichment_runner.py`):
   - Keyword extraction + tagging (rarely enabled now; most clients skip it).
5. **Label Studio orchestration** (`backend/pipeline/service.py` + `backend/integrations/labelstudio.py`):
   - Reuses/creates projects, configures storage, pushes tasks.
6. **Post-processing** (`backend/postprocessing/workflow.py`):
   - Merge human + machine output, optional webhook, final deliveries.

## 5. Debugging Checklist

- **Audio format mismatch?** Confirm `preprocessing.clearAudio.enabled: true` so the ffmpeg standardization runs; otherwise AssemblyAI receives mixed formats.
- **Sentiment/“emotion” data missing?** Toggle `annotation.transcription.sentiment.enabled` and rerun the ASR step.
- **Pipeline aborts mid-run?** Inspect `logs/pipeline.log` (if enabled) or console for stack traces. Most failures trace back to network (S3/Label Studio) or malformed CSV rows.
- **Label Studio rejects tasks?** Run `scripts/ls_debug.py` to inspect payload; mismatched schema fields will surface here.
- **AssemblyAI errors?** They return HTTP 401/422; see `backend/integrations/assemblyai.py` for retry/backoff logic.

## 6. Testing Expectations

- No automated tests yet; rely on:
  - `scripts/run_pipeline.py` against `data/user_responses.csv` (smoke).
  - Manual review in Label Studio (confirm template + metadata).
  - Sanity check exports using `scripts/export_deliverables.py`.
- When touching config/data contracts, add a note to `ARCHITECTURE.md` + ping reviewers.

## 7. Contribution Flow

1. Branch off `main`.
2. Keep changes config-driven when possible.
3. Run formatter (`ruff`, `black`, etc. if configured) before PR.
4. Update docs (`docs/architecture.md`, this guide) whenever you change runtime behavior or config surfaces.
5. Provide pipeline output snippets in PR description so reviewers can confirm flow.

## 8. Useful Links

- Label Studio docs: https://labelstud.io/guide/
- AssemblyAI API: https://www.assemblyai.com/docs
- AWS S3 CLI reference: https://docs.aws.amazon.com/cli/latest/reference/s3/

Ping #audio-annotation Slack channel when you add new clients or flip sentiment/QA schema fields so ops can sync templates.
