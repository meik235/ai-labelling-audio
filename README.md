<!-- @format -->

# Label Studio ↔ AssemblyAI ML Backend

This repo gives you a drop-in FastAPI service that Label Studio can call to transcribe audio tasks with AssemblyAI. The service fetches an audio file from the task, uploads it to AssemblyAI, waits for the transcript, and responds with a Label-Studio-ready prediction payload.

## 1. Prerequisites

- Python 3.10+ (tested with 3.11)
- An AssemblyAI API key ([dashboard](https://www.assemblyai.com/dashboard/))
- Label Studio 1.12+ (self-hosted)

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure

Create `.env` in the project root with:

```
ASSEMBLYAI_API_KEY=your_real_key
# Optional overrides:
# ASSEMBLYAI_BASE_URL=https://api.assemblyai.com/v2
# ASSEMBLYAI_POLL_INTERVAL=3.0
# ASSEMBLYAI_AUTO_CHAPTERS=false
# LABEL_STUDIO_AUDIO_FIELD=audio

```

You can also export the same values in your shell if you prefer.

## 4. Run the backend

### Option A: Local Development (FastAPI/Uvicorn)

```bash
source .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 9090
```

### Option B: AWS Lambda Deployment

See [LAMBDA_DEPLOYMENT.md](./LAMBDA_DEPLOYMENT.md) for complete Lambda setup.

Quick deploy:

```bash
npm install
serverless deploy
```

The service exposes:

- `GET /health` → `{ "status": "ok" }`
- `POST /predict` → Label Studio ML backend contract

## 5. Wire it into Label Studio

1. Start Label Studio (`label-studio start` or via Docker).
2. In the UI, open _Settings → Machine Learning_.
3. Add a new model with the URL `http://localhost:9090/predict`.
4. Make sure your project labeling config looks like:

```
<View>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio" placeholder="Auto transcript will appear here"/>
</View>
```

If your data uses a different field than `$audio`, either update the labeling config or set `LABEL_STUDIO_AUDIO_FIELD` in `.env`.

## 6. Quick smoke test

With the server running:

```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task": {
      "id": 1,
      "data": {
        "audio": "https://storage.googleapis.com/aai-web-samples/espn-birds.m4a"
      }
    }
  }'
```

You should get a JSON response containing `predictions[0].result[0].value.text[0]` with the transcript.

## 7. Deploying

- For production, run Uvicorn behind a process manager (e.g., `gunicorn -k uvicorn.workers.UvicornWorker`).
- Ensure your server has outbound HTTPS access to AssemblyAI.
- Tighten `Label Studio` ↔ `backend` networking (e.g., private VPC, firewall rules).

## 8. Troubleshooting

- `422` errors → request payload missing `task.data.audio`.
- `502` errors → AssemblyAI or audio fetch failed; check logs for specifics.
- Enable debug logging by starting Uvicorn with `--log-level debug`.

## 9. Automated Pipeline (CSV → Label Studio → S3)

All modules can be executed end-to-end via a YAML config.

1. Copy the sample config:

```bash
cp config/modules.example.yaml config/modules.yaml
```

2. Edit `config/modules.yaml` (set `clientId`, `preprocessing`, `annotation`, `delivery`, `lsProject`, and `rejectionHandler`). A minimal example is provided in `config/modules.example.yaml`.
   - `preprocessing.csv_path` → Google Sheet CSV export
   - `preprocessing.s3_prefix` → base prefix (e.g., `client/project`)
   - `annotation.label_config` → inline XML or path
   - `annotation.ml_backend_url` → running FastAPI/Lambda endpoint
   - `annotation.webhook_url` → optional webhook receiver
   - `delivery.final_csv_path` → where to materialize final report
   - `annotation.preAnnotation.diarization.model` → set to `pyannote` for neural diarization (falls back to `heuristic` if pyannote fails)
3. Ensure `.env` (or environment) contains:
   - `LABEL_STUDIO_BASE_URL`, `LABEL_STUDIO_API_KEY`
   - `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - Optional: `GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH`

### Pyannote (neural) diarization

High-quality speaker boundaries are powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio). Because the model lives behind Hugging Face’s gated repos you must:

1. Accept the terms for:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
   - https://hf.co/pyannote/embedding
2. Create a Hugging Face token (`Read` scope) and set it in your environment:

```bash
export HUGGINGFACE_TOKEN=hf_your_token
```

3. (Optional) put the token directly into `annotation.preAnnotation.diarization.huggingFaceToken`.

If the token or pyannote packages are missing the pipeline automatically falls back to the lightweight heuristic diarizer so you still get baseline segments.

Run the pipeline:

```bash
python scripts/run_pipeline.py --config config/modules.yaml
```

What happens:

- **Preprocessing** downloads Google Sheet audio → standardizes WAV → uploads to `raw/`, `processed/`, `ml-ready/`
- **Annotation** creates Label Studio project, bulk uploads tasks, attaches ML backend, exports annotations, saves transcripts & audio segments to S3
- **Delivery** cleans transcripts, packages deliverables, uploads final CSV + JSON to `annotations/`

## 10. Lambda Deployment

For production deployment on AWS Lambda, see [LAMBDA_DEPLOYMENT.md](./LAMBDA_DEPLOYMENT.md).

**Key points:**

- Uses Mangum adapter to wrap FastAPI for Lambda
- Configured with 15-minute timeout (max Lambda limit)
- Deploy via Serverless Framework or AWS SAM
- See [WORKFLOW.md](./WORKFLOW.md) for complete workflow documentation

## 11. Model Evaluation & Audio Processing

For comprehensive model evaluation, benchmarking, and audio processing pipelines:

- **[MODEL_EVALUATION.md](./MODEL_EVALUATION.md)** - Model scouting, benchmarking framework, and audio metrics (WER, CER, DER, RTF, etc.)
- **[AUDIO_PROCESSING.md](./AUDIO_PROCESSING.md)** - Complete pre-processing and post-processing pipeline for audio annotation

## 12. Next steps (optional)

- Cache completed transcripts keyed by audio URL to avoid reprocessing.
- Add support for diarization or sentiment by passing extra params to AssemblyAI.
- Store transcripts in persistent storage (S3, database) for audit trails.
- Implement async processing for very long audio files (SQS + webhooks).
- Integrate model evaluation metrics into ML Backend response.
- Add pre-processing pipeline to Lambda function for audio standardization.
