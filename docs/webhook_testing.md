# Webhook Testing - Step by Step Commands

## Prerequisites Check

```bash
# Make sure you're in the project directory
source .venv/bin/activate

# Verify environment variables are loaded
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:+SET}"
echo "LABEL_STUDIO_API_KEY: ${LABEL_STUDIO_API_KEY:+SET}"
```

## Step 1: Start Webhook Server

**Terminal 1 - Start the webhook server:**

```bash
source .venv/bin/activate
python webhooks/server.py
```

You should see:
```
================================================================================
Starting unified webhook server (LOCAL mode)
================================================================================
Preprocessing rejections: http://localhost:8000/webhooks/label-studio
Label Studio events: http://localhost:8000/webhooks/label-studio/events
Health check: http://localhost:8000/health
Host: 0.0.0.0:8000
Security: Webhook secret verification DISABLED (local mode)
================================================================================
```

**Keep this terminal open** - the server needs to keep running.

## Step 2: Test Webhook Server Health

**Terminal 2 - Test the server is running:**

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"ok","service":"webhook-server","mode":"local","timestamp":"..."}
```

## Step 3: Test Preprocessing Rejection Webhook

**Terminal 2 - Send a test rejection:**

```bash
curl -X POST http://localhost:8000/webhooks/label-studio \
  -H "Content-Type: application/json" \
  -d '{
    "row_index": 1,
    "channel": "speaker_a",
    "reason": "duration_too_long:673.5",
    "module": "preprocessing",
    "metadata": {"test": "data"}
  }'
```

**Check Terminal 1** - You should see:
```
================================================================================
PREPROCESSING REJECTION at 2025-...
================================================================================
Row Index: 1
Channel: speaker_a
Reason: duration_too_long:673.5
Module: preprocessing

Full payload:
{
  "row_index": 1,
  "channel": "speaker_a",
  ...
}
================================================================================
```

## Step 4: Run Pipeline (Test Real Rejections)

**Terminal 2 - Run the pipeline:**

```bash
source .venv/bin/activate
python scripts/run_pipeline.py --config config/modules.yaml
```

**Watch Terminal 1** - You'll see rejection webhooks appear in real-time as the pipeline processes audio files.

## Step 5: Test Label Studio Events Webhook

**Terminal 2 - Test Label Studio event:**

```bash
curl -X POST http://localhost:8000/webhooks/label-studio/events \
  -H "Content-Type: application/json" \
  -d '{
    "action": "ANNOTATION_REJECTED",
    "task": {
      "id": 123,
      "project": 45,
      "data": {"audio": "https://example.com/audio.wav"}
    },
    "annotation": {
      "id": 456,
      "created_by": {"username": "reviewer1"},
      "result": "Quality issues found"
    }
  }'
```

**Check Terminal 1** - You should see:
```
================================================================================
LABEL STUDIO EVENT at 2025-...
================================================================================
Event Type: ANNOTATION_REJECTED
QC REJECTION DETECTED
Task ID: 123
Project ID: 45
Rejected by: reviewer1
Reason: Quality issues found
...
================================================================================
```

## Step 6: Register Webhook in Label Studio (Optional)

If you want Label Studio to automatically send events:

**Via Python script:**

```bash
python -c "
import asyncio
from backend.integrations import LabelStudioClient
import os

async def register():
    client = LabelStudioClient(
        os.getenv('LABEL_STUDIO_BASE_URL', 'http://localhost:8080'),
        os.getenv('LABEL_STUDIO_API_KEY', '')
    )
    result = await client.register_webhook(
        project_id=YOUR_PROJECT_ID,
        target_url='http://localhost:8000/webhooks/label-studio/events',
        events=['ANNOTATION_REJECTED', 'TASK_REJECTED', 'ANNOTATION_UPDATED']
    )
    print('Webhook registered:', result)

asyncio.run(register())
"
```

## Quick Test Summary

```bash
# Terminal 1: Start server
python webhooks/server.py

# Terminal 2: Test health
curl http://localhost:8000/health

# Terminal 2: Test preprocessing rejection
curl -X POST http://localhost:8000/webhooks/label-studio \
  -H "Content-Type: application/json" \
  -d '{"row_index": 1, "channel": "test", "reason": "test"}'

# Terminal 2: Test LS event
curl -X POST http://localhost:8000/webhooks/label-studio/events \
  -H "Content-Type: application/json" \
  -d '{"action": "ANNOTATION_REJECTED", "task": {"id": 123}}'

# Terminal 2: Run pipeline (will trigger real rejections)
python scripts/run_pipeline.py --config config/modules.yaml
```

## Troubleshooting

**Webhook server not starting:**
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>
```

**Connection refused:**
- Make sure webhook server is running in Terminal 1
- Check the URL in `config/modules.yaml` matches: `http://localhost:8000/webhooks/label-studio`

**No webhooks appearing:**
- Verify webhook URL in config is correct
- Check Terminal 1 for any error messages
- Make sure pipeline is actually processing files (check for rejections in pipeline output)

