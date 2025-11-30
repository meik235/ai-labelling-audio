<!-- @format -->

# Webhooks

This folder contains all webhook-related code for the audio annotation pipeline.

## Structure

- **`server.py`** - Unified webhook server handling both preprocessing rejections and Label Studio events
- **`__init__.py`** - Package initialization

## Usage

### Unified Webhook Server

The server handles two types of webhooks:

1. **Preprocessing rejections** - When audio validation fails during preprocessing
2. **Label Studio events** - QC rejections, task updates, etc. from Label Studio

### Local Mode (No Authentication)

Perfect for development and testing:

```bash
# Start server (no authentication required)
python webhooks/server.py
```

The server will:

- Listen on `http://localhost:8000`
- Accept preprocessing rejections at `/webhooks/label-studio`
- Accept Label Studio events at `/webhooks/label-studio/events`
- Print all received payloads to console with pretty formatting
- **No webhook secret required**

### Production Mode (With Authentication)

For production deployment:

```bash
# Set webhook secret
export WEBHOOK_SECRET=$(openssl rand -hex 32)

# Optional: configure host/port
export WEBHOOK_HOST=0.0.0.0
export WEBHOOK_PORT=8000
export CORS_ORIGINS=https://your-domain.com

# Start server
python webhooks/server.py
```

The server will:

- Verify `X-Webhook-Secret` header matches `WEBHOOK_SECRET`
- Use structured logging instead of console output
- Reject requests without valid secret (401 Unauthorized)

## Configuration

### Local Testing

Update `config/modules.yaml`:

```yaml
preprocessing:
  validation:
    rejectionWebhook:
      url: http://localhost:8000/webhooks/label-studio
      secret: "" # Empty for local mode
```

### Production

Update `config/modules.yaml`:

```yaml
preprocessing:
  validation:
    rejectionWebhook:
      url: https://your-production-url.com/webhooks/label-studio
      secret: ${WEBHOOK_SECRET} # Must match server's WEBHOOK_SECRET
```

**Important:** The secret in your config must match the `WEBHOOK_SECRET` environment variable on the server.

## Environment Variables

| Variable         | Default   | Description                                                        |
| ---------------- | --------- | ------------------------------------------------------------------ |
| `WEBHOOK_SECRET` | (empty)   | Webhook secret for authentication. If set, enables production mode |
| `WEBHOOK_HOST`   | `0.0.0.0` | Host to bind to                                                    |
| `WEBHOOK_PORT`   | `8000`    | Port to listen on                                                  |
| `CORS_ORIGINS`   | `*`       | Comma-separated list of allowed CORS origins                       |

## Testing

### Test Local Mode

```bash
# Start server
python webhooks/server.py

# In another terminal, test preprocessing rejection webhook
curl -X POST http://localhost:8000/webhooks/label-studio \
  -H "Content-Type: application/json" \
  -d '{"row_index": 1, "channel": "speaker_a", "reason": "test"}'

# Test Label Studio events webhook
curl -X POST http://localhost:8000/webhooks/label-studio/events \
  -H "Content-Type: application/json" \
  -d '{"action": "ANNOTATION_REJECTED", "task": {"id": 123}, "annotation": {"id": 456}}'
```

### Test Production Mode

```bash
# Set secret
export WEBHOOK_SECRET=test_secret_123

# Start server
python webhooks/server.py

# Test without secret (should fail)
curl -X POST http://localhost:8000/webhooks/label-studio \
  -H "Content-Type: application/json" \
  -d '{"row_index": 1, "channel": "speaker_a", "reason": "test"}'
# Expected: 401 Unauthorized

# Test with correct secret (should succeed)
curl -X POST http://localhost:8000/webhooks/label-studio \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: test_secret_123" \
  -d '{"row_index": 1, "channel": "speaker_a", "reason": "test"}'
# Expected: 200 OK
```

## Deployment

### Option 1: Standalone Server

Deploy `server.py` to your server (EC2, ECS, Docker, etc.):

```bash
# Set production environment variables
export WEBHOOK_SECRET=your_secure_secret
export WEBHOOK_HOST=0.0.0.0
export WEBHOOK_PORT=8000

# Run with process manager (systemd, supervisor, etc.)
python webhooks/server.py
```

### Option 2: AWS Lambda

Use `lambda/lambda_webhook_handler.py` for serverless deployment. See `serverless.yml` for configuration.

## Label Studio Events (QC Rejections)

To receive notifications when QC rejects tasks in Label Studio, use the same unified webhook server.

### Register Webhook in Label Studio

1. **Via Label Studio UI:**

   - Go to your project → Settings → Webhooks
   - Add webhook URL: `http://your-server:8000/webhooks/label-studio/events`
   - Select events: `ANNOTATION_REJECTED`, `TASK_REJECTED`, `ANNOTATION_UPDATED`
   - Save

2. **Via API (programmatic):**

   ```python
   from backend.integrations import LabelStudioClient

   client = LabelStudioClient(base_url, api_key)
   await client.register_webhook(
       project_id=123,
       target_url="http://your-server:8000/webhooks/label-studio/events",
       events=["ANNOTATION_REJECTED", "TASK_REJECTED", "ANNOTATION_UPDATED"]
   )
   ```

### What You'll Receive

When a reviewer rejects a task in Label Studio, you'll get:

- Task ID
- Project ID
- Who rejected it
- Rejection reason
- Task data

You can then:

- Update your QC sheet
- Create a new task with corrected data
- Send notifications
- Trigger re-processing

## Security Notes

- **Local mode**: No authentication - only use for development
- **Production mode**: Always set `WEBHOOK_SECRET` - never leave it empty
- **HTTPS**: Use HTTPS in production (via reverse proxy, API Gateway, etc.)
- **Secret generation**: Use `openssl rand -hex 32` to generate secure secrets
