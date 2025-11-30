<!-- @format -->

# Webhook Calls - Complete Reference

This document lists all webhook calls in the pipeline and when they are triggered.

## 1. Preprocessing Rejection Webhook

**When Called:**

- During preprocessing validation when an audio file is rejected
- Triggered in `backend/preprocessing/pipeline.py` → `_send_rejection_webhook()`

**Conditions:**

- `validation.rejection_webhook_url` must be configured
- Audio file fails validation (duration, size, format, etc.)

**Config Location:**

```yaml
clientConfig:
  preprocessing:
    validation:
      rejectionWebhook:
        url: "http://localhost:8000/webhooks/label-studio"
        secret: ""
```

**Alternative Config:**

```yaml
devConfig:
  webhooks:
    rejection:
      url: http://localhost:8000/webhooks/label-studio
      secret: ""
```

**Payload:**

```json
{
  "row_index": 1,
  "channel": "channel_1",
  "reason": "duration_too_long:673.5",
  "module": "preprocessing",
  "metadata": {...}
}
```

**Endpoint:** `/webhooks/label-studio` (preprocessing rejection endpoint)

---

## 2. Label Studio Events Webhook (Incoming from Label Studio)

**When Called:**

- Automatically by Label Studio when events occur
- Received at `/webhooks/label-studio/events`

**Events Triggered:**

- `TASK_CREATED` - When a task is created
- `ANNOTATION_CREATED` - When an annotation is created
- `ANNOTATION_UPDATED` - When an annotation is updated
- `TASK_COMPLETED` - When a task is marked as completed
- `ANNOTATION_REJECTED` - When an annotation is rejected
- `TASK_REJECTED` - When a task is rejected

**Config Location:**

```yaml
devConfig:
  labelStudio:
    webhook:
      enabled: true
      url: "http://localhost:8000/webhooks/label-studio/events"
      events: ["TASK_COMPLETED", "ANNOTATION_CREATED", "ANNOTATION_UPDATED"]
```

**Auto-Export Trigger:**

- When `TASK_COMPLETED`, `ANNOTATION_CREATED`, or `ANNOTATION_UPDATED` events are received
- Automatically triggers `run_export_only()` in the background
- This happens in `webhooks/server.py` → `_process_label_studio_event()`

**Endpoint:** `/webhooks/label-studio/events` (Label Studio events endpoint)

---

## 3. Post-Processing Completion Webhook

**When Called:**

- After successful post-processing when deliverables are built
- Called in both `run_with_config()` (full pipeline) and `run_export_only()` (export-only)

**Conditions:**

- `config.postprocessing.completion_webhook_url` must be configured
- Post-processing must complete successfully
- Deliverables must be built (transcripts/annotations exist)

**Config Location:**

```yaml
devConfig:
  postProcessing:
    completionWebhook:
      url: "http://localhost:8000/webhooks/pipeline-complete"
      secret: ""
```

**Alternative Config:**

```yaml
clientConfig:
  postProcessing:
    completionWebhook:
      url: "https://demo.completion.webhook"
      secret: "secret"
```

**Payload (Full Pipeline - `run_with_config()`):**

```json
{
	"client_id": "Testing18",
	"project_id": 174,
	"s3_prefix": "testing_output",
	"preprocessing_tasks": 8,
	"annotation_tasks": 8,
	"export": {
		"task_count": 8
	},
	"deliverables": {
		"transcripts": 8,
		"annotations": 8
	}
}
```

**Payload (Export-Only - `run_export_only()`):**

```json
{
	"project_id": 176,
	"s3_prefix": "testing1",
	"export": {
		"task_count": 1,
		"processed_count": 1,
		"skipped_count": 0
	},
	"deliverables": {
		"transcripts": 1,
		"annotations": 1,
		"final_csv": "https://kgen-ai-labelling.s3.ap-south-1.amazonaws.com/Testing19/testing1/annotations/final_report.csv"
	}
}
```

**Implementation:**

- ✅ **Called in both methods:**
  - `run_with_config()` - After full pipeline run
  - `run_export_only()` - After export-only completes (fixed in latest version)
- Checks both stored metadata and current config for webhook URL

**Endpoint:** Configured URL (e.g., `/webhooks/pipeline-complete`)

---

## Summary Table

| Webhook Type                | When Called               | Method                           | Config Required                        |
| --------------------------- | ------------------------- | -------------------------------- | -------------------------------------- |
| **Preprocessing Rejection** | Audio validation fails    | `_send_rejection_webhook()`      | `validation.rejectionWebhook.url`      |
| **Label Studio Events**     | LS events occur (auto)    | Received by webhook server       | `labelStudio.webhook.enabled=true`     |
| **Completion Webhook**      | Post-processing completes | `_send_completion_webhook()`     | `postProcessing.completionWebhook.url` |
| **Auto-Export**             | LS task completed         | `run_export_only()` (background) | `labelStudio.webhook.enabled=true`     |

---

## Configuration Notes

1. **Completion webhook requires URL configuration**

   - Must have `completionWebhook.url` set in config
   - Can be set in either `devConfig.postProcessing.completionWebhook.url` or `clientConfig.postProcessing.completionWebhook.url`
   - If URL is empty, webhook will not be called (silently skipped)

2. **Webhook is called in both scenarios:**

   - ✅ Full pipeline run (`run_with_config()`)
   - ✅ Export-only run (`run_export_only()`)
   - Both check for webhook URL in metadata and current config

3. **Webhook payload differences:**
   - `run_with_config()` includes: `client_id`, `preprocessing_tasks`, `annotation_tasks`
   - `run_export_only()` includes: `project_id`, `export` details (task_count, processed_count, skipped_count)
   - Both include: `s3_prefix`, `deliverables` information
