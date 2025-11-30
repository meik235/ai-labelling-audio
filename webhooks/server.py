#!/usr/bin/env python3
"""Unified webhook server for preprocessing rejections and Label Studio events."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.getLogger(__name__).info("Loaded environment variables from %s", env_path)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
PORT = int(os.getenv("WEBHOOK_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

IS_PRODUCTION = bool(WEBHOOK_SECRET)
MODE = "PRODUCTION" if IS_PRODUCTION else "LOCAL"

app = FastAPI(
    title="Webhook Server",
    description=f"Unified webhook endpoint for preprocessing rejections and Label Studio events ({MODE} mode)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def verify_webhook_secret(secret: Optional[str]) -> bool:
    """Verify webhook secret if in production mode."""
    if not IS_PRODUCTION:
        return True
    return secret == WEBHOOK_SECRET


def _extract_project_id_from_payload(payload: Dict[str, Any]) -> Optional[int]:
    """Extract project ID from Label Studio webhook payload."""
    if "project_id" in payload:
        return payload["project_id"]
    if "project" in payload:
        if isinstance(payload["project"], dict):
            return payload["project"].get("id")
        if isinstance(payload["project"], int):
            return payload["project"]
    if "task" in payload and isinstance(payload["task"], dict):
        project = payload["task"].get("project")
        if isinstance(project, dict):
            return project.get("id")
        if isinstance(project, int):
            return project
    return None


def extract_ls_rejection_info(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract rejection information from Label Studio webhook payload."""
    event_type = payload.get("action") or payload.get("event_type")
    
    if event_type in ["ANNOTATION_REJECTED", "TASK_REJECTED"]:
        task = payload.get("task", {})
        annotation = payload.get("annotation", {})
        
        return {
            "event_type": event_type,
            "task_id": task.get("id"),
            "project_id": task.get("project"),
            "annotation_id": annotation.get("id") if annotation else None,
            "rejected_by": annotation.get("created_by", {}).get("username") if annotation else None,
            "rejection_reason": annotation.get("result") or payload.get("reason"),
            "task_data": task.get("data", {}),
        }
    
    return None


async def _process_label_studio_event(payload: Dict[str, Any]) -> JSONResponse:
    """Process a Label Studio event payload and return response."""
    event_type = payload.get("action") or payload.get("event_type")
    
    logger.info("Label Studio webhook received: %s", event_type)
    
    rejection_info = extract_ls_rejection_info(payload)
    
    if rejection_info:
        logger.warning(
            "QC rejection detected task_id=%s project_id=%s rejected_by=%s reason=%s",
            rejection_info["task_id"],
            rejection_info["project_id"],
            rejection_info.get("rejected_by", "unknown"),
            rejection_info.get("rejection_reason", "N/A"),
        )
        logger.debug("Rejection payload: %s", json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        logger.debug("Label Studio payload: %s", json.dumps(payload, indent=2, ensure_ascii=False))
    
    # Auto-trigger export on task completion (run in background)
    if event_type in ["TASK_COMPLETED", "ANNOTATION_CREATED", "ANNOTATION_UPDATED"]:
        project_id = _extract_project_id_from_payload(payload)
        if project_id:
            logger.info("Triggering export for project %s (event=%s)", project_id, event_type)
            
            # Run export in background using BackgroundTasks
            from fastapi import BackgroundTasks
            background_tasks = BackgroundTasks()
            
            async def trigger_export():
                try:
                    from pathlib import Path
                    import sys
                    project_root = Path(__file__).parent.parent
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    
                    # Load environment variables (same as pipeline scripts)
                    try:
                        from dotenv import load_dotenv
                        env_path = project_root / ".env"
                        if env_path.exists():
                            load_dotenv(env_path)
                            logger.debug("Background export loaded env from %s", env_path)
                    except ImportError:
                        pass
                    
                    from backend.pipeline import PipelineService
                    
                    service = PipelineService.from_settings()
                    await service.run_export_only(project_id=project_id)
                    
                    logger.info("Export completed for project %s", project_id)
                except Exception as export_error:
                    error_msg = f"Failed to trigger export: {export_error}"
                    logger.error(error_msg, exc_info=True)
                    logger.info(
                        "Troubleshooting tips: ensure S3 bucket + AWS credentials are configured and .env exists."
                    )
            
            # Schedule background task
            import asyncio
            asyncio.create_task(trigger_export())
            
            logger.info("Export task scheduled for project %s", project_id)
    
    return JSONResponse(
        content={
            "status": "received",
            "type": "label_studio_event",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
        },
        status_code=200,
    )


@app.post("/webhooks/label-studio")
async def preprocessing_rejection_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """Receive preprocessing rejection webhooks from the pipeline."""
    if not verify_webhook_secret(x_webhook_secret):
        logger.warning(f"Invalid webhook secret from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )

    try:
        payload = await request.json()
        
        # Check if this looks like a Label Studio event (misrouted)
        is_label_studio_event = (
            isinstance(payload, dict) and
            ("action" in payload or "event_type" in payload) and
            ("task" in payload or "annotation" in payload or "project" in payload)
        )
        
        if is_label_studio_event:
            # This is a Label Studio event sent to the wrong endpoint
            event_type = payload.get("action") or payload.get("event_type", "unknown")
            logger.warning(
                "Label Studio event received at preprocessing endpoint (misrouted). "
                "Event type: %s. "
                "Label Studio events should be sent to /webhooks/label-studio/events instead. "
                "Auto-routing to correct handler.",
                event_type,
            )
            logger.debug("Label Studio payload: %s", json.dumps(payload, indent=2, ensure_ascii=False))
            
            # Process as Label Studio event
            return await _process_label_studio_event(payload)
        
        # Check if this looks like a preprocessing rejection payload
        row_index = payload.get("row_index")
        channel = payload.get("channel")
        reason = payload.get("reason")
        
        if row_index is None and channel is None and reason is None:
            # Unknown payload structure
            logger.warning(
                "Received webhook at preprocessing endpoint but payload missing expected fields (row_index, channel, reason). "
                "Payload keys: %s",
                list(payload.keys()) if isinstance(payload, dict) else "not a dict",
            )
            logger.debug("Unexpected payload: %s", json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            logger.info(
                "Preprocessing rejection received row_index=%s channel=%s reason=%s",
                row_index,
                channel,
                reason,
            )
            logger.debug("Preprocessing payload: %s", json.dumps(payload, indent=2, ensure_ascii=False))
        
        return JSONResponse(
            content={
                "status": "received",
                "type": "preprocessing_rejection",
                "timestamp": datetime.now().isoformat(),
                "row_index": payload.get("row_index"),
            },
            status_code=200,
        )
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in webhook payload: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )
    except Exception as e:
        error_msg = f"Error processing webhook: {e}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@app.post("/webhooks/label-studio/events")
async def label_studio_events_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """Receive webhook events from Label Studio and trigger export on task completion."""
    if not verify_webhook_secret(x_webhook_secret):
        logger.warning(f"Invalid webhook secret from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )

    try:
        payload = await request.json()
        return await _process_label_studio_event(payload)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in webhook payload: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )
    except Exception as e:
        error_msg = f"Error processing webhook: {e}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "webhook-server",
        "mode": MODE.lower(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "webhook-server",
        "version": "1.0.0",
        "mode": MODE.lower(),
        "endpoints": {
            "preprocessing_rejections": "/webhooks/label-studio",
            "label_studio_events": "/webhooks/label-studio/events",
            "health": "/health",
        },
        "security": {
            "webhook_secret_required": IS_PRODUCTION,
            "webhook_secret_set": bool(WEBHOOK_SECRET),
        },
    }


if __name__ == "__main__":
    protocol = "https" if IS_PRODUCTION else "http"
    base_url = f"{protocol}://{HOST if HOST != '0.0.0.0' else 'localhost'}:{PORT}"

    logger.info("Starting unified webhook server (%s mode)", MODE)
    logger.info("Preprocessing endpoint: %s/webhooks/label-studio", base_url)
    logger.info("Event endpoint: %s/webhooks/label-studio/events", base_url)
    logger.info("Health endpoint: %s/health", base_url)
    logger.info("Host binding: %s:%s", HOST, PORT)

    if IS_PRODUCTION:
        logger.info("Webhook secret verification ENABLED")
    else:
        logger.info("Webhook secret verification DISABLED (local mode)")
        logger.info("Set WEBHOOK_SECRET to enable production security")

    logger.info("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info" if IS_PRODUCTION else "warning",
        access_log=IS_PRODUCTION,
    )
