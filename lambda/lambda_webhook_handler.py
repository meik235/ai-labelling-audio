"""Lambda handler to receive Label Studio webhooks and trigger export.

For Lambda deployment, this file must be self-contained.
For local development, see webhooks/lambda_handler.py (shared implementation).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path for Lambda deployment
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.pipeline import PipelineService


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle API Gateway webhook events."""
    logger.info("Lambda webhook handler invoked")
    logger.debug("Event received: %s", json.dumps(event, indent=2, default=str))
    
    try:
        body = event.get("body") or "{}"
        logger.debug("Parsing webhook body: %s", body[:500] if len(body) > 500 else body)
        data = json.loads(body)
        logger.info("Webhook payload parsed successfully")
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in webhook body: {e}"
        logger.error(error_msg)
        logger.debug("Failed body content: %s", body[:500] if len(body) > 500 else body)
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON body"})}

    project_id = _extract_project_id(data)
    logger.debug("Extracted project_id: %s", project_id)
    
    if not project_id:
        error_msg = "project_id missing from webhook payload"
        logger.warning(error_msg)
        logger.debug("Payload structure: %s", json.dumps(data, indent=2, default=str))
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "project_id missing from webhook"}),
        }

    logger.info("Starting export for project_id=%s", project_id)
    
    try:
        service = PipelineService.from_settings()
        logger.debug("PipelineService initialized")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("Running export for project_id=%s", project_id)
            result = loop.run_until_complete(
                service.run_export_only(project_id=int(project_id))
            )
            logger.info("Export completed successfully for project_id=%s", project_id)
            logger.debug("Export result: %s", json.dumps(result, indent=2, default=str))
        finally:
            loop.close()
            logger.debug("Event loop closed")
    except Exception as exc:
        error_msg = f"Export failed for project_id={project_id}: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "export failed", "message": str(exc)}),
        }

    response = {
        "message": "Export triggered",
        "project_id": project_id,
        "deliverables": result.get("deliverables", {}),
    }
    logger.info("Returning success response for project_id=%s", project_id)
    logger.debug("Response: %s", json.dumps(response, indent=2, default=str))
    
    return {
        "statusCode": 200,
        "body": json.dumps(response),
    }


def _extract_project_id(payload: Dict[str, Any]) -> int | None:
    """Support multiple LS webhook payload shapes."""
    logger.debug("Extracting project_id from payload")
    
    if "project_id" in payload:
        project_id = payload["project_id"]
        logger.debug("Found project_id in payload root: %s", project_id)
        return project_id
    
    if "project" in payload and isinstance(payload["project"], dict):
        project_id = payload["project"].get("id")
        logger.debug("Found project_id in payload.project: %s", project_id)
        return project_id
    
    if "task" in payload and isinstance(payload["task"], dict):
        project = payload["task"].get("project")
        if isinstance(project, dict):
            project_id = project.get("id")
            logger.debug("Found project_id in payload.task.project: %s", project_id)
            return project_id
    
    logger.debug("No project_id found in payload")
    return None
