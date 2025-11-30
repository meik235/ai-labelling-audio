"""AWS Lambda handler for CSV upload-triggered pipeline execution."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import boto3
from backend.pipeline import PipelineService, PipelineConfig

logger = logging.getLogger(__name__)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler triggered by S3 CSV upload.
    
    Expected S3 event structure:
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "bucket-name"},
                "object": {"key": "csvUpload/filename.csv"}
            }
        }]
    }
    
    Args:
        event: Lambda event from S3 trigger
        context: Lambda context
        
    Returns:
        Response dict with status and details
    """
    s3_client = boto3.client("s3")
    
    # Extract S3 bucket and key from event
    try:
        records = event.get("Records", [])
        if not records:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No S3 records in event"}),
            }
        
        s3_record = records[0].get("s3", {})
        bucket_name = s3_record.get("bucket", {}).get("name")
        object_key = s3_record.get("object", {}).get("key")
        
        if not bucket_name or not object_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing bucket or key in S3 event"}),
            }
        
        # Verify CSV is in csvUpload/ folder
        if not object_key.startswith("csvUpload/"):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": f"CSV must be in csvUpload/ folder. Got: {object_key}",
                }),
            }
        
        logger.info("Processing CSV s3://%s/%s", bucket_name, object_key)
        
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to parse S3 event: {str(e)}"}),
        }
    
    # Download CSV to temporary file
    csv_filename = Path(object_key).name
    temp_csv_path = None
    
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".csv", delete=False) as tmp_file:
            temp_csv_path = tmp_file.name
            s3_client.download_fileobj(bucket_name, object_key, tmp_file)
        
        logger.info("Downloaded CSV to %s", temp_csv_path)
        
        # Extract CSV folder name (one CSV = one folder = one project)
        csv_folder_name = Path(object_key).stem
        # Sanitize folder name
        import re
        csv_folder_name = re.sub(r'[^a-zA-Z0-9_-]', '_', csv_folder_name)
        csv_folder_name = re.sub(r'_+', '_', csv_folder_name).strip('_') or "csv_data"
        
        # Create pipeline config with production settings
        # s3_prefix will be set to csv_folder_name by pipeline
        # Structure: {csv_folder_name}/raw/, {csv_folder_name}/processed/, etc.
        config_dict = {
            "clientId": "default",
            "preprocessing": {
                "csv_path": temp_csv_path,
                "s3_prefix": "",
            },
            "annotation": {
                "labelConfig": """<View>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio" rows="5" label="Transcript"/>
</View>""",
            },
            "delivery": {
                "final_csv_path": "/tmp/final_report.csv",
                "s3_prefix": "",
                "extract_audio_segments": True,
            },
            "lsProject": {
                "projectNamePrefix": f"AUDIO_{csv_folder_name}_",
                "description": f"Automated pipeline triggered by CSV upload: {object_key}",
                "batchSize": 25,
                "mlBackend": {
                    "enabled": True,
                    "url": os.getenv("ML_BACKEND_URL", "http://localhost:9090/predict"),
                },
                "runPreAnnotate": True,
                "reviewerIds": [],
                "export": {"format": "json"},
            },
        }
        
        config = PipelineConfig.from_dict(config_dict)
        service = PipelineService.from_settings()
        
        # Run pipeline (async)
        import asyncio
        
        # Create new event loop for Lambda
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(service.run_with_config(config))
        finally:
            loop.close()
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Pipeline completed successfully",
                "csv_file": object_key,
                "tasks_processed": result.get("preprocessing_tasks", 0),
                "project_id": result.get("project", {}).get("id"),
                "exported_tasks": result.get("export", {}).get("task_count", 0),
            }),
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Pipeline failed: %s", error_details)
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Pipeline execution failed",
                "message": str(e),
                "csv_file": object_key,
            }),
        }
        
    finally:
        # Clean up temporary CSV file
        if temp_csv_path and Path(temp_csv_path).exists():
            try:
                Path(temp_csv_path).unlink()
            except Exception as e:
                logger.warning("Failed to delete temp file %s: %s", temp_csv_path, e)

