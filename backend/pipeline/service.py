"""Config-driven orchestration pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

from backend.annotation.enrichment import EnrichmentRunner
from backend.annotation.preannotation import PreAnnotationRunner
from backend.annotation.transcription import TranscriptionRunner
from backend.config import get_config, get_value
from backend.postprocessing import PostProcessingPipeline
from backend.postprocessing.workflow import AnnotationWorkflow
from httpx import HTTPStatusError

from backend.integrations import (
    AssemblyAIClient,
    GoogleDriveDownloader,
    LabelStudioClient,
    S3Uploader,
)
from backend.pipeline.config import (
    PostProcessingConfig,
    LabelStudioProjectConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from backend.pipeline.data_source import DataSourceResolver, validate_audio_location
from backend.pipeline.metadata_store import ProjectMetadata, ProjectMetadataStore
from backend.pipeline.task_operation import TaskOperation


logger = logging.getLogger(__name__)


class PipelineService:
    """High-level orchestration service following the documented flow."""

    def __init__(self, modules_config: Optional[Dict[str, Any]] = None) -> None:
        self.modules_config = modules_config or get_config()

        self.s3 = self._build_s3_uploader()
        self.labelstudio = self._build_labelstudio_client()
        self.postprocessing = PostProcessingPipeline(s3_uploader=self.s3)

        gdrive_downloader = self._build_google_drive_downloader()
        from backend.preprocessing.pipeline import PreprocessingPipeline
        self.preprocess = PreprocessingPipeline(
            s3_uploader=self.s3,
            google_drive_downloader=gdrive_downloader,
        )

        self.metadata_store = ProjectMetadataStore(self.s3)
        self.preannotation_runner = PreAnnotationRunner()

        self.assembly_client = self._build_assemblyai_client()
        self.transcription_runner = TranscriptionRunner(self.assembly_client)
        self.enrichment_runner = EnrichmentRunner()

        # Cache LS user info for annotations
        self._labelstudio_user_info: Optional[Dict[str, Any]] = None
        self._labelstudio_default_user_id: Optional[int] = None

    @classmethod
    def from_settings(cls, modules_path: Optional[str | Path] = None) -> "PipelineService":
        config = get_config(modules_path)
        return cls(modules_config=config)

    def _get_section(self, *keys: str) -> Optional[Dict[str, Any]]:
        data = get_value(*keys, config=self.modules_config)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _require_field(section: Optional[Dict[str, Any]], field: str, context: str) -> Any:
        if not section:
            raise ValueError(f"{context} is missing in modules.yaml")
        value = section.get(field)
        if value in (None, "", []):
            raise ValueError(f"{context}.{field} is missing in modules.yaml")
        return value

    def _build_s3_uploader(self) -> S3Uploader:
        bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
        if not bucket:
            raise ValueError("S3 bucket is required (set S3_BUCKET or AWS_S3_BUCKET environment variable)")
        region = os.getenv("AWS_REGION", "ap-south-1")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        return S3Uploader(
            bucket_name=str(bucket),
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=str(region),
        )

    def _build_labelstudio_client(self) -> LabelStudioClient:
        base_url = os.getenv("LABEL_STUDIO_BASE_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        if not base_url or not api_key:
            raise ValueError("LABEL_STUDIO_BASE_URL and LABEL_STUDIO_API_KEY must be set in .env file")
        return LabelStudioClient(str(base_url), str(api_key))

    def _build_google_drive_downloader(self) -> Optional[GoogleDriveDownloader]:
        service_account = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH")
        api_key = os.getenv("GOOGLE_DRIVE_API_KEY")
        if not service_account and not api_key:
            return None
        return GoogleDriveDownloader(
            api_key=api_key,
            service_account_path=service_account,
        )

    def _build_assemblyai_client(self) -> AssemblyAIClient:
        base_url = os.getenv("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2")
        poll_interval = float(os.getenv("ASSEMBLYAI_POLL_INTERVAL", "3.0"))
        auto_chapters_str = os.getenv("ASSEMBLYAI_AUTO_CHAPTERS", "false")
        auto_chapters = auto_chapters_str.lower() in ("true", "yes", "1", "on")
        speaker_labels_str = os.getenv("ASSEMBLYAI_SPEAKER_LABELS", "true")
        speaker_labels = speaker_labels_str.lower() in ("true", "yes", "1", "on")
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY must be set in .env file")

        return AssemblyAIClient(
            api_key=str(api_key),
            base_url=str(base_url),
            poll_interval=poll_interval,
            auto_chapters=auto_chapters,
            speaker_labels=speaker_labels,
        )

    def _get_csv_folder_name(self, csv_path: str) -> str:
        """
        Extract folder name from CSV path for organizing data (metadata only).
        """
        csv_file = Path(csv_path)
        folder_name = csv_file.stem
        import re

        folder_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
        folder_name = re.sub(r'_+', '_', folder_name)
        folder_name = folder_name.strip('_')
        return folder_name or "csv_data"

    async def run_with_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Execute preprocessing → annotation setup → delivery via config."""
        pipeline_start = time.perf_counter()
        logger.info("Pipeline run started")
        
        # Load data using data source resolver
        step_start = time.perf_counter()
        client_config = self.modules_config.get("clientConfig", {})
        dev_config = self.modules_config.get("devConfig", {})
        
        # Support both new structure and legacy flat structure
        if not client_config:
            # Legacy: assume CSV from preprocessing.csv_path
            source_type = "csv"
            data_source_config = {"csv": {"path": config.preprocessing.csv_path}}
            logger.info("clientConfig missing; defaulting to CSV data source")
        else:
            data_source_config = client_config.get("dataSource", {})
            if not data_source_config:
                # Fallback: use CSV from preprocessing config
                source_type = "csv"
                data_source_config = {"csv": {"path": config.preprocessing.csv_path}}
                logger.info("clientConfig.dataSource missing; defaulting to CSV data source")
            else:
                source_type = data_source_config.get("type", "csv")
        
        logger.info("Resolved data source", extra={"source_type": source_type})
        
        resolver = DataSourceResolver(
            source_type=source_type,
            source_config=data_source_config,
            dev_config=dev_config,
        )
        
        try:
            data_rows = resolver.load_data()
            step_duration = time.perf_counter() - step_start
            logger.info(
                "Loaded data rows",
                extra={
                    "count": len(data_rows),
                    "source_type": source_type,
                    "duration_sec": round(step_duration, 2),
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {source_type}: {e}")
        
        # For CSV path display, use the resolved path or fallback
        csv_display_path = config.preprocessing.csv_path
        if source_type != "csv":
            csv_display_path = f"{source_type}://{data_source_config.get(source_type, {}).get('tableName') or data_source_config.get(source_type, {}).get('endpoint', 'unknown')}"
        
        csv_folder_name = self._get_csv_folder_name(csv_display_path)
        
        base_prefix = (config.preprocessing.s3_prefix or "").strip()
        if not base_prefix:
            # Try to get from dev config
            dev_ingestion = dev_config.get("ingestion", {})
            s3_config = dev_ingestion.get("s3", {})
            base_prefix = s3_config.get("rawPrefix", "").replace("/raw", "").rstrip("/")
            if not base_prefix:
                raise ValueError("preprocessing.s3_prefix must be provided in modules.yaml or devConfig.ingestion.s3.rawPrefix")
        
        client_id = config.client_id
        s3_prefix = f"{client_id}/{base_prefix.rstrip('/')}"
        
        if config.preprocessing.version:
            s3_prefix = f"{s3_prefix}/{config.preprocessing.version}"
        elif config.preprocessing.add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_prefix = f"{s3_prefix}/{timestamp}"
        
        logger.info(
            "Resolved paths",
            extra={"data_source": csv_display_path, "s3_prefix": s3_prefix},
        )

        config.preprocessing.store_processed_artifacts = bool(config.preprocessing.clear_audio)
        config.preprocessing.store_ml_ready_artifacts = bool(
            config.annotation.preannotation
            or config.annotation.transcription
            or config.annotation.enrichment
        )
        
        # Get audio location preferences for validation
        audio_preferences = data_source_config.get("audioLocationPreference", {})
        runtime_checks = dev_config.get("ingestion", {}).get("runtimeChecks", {})
        
        step_start = time.perf_counter()
        logger.info("Preprocessing started")
        preprocessing_result = await self.preprocess_inputs(
            data_rows=data_rows,
            s3_prefix=s3_prefix,
            options=config.preprocessing,
            audio_preferences=audio_preferences,
            runtime_checks=runtime_checks,
        )
        if isinstance(preprocessing_result, tuple):
            preprocessing_tasks, speaker_tracker = preprocessing_result
        else:
            # Backward compatibility
            preprocessing_tasks = preprocessing_result
            speaker_tracker = None
        step_duration = time.perf_counter() - step_start
        logger.info(
            "Preprocessing finished",
            extra={
                "task_count": len(preprocessing_tasks),
                "duration_sec": round(step_duration, 2),
                "avg_time_per_task_sec": round(step_duration / len(preprocessing_tasks), 2) if preprocessing_tasks else 0,
            },
        )

        # ---------------- Pre-validate UPDATE tasks (before expensive annotation processing) ----------------
        # For UPDATE operations, verify tasks exist in Label Studio before processing annotations
        # First, try to determine project_id if we have project title
        validation_project_id = None
        if preprocessing_tasks:
            # Check if we have UPDATE tasks that need validation
            has_update_tasks = False
            for task in preprocessing_tasks:
                task_operation = task.get("task_operation")
                if not task_operation:
                    metadata = task.get("metadata", {})
                    if isinstance(metadata, dict):
                        task_operation = metadata.get("task_operation")
                if task_operation and TaskOperation.from_string(task_operation) == TaskOperation.UPDATE:
                    has_update_tasks = True
                    break
            
            if has_update_tasks:
                # Try to find project by title to get project_id for validation
                ls_project_cfg = config.ls_project
                project_title = ls_project_cfg.project_name_prefix
                try:
                    existing = await self.labelstudio.find_project_by_title(project_title)
                    if existing:
                        validation_project_id = existing["id"]
                        logger.info("Found existing project for UPDATE validation", extra={"project_id": validation_project_id})
                except Exception as e:
                    logger.debug("Could not find project for validation", extra={"error": str(e)})
        
        if preprocessing_tasks and validation_project_id:
            logger.info("Pre-validating UPDATE tasks before annotation processing")
            validated_tasks = []
            skipped_update_tasks = []
            
            for task in preprocessing_tasks:
                task_operation = task.get("task_operation")
                if not task_operation:
                    metadata = task.get("metadata", {})
                    if isinstance(metadata, dict):
                        task_operation = metadata.get("task_operation")
                
                # Only validate UPDATE operations
                if task_operation and TaskOperation.from_string(task_operation) == TaskOperation.UPDATE:
                    task_id = task.get("task_id")
                    if not task_id:
                        metadata = task.get("metadata", {})
                        if isinstance(metadata, dict):
                            task_id = metadata.get("task_id")
                    
                    if task_id:
                        try:
                            existing_task = await self.labelstudio.get_task(task_id)
                            task_project_id = existing_task.get("project")
                            
                            # Verify task belongs to current project
                            if task_project_id and task_project_id != validation_project_id:
                                logger.warning(
                                    f"Row {task.get('row_index', 'unknown')}: Task ID {task_id} belongs to a different project (Project {task_project_id}). "
                                    f"Cannot update tasks from other projects. This task will be skipped.",
                                    extra={
                                        "row_index": task.get("row_index"),
                                        "task_id": task_id,
                                        "task_project_id": task_project_id,
                                        "current_project_id": validation_project_id,
                                        "error": "task_wrong_project",
                                    },
                                )
                                skipped_update_tasks.append(task)
                                continue
                            
                            # Task exists and belongs to project - proceed
                            validated_tasks.append(task)
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 404:
                                logger.warning(
                                    f"Row {task.get('row_index', 'unknown')}: Task ID {task_id} does not exist in Label Studio. "
                                    f"Cannot update a task that doesn't exist. This task will be skipped.",
                                    extra={
                                        "row_index": task.get("row_index"),
                                        "task_id": task_id,
                                        "error": "task_not_found",
                                        "status_code": 404,
                                    },
                                )
                                skipped_update_tasks.append(task)
                                continue
                            else:
                                # For other HTTP errors, still proceed (might be temporary)
                                logger.warning(
                                    f"Row {task.get('row_index', 'unknown')}: Could not verify task {task_id} (Error {e.response.status_code}). Proceeding anyway, but please verify the task exists.",
                                    extra={
                                        "row_index": task.get("row_index"),
                                        "task_id": task_id,
                                        "status_code": e.response.status_code,
                                    },
                                )
                                validated_tasks.append(task)
                        except Exception as e:
                            # For other errors, log but proceed (might be network issue)
                            logger.warning(
                                f"Row {task.get('row_index', 'unknown')}: Could not verify task {task_id} due to an error. Proceeding anyway, but please verify the task exists.",
                                extra={
                                    "row_index": task.get("row_index"),
                                    "task_id": task_id,
                                    "error": str(e),
                                },
                            )
                            validated_tasks.append(task)
                    else:
                        # No task_id - will be caught later, but proceed for now
                        validated_tasks.append(task)
                else:
                    # Not UPDATE operation, proceed normally
                    validated_tasks.append(task)
            
            if skipped_update_tasks:
                logger.warning(
                    f"Skipped {len(skipped_update_tasks)} UPDATE task(s) because the task IDs were not found or belong to different projects.",
                    extra={"skipped_count": len(skipped_update_tasks)},
                )
            
            preprocessing_tasks = validated_tasks

        # ---------------- Annotation chain ----------------
        if config.annotation.preannotation:
            step_start = time.perf_counter()
            logger.info("Pre-annotation enabled")
            annotated_tasks = self.preannotation_runner.run(
                preprocessing_tasks,
                config.annotation.preannotation,
            )
            step_duration = time.perf_counter() - step_start
            logger.info(
                "Pre-annotation completed",
                extra={
                    "task_count": len(annotated_tasks),
                    "duration_sec": round(step_duration, 2),
                },
            )
        else:
            logger.info("Pre-annotation disabled")
            annotated_tasks = preprocessing_tasks

        if config.annotation.transcription:
            step_start = time.perf_counter()
            logger.info("Transcription enabled")
            transcribed_tasks = await self.transcription_runner.run(
                annotated_tasks,
                config.annotation.transcription,
            )
            step_duration = time.perf_counter() - step_start
            logger.info(
                "Transcription completed",
                extra={
                    "task_count": len(transcribed_tasks),
                    "duration_sec": round(step_duration, 2),
                    "avg_time_per_task_sec": round(step_duration / len(transcribed_tasks), 2) if transcribed_tasks else 0,
                },
            )
        else:
            logger.info("Transcription disabled")
            transcribed_tasks = annotated_tasks

        if config.annotation.enrichment:
            step_start = time.perf_counter()
            logger.info("Additional annotation enabled")
            enriched_tasks = self.enrichment_runner.run(
                transcribed_tasks,
                config.annotation.enrichment,
            )
            step_duration = time.perf_counter() - step_start
            logger.info(
                "Enrichment completed",
                extra={
                    "task_count": len(enriched_tasks),
                    "duration_sec": round(step_duration, 2),
                },
            )
        else:
            logger.info("Additional annotation disabled")
            enriched_tasks = transcribed_tasks

        export_prefix = s3_prefix

        ls_project_cfg = config.ls_project
        project_title = ls_project_cfg.project_name_prefix
        project_description = f"{ls_project_cfg.description}\nCSV: {Path(config.preprocessing.csv_path).name}"

        step_start = time.perf_counter()
        project_id = None
        existing = await self.labelstudio.find_project_by_title(project_title)
        if existing:
            logger.info(
                "Reusing Label Studio project",
                extra={"project_title": project_title, "project_id": existing["id"]},
            )
            project_id = existing["id"]
        else:
            logger.info("Creating Label Studio project", extra={"project_title": project_title})

        # Generate dynamic labelConfig based on detected speakers
        dynamic_label_config = self._generate_dynamic_label_config(
            config.annotation.label_config,
            speaker_tracker,
            enriched_tasks,
            config.annotation.preannotation.label_schema.fields if config.annotation.preannotation else [],
        )
        
        project_info = await self.setup_labelstudio_project(
            project_title=project_title,
            description=project_description,
            label_config=dynamic_label_config,
            tasks=enriched_tasks,
            ls_project=ls_project_cfg,
            existing_project_id=project_id,
            s3_prefix=s3_prefix,
        )
        step_duration = time.perf_counter() - step_start
        logger.info(
            "Label Studio project setup completed",
            extra={
                "project_id": project_info["project"]["id"],
                "task_count": len(enriched_tasks),
                "duration_sec": round(step_duration, 2),
            },
        )

        self._persist_project_metadata(
            project_info=project_info,
            csv_path=config.preprocessing.csv_path,
            preprocessing_prefix=s3_prefix,
            export_prefix=export_prefix,
            config=config,
        )

        export_info: Dict[str, Any] = {
            "project_id": project_info["project"]["id"],
            "task_count": 0,
            "transcripts": [],
            "annotations": [],
            "s3_prefix": export_prefix,
        }
        deliverables = None

        if config.postprocessing:
            step_start = time.perf_counter()
            logger.info("Post-processing enabled; exporting project data")
            export_info = await self.export_project_data(
                project_id=project_info["project"]["id"],
                s3_prefix=export_prefix,
                output_dir="./.tmp/export",
                extract_audio_segments=config.postprocessing.extract_audio_segments,
                output_fields=config.postprocessing.output_fields,
            )
            export_duration = time.perf_counter() - step_start
            logger.info(
                "Export completed",
                extra={
                    "task_count": export_info.get("task_count", 0),
                    "duration_sec": round(export_duration, 2),
                },
            )

            transcripts = export_info.get("transcripts", [])
            annotations = export_info.get("annotations", [])

            # Only build deliverables if there are user annotations (not just pre-annotation transcripts)
            # This prevents creating incomplete final_report.csv on initial run
            if not annotations:
                logger.warning(
                    "No user annotations found during post-processing; complete QC in Label Studio before exporting",
                    extra={
                        "project_id": project_info["project"]["id"],
                        "task_count": export_info.get("task_count"),
                        "transcript_count": len(transcripts),
                        "annotation_count": len(annotations),
                    },
                )
                deliverables = {
                    "transcripts": transcripts,  # Keep transcripts for reference
                    "annotations": [],
                    "final_csv": None,
                    "note": "No user annotations found - complete QC in Label Studio first. Run export-only after annotations are submitted.",
                }
            else:
                step_start = time.perf_counter()
                logger.info("Building deliverables bundle")
                deliverables = await self.build_deliverables(
                    s3_prefix=config.postprocessing.s3_prefix or export_prefix,
                    transcripts=transcripts,
                    annotations=annotations,
                    final_csv_path=config.postprocessing.final_csv_path,
                    postprocessing_config=config.postprocessing,
                )
                deliverables_duration = time.perf_counter() - step_start
                logger.info(
                    "Deliverables built",
                    extra={
                        "transcripts": len(transcripts),
                        "annotations": len(annotations),
                        "duration_sec": round(deliverables_duration, 2),
                    },
                )

                if config.postprocessing.completion_webhook_url:
                    logger.info("Triggering completion webhook")
                    await self._send_completion_webhook(
                        url=config.postprocessing.completion_webhook_url,
                        secret=config.postprocessing.completion_webhook_secret,
                        payload={
                            "client_id": config.client_id,
                            "project_id": project_info["project"]["id"],
                            "s3_prefix": config.postprocessing.s3_prefix or export_prefix,
                            "preprocessing_tasks": len(preprocessing_tasks),
                            "annotation_tasks": len(enriched_tasks),
                            "export": {
                                "task_count": export_info.get("task_count", 0),
                            },
                            "deliverables": {
                                "transcripts": len(transcripts),
                                "annotations": len(annotations),
                            },
                        },
                    )

        total_duration = time.perf_counter() - pipeline_start
        logger.info(
            "Pipeline run completed",
            extra={
                "total_duration_sec": round(total_duration, 2),
                "total_duration_min": round(total_duration / 60, 2),
            },
        )
        return {
            "preprocessing_tasks": len(preprocessing_tasks),
            "annotation_tasks": len(enriched_tasks),
            "project": project_info["project"],
            "project_info": project_info,  # Include full project_info with task counts
            "export": export_info,
            "deliverables": deliverables,
        }
    
    def _cleanup(self) -> None:
        """Cleanup resources like thread pools and connections."""
        if hasattr(self, "s3") and self.s3:
            self.s3.shutdown(wait=True)

    async def run_export_only(self, *, project_id: int) -> Dict[str, Any]:
        """Re-export annotations and rebuild deliverables using stored metadata."""
        metadata = self.metadata_store.load(project_id)
        export_prefix = metadata.export_s3_prefix or metadata.preprocessing_s3_prefix
        output_dir = metadata.export_output_dir or "./.tmp/export"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get output_fields from metadata if available
        output_fields = None
        if metadata.postprocessing:
            output_fields = metadata.postprocessing.get("output_fields")

        export_info = await self.export_project_data(
            project_id=project_id,
            s3_prefix=export_prefix,
            output_dir=output_dir,
            extract_audio_segments=metadata.postprocessing.get("extract_audio_segments", True) if metadata.postprocessing else True,
            output_fields=output_fields,
        )

        deliverables = None
        transcripts = export_info.get("transcripts", [])
        annotations = export_info.get("annotations", [])
        
        # Build deliverables if we have processed transcripts/annotations
        # Even if postprocessing metadata is missing (e.g., project created before postprocessing was enabled)
        if transcripts or annotations:
            # Use postprocessing metadata if available, otherwise use defaults
            if metadata.postprocessing:
                s3_prefix = metadata.postprocessing.get("s3_prefix") or export_prefix
                final_csv_path = metadata.postprocessing.get("final_csv_path") or "./.tmp/final_report.csv"
                extract_audio_segments = metadata.postprocessing.get("extract_audio_segments", True)
            else:
                # Use defaults if postprocessing metadata not available
                logger.info(
                    "Postprocessing metadata not found; using defaults",
                    extra={"project_id": project_id},
                )
                s3_prefix = export_prefix
                final_csv_path = "./.tmp/final_report.csv"
                extract_audio_segments = True
            
            Path(final_csv_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Reconstruct PostProcessingConfig
            postprocessing_config = PostProcessingConfig(
                final_csv_path=final_csv_path,
                s3_prefix=s3_prefix,
                extract_audio_segments=extract_audio_segments,
                # Use defaults for other fields
                post_merge_strategy="prefer-human",
                post_output_formats=["json", "csv"],
            )
            
            deliverables = await self.build_deliverables(
                s3_prefix=s3_prefix,
                transcripts=transcripts,
                annotations=annotations,
                final_csv_path=final_csv_path,
                postprocessing_config=postprocessing_config,
            )
            
            # Send completion webhook if configured
            # Check both metadata and current config for webhook URL
            completion_webhook_url = None
            completion_webhook_secret = None
            
            if metadata.postprocessing:
                completion_webhook_url = metadata.postprocessing.get("completion_webhook_url")
                completion_webhook_secret = metadata.postprocessing.get("completion_webhook_secret")
            
            # Fallback to current config if not in metadata
            if not completion_webhook_url:
                try:
                    from backend.pipeline.config import PipelineConfig
                    current_config = PipelineConfig.load_from_dict(self.modules_config) if self.modules_config else None
                    if current_config and current_config.postprocessing:
                        completion_webhook_url = current_config.postprocessing.completion_webhook_url
                        completion_webhook_secret = current_config.postprocessing.completion_webhook_secret
                except Exception as e:
                    logger.debug("Could not load config for completion webhook", extra={"error": str(e)})
            
            if completion_webhook_url:
                logger.info("Triggering completion webhook after export")
                await self._send_completion_webhook(
                    url=completion_webhook_url,
                    secret=completion_webhook_secret,
                    payload={
                        "project_id": project_id,
                        "s3_prefix": s3_prefix,
                        "export": {
                            "task_count": export_info.get("task_count", 0),
                            "processed_count": export_info.get("processed_count", 0),
                            "skipped_count": export_info.get("skipped_count", 0),
                        },
                        "deliverables": {
                            "transcripts": len(transcripts),
                            "annotations": len(annotations),
                            "final_csv": str(final_csv_path) if final_csv_path else None,
                        },
                    },
                )
        elif metadata.postprocessing:
            # No transcripts/annotations but postprocessing was configured
            logger.warning(
                "No annotations found during export; complete QC in Label Studio first",
                extra={
                    "project_id": project_id,
                    "task_count": export_info.get("task_count"),
                },
            )
            deliverables = {
                "transcripts": [],
                "annotations": [],
                "final_csv": None,
                "note": "No annotations found - complete QC in Label Studio first",
            }

        return {
            "project_id": project_id,
            "export": export_info,
            "deliverables": deliverables,
        }

    def _format_task_with_preannotation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Format a task with pre-annotation including diarization and transcription segments.
        
        Transcription segments are always added as labelable regions (with text) to make audio labeling easier.
        Diarization segments are also added when available, with transcription text merged by time overlap.
        Duplicate segments (>80% overlap) are avoided to prevent redundancy.
        """
        formatted = task.copy()
        formatted.pop("transcription", None)
        formatted.pop("segment_feed", None)
        formatted.pop("segments", None)
        formatted.pop("transcription_chat", None)
        results = []

        # Get transcription segments (from ASR) - these will be used for labeling and stored back
        transcription_segments = task.get("segments", [])
        normalized_segments: List[Dict[str, Any]] = []
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Get diarization segments (with speaker labels)
        diarization_segments = task.get("diarization", {}).get("segments", [])
        
        logger.info(
            "Formatting task with preannotation",
            extra={
                "diarization_segment_count": len(diarization_segments),
                "transcription_segment_count": len(transcription_segments),
                "has_transcript": bool(task.get("transcription")),
                "task_keys": list(task.keys()),
            },
        )
        
        segment_idx = 0
        
        # Always add transcription segments as labelable regions (with text for easy editing)
        # This makes it easy to label audio with transcript text
        transcription_segment_map = {}
        for idx, trans_seg in enumerate(transcription_segments):
            start = float(trans_seg.get("start", 0))
            end = float(trans_seg.get("end", start + 1))
            text = (trans_seg.get("text") or "").strip()
            if not text or end <= start:
                continue
            
            # Try to get speaker from transcription segment (keep Speaker_N format)
            speaker_label = trans_seg.get("speaker") or trans_seg.get("label")
            if not speaker_label:
                # Default to Speaker_1 if no speaker info
                speaker_label = "Speaker_1"
            
            normalized_segments.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker_label,
            })

            # Store transcription segment for matching with diarization
            transcription_segment_map[(start, end)] = {
                "text": text,
                "speaker": speaker_label,
            }
            
            # Use the same ID for both label and textarea so they're treated as one region
            region_id = f"transcription-segment-{idx}"
            
            # Get language from task metadata or transcript metadata
            segment_language = ""
            transcript_metadata = task.get("transcript_metadata", {})
            if isinstance(transcript_metadata, dict):
                segment_language = transcript_metadata.get("language_code", "")
            if not segment_language:
                task_data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
                segment_language = (
                    task_data.get("Language") or
                    task_data.get("language") or
                    task_data.get("lang") or
                    ""
                )
            
            # Check if this segment ends at an utterance boundary (for end_of_speech)
            segment_end_of_speech = False
            if isinstance(transcript_metadata, dict):
                utterances = transcript_metadata.get("utterances", [])
                if isinstance(utterances, list) and utterances:
                    for utterance in utterances:
                        utt_end_ms = utterance.get("end", 0)
                        utt_end_sec = utt_end_ms / 1000.0 if utt_end_ms else 0
                        if utt_end_sec > 0 and abs(end - utt_end_sec) < 0.1:
                            segment_end_of_speech = True
                            break
            
            # Add transcription segment as a labelable region with text
            results.append({
                "id": region_id,
                "from_name": "labels",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": [speaker_label],
                    "text": text,
                },
            })
            # Add TextArea result with same ID for perRegion TextArea (transcription_region) to populate
            results.append({
                "id": region_id,
                "from_name": "transcription_region",
                "to_name": "audio",
                "type": "textarea",
                "value": {
                    "start": start,
                    "end": end,
                    "text": [text],
                },
            })
            # Add Language field
            if segment_language:
                results.append({
                    "id": region_id,
                    "from_name": "language",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": [segment_language],
                    },
                })
            # Add End of Speech field
            if segment_end_of_speech:
                results.append({
                    "id": region_id,
                    "from_name": "end_of_speech",
                    "to_name": "audio",
                    "type": "choices",
                    "value": {
                        "start": start,
                        "end": end,
                        "choices": ["true"],
                    },
                })
            segment_idx += 1
        
        # Also add diarization segments (they may have better speaker detection)
        # Merge transcription text into diarization segments when they overlap
        for diar_segment in diarization_segments:
            start = float(diar_segment.get("start", 0))
            end = float(diar_segment.get("end", start + 1))
            if end <= start:
                continue
            
            # Get speaker label (keep Speaker_N format)
            speaker_label = diar_segment.get("speaker") or diar_segment.get("label")
            if not speaker_label:
                # Try to get speaker_index and convert to Speaker_N format
                speaker_idx = diar_segment.get("speaker_index", 0)
                speaker_label = f"Speaker_{speaker_idx + 1}"
            
            # Check if this diarization segment overlaps with any transcription segment
            # If there's an exact match, skip to avoid duplicates
            is_duplicate = False
            for (trans_start, trans_end), trans_data in transcription_segment_map.items():
                # Check for significant overlap (>80% of either segment)
                overlap_start = max(start, trans_start)
                overlap_end = min(end, trans_end)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    diar_duration = end - start
                    trans_duration = trans_end - trans_start
                    overlap_ratio_diar = overlap_duration / diar_duration if diar_duration > 0 else 0
                    overlap_ratio_trans = overlap_duration / trans_duration if trans_duration > 0 else 0
                    
                    # If >80% overlap, consider it a duplicate and skip diarization segment
                    if overlap_ratio_diar > 0.8 and overlap_ratio_trans > 0.8:
                        is_duplicate = True
                        break
            
            if is_duplicate:
                continue
            
            # Match transcription text to this diarization segment by time overlap
            segment_text = ""
            for trans_seg in transcription_segments:
                trans_start = float(trans_seg.get("start", 0))
                trans_end = float(trans_seg.get("end", trans_start + 1))
                trans_text = (trans_seg.get("text") or "").strip()
                
                # Check if transcription segment overlaps with diarization segment
                if trans_text and trans_end > start and trans_start < end:
                    # Calculate overlap
                    overlap_start = max(start, trans_start)
                    overlap_end = min(end, trans_end)
                    overlap_ratio = (overlap_end - overlap_start) / (end - start)
                    
                    # If significant overlap (>50%), include this text
                    if overlap_ratio > 0.5:
                        if segment_text:
                            segment_text += " "
                        segment_text += trans_text
            
            # If no transcription matched, try to get text from diarization segment itself
            if not segment_text:
                segment_text = diar_segment.get("text", "").strip()
            
            # If still no text, use empty string (user can add it in Label Studio)
            if not segment_text:
                segment_text = ""
            
            # Get language from task metadata or transcript metadata
            segment_language = ""
            transcript_metadata = task.get("transcript_metadata", {})
            if isinstance(transcript_metadata, dict):
                segment_language = transcript_metadata.get("language_code", "")
            if not segment_language:
                task_data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
                segment_language = (
                    task_data.get("Language") or
                    task_data.get("language") or
                    task_data.get("lang") or
                    ""
                )
            
            # Check if this segment ends at an utterance boundary (for end_of_speech)
            segment_end_of_speech = False
            if isinstance(transcript_metadata, dict):
                utterances = transcript_metadata.get("utterances", [])
                if isinstance(utterances, list) and utterances:
                    for utterance in utterances:
                        utt_end_ms = utterance.get("end", 0)
                        utt_end_sec = utt_end_ms / 1000.0 if utt_end_ms else 0
                        if utt_end_sec > 0 and abs(end - utt_end_sec) < 0.1:
                            segment_end_of_speech = True
                            break
            
            # Use the same ID for both label and textarea so they're treated as one region
            region_id = f"diarization-segment-{segment_idx}"
            
            results.append({
                "id": region_id,
                "from_name": "labels",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": [speaker_label],
                    "text": [segment_text],
                },
            })
            # Add TextArea result with same ID for perRegion TextArea (transcription_region) to populate
            if segment_text:
                results.append({
                    "id": region_id,
                    "from_name": "transcription_region",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": [segment_text] if isinstance(segment_text, str) else segment_text,
                    },
                })
            # Add Language field
            if segment_language:
                results.append({
                    "id": region_id,
                    "from_name": "language",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": [segment_language],
                    },
                })
            # Add End of Speech field
            if segment_end_of_speech:
                results.append({
                    "id": region_id,
                    "from_name": "end_of_speech",
                    "to_name": "audio",
                    "type": "choices",
                    "value": {
                        "start": start,
                        "end": end,
                        "choices": ["true"],
                    },
                })
            segment_idx += 1
        
        if results:
            prediction_entry = {
                "result": results,
                "score": 1.0,
                "model_version": task.get("transcript_metadata", {}).get("transcript_id", "preannotation"),
            }
            formatted["predictions"] = [prediction_entry]
            # NOTE: Do NOT create annotations for pre-annotations - only create predictions.
            # Annotations should only be created when users actually submit their work in Label Studio.
            # Creating annotations here marks tasks as "completed" immediately, which is incorrect.
        formatted["segments"] = normalized_segments
        return formatted

    @staticmethod
    def _normalize_speaker_label(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        value = str(raw).strip()
        if not value:
            return None
        lower = value.lower()
        
        # Handle "Speaker one", "Speaker two", "Speaker_1", "Speaker_2" formats
        if lower in ("speaker one", "speaker 1", "speaker_1"):
            return "Speaker_1"
        if lower in ("speaker two", "speaker 2", "speaker_2"):
            return "Speaker_2"
        
        if lower.startswith("speaker"):
            parts = value.split()
            suffix = parts[-1] if parts else ""
            # Handle numeric suffixes (1, 2, 3...) - convert to Speaker_N
            if suffix.isdigit():
                idx = int(suffix)
                return f"Speaker_{idx}"
            # Handle "one", "two", etc.
            if suffix.lower() == "one":
                return "Speaker_1"
            if suffix.lower() == "two":
                return "Speaker_2"
            # Handle Speaker_N format directly
            if "_" in value:
                return value  # Already in Speaker_N format
            if len(suffix) == 1 and suffix.isalpha():
                return f"Speaker_{suffix.upper()}"
            # Default: preserve the original format
            return value
        if len(value) == 1 and value.isalpha():
            return f"Speaker_{value.upper()}"
        if value.isdigit():
            idx = int(value)
            return f"Speaker_{idx + 1}"
        if lower.startswith("spk"):
            suffix = value[3:]
            return f"Speaker {suffix}".strip()
        mapping = {
            "a": "Speaker A",
            "b": "Speaker B",
            "c": "Speaker C",
        }
        return mapping.get(lower, value.title())

    @staticmethod
    def _canonical_speaker_label(label: Optional[str]) -> str:
        """Map any speaker label to the limited set supported by the LS project."""
        if not label:
            return "Speaker one"
        normalized = " ".join(str(label).strip().split()).lower()
        synonyms = {
            "speaker one": 0,
            "speaker 1": 0,
            "speaker 01": 0,
            "speaker a": 0,
            "speaker zero": 0,
            "speaker 0": 0,
            "speaker two": 1,
            "speaker 2": 1,
            "speaker 02": 1,
            "speaker b": 1,
        }
        if normalized in synonyms:
            idx = synonyms[normalized]
        else:
            tail = normalized.split(" ")[-1]
            idx = 0
            if tail.isdigit():
                idx = max(0, int(tail) - 1)
            elif tail in {"one", "first"}:
                idx = 0
            elif tail in {"two", "second"}:
                idx = 1
            elif len(tail) == 1 and tail.isalpha():
                idx = max(0, ord(tail) - ord("a"))
            else:
                match = re.search(r"(\d+)", normalized)
                if match:
                    idx = max(0, int(match.group(1)) - 1)
        return f"Speaker_{idx + 1}" if idx >= 0 else "Speaker_1"

    def _generate_dynamic_label_config(
        self,
        base_label_config: str,
        speaker_tracker: Optional[Any],
        tasks: List[Dict[str, Any]],
        label_schema_fields: List[str],
    ) -> str:
        """Generate dynamic Label Studio labelConfig based on detected speakers."""
        try:
            import xml.etree.ElementTree as ET
            from xml.dom import minidom
        except ImportError:
            logger.warning("XML parsing not available, using base label config")
            return base_label_config
        
        # Extract all unique speaker labels from tasks
        all_speakers: Set[str] = set()
        for task in tasks:
            diarization_segments = task.get("diarization", {}).get("segments", [])
            for seg in diarization_segments:
                speaker = seg.get("speaker") or seg.get("label")
                if speaker:
                    all_speakers.add(speaker)
        
        # If speaker_tracker is available, use its tracked speakers
        if speaker_tracker:
            tracked_speakers = speaker_tracker.get_all_speaker_labels()
            if tracked_speakers:
                all_speakers.update(tracked_speakers)
        
        # Sort speakers: Speaker_1, Speaker_2, etc.
        sorted_speakers = sorted(
            all_speakers,
            key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 999
        )
        
        # If no speakers detected, use default
        if not sorted_speakers:
            sorted_speakers = ["Speaker_1", "Speaker_2"]
        
        # Parse base config or create new one
        try:
            root = ET.fromstring(base_label_config)
        except ET.ParseError:
            # If parsing fails, create a new config
            root = ET.Element("View")
        
        # Find or create Labels element for speakers
        labels_elem = None
        for elem in root.findall(".//Labels[@name='labels']"):
            labels_elem = elem
            break
        
        if labels_elem is None:
            # Create new Labels element
            labels_elem = ET.SubElement(root, "Labels", name="labels", toName="audio")
        
        # Clear existing speaker labels and add dynamic ones
        for child in list(labels_elem):
            if child.tag == "Label" and (child.get("value", "").startswith("Speaker") or "speaker" in child.get("value", "").lower()):
                labels_elem.remove(child)
        
        # Add dynamic speaker labels with colors
        colors = ["#D186FF", "#1CE6FF", "#86FFD1", "#FFD186", "#86D1FF", "#FF86D1"]
        for idx, speaker in enumerate(sorted_speakers):
            color = colors[idx % len(colors)]
            hotkey = str((idx % 9) + 1) if idx < 9 else None
            label_attrs = {
                "value": speaker,
                "background": color,
            }
            if hotkey:
                label_attrs["hotkey"] = hotkey
            ET.SubElement(labels_elem, "Label", **label_attrs)
        
        # Ensure Audio element exists
        audio_elem = None
        for elem in root.findall(".//Audio[@name='audio']"):
            audio_elem = elem
            break
        if audio_elem is None:
            ET.SubElement(root, "Audio", name="audio", value="$audio")
        
        # Ensure per-region TextArea element for segment transcription exists
        transcription_region_elem = None
        for elem in root.findall(".//TextArea[@name='transcription_region']"):
            transcription_region_elem = elem
            break
        if transcription_region_elem is None:
            ET.SubElement(
                root,
                "TextArea",
                name="transcription_region",
                toName="audio",
                perRegion="true",
                rows="3",
                editable="true",
                placeholder="Segment transcription"
            )
        
        # Add per-region Language field
        language_elem = None
        for elem in root.findall(".//TextArea[@name='language']"):
            language_elem = elem
            break
        if language_elem is None:
            ET.SubElement(
                root,
                "TextArea",
                name="language",
                toName="audio",
                perRegion="true",
                rows="1",
                editable="true",
                placeholder="Language (e.g., en, hi, bn)"
            )
        
        # Add per-region End of Speech checkbox
        end_of_speech_elem = None
        for elem in root.findall(".//Choices[@name='end_of_speech']"):
            end_of_speech_elem = elem
            break
        if end_of_speech_elem is None:
            end_of_speech_elem = ET.SubElement(
                root,
                "Choices",
                name="end_of_speech",
                toName="audio",
                perRegion="true"
            )
            ET.SubElement(end_of_speech_elem, "Choice", value="true")
        
        # Convert to string
        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        # Remove XML declaration line
        lines = pretty.split("\n")
        if lines and lines[0].startswith("<?xml"):
            return "\n".join(lines[1:]).strip()
        return pretty.strip()

    @staticmethod
    def _extract_labelstudio_user_id(user_info: Optional[Dict[str, Any]]) -> Optional[int]:
        if not user_info or not isinstance(user_info, dict):
            return None

        candidate_keys = ("id", "pk", "user_id")
        for key in candidate_keys:
            value = user_info.get(key)
            if isinstance(value, int):
                return value

        nested = user_info.get("user")
        if isinstance(nested, dict):
            for key in candidate_keys:
                value = nested.get(key)
                if isinstance(value, int):
                    return value

        return None
    
    def _format_tasks_with_preannotations(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format multiple tasks with pre-annotations if transcripts are available."""
        return [self._format_task_with_preannotation(task) for task in tasks]

    async def preprocess_inputs(
        self,
        data_rows: List[Dict[str, str]],
        *,
        s3_prefix: str,
        options: PreprocessingConfig,
        audio_preferences: Optional[Dict[str, Any]] = None,
        runtime_checks: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Preprocessing module: process data rows, clean audio, upload artifacts."""
        return await self.preprocess.run(
            data_rows=data_rows,
            s3_prefix=s3_prefix,
            options=options,
            audio_preferences=audio_preferences or {},
            runtime_checks=runtime_checks or {},
        )

    async def setup_labelstudio_project(
        self,
        *,
        project_title: str,
        description: str,
        label_config: str,
        tasks: List[Dict[str, Any]],
        ls_project: LabelStudioProjectConfig,
        existing_project_id: Optional[int] = None,
        s3_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create/update Label Studio project, upload tasks, attach integrations."""
        try:
            user_info = await self.labelstudio.verify_connection()
            self._labelstudio_user_info = user_info
            self._labelstudio_default_user_id = self._extract_labelstudio_user_id(user_info)
            username = user_info.get("username") or user_info.get("email")
            if username:
                logger.info(
                    "Connected to Label Studio",
                    extra={"user": username, "user_id": self._labelstudio_default_user_id},
                )
            else:
                logger.info("Label Studio token verified")
        except (ConnectionError, RuntimeError) as e:
            logger.error("Label Studio connection failed", exc_info=e)
            raise
        
        project: Dict[str, Any]
        if existing_project_id:
            logger.info("Fetching existing project", extra={"project_id": existing_project_id})
            project = await self.labelstudio.get_project(existing_project_id)
            logger.info("Reusing project", extra={"project_title": project["title"], "project_id": project["id"]})
        else:
            project = await self.labelstudio.create_project(
                title=project_title,
                description=description,
                label_config=label_config,
            )
            logger.info("Project created", extra={"project_title": project["title"], "project_id": project["id"]})
        project_id = project["id"]

        existing_label_config = (project.get("label_config") or "").strip()
        desired_label_config = label_config.strip()
        if existing_label_config != desired_label_config:
            try:
                logger.info("Updating Label Studio label config")
                project = await self.labelstudio.update_project(
                    project_id,
                    label_config=label_config,
                )
                logger.info("Label config updated")
            except HTTPStatusError as exc:
                logger.error("Label config update failed", exc_info=exc)
                raise

        # Process tasks based on task_operation field (CREATE, UPDATE, SKIP)
        # This allows CSV/DynamoDB to control which tasks to create, update, or skip
        tasks_to_create: List[Dict[str, Any]] = []
        tasks_to_update: List[Dict[str, Any]] = []
        tasks_to_skip: List[Dict[str, Any]] = []
        
        for task in tasks:
            # Get task_operation from top level first, then fallback to metadata
            task_operation = task.get("task_operation")
            if not task_operation:
                # Fallback: check metadata if task_operation not at top level
                metadata = task.get("metadata", {})
                if isinstance(metadata, dict):
                    task_operation = metadata.get("task_operation")
            
            # Get task_id from top level first, then fallback to metadata
            task_id = task.get("task_id")
            if not task_id:
                # Fallback: check metadata if task_id not at top level
                metadata = task.get("metadata", {})
                if isinstance(metadata, dict):
                    task_id = metadata.get("task_id")
            
            row_index = task.get("row_index", "unknown")
            
            # Validation: task_operation must be provided
            if not task_operation or (isinstance(task_operation, str) and not task_operation.strip()):
                logger.warning(
                    f"Row {row_index}: Missing 'task_operation' value. Please add CREATE, UPDATE, or SKIP. This task will be skipped.",
                    extra={
                        "row_index": row_index,
                        "reason": "missing_task_operation",
                    },
                )
                tasks_to_skip.append(task)
                continue  # Skip this task
            
            # Validate and convert to TaskOperation enum
            if isinstance(task_operation, str):
                task_operation_stripped = task_operation.strip().upper()
                # Check if it's a valid enum value
                try:
                    task_operation = TaskOperation(task_operation_stripped)
                except ValueError:
                    # Invalid value provided
                    logger.warning(
                        f"Row {row_index}: Invalid 'task_operation' value '{task_operation}'. "
                        f"Valid values are: CREATE, UPDATE, or SKIP. This task will be skipped.",
                        extra={
                            "row_index": row_index,
                            "task_operation": task_operation,
                            "reason": "invalid_task_operation",
                        },
                    )
                    tasks_to_skip.append(task)
                    continue  # Skip this task
            elif isinstance(task_operation, TaskOperation):
                pass  # Already an enum
            else:
                task_operation = TaskOperation.CREATE
            
            # Validation: CREATE operation cannot have task_id
            if task_operation == TaskOperation.CREATE and task_id:
                logger.warning(
                    f"Row {row_index}: Cannot create a new task with an existing task ID ({task_id}). "
                    f"To create a new task: leave 'task_id' empty. To update an existing task: change 'task_operation' to UPDATE. This task will be skipped.",
                    extra={
                        "row_index": row_index,
                        "task_operation": "CREATE",
                        "task_id": task_id,
                        "reason": "create_with_task_id",
                    },
                )
                tasks_to_skip.append(task)
                continue  # Skip this task
            
            # Validation: UPDATE operation must have task_id
            if task_operation == TaskOperation.UPDATE and not task_id:
                logger.warning(
                    f"Row {row_index}: Cannot update a task without a task ID. "
                    f"Please provide the task ID in the 'task_id' column, or change 'task_operation' to CREATE to create a new task. This task will be skipped.",
                    extra={
                        "row_index": row_index,
                        "task_operation": "UPDATE",
                        "reason": "update_without_task_id",
                    },
                )
                tasks_to_skip.append(task)
                continue  # Skip this task
            
            # Route task to appropriate list based on operation
            if task_operation == TaskOperation.SKIP:
                tasks_to_skip.append(task)
                logger.debug("Task skipped based on task_operation=SKIP", extra={"row_index": row_index})
            elif task_operation == TaskOperation.UPDATE:
                tasks_to_update.append(task)
            else:  # CREATE (validated above)
                tasks_to_create.append(task)
        
        
        # Log operation summary
        logger.info(
            "Task operation summary",
            extra={
                "total_tasks": len(tasks),
                "to_create": len(tasks_to_create),
                "to_update": len(tasks_to_update),
                "to_skip": len(tasks_to_skip),
            },
        )
        
        # Track successful task operations
        tasks_created_count = 0
        tasks_updated_count = 0
        
        # Handle CREATE operations
        if tasks_to_create:
            logger.info("Creating new tasks", extra={"count": len(tasks_to_create)})
            try:
                payload = (
                    self._format_tasks_with_preannotations(tasks_to_create)
                    if ls_project.run_pre_annotate
                    else tasks_to_create
                )
                batch_size = max(1, ls_project.batch_size)
                for start in range(0, len(payload), batch_size):
                    batch = payload[start : start + batch_size]
                    await self.labelstudio.bulk_upload_tasks(project_id, batch)
                tasks_created_count = len(tasks_to_create)
                logger.info("Bulk upload complete", extra={"task_count": tasks_created_count})
            except RuntimeError as e:
                logger.warning("Bulk upload failed; falling back to per-task creation", extra={"error": str(e)[:200]})
                for idx, task in enumerate(tasks_to_create, 1):
                    try:
                        formatted_task = (
                            self._format_task_with_preannotation(task)
                            if ls_project.run_pre_annotate
                            else task
                        )
                        metadata = {
                            k: v
                            for k, v in formatted_task.items()
                            if k not in {"audio", "annotations", "predictions", "task_operation", "task_id"}
                        }
                        await self.labelstudio.create_task(
                            project_id=project_id,
                            audio_url=formatted_task.get("audio", ""),
                            metadata=metadata,
                            predictions=formatted_task.get("predictions"),
                        )
                        tasks_created_count += 1
                        if idx % 25 == 0:
                            logger.debug("Uploaded task chunk", extra={"uploaded": idx, "total": len(tasks_to_create)})
                    except Exception as task_error:
                        logger.error("Per-task upload failed", extra={"task_index": idx, "error": str(task_error)})
                logger.info("Per-task upload complete", extra={"task_count": tasks_created_count, "total_attempted": len(tasks_to_create)})
        
        # Handle UPDATE operations
        if tasks_to_update:
            logger.info("Updating existing tasks", extra={"count": len(tasks_to_update)})
            for idx, task in enumerate(tasks_to_update, 1):
                try:
                    task_id = task.get("task_id")
                    if not task_id:
                        logger.warning(
                            f"Row {task.get('row_index', 'unknown')}: Cannot update task without a task ID. This task will be skipped.",
                            extra={"row_index": task.get("row_index")}
                        )
                        continue
                    
                    formatted_task = (
                        self._format_task_with_preannotation(task)
                        if ls_project.run_pre_annotate
                        else task
                    )
                    
                    metadata = {
                        k: v
                        for k, v in formatted_task.items()
                        if k not in {"audio", "annotations", "predictions", "task_operation", "task_id"}
                    }
                    
                    # Try to verify task exists and belongs to this project before updating
                    try:
                        existing_task = await self.labelstudio.get_task(task_id)
                        # Verify task belongs to current project
                        task_project_id = existing_task.get("project")
                        if task_project_id and task_project_id != project_id:
                            logger.error(
                                f"Row {task.get('row_index', 'unknown')}: Task ID {task_id} belongs to a different project (Project {task_project_id}). "
                                f"Cannot update tasks from other projects. This task will be skipped.",
                                extra={
                                    "row_index": task.get("row_index"),
                                    "task_id": task_id,
                                    "task_project_id": task_project_id,
                                    "current_project_id": project_id,
                                    "error": "task_wrong_project",
                                },
                            )
                            continue  # Skip this task
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            logger.error(
                                f"Row {task.get('row_index', 'unknown')}: Task ID {task_id} does not exist in Label Studio. "
                                f"Cannot update a task that doesn't exist. This task will be skipped.",
                                extra={
                                    "row_index": task.get("row_index"),
                                    "task_id": task_id,
                                    "error": "task_not_found",
                                    "status_code": 404,
                                },
                            )
                            continue  # Skip this task
                        else:
                            raise  # Re-raise if it's not a 404
                    
                    # Task exists, proceed with update
                    await self.labelstudio.update_task(
                        task_id=task_id,
                        audio_url=formatted_task.get("audio"),
                        metadata=metadata,
                        predictions=formatted_task.get("predictions"),
                    )
                    tasks_updated_count += 1
                    logger.info(
                        f"Row {task.get('row_index', 'unknown')}: Successfully updated task {task_id} in Label Studio.",
                        extra={"row_index": task.get("row_index"), "task_id": task_id},
                    )
                    
                    if idx % 25 == 0:
                        logger.debug("Updated task chunk", extra={"updated": idx, "total": len(tasks_to_update)})
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        logger.error(
                            f"Row {task.get('row_index', 'unknown')}: Task ID {task_id} does not exist in Label Studio. "
                            f"Cannot update a task that doesn't exist.",
                            extra={
                                "row_index": task.get("row_index"),
                                "task_id": task_id,
                                "error": "task_not_found",
                                "status_code": 404,
                            },
                        )
                    else:
                        logger.error(
                            f"Row {task.get('row_index', 'unknown')}: Failed to update task {task_id}. Please check the task ID and try again.",
                            extra={
                                "row_index": task.get("row_index"),
                                "task_id": task_id,
                                "error": str(e),
                                "status_code": e.response.status_code,
                            },
                        )
                except Exception as task_error:
                    logger.error(
                        f"Row {task.get('row_index', 'unknown')}: Task update failed for task {task_id}",
                        extra={
                            "row_index": task.get("row_index"),
                            "task_id": task_id,
                            "error": str(task_error),
                        },
                    )
            logger.info("Task updates complete", extra={"task_count": tasks_updated_count, "total_attempted": len(tasks_to_update)})
        
        if tasks_to_skip:
            logger.info("Skipped tasks", extra={"count": len(tasks_to_skip)})

        ml_backend_info = None
        backend_cfg = ls_project.ml_backend
        if backend_cfg.enabled and backend_cfg.url:
            ml_backend_url = backend_cfg.url
            logger.info("Verifying ML backend", extra={"url": ml_backend_url})
            await self._verify_ml_backend(ml_backend_url)
            try:
                logger.info("Checking for existing ML backend entry")
                existing_backend = await self.labelstudio.find_ml_backend(url=ml_backend_url)
            except Exception:
                existing_backend = None

            if existing_backend:
                ml_backend_info = existing_backend
                logger.info("Existing ML backend found", extra={"ml_backend_id": ml_backend_info.get("id")})
            else:
                try:
                    logger.info("Registering ML backend", extra={"url": ml_backend_url})
                    ml_backend_info = await self.labelstudio.register_ml_backend(
                        title=f"{project_title} Backend",
                        url=ml_backend_url,
                    )
                    logger.info("ML backend registered", extra={"ml_backend_id": ml_backend_info.get("id")})
                except Exception as e:
                    logger.warning("ML backend registration failed; attempting reuse", exc_info=e)
                    try:
                        ml_backend_info = await self.labelstudio.find_ml_backend(url=ml_backend_url)
                    except Exception as find_err:
                        ml_backend_info = None
                        logger.error("Unable to reuse ML backend entry", exc_info=find_err)
            
            if ml_backend_info:
                try:
                    logger.info(
                        "Attaching ML backend",
                        extra={"ml_backend_id": ml_backend_info["id"], "project_id": project_id},
                    )
                    await self.labelstudio.attach_ml_backend_to_project(
                        project_id,
                        ml_backend_info["id"],
                    )
                    await self._confirm_ml_attachment(project_id, ml_backend_info["id"])
                    logger.info("ML backend attachment verified")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to attach ML backend automatically. "
                        f"Error: {e}. Check that Label Studio can reach {ml_backend_url} and rerun."
                    ) from e

        storage_info = None
        target_storage_info = None
        storage_cfg = ls_project.storage
        if storage_cfg.enabled:
            try:
                bucket = storage_cfg.bucket or os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
                if not bucket:
                    logger.warning("Storage bucket missing; set storage.bucket or S3_BUCKET env var")
                else:
                    aws_access_key = storage_cfg.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
                    aws_secret_key = storage_cfg.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
                    region = storage_cfg.region or os.getenv("AWS_REGION", "ap-south-1")
                    
                    if storage_cfg.source_enabled:
                        logger.info(
                            "Configuring Label Studio SOURCE storage",
                            extra={"type": storage_cfg.storage_type, "bucket": bucket},
                        )
                        source_prefix = storage_cfg.source_prefix or storage_cfg.prefix
                        if not source_prefix and s3_prefix:
                            source_prefix = f"{s3_prefix}/raw"
                            logger.info("SOURCE prefix auto-detected", extra={"prefix": source_prefix})
                        elif source_prefix:
                            logger.info("SOURCE prefix configured", extra={"prefix": source_prefix})
                        
                        existing_storage = await self.labelstudio.find_cloud_storage(
                            project_id=project_id,
                            storage_type=storage_cfg.storage_type,
                            bucket=bucket,
                            prefix=source_prefix,
                            use_blob_urls=storage_cfg.use_blob_urls,
                        )
                        
                        if existing_storage:
                            storage_info = existing_storage
                            logger.info("Existing SOURCE storage found", extra={"storage_id": existing_storage.get("id")})
                        else:
                            logger.info("Creating SOURCE storage")
                            title_prefix = s3_prefix if s3_prefix else (storage_cfg.source_prefix or storage_cfg.prefix or source_prefix)
                            if title_prefix and title_prefix.endswith("/raw"):
                                title_prefix = title_prefix[:-4]
                            storage_title = storage_cfg.title or f"S3: {bucket}/{title_prefix or ''}"
                            
                            storage_info = await self.labelstudio.configure_cloud_storage(
                                project_id=project_id,
                                storage_type=storage_cfg.storage_type,
                                bucket=bucket,
                                prefix=source_prefix,
                                region=region,
                                aws_access_key_id=aws_access_key,
                                aws_secret_access_key=aws_secret_key,
                                use_blob_urls=storage_cfg.use_blob_urls,
                                title=storage_title,
                                is_source=True,
                                is_target=False,
                            )
                            logger.info("SOURCE storage configured")
                    
                    if storage_cfg.target_enabled:
                        logger.info(
                            "Configuring Label Studio TARGET storage",
                            extra={"type": storage_cfg.storage_type, "bucket": bucket},
                        )
                        target_prefix = storage_cfg.target_prefix or storage_cfg.prefix
                        if not target_prefix and s3_prefix:
                            target_prefix = f"{s3_prefix}/annotations"
                            logger.info("TARGET prefix auto-detected", extra={"prefix": target_prefix})
                        elif target_prefix:
                            logger.info("TARGET prefix configured", extra={"prefix": target_prefix})
                        
                        existing_target = await self.labelstudio.find_cloud_storage(
                            project_id=project_id,
                            storage_type=storage_cfg.storage_type,
                            bucket=bucket,
                            prefix=target_prefix,
                            use_blob_urls=storage_cfg.use_blob_urls,
                        )
                        
                        if existing_target:
                            target_storage_info = existing_target
                            logger.info("Existing TARGET storage found", extra={"storage_id": existing_target.get("id")})
                        else:
                            logger.info("Creating TARGET storage")
                            target_title = f"Target S3: {bucket}/{target_prefix or ''}"
                            
                            target_storage_info = await self.labelstudio.configure_cloud_storage(
                                project_id=project_id,
                                storage_type=storage_cfg.storage_type,
                                bucket=bucket,
                                prefix=target_prefix,
                                region=region,
                                aws_access_key_id=aws_access_key,
                                aws_secret_access_key=aws_secret_key,
                                use_blob_urls=False,  # Target storage typically doesn't need blob URLs
                                title=target_title,
                                is_source=False,
                                is_target=True,
                            )
                            logger.info("TARGET storage configured")
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "not available" in error_msg.lower():
                    logger.warning(
                        "Label Studio storage API unavailable; configure manually",
                        extra={"project_id": project_id},
                    )
                else:
                    logger.error("Failed to configure Label Studio storage automatically", exc_info=e)

        webhook_info = None
        webhook_cfg = ls_project.webhook
        if webhook_cfg.enabled and webhook_cfg.url:
            try:
                logger.info(
                    "Ensuring Label Studio webhook",
                    extra={"url": webhook_cfg.url, "events": webhook_cfg.events},
                )
                existing_webhook = await self.labelstudio.find_webhook(
                    project_id=project_id,
                    target_url=webhook_cfg.url,
                    events=webhook_cfg.events,
                )
                if existing_webhook:
                    webhook_info = existing_webhook
                    logger.info("Existing webhook found", extra={"webhook_id": existing_webhook.get("id")})
                if not webhook_info:
                    webhook_info = await self.labelstudio.register_webhook(
                        project_id=project_id,
                        target_url=webhook_cfg.url,
                        events=webhook_cfg.events,
                        is_active=webhook_cfg.is_active,
                    )
                    logger.info("Webhook registered", extra={"webhook_id": webhook_info.get("id")})
            except Exception as e:
                logger.error("Failed to configure webhook automatically", exc_info=e)

        return {
            "project": project,
            "ml_backend": ml_backend_info,
            "storage": storage_info,
            "target_storage": target_storage_info,
            "webhook": webhook_info,
            "tasks_created": tasks_created_count,
            "tasks_updated": tasks_updated_count,
            "tasks_skipped": len(tasks_to_skip),
        }

    async def _verify_ml_backend(self, backend_url: str) -> None:
        """Ping the ML backend /health endpoint before attaching."""
        from urllib.parse import urlparse
        import httpx

        parsed = urlparse(backend_url)
        base_url = backend_url
        if parsed.path and parsed.path.endswith("/predict"):
            base_url = backend_url[: backend_url.rfind("/predict")]
        health_url = f"{base_url}/health"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(health_url)
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"ML backend health check failed at {health_url}. "
                f"Ensure the backend is running before running the pipeline. Error: {exc}"
            ) from exc

    async def _confirm_ml_attachment(self, project_id: int, backend_id: int) -> None:
        """Verify the backend is attached to the project, retry once if needed."""
        project = await self.labelstudio.get_project(project_id)
        attached = project.get("ml_backends") or []
        if backend_id in attached:
            return

        await self.labelstudio.attach_ml_backend_to_project(project_id, backend_id)
        project = await self.labelstudio.get_project(project_id)
        attached = project.get("ml_backends") or []
        if backend_id not in attached:
            raise RuntimeError(
                f"ML backend {backend_id} could not be attached to project {project_id}. "
                "Attach it manually in Label Studio and rerun."
            )

    def _persist_project_metadata(
        self,
        *,
        project_info: Dict[str, Any],
        csv_path: str,
        preprocessing_prefix: str,
        export_prefix: str,
        config: PipelineConfig,
    ) -> None:
        project = project_info["project"]
        metadata = ProjectMetadata(
            project_id=project["id"],
            project_title=project.get("title", ""),
            csv_filename=Path(csv_path).name,
            preprocessing_s3_prefix=preprocessing_prefix,
            export_s3_prefix=export_prefix,
            export_output_dir="./.tmp/export",
            postprocessing=(
                {
                    "s3_prefix": config.postprocessing.s3_prefix if config.postprocessing else None,
                    "final_csv_path": config.postprocessing.final_csv_path if config.postprocessing else None,
                    "extract_audio_segments": config.postprocessing.extract_audio_segments if config.postprocessing else False,
                    "output_fields": config.postprocessing.output_fields if config.postprocessing else None,
                }
                if config.postprocessing
                else None
            ),
            ml_backend_url=config.ls_project.ml_backend.url if config.ls_project.ml_backend.enabled else None,
            webhook_url=config.ls_project.webhook.url if config.ls_project.webhook.enabled else None,
        )
        self.metadata_store.save(metadata)

    async def export_project_data(
        self,
        project_id: int,
        *,
        s3_prefix: str,
        output_dir: str,
        extract_audio_segments: bool = True,
        output_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Export final annotations and upload transcripts/segments to S3.
        
        Args:
            project_id: Label Studio project ID
            s3_prefix: S3 prefix for storing exports
            output_dir: Local directory for temporary files
            extract_audio_segments: Whether to extract audio segments
            output_fields: List of field names to include in final JSON (defaults to standard fields)
        """
        workflow = AnnotationWorkflow(  # type: ignore[name-defined]
            label_studio_url=str(self.labelstudio.base_url),
            label_studio_api_key=self.labelstudio.api_key,
            s3_bucket=self.s3.bucket_name,
            s3_access_key=self.s3.access_key_id,
            s3_secret_key=self.s3.secret_access_key,
        )
        
        return await workflow.export_and_save(
            project_id=project_id,
            output_dir=output_dir,
            s3_prefix=s3_prefix,
            extract_audio_segments=extract_audio_segments,
            output_fields=output_fields,
        )

    async def build_deliverables(
        self,
        *,
        s3_prefix: str,
        transcripts: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        final_csv_path: str,
        postprocessing_config: PostProcessingConfig,
    ) -> Dict[str, Any]:
        """Cleanup, build final outputs (CSV/SRT/TXT), upload deliverables."""
        output_formats = postprocessing_config.post_output_formats or ["json", "csv"]
        output_dir = Path(final_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if postprocessing_config.post_schema_enabled and postprocessing_config.post_schema_id:
            self.postprocessing.validate_schema(
                items=transcripts,
                schema_id=postprocessing_config.post_schema_id,
            )

        final_csv_path_obj: Optional[Path] = None
        if "csv" in output_formats:
            final_csv_path_obj = self.postprocessing.build_final_csv(
                transcripts,
                Path(final_csv_path),
            )

        extra_files = self.postprocessing.build_additional_outputs(
            transcripts=transcripts,
            output_dir=output_dir,
            formats=output_formats,
        )

        return self.postprocessing.upload_deliverables(
            s3_prefix=s3_prefix,
            transcripts=transcripts,
            annotations=annotations,
            final_csv=final_csv_path_obj,
            extra_files=extra_files,
            bucket_override=postprocessing_config.post_s3_bucket,
        )

    async def _send_completion_webhook(
        self,
        *,
        url: str,
        secret: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        """Send completion webhook after successful post-processing."""
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Webhook-Secret"] = secret

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json=payload, headers=headers)
        except Exception as exc:
            logger.warning("Completion webhook failed", extra={"url": url, "error": str(exc)})