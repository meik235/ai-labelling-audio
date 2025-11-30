"""Config-driven preprocessing pipeline."""

from __future__ import annotations

import asyncio
import io
import csv
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from backend.integrations import GoogleDriveDownloader, S3Uploader
from backend.pipeline.config import PreprocessingConfig, ValidationConfig
from backend.pipeline.data_source import validate_audio_location
from backend.preprocessing.audio_ops import AudioPreprocessor
from backend.storage import StorageError, fetch_audio_bytes


logger = logging.getLogger(__name__)


class SpeakerTracker:
    """Tracks speakers per row using metadata to assign consistent Speaker_N labels."""
    
    def __init__(self):
        # Map: (row_index, user_identifier) -> speaker_number
        self._speaker_map: Dict[Tuple[int, str], int] = {}
        # Track max speaker number per row
        self._row_max_speaker: Dict[int, int] = {}
        # Track all unique speakers across all rows for label generation
        self._all_speakers: Set[str] = set()
    
    def get_speaker_label(
        self,
        row_index: int,
        channel_name: str,
        metadata: Dict[str, Any],
        detected_speakers: int = 1,
    ) -> Tuple[str, int]:
        """
        Get speaker label for a channel based on row metadata and channel type.
        
        Returns:
            (speaker_label, speaker_index) e.g., ("Speaker_1", 0)
        """
        # Try to identify user from metadata
        user_id = self._extract_user_id(metadata, channel_name)
        
        # For single speaker audio
        if channel_name == "speaker_single":
            speaker_num = 1
            speaker_label = f"Speaker_{speaker_num}"
            self._all_speakers.add(speaker_label)
            self._row_max_speaker[row_index] = max(self._row_max_speaker.get(row_index, 0), speaker_num)
            return speaker_label, 0
        
        # For mixed audio (multiple speakers), use Speaker_1 and Speaker_2
        elif channel_name == "speaker_mix":
            self._all_speakers.add("Speaker_1")
            self._all_speakers.add("Speaker_2")
            return "Speaker_1", 0
        
        # For unknown/additional channels (e.g., speaker_c, speaker_d, or generic audio)
        # Use user tracking or detected speakers, allowing for more than 2 speakers
        if user_id:
            key = (row_index, user_id)
            if key not in self._speaker_map:
                # Assign next available speaker number for this row
                # This allows Speaker_3, Speaker_4, etc. for additional speakers
                max_speaker = self._row_max_speaker.get(row_index, 0)
                speaker_num = max_speaker + 1
                self._speaker_map[key] = speaker_num
                self._row_max_speaker[row_index] = speaker_num
            else:
                speaker_num = self._speaker_map[key]
        else:
            # No user ID, use detected speaker count
            # Allow more than 2 speakers for cases with additional audio files
            # But ensure at least 1 speaker
            speaker_num = max(1, min(detected_speakers, 10))  # Allow up to 10 speakers, minimum 1
        
        speaker_label = f"Speaker_{speaker_num}"
        self._all_speakers.add(speaker_label)
        return speaker_label, speaker_num - 1
    
    def get_all_speaker_labels(self) -> List[str]:
        """Get all unique speaker labels for labelConfig generation."""
        return sorted(self._all_speakers, key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
    
    @staticmethod
    def _extract_user_id(metadata: Dict[str, Any], channel_name: str) -> Optional[str]:
        """Extract user identifier from metadata (email, username, etc.)."""
        # Try common metadata fields
        for key in ["Email address", "KGeN Login Email / Registered Mobile No", 
                   "KGeN Username", "Name", "email", "username", "Email"]:
            if key in metadata and metadata[key]:
                return str(metadata[key]).strip()
        
        # If channel has user info (e.g., speaker_a might have Name, speaker_b has Name__2)
        if channel_name == "speaker_a":
            if "Name" in metadata and metadata.get("Name"):
                return str(metadata["Name"]).strip()
        elif channel_name == "speaker_b":
            if "Name__2" in metadata and metadata.get("Name__2"):
                return str(metadata["Name__2"]).strip()
        
        return None


def _constrain_segments_by_duration(
    segments: List[Dict[str, Any]],
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Split segments that exceed max_duration and filter out segments below min_duration."""
    if not segments:
        return segments
    
    constrained: List[Dict[str, Any]] = []
    
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        duration = end - start
        
        # Filter out segments below minimum duration
        if min_duration and duration < min_duration:
            continue
        
        # Split segments that exceed maximum duration
        if max_duration and duration > max_duration:
            num_splits = int(duration / max_duration) + 1
            split_duration = duration / num_splits
            for i in range(num_splits):
                split_start = start + (i * split_duration)
                split_end = start + ((i + 1) * split_duration) if i < num_splits - 1 else end
                constrained.append({
                    **seg,
                    "start": round(split_start, 3),
                    "end": round(split_end, 3),
                })
        else:
            constrained.append(seg)
    
    return constrained


def _speaker_label_from_index(index: int) -> str:
    """Return canonical speaker labels matching Label Studio config."""
    if index == 0:
        return "Speaker one"
    if index == 1:
        return "Speaker two"
    return f"Speaker {index + 1}"


class PreprocessingPipeline:
    """Ingest CSV rows, validate audio, clean audio, upload artifacts."""

    def __init__(
        self,
        *,
        s3_uploader: S3Uploader,
        google_drive_downloader: Optional[GoogleDriveDownloader] = None,
    ) -> None:
        self.s3 = s3_uploader
        self.gdrive = google_drive_downloader

    async def run(
        self,
        data_rows: Optional[List[Dict[str, str]]] = None,
        csv_path: Optional[str] = None,
        *,
        s3_prefix: str,
        options: PreprocessingConfig,
        temp_dir: Optional[str] = None,
        audio_preferences: Optional[Dict[str, Any]] = None,
        runtime_checks: Optional[Dict[str, Any]] = None,
        diarization_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], SpeakerTracker]:
        """Execute validation + clear-audio modules as configured.
        
        Returns:
            (tasks, speaker_tracker) - List of tasks and speaker tracker for dynamic label generation
        """
        temp_root = Path(temp_dir or tempfile.mkdtemp(prefix="preprocess_"))
        temp_root.mkdir(parents=True, exist_ok=True)
        s3_prefix_clean = s3_prefix.rstrip("/")
        
        # Initialize speaker tracker for per-row speaker assignment
        speaker_tracker = SpeakerTracker()
        
        # Support both new data_rows parameter and legacy csv_path
        if data_rows is not None:
            rows = data_rows
            logger.info("Loaded rows from resolver", extra={"count": len(rows)})
        elif csv_path:
            rows = self._read_csv(csv_path)
            logger.info("Loaded rows from CSV", extra={"csv_path": csv_path, "count": len(rows)})
        else:
            raise ValueError("Either data_rows or csv_path must be provided")
        
        audio_preferences = audio_preferences or {}
        runtime_checks = runtime_checks or {}

        logger.info("Preprocessing module started")
        self._log_validation_status(options.validation)
        self._log_clear_audio_status(options.clear_audio)

        tasks: List[Dict[str, Any]] = []

        # Build jobs for each audio channel
        jobs: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []  # Track skipped rows for logging
        
        logger.info("Starting CSV row processing", extra={"total_rows": len(rows)})
        
        for idx, row in enumerate(rows, start=2):
            metadata = self._extract_metadata(row)
            
            # Extract email or identifier for logging
            row_identifier = metadata.get("Email address") or metadata.get("Email") or f"Row {idx}"
            
            # Check task_operation BEFORE processing audio (early exit for SKIP or invalid config)
            task_operation = metadata.get("task_operation")
            task_id = metadata.get("task_id")
            
            # Import TaskOperation here to avoid circular imports
            from backend.pipeline.task_operation import TaskOperation
            
            # Validation: task_operation must be provided
            if not task_operation or (isinstance(task_operation, str) and not task_operation.strip()):
                logger.warning(
                    f"Row {idx} ({row_identifier}): Missing 'task_operation' column value. "
                    f"Please add CREATE, UPDATE, or SKIP in the 'task_operation' column. This row will be skipped.",
                    extra={
                        "row_index": idx,
                        "row_identifier": row_identifier,
                        "reason": "missing_task_operation",
                    },
                )
                skipped_rows.append({
                    "row_index": idx,
                    "identifier": row_identifier,
                    "reason": "missing_task_operation",
                })
                continue  # Skip all processing for this row
            
            # Validate and normalize task_operation
            if isinstance(task_operation, str):
                task_operation_stripped = task_operation.strip().upper()
                # Check if it's a valid enum value
                try:
                    task_operation_enum = TaskOperation(task_operation_stripped)
                except ValueError:
                    # Invalid value provided
                    logger.warning(
                        f"Row {idx} ({row_identifier}): Invalid 'task_operation' value '{task_operation}'. "
                        f"Valid values are: CREATE, UPDATE, or SKIP. This row will be skipped.",
                        extra={
                            "row_index": idx,
                            "row_identifier": row_identifier,
                            "task_operation": task_operation,
                            "reason": "invalid_task_operation",
                        },
                    )
                    skipped_rows.append({
                        "row_index": idx,
                        "identifier": row_identifier,
                        "reason": "invalid_task_operation",
                    })
                    continue  # Skip all processing for this row
            else:
                task_operation_enum = TaskOperation.CREATE
            
            # Skip entire row if task_operation is SKIP
            if task_operation_enum == TaskOperation.SKIP:
                logger.info(
                    f"Row {idx} ({row_identifier}): Skipped as requested (task_operation=SKIP).",
                    extra={
                        "row_index": idx,
                        "row_identifier": row_identifier,
                        "reason": "task_operation_skip",
                    },
                )
                skipped_rows.append({
                    "row_index": idx,
                    "identifier": row_identifier,
                    "reason": "task_operation_skip",
                })
                continue  # Skip all processing for this row
            
            # Validation: CREATE cannot have task_id
            if task_operation_enum == TaskOperation.CREATE and task_id:
                logger.warning(
                    f"Row {idx} ({row_identifier}): Cannot create a new task with an existing task ID ({task_id}). "
                    f"To create a new task: leave 'task_id' empty. To update an existing task: change 'task_operation' to UPDATE. This row will be skipped.",
                    extra={
                        "row_index": idx,
                        "row_identifier": row_identifier,
                        "task_operation": "CREATE",
                        "task_id": task_id,
                        "reason": "create_with_task_id",
                    },
                )
                skipped_rows.append({
                    "row_index": idx,
                    "identifier": row_identifier,
                    "reason": "create_with_task_id",
                    "task_id": task_id,
                })
                continue  # Skip all processing for this row
            
            # Validation: UPDATE must have task_id
            if task_operation_enum == TaskOperation.UPDATE and not task_id:
                logger.warning(
                    f"Row {idx} ({row_identifier}): Cannot update a task without a task ID. "
                    f"Please provide the task ID in the 'task_id' column, or change 'task_operation' to CREATE to create a new task. This row will be skipped.",
                    extra={
                        "row_index": idx,
                        "row_identifier": row_identifier,
                        "task_operation": "UPDATE",
                        "reason": "update_without_task_id",
                    },
                )
                skipped_rows.append({
                    "row_index": idx,
                    "identifier": row_identifier,
                    "reason": "update_without_task_id",
                })
                continue  # Skip all processing for this row
            
            # Now extract audio sources and continue with normal processing
            audio_sources = self._extract_audio_columns(row)

            if not audio_sources:
                logger.warning(
                    f"Row {idx} ({row_identifier}): No audio file URLs found. Please add audio URLs in the audio upload columns. This row will be skipped.",
                    extra={
                        "row_index": idx,
                        "row_identifier": row_identifier,
                        "reason": "no_audio_urls",
                    },
                )
                skipped_rows.append({
                    "row_index": idx,
                    "identifier": row_identifier,
                    "reason": "no_audio_urls",
                })
                continue
            
            logger.info(
                "✓ CSV Row PROCESSING: Found audio sources",
                extra={
                    "row_index": idx,
                    "row_identifier": row_identifier,
                    "audio_sources": list(audio_sources.keys()),
                },
            )

            logger.debug(
                "Detected audio sources",
                extra={"row_index": idx, "sources": list(audio_sources.keys())},
            )

            # Persist original row metadata for traceability under raw/
            raw_metadata_key = f"{s3_prefix_clean}/raw/metadata/row_{idx}.json"
            self.s3.upload_bytes(
                json.dumps(
                    {
                        "row_index": idx,
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
                raw_metadata_key,
                metadata={"source_row": str(idx), "type": "metadata"},
            )

            for channel_name, audio_url in audio_sources.items():
                if not audio_url:
                    continue
                
                # Validate audio location if runtime checks are enabled
                if runtime_checks.get("requireKnownSource", True):
                    if not validate_audio_location(audio_url, audio_preferences, runtime_checks):
                        row_identifier = metadata.get("Email address") or metadata.get("Email") or f"Row {idx}"
                        logger.warning(
                            "CSV Row REJECTED: Audio URL rejected due to location policy",
                            extra={
                                "row_index": idx,
                                "row_identifier": row_identifier,
                                "channel": channel_name,
                                "audio_url": audio_url,
                                "reason": "location_policy_rejection",
                            },
                        )
                        skipped_rows.append({
                            "row_index": idx,
                            "identifier": row_identifier,
                            "channel": channel_name,
                            "reason": "location_policy_rejection",
                            "audio_url": audio_url,
                        })
                        continue
                
                jobs.append(
                    {
                        "row_index": idx,
                        "channel_name": channel_name,
                        "audio_url": audio_url,
                        "metadata": metadata,
                    }
                )

        # Process all jobs in parallel with a concurrency limit
        semaphore = asyncio.Semaphore(5)

        async def _process_job(job: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            async with semaphore:
                row_index = job["row_index"]
                channel_name = job["channel_name"]
                audio_url = job["audio_url"]
                metadata = job["metadata"]
                store_processed = getattr(options, "store_processed_artifacts", False)
                store_ml_ready = getattr(options, "store_ml_ready_artifacts", False)

                logger.info(
                    "Processing audio row",
                    extra={"row_index": row_index, "channel": channel_name},
                )
                try:
                    raw_bytes = await self._download_audio_with_policy(
                        audio_url=audio_url,
                        validation=options.validation,
                    )
                except Exception as exc:
                    row_identifier = metadata.get("Email address") or metadata.get("Email") or f"Row {row_index}"
                    logger.warning(
                        f"CSV Row {row_index} REJECTED ({row_identifier}): Audio download failed - {str(exc)}",
                        extra={
                            "row_index": row_index,
                            "row_identifier": row_identifier,
                            "channel": channel_name,
                            "error": str(exc),
                            "reason": "download_failed",
                        },
                    )
                    await self._handle_rejection(
                        row_index=row_index,
                        channel_name=channel_name,
                        reason=str(exc),
                        metadata=metadata,
                        validation=options.validation,
                    )
                    return None

                channel_variants = self._split_channel_variants(raw_bytes, channel_name)
                variant_tasks: List[Dict[str, Any]] = []
                for variant in channel_variants:
                    task = await self._process_audio_variant(
                        audio_bytes=variant["bytes"],
                        row_index=row_index,
                        channel_name=variant["name"],
                        audio_url=audio_url,
                        metadata=metadata,
                        options=options,
                        s3_prefix_clean=s3_prefix_clean,
                        store_processed=store_processed,
                        store_ml_ready=store_ml_ready,
                        diarization_config=diarization_config,
                        channel_info=variant.get("channel_info"),
                        speaker_tracker=speaker_tracker,
                    )
                    if task:
                        variant_tasks.append(task)
                
                if variant_tasks:
                    row_identifier = metadata.get("Email address") or metadata.get("Email") or f"Row {row_index}"
                    logger.info(
                        "✓ CSV Row SUCCESS: Task(s) created",
                        extra={
                            "row_index": row_index,
                            "row_identifier": row_identifier,
                            "channel": channel_name,
                            "tasks_created": len(variant_tasks),
                        },
                    )

                return variant_tasks

        if jobs:
            results = await asyncio.gather(*[_process_job(job) for job in jobs])
            tasks = [
                task
                for job_tasks in results
                if job_tasks
                for task in job_tasks
            ]
        
        # Log summary of CSV row processing
        successful_rows = set()
        for task in tasks:
            row_idx = task.get("row_index")
            if row_idx:
                successful_rows.add(row_idx)
        
        rows_with_audio = len([r for r in rows if self._extract_audio_columns(r)])
        
        logger.info("=" * 80)
        logger.info(
            "CSV PROCESSING SUMMARY",
            extra={
                "total_csv_rows": len(rows),
                "rows_with_audio": rows_with_audio,
                "rows_processed": len(successful_rows),
                "rows_skipped": len(skipped_rows),
                "tasks_created": len(tasks),
            },
        )
        logger.info(f"Total CSV rows: {len(rows)}")
        logger.info(f"Rows with audio URLs: {rows_with_audio}")
        logger.info(f"Rows successfully processed: {len(successful_rows)}")
        logger.info(f"Rows skipped/rejected: {len(skipped_rows)}")
        logger.info(f"Tasks created: {len(tasks)}")
        
        if skipped_rows:
            logger.warning("=" * 60)
            logger.warning("SUMMARY: Rows that were skipped:")
            logger.warning("=" * 60)
            for skipped in skipped_rows:
                reason = skipped.get('reason', 'unknown reason')
                reason_messages = {
                    "missing_task_operation": "Missing 'task_operation' value",
                    "invalid_task_operation": f"Invalid 'task_operation' value (valid: CREATE, UPDATE, SKIP)",
                    "task_operation_skip": "Intentionally skipped (task_operation=SKIP)",
                    "create_with_task_id": f"Cannot create new task with existing ID {skipped.get('task_id', '')}",
                    "update_without_task_id": "UPDATE operation requires 'task_id'",
                    "no_audio_urls": "No audio file URLs found",
                }
                reason_msg = reason_messages.get(reason, reason)
                logger.warning(
                    f"  • Row {skipped['row_index']} ({skipped.get('identifier', 'N/A')}): {reason_msg}",
                )
            logger.warning("=" * 60)
        if successful_rows:
            logger.info(f"SUCCESSFUL ROWS (tasks created): {sorted(successful_rows)}")
        logger.info("=" * 80)

        logger.info("Preprocessing complete", extra={"task_count": len(tasks)})
        return tasks

    def _split_channel_variants(self, audio_bytes: bytes, channel_name: str) -> List[Dict[str, Any]]:
        """Return list of audio variants (no channel splitting; we keep a single task per row)."""
        channel_info = {"index": -1, "count": 1, "names": ["channel_1"]}
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.warning("soundfile not installed; unable to inspect channels")
            return [{"name": channel_name, "bytes": audio_bytes, "channel_info": channel_info}]

        try:
            buffer = io.BytesIO(audio_bytes)
            data, _ = sf.read(buffer, always_2d=True)
            total_channels = data.shape[1] if len(data.shape) > 1 else 1
            channel_info = {
                "index": -1,
                "count": total_channels,
                "names": [f"channel_{idx + 1}" for idx in range(total_channels)],
            }
        except Exception as exc:
            logger.warning("Failed to inspect audio channels: %s", exc)
        return [{"name": channel_name, "bytes": audio_bytes, "channel_info": channel_info}]

    async def _process_audio_variant(
        self,
        *,
        audio_bytes: bytes,
        row_index: int,
        channel_name: str,
        audio_url: str,
        metadata: Dict[str, Any],
        options: PreprocessingConfig,
        s3_prefix_clean: str,
        store_processed: bool,
        store_ml_ready: bool,
        diarization_config: Optional[Dict[str, Any]],
        channel_info: Optional[Dict[str, Any]] = None,
        speaker_tracker: Optional[SpeakerTracker] = None,
    ) -> Optional[Dict[str, Any]]:
        raw_bytes = audio_bytes
        validation_summary = await self._validate_audio(
            raw_bytes=raw_bytes,
            source_url=audio_url,
            row_index=row_index,
            channel_name=channel_name,
            validation=options.validation,
            metadata=metadata,
        )
        if not validation_summary:
            return None

        original_duration = validation_summary.get("duration_sec", 0.0)
        
        # For diarization, we need full audio (cleaned but NOT trimmed) to analyze entire duration
        diarization_audio_bytes = raw_bytes
        
        if options.clear_audio:
            audio_processor = AudioPreprocessor(
                clear_config=options.clear_audio,
                enable_noise_check=False,
                enable_segmentation=False,
            )
            step_start = time.perf_counter()
            processed_bytes, preprocessing_meta = await audio_processor.preprocess_audio(
                raw_bytes,
                filename=os.path.basename(audio_url),
                original_duration=original_duration,
            )
            step_duration = time.perf_counter() - step_start
            logger.info(
                "Audio preprocessing completed",
                extra={
                    "row_index": row_index,
                    "channel": channel_name,
                    "duration_sec": round(step_duration, 2),
                    "audio_duration_sec": round(original_duration, 2),
                },
            )
            diarization_audio_bytes = processed_bytes
        else:
            processed_bytes = raw_bytes
            preprocessing_meta = {
                "preprocessing_applied": [],
                "note": "clearAudio not configured; audio preprocessing skipped",
            }
            diarization_audio_bytes = raw_bytes

        diarization_segments: List[Dict[str, Any]] = []
        diarization_model: Optional[str] = "assemblyai"

        if diarization_config and diarization_config.get("enabled"):
            logger.info(
                "Ignoring preprocessing diarization config; AssemblyAI diarization will be used instead",
                extra={"row_index": row_index, "channel": channel_name},
            )

        logger.info(
            "Skipping preprocessing diarization in favor of AssemblyAI output",
            extra={"row_index": row_index, "channel": channel_name},
        )
        
        # Always set diarization_meta, even if empty
        diarization_meta = preprocessing_meta.setdefault("diarization", {})
        speaker_ready_segments: List[Dict[str, Any]] = []
        
        if diarization_segments:
            # Use speaker tracker to get consistent Speaker_N labels per row
            detected_speaker_count = len({seg.get("speaker_index", 0) for seg in diarization_segments})
            if speaker_tracker:
                # Get speaker label for this channel/row
                speaker_label, speaker_base_idx = speaker_tracker.get_speaker_label(
                    row_index=row_index,
                    channel_name=channel_name,
                    metadata=metadata,
                    detected_speakers=max(detected_speaker_count, 1),
                )
                
                # For single speaker audio, use the assigned label
                if channel_name == "speaker_single":
                    for seg in diarization_segments:
                        speaker_ready_segments.append({
                            **seg,
                            "label": speaker_label,
                            "speaker": speaker_label,
                            "speaker_index": speaker_base_idx,
                        })
                else:
                    # For mix audio, map diarization speakers to tracked speakers
                    speaker_map: Dict[int, str] = {}
                    for seg in diarization_segments:
                        seg_speaker_idx = int(seg.get("speaker_index", 0))
                        if seg_speaker_idx not in speaker_map:
                            if seg_speaker_idx == 0:
                                speaker_map[seg_speaker_idx] = speaker_label
                            else:
                                additional_label, _ = speaker_tracker.get_speaker_label(
                                    row_index=row_index,
                                    channel_name=f"{channel_name}_spk{seg_speaker_idx}",
                                    metadata=metadata,
                                    detected_speakers=seg_speaker_idx + 1,
                                )
                                speaker_map[seg_speaker_idx] = additional_label
                        speaker_ready_segments.append({
                            **seg,
                            "label": speaker_map[seg_speaker_idx],
                            "speaker": speaker_map[seg_speaker_idx],
                            "speaker_index": seg_speaker_idx,
                        })
            else:
                # Fallback: use Speaker_N format based on detected speakers
                if channel_name == "speaker_single":
                    fixed_speaker = "Speaker_1"
                    for seg in diarization_segments:
                        speaker_ready_segments.append({
                            **seg,
                            "label": fixed_speaker,
                            "speaker": fixed_speaker,
                            "speaker_index": 0,
                        })
                else:
                    for seg in diarization_segments:
                        speaker_idx = int(seg.get("speaker_index", 0))
                        speaker_name = f"Speaker_{speaker_idx + 1}"
                        speaker_ready_segments.append({**seg, "label": speaker_name, "speaker": speaker_name})

            diarization_meta["segments"] = speaker_ready_segments
            diarization_meta["speaker_count"] = len({seg["speaker"] for seg in speaker_ready_segments})
            diarization_meta["source"] = diarization_model or "assemblyai"
            diarization_segments = speaker_ready_segments

            logger.info(
                "Diarization completed",
                extra={
                    "model": diarization_model,
                    "speaker_count": diarization_meta["speaker_count"],
                    "segment_count": len(speaker_ready_segments),
                    "audio_duration": validation_summary.get("duration_sec", 0.0),
                },
            )
        else:
            diarization_meta["segments"] = []
            diarization_meta["speaker_count"] = 0
            diarization_meta["source"] = "assemblyai"
            diarization_meta["note"] = "Preprocessing diarization disabled; AssemblyAI speaker labels will be used."

        preprocessing_meta["validation"] = validation_summary
        include_fields = options.clear_audio.metadata_include_fields if options.clear_audio else []
        metadata_fields = self._select_metadata_fields(
            include_fields=include_fields,
            size_bytes=len(raw_bytes),
            validation_summary=validation_summary,
        )
        preprocessing_meta["metadata_fields"] = metadata_fields
        preprocessing_meta["metadata"] = self._format_metadata_lines(metadata_fields)

        raw_key = f"{s3_prefix_clean}/raw/{channel_name}_{row_index}.wav"
        processed_key = f"{s3_prefix_clean}/preprocessed/{channel_name}_{row_index}_processed.wav"
        processed_meta_key = f"{s3_prefix_clean}/preprocessed/metadata/{channel_name}_{row_index}.json"
        ml_ready_audio_key = f"{s3_prefix_clean}/ml-ready/audio/{channel_name}_{row_index}_clean.wav"

        raw_metadata = {
            "source_row": str(row_index),
            "channel": channel_name,
        }

        processed_meta_payload = json.dumps(
            {
                "row_index": row_index,
                "channel": channel_name,
                "preprocessing": preprocessing_meta,
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        upload_tasks: List[Any] = []
        index_map: Dict[str, int] = {}

        def _enqueue(name: str, coro):
            index_map[name] = len(upload_tasks)
            upload_tasks.append(coro)

        _enqueue("raw", self.s3.upload_bytes_async(raw_bytes, raw_key, metadata=raw_metadata))

        if store_processed:
            _enqueue(
                "processed",
                self.s3.upload_bytes_async(
                    processed_bytes,
                    processed_key,
                    metadata=raw_metadata,
                ),
            )
            _enqueue(
                "processed_meta",
                self.s3.upload_bytes_async(
                    processed_meta_payload,
                    processed_meta_key,
                    metadata=raw_metadata,
                ),
            )

        if store_ml_ready:
            _enqueue(
                "ml_ready",
                self.s3.upload_bytes_async(
                    processed_bytes,
                    ml_ready_audio_key,
                    metadata=raw_metadata,
                ),
            )

        results = await asyncio.gather(*upload_tasks)
        raw_url = results[index_map["raw"]]
        processed_url = results[index_map["processed"]] if store_processed else raw_url
        ml_ready_audio_url = results[index_map["ml_ready"]] if store_ml_ready else None

        metadata_enriched = dict(metadata)
        if channel_info:
            metadata_enriched["channel_info"] = channel_info
        self._inject_sidebar_metadata(
            metadata_enriched,
            preprocessing_meta.get("metadata_fields") or {},
            row_index=row_index,
            channel=channel_name,
        )

        display_channel = channel_name
        if channel_info:
            channel_names = channel_info.get("names")
            if channel_names:
                display_channel = ",".join(channel_names)
            else:
                count = channel_info.get("count", 1)
                if count > 1:
                    display_channel = ",".join([f"channel_{idx + 1}" for idx in range(count)])

        # Use cleaned audio for transcription if audio was cleaned, otherwise use original
        # However, if processed audio is too short (likely over-trimmed), prefer original for transcription
        if options.clear_audio and store_processed:
            # Check if processed audio is significantly shorter than original (likely over-trimmed)
            processed_size = len(processed_bytes) if processed_bytes else 0
            original_size = len(raw_bytes) if raw_bytes else 0
            size_ratio = processed_size / original_size if original_size > 0 else 0
            
            # If processed audio is less than 10% of original size, it's likely over-trimmed
            # Use original audio for transcription to ensure we get the full content
            if size_ratio < 0.1:
                logger.warning(
                    "Processed audio is too short (%.1f%% of original), using original audio for transcription",
                    size_ratio * 100,
                    extra={
                        "row_index": row_index,
                        "channel": channel_name,
                        "original_size_mb": round(original_size / (1024 * 1024), 2),
                        "processed_size_mb": round(processed_size / (1024 * 1024), 2),
                    },
                )
                transcription_audio_url = raw_url
            else:
                transcription_audio_url = processed_url
        else:
            transcription_audio_url = raw_url
        
        task_data = {
            "audio": raw_url,
            "channel": display_channel,
            "row_index": row_index,
            "metadata": metadata_enriched,
            "transcription_audio": transcription_audio_url,
        }
        
        # Store task_operation and task_id in task data for later use
        task_operation = metadata_enriched.get("task_operation", "CREATE")
        task_id = metadata_enriched.get("task_id")
        if task_operation:
            task_data["task_operation"] = task_operation
        if task_id:
            task_data["task_id"] = task_id
        task_data["transcription"] = ""
        if options.clear_audio:
            task_data["preprocessing"] = preprocessing_meta

        if diarization_segments:
            task_data["diarization"] = {"segments": diarization_segments}
            speaker_segments = task_data.setdefault("segments", [])
            for segment in diarization_segments:
                speaker_segments.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment.get("speaker") or segment.get("label", ""),
                        "speaker": segment.get("speaker") or segment.get("label", ""),
                    }
                )

        if store_ml_ready and ml_ready_audio_url:
            ml_payload = {
                **task_data,
                "raw_audio": raw_url,
                "ml_ready_audio": ml_ready_audio_url,
            }
            if store_processed:
                ml_payload["preprocessed_audio"] = processed_url
            self._upload_ml_ready_payload(
                payload=ml_payload,
                s3_prefix=s3_prefix_clean,
                row_index=row_index,
                channel=channel_name,
            )

        logger.info(
            "Preprocessing task created",
            extra={"row_index": row_index, "channel": channel_name},
        )
        return task_data

    def _log_validation_status(self, validation: ValidationConfig) -> None:
        """Log validation configuration status once at the start."""
        if not getattr(validation, "enabled", True):
            logger.info("[Validation] Disabled via config")
            return
        logger.debug(
            "[Validation] Enabled",
            extra={
                "allowed_formats": validation.allowed_formats or "ANY",
                "duration_min": validation.duration_min,
                "duration_max": validation.duration_max,
                "size_max_mb": validation.size_max_mb,
                "sample_rates": validation.sample_rate_allowed or "ANY",
                "download_timeout_sec": validation.download_timeout,
                "retry_count": validation.retry_count,
                "retry_delay_sec": validation.retry_delay,
                "rejection_webhook": validation.rejection_webhook_url or "disabled",
            },
        )

    def _log_clear_audio_status(self, clear_config) -> None:
        """Log clear-audio configuration (or skip message)."""
        if not clear_config:
            logger.info("[ClearAudio] Disabled")
            return
        logger.debug(
            "[ClearAudio] Enabled",
            extra={
                "target_format": clear_config.target_format,
                "target_sample_rate": clear_config.target_sample_rate,
                "normalization": {
                    "enabled": clear_config.normalization_enabled,
                    "level_db": clear_config.normalization_level_db,
                },
                "denoise": {
                    "enabled": clear_config.denoise_enabled,
                    "intensity": clear_config.denoise_intensity,
                },
                "trim": {
                    "enabled": clear_config.trim_enabled,
                    "silence_threshold": clear_config.trim_silence_threshold,
                    "min_silence": clear_config.trim_min_silence_duration,
                },
                "chunking": {
                    "enabled": clear_config.chunking_enabled,
                    "max_duration_sec": clear_config.chunking_max_duration_sec,
                    "overlap_sec": clear_config.chunking_overlap_sec,
                },
                "metadata_fields": clear_config.metadata_include_fields,
            },
        )

    def _read_csv(self, csv_path: str) -> List[Dict[str, str]]:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return []

        header = rows[0]
        deduped_header: List[str] = []
        seen: Dict[str, int] = {}

        for idx, col in enumerate(header):
            clean_name = (col or "").strip()
            if not clean_name:
                clean_name = f"column_{idx}"

            count = seen.get(clean_name, 0)
            if count > 0:
                deduped_name = f"{clean_name}__{count+1}"
            else:
                deduped_name = clean_name
            seen[clean_name] = count + 1
            deduped_header.append(deduped_name)

        records: List[Dict[str, str]] = []
        for row in rows[1:]:
            # Pad or trim row to match header length
            if len(row) < len(deduped_header):
                row = row + [""] * (len(deduped_header) - len(row))
            elif len(row) > len(deduped_header):
                row = row[: len(deduped_header)]
            records.append(dict(zip(deduped_header, row)))

        return records

    def _extract_audio_columns(self, row: Dict[str, str]) -> Dict[str, str]:
        """Extract all audio URLs from CSV.
        
        Creates separate tasks for each audio URL found.
        Channel type (single/mixed) will be determined later from AssemblyAI diarization results.
        """
        audio_columns = {}
        audio_index = 0
        
        for key, value in row.items():
            if not value or not isinstance(value, str):
                continue
            
            if not (value.startswith("http://") or value.startswith("https://")):
                continue
            
            key_lower = key.lower()
            is_audio_column = (
                "audio" in key_lower or 
                "upload" in key_lower or 
                "recording" in key_lower or
                "url" in key_lower
            )
            
            if is_audio_column:
                if audio_index == 0:
                    audio_columns["audio"] = value
                else:
                    audio_columns[f"audio_{audio_index}"] = value
                audio_index += 1
        
        return audio_columns

    def _extract_metadata(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Extract metadata from CSV row, including task operation and task_id.
        
        Special columns:
        - task_operation: CREATE, UPDATE, or SKIP (case-insensitive)
        - task_id: Label Studio task ID (required for UPDATE operations)
        """
        metadata = {}
        for key, value in row.items():
            # Always include task_operation and task_id columns (even if empty) for validation
            if key in ("task_operation", "task_id"):
                metadata[key] = value if value is not None else ""
            elif value:
                metadata[key] = value
        
        # Extract task_operation (preserve empty string for validation, don't default to CREATE)
        # Validation will check if it's missing/empty and skip the row
        task_operation = metadata.get("task_operation", "")
        if task_operation:
            task_operation = task_operation.strip()
        # Keep original value (empty string or actual value) - don't default to CREATE
        metadata["task_operation"] = task_operation
        
        # Extract task_id if provided
        task_id = metadata.get("task_id", "")
        if task_id:
            task_id = task_id.strip()
            # Try to convert to int if it's numeric
            try:
                metadata["task_id"] = int(task_id)
            except (ValueError, TypeError):
                # Keep as string if not numeric
                metadata["task_id"] = task_id
        else:
            metadata["task_id"] = None
        
        return metadata

    async def _download_audio_with_policy(
        self,
        *,
        audio_url: str,
        validation: ValidationConfig,
    ) -> bytes:
        parsed = urlparse(audio_url)
        timeout = httpx.Timeout(validation.download_timeout, connect=10.0)
        attempts = validation.retry_count + 1

        for attempt in range(attempts):
            try:
                if "drive.google.com" in audio_url and self.gdrive:
                    return await self.gdrive.download_file(audio_url)

                if parsed.scheme in {"http", "https"}:
                    return await fetch_audio_bytes(audio_url, timeout=timeout)

                if Path(audio_url).exists():
                    return Path(audio_url).read_bytes()

                raise RuntimeError(f"Unsupported audio source: {audio_url}")
            except (StorageError, RuntimeError) as exc:
                if attempt >= validation.retry_count:
                    raise
                await asyncio.sleep(validation.retry_delay)

        raise RuntimeError(f"Failed to download {audio_url} after retries.")

    async def _validate_audio(
        self,
        *,
        raw_bytes: bytes,
        source_url: str,
        row_index: int,
        channel_name: str,
        validation: ValidationConfig,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        # If validation is disabled, still probe basic metadata but skip all checks/rejections.
        if not getattr(validation, "enabled", True):
            summary = self._probe_audio(raw_bytes)
            summary["size_mb"] = round(len(raw_bytes) / (1024 * 1024), 3)
            logger.debug(
                "[Validation] Skipped (disabled)",
                extra={"row_index": row_index, "channel": channel_name, "probe": summary},
            )
            return summary

        summary = self._probe_audio(raw_bytes)
        summary["size_mb"] = round(len(raw_bytes) / (1024 * 1024), 3)
        
        # Extract row identifier for logging
        row_identifier = metadata.get("Email address") or metadata.get("Email") or f"Row {row_index}"

        def _reject(reason: str) -> None:
            asyncio.create_task(
                self._handle_rejection(
                    row_index=row_index,
                    channel_name=channel_name,
                    reason=reason,
                    metadata={
                        **metadata,
                        "source_url": source_url,
                        "probe": summary,
                    },
                    validation=validation,
                )
            )

        if validation.allowed_formats:
            allowed_formats = {fmt.lower() for fmt in validation.allowed_formats}
            if summary.get("format") and summary["format"].lower() not in allowed_formats:
                reason = f"Invalid audio format - got '{summary.get('format')}', allowed: {list(allowed_formats)}"
                _reject(reason)
                return None

        duration = summary.get("duration_sec") or 0.0
        if validation.duration_min is not None:
            if duration < validation.duration_min:
                reason = f"Audio duration too short - {duration:.2f}s (min: {validation.duration_min}s)"
                _reject(reason)
                return None

        if validation.duration_max is not None:
            if duration > validation.duration_max:
                reason = f"Audio duration too long - {duration:.2f}s (max: {validation.duration_max}s)"
                _reject(reason)
                return None

        if validation.size_max_mb is not None:
            if summary["size_mb"] > validation.size_max_mb:
                reason = f"Audio file too large - {summary['size_mb']:.2f}MB (max: {validation.size_max_mb}MB)"
                _reject(reason)
                return None

        if validation.sample_rate_allowed:
            allowed_rates = set(validation.sample_rate_allowed)
            if summary.get("sample_rate") not in allowed_rates:
                reason = f"Invalid sample rate - {summary.get('sample_rate')}Hz (allowed: {sorted(allowed_rates)})"
                _reject(reason)
                return None

        logger.debug(
            "[Validation] Passed",
            extra={"row_index": row_index, "channel": channel_name, "summary": summary},
        )
        return summary

    async def _handle_rejection(
        self,
        *,
        row_index: int,
        channel_name: str,
        reason: str,
        metadata: Dict[str, Any],
        validation: ValidationConfig,
    ) -> None:
        row_identifier = metadata.get("Email address") or metadata.get("Email") or f"Row {row_index}"
        logger.warning(
            f"CSV Row {row_index} REJECTED ({row_identifier}): {reason}",
            extra={
                "row_index": row_index,
                "row_identifier": row_identifier,
                "channel": channel_name,
                "reason": reason,
            },
        )
        if not validation.rejection_webhook_url:
            return

        payload = {
            "row_index": row_index,
            "channel": channel_name,
            "reason": reason,
            "module": "preprocessing",
            "metadata": metadata,
        }

        headers = {"Content-Type": "application/json"}
        if validation.rejection_webhook_secret:
            headers["X-Webhook-Secret"] = validation.rejection_webhook_secret

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    validation.rejection_webhook_url,
                    json=payload,
                    headers=headers,
                )
        except httpx.ConnectError:
            logger.warning(
                "Rejection webhook unreachable",
                extra={"url": validation.rejection_webhook_url},
            )
        except httpx.HTTPError as exc:
            logger.warning("Rejection webhook returned error", extra={"error": str(exc)})
        except Exception as exc:
            logger.warning("Unexpected error calling rejection webhook", extra={"error": str(exc)})

    def _probe_audio(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Use ffprobe to read audio metadata."""
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout or "{}")
            format_info = data.get("format", {})
            streams = data.get("streams", [])
            first_stream = streams[0] if streams else {}
            format_name = (format_info.get("format_name") or "").split(",")[0]

            channels = int(first_stream.get("channels") or 1)
            return {
                "format": format_name.lower() if format_name else None,
                "duration_sec": float(format_info.get("duration") or 0.0),
                "sample_rate": int(first_stream.get("sample_rate") or 0),
                "channels": channels,
            }
        except subprocess.CalledProcessError as exc:
            logger.warning("ffprobe failed", extra={"error": exc.stderr})
            return {}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _select_metadata_fields(
        self,
        *,
        include_fields: List[str],
        size_bytes: int,
        validation_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        summary = validation_summary or {}
        field_map = {
            "duration": summary.get("duration_sec"),
            "sample_rate": summary.get("sample_rate"),
            "format": summary.get("format"),
            "size": round(size_bytes / (1024 * 1024), 3),
        }

        for field in include_fields:
            key = field.lower()
            if key in field_map and field_map[key] is not None:
                info[key] = field_map[key]

        return info

    def _format_metadata_lines(self, metadata_fields: Dict[str, Any]) -> str:
        """Convert metadata dict into human-readable string for UI Paragraphs."""
        label_map = {
            "duration": "Duration (s)",
            "size": "Size (MB)",
            "sample_rate": "Sample Rate (Hz)",
            "format": "Format",
            "snr": "SNR (dB)",
        }
        lines = []
        for key, value in metadata_fields.items():
            label = label_map.get(key, key.replace("_", " ").title())
            lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def _inject_sidebar_metadata(
        self,
        metadata: Dict[str, Any],
        metadata_fields: Dict[str, Any],
        *,
        row_index: int,
        channel: str,
    ) -> None:
        """Populate common sidebar fields expected by the LS template."""
        duration = metadata_fields.get("duration")
        metadata["duration_s"] = str(duration) if duration is not None else ""
        sample_rate = metadata_fields.get("sample_rate")
        metadata["sample_rate"] = str(sample_rate) if sample_rate is not None else ""
        size_mb = metadata_fields.get("size")
        metadata["size_mb"] = str(size_mb) if size_mb is not None else ""
        if "snr_db" not in metadata:
            snr = metadata_fields.get("snr")
            metadata["snr_db"] = str(snr) if snr is not None else ""
        metadata["chunk_id"] = f"{row_index}_{channel}"


    def _upload_ml_ready_payload(
        self,
        *,
        payload: Dict[str, Any],
        s3_prefix: str,
        row_index: int,
        channel: str,
    ) -> None:
        """Upload task metadata to ml-ready folder for traceability."""
        # Structure: {s3_prefix}/ml-ready/tasks/
        key = f"{s3_prefix}/ml-ready/tasks/row{row_index}_{channel}.json"
        self.s3.upload_bytes(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),  # type: ignore[name-defined]
            key,
            metadata={
                "row": str(row_index),
                "channel": channel,
            },
        )


