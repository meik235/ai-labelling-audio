"""Post-processing workflow: export from Label Studio and derive transcripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.audio_segmenter import AudioSegmenter
from backend.integrations import LabelStudioClient, S3Uploader
from backend.qc_sheet import QCSheetUpdater


logger = logging.getLogger(__name__)


# Language code to full name mapping
LANGUAGE_CODE_TO_NAME = {
    "en": "English",
    "en_us": "English (US)",
    "en_uk": "English (UK)",
    "en_au": "English (Australia)",
    "en_za": "English (South Africa)",
    "hi": "Hindi",
    "bn": "Bengali",
    "es": "Spanish",
    "es_419": "Spanish (Latin America)",
    "es_es": "Spanish (Spain)",
    "fr": "French",
    "fr_ca": "French (Canada)",
    "de": "German",
    "pt": "Portuguese",
    "pt_br": "Portuguese (Brazil)",
    "it": "Italian",
    "nl": "Dutch",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "zh_cn": "Chinese (Simplified)",
    "zh_tw": "Chinese (Traditional)",
    "pl": "Polish",
    "ru": "Russian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}


def _get_language_full_name(language_code: str) -> str:
    """Convert language code to full language name.
    
    Args:
        language_code: Language code (e.g., "bn", "en", "hi") or full name
    
    Returns:
        Full language name if mapping exists, otherwise returns the original value
    """
    if not language_code or not isinstance(language_code, str):
        return ""
    
    original = language_code.strip()
    language_code_lower = original.lower()
    
    # Check if it's already a full name (contains spaces or starts with uppercase)
    if " " in original or (original and original[0].isupper()):
        return original
    
    # Check exact match in mapping
    if language_code_lower in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[language_code_lower]
    
    # Return original as-is if no mapping found
    return original


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_timestamp(timestamp: Any) -> float:
    """Parse timestamp from various formats (seconds, HH:MM:SS, etc.) to seconds."""
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if isinstance(timestamp, str):
        # Try HH:MM:SS format
        if ":" in timestamp:
            parts = timestamp.split(":")
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
        # Try to parse as float
        try:
            return float(timestamp)
        except ValueError:
            return 0.0
    return 0.0


class AnnotationWorkflow:
    """Orchestrates the post-processing workflow around Label Studio export."""

    def __init__(
        self,
        label_studio_url: str,
        label_studio_api_key: str,
        s3_bucket: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        qc_sheet_path: Optional[str] = None,
        qc_updater_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Initialize workflow orchestrator.

        Args:
            label_studio_url: Label Studio server URL
            label_studio_api_key: Label Studio API token
            s3_bucket: S3 bucket name for final data
            s3_access_key: AWS access key ID
            s3_secret_key: AWS secret access key
            qc_sheet_path: Path to QC tracking CSV sheet (optional)
            qc_updater_callback: Custom callback function for updating QC metadata
                Signature: callback(task_id: str, metadata: Dict[str, Any]) -> None
        """
        self.label_studio = LabelStudioClient(label_studio_url, label_studio_api_key)

        self.s3_uploader = None
        if s3_bucket and s3_access_key and s3_secret_key:
            self.s3_uploader = S3Uploader(s3_bucket, s3_access_key, s3_secret_key)

        self.audio_segmenter = AudioSegmenter()

        self.qc_updater = None
        if qc_sheet_path:
            self.qc_updater = QCSheetUpdater(qc_sheet_path)

        self.qc_updater_callback = qc_updater_callback

    async def create_tasks(
        self,
        project_id: int,
        tasks: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Create tasks in Label Studio.

        Args:
            project_id: Label Studio project ID
            tasks: List of task data with audio URLs and metadata

        Returns:
            Created task data
        """
        created_tasks = await self.label_studio.create_tasks_batch(project_id, tasks)

        # Update QC sheet if available
        if self.qc_updater:
            for task in created_tasks:
                task_id = str(task.get("id", ""))
                self.qc_updater.update_task_status(task_id, "pending")

        return created_tasks

    async def export_and_save(
        self,
        project_id: int,
        output_dir: str | Path,
        s3_prefix: Optional[str] = None,
        extract_audio_segments: bool = True,
        output_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Export final data from Label Studio and store in S3.
        """
        if not self.s3_uploader:
            raise ValueError("S3 uploader not configured. Provide s3_bucket and credentials.")

        if not s3_prefix:
            raise ValueError("s3_prefix is required for storing final data.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export from Label Studio
        export_data = await self.label_studio.export_project(
            project_id, export_type="JSON", download_all=True
        )

        # Parse export data
        annotations = json.loads(export_data)

        logger.info("Label Studio export returned %s task(s)", len(annotations))

        if not annotations:
            logger.warning("No annotations found in Label Studio export. Complete QC before exporting.")
            return {
                "project_id": project_id,
                "task_count": 0,
                "transcript_urls": [],
                "segment_urls": [],
                "s3_prefix": s3_prefix,
                "total_segments": 0,
                "transcripts": [],
                "annotations": [],
            }

        # Process each annotation
        transcript_urls: List[str] = []
        transcript_records: List[Dict[str, Any]] = []
        segment_urls: List[Dict[str, Any]] = []
        skipped_count = 0
        processed_count = 0

        for annotation in annotations:
            # Skip annotations that already have "unknown" as task_id
            if annotation.get("task_id", "").strip().lower() == "unknown":
                logger.warning(
                    "Skipping annotation with pre-existing 'unknown' task_id",
                    extra={"annotation_keys": list(annotation.keys())},
                )
                skipped_count += 1
                continue
            
            # Try multiple ways to extract task_id (Label Studio export structure can vary)
            task_data = annotation.get("task", {})
            task_id = None
            
            # Try task.id first
            if isinstance(task_data, dict) and task_data:
                task_id = task_data.get("id")
            
            # Try annotation.id as fallback
            if not task_id:
                task_id = annotation.get("id")
            
            # Try task_data directly if it's a number
            if not task_id and isinstance(task_data, (int, str)):
                task_id = task_data
            
            # Last resort: try to find id anywhere in the annotation
            if not task_id:
                task_id = annotation.get("task_id") or (annotation.get("task", {}) or {}).get("id")
            
            # Check if task_id was extracted as the literal string "unknown" (should never happen)
            if task_id and str(task_id).strip().lower() == "unknown":
                logger.warning(
                    "Task ID extracted as literal 'unknown' string - this should not happen; skipping annotation",
                    extra={
                        "annotation_keys": list(annotation.keys()),
                        "task_data": str(task_data)[:200] if task_data else None,
                    },
                )
                skipped_count += 1
                continue
            
            if not task_id:
                logger.warning(
                    "Could not extract task_id from annotation; keys=%s task_data_type=%s",
                    list(annotation.keys()),
                    type(task_data),
                )
                # Skip this annotation if we can't get task_id
                skipped_count += 1
                continue
            
            # Ensure task_id is valid (not None, not empty, not "unknown")
            task_id = str(task_id).strip()
            if not task_id or task_id.lower() == "unknown" or task_id == "None":
                logger.warning(
                    "Invalid task_id extracted: %s; skipping annotation",
                    task_id,
                    extra={"annotation_keys": list(annotation.keys())},
                )
                skipped_count += 1
                continue
            
            # Extract audio_url and task data - try multiple locations
            # Label Studio export structure can vary:
            # 1. annotation.task.data.audio (full task object)
            # 2. annotation.data.audio (task data at top level)
            # 3. annotation.task is just an ID reference, need to fetch separately
            audio_url = ""
            full_task_data = None
            
            # Try task.data.audio first
            if isinstance(task_data, dict) and task_data:
                full_task_data = task_data
                audio_url = task_data.get("data", {}).get("audio", "")
            
            # Try annotation.data.audio (task data at top level)
            if not audio_url:
                top_level_data = annotation.get("data", {})
                if top_level_data:
                    audio_url = top_level_data.get("audio", "")
                    if top_level_data and not full_task_data:
                        # Use top-level data as task_data if task is empty
                        full_task_data = {"id": task_id, "data": top_level_data}
            
            # If task_data is empty but we have task_id, the task object might just be a reference
            # In that case, we'll use what we have from the annotation
            if not full_task_data and task_id:
                # Construct minimal task_data from what we have
                full_task_data = {
                    "id": int(task_id),
                    "data": annotation.get("data", {})
                }
            
            if not audio_url:
                logger.warning("Could not find audio_url for task %s", task_id)

            logger.info("Processing task %s", task_id)

            # Extract transcript from annotation
            annotation_result = annotation.get("annotations", [{}])
            if not annotation_result:
                # No annotations for this task yet
                logger.info("Task %s has no annotations. Skipping.", task_id)
                skipped_count += 1
                continue

            result_items = annotation_result[0].get("result", [])

            if not result_items:
                # Annotations exist but no transcript data yet
                logger.info("Task %s annotations lack transcript data. Skipping.", task_id)
                skipped_count += 1
                continue

            # Get original ASR segments from task data (if available) to preserve timestamps
            original_segments = []
            if full_task_data:
                # Try to get original ASR segments from task data
                task_data_obj = full_task_data.get("data", {})
                if isinstance(task_data_obj, dict):
                    # Check for segments in various locations
                    original_segments = (
                        task_data_obj.get("segments", []) or
                        task_data_obj.get("transcription_segments", []) or
                        []
                    )
                    
                    # Also check preprocessing metadata for original segments
                    if not original_segments:
                        preprocessing = task_data_obj.get("preprocessing", {})
                        if isinstance(preprocessing, dict):
                            transcription_meta = preprocessing.get("transcription", {})
                            if isinstance(transcription_meta, dict):
                                original_segments = transcription_meta.get("segments", []) or []
            
            # Also check annotation's task data directly
            if not original_segments and isinstance(task_data, dict):
                task_data_segments = task_data.get("data", {}).get("segments", [])
                if task_data_segments:
                    original_segments = task_data_segments
            
            if original_segments:
                logger.debug("Found %s original ASR segments for task %s", len(original_segments), task_id)
            else:
                logger.debug("No original ASR segments found for task %s - timestamps may default to 00:00:00", task_id)
            
            # Format transcript segments in the required format
            # Fields included are configurable via output_fields parameter
            # Default: [{start, end, speaker, text, language, end_of_speech}, ...]
            formatted_segments = []
            full_transcript = []
            
            # Default output fields if not specified
            if output_fields is None:
                output_fields = ["start", "end", "speaker", "text", "language", "end_of_speech"]
            
            # Fields that should be skipped (they're metadata, not segments)
            skip_fields = {"language", "end_of_speech"}
            
            # Build a map of original segments by text (for matching)
            original_segment_map = {}
            for orig_seg in original_segments:
                if isinstance(orig_seg, dict):
                    orig_text = orig_seg.get("text", "").strip()
                    if orig_text:
                        # Use text as key for matching (first occurrence)
                        if orig_text not in original_segment_map:
                            original_segment_map[orig_text] = orig_seg

            for item in result_items:
                # Skip perRegion metadata fields that are in skip_fields - they're not segments
                from_name = item.get("from_name", "")
                if from_name in skip_fields:
                    continue  # These are metadata fields, not segments
                
                value = item.get("value", {})
                
                # Extract text
                text = ""
                if isinstance(value, dict):
                    # Handle different text formats from Label Studio
                    if "text" in value:
                        text_data = value.get("text")
                        if isinstance(text_data, list):
                            # Join all text elements (e.g., main text + QC notes)
                            text = " ".join(str(t) for t in text_data if t and str(t).strip())
                        elif isinstance(text_data, str):
                            text = text_data
                        else:
                            text = str(text_data) if text_data else ""
                elif isinstance(value, str):
                    text = value
                
                if not text or not text.strip():
                    continue
                
                text = text.strip()
                
                # Extract timestamps from user annotation
                start_seconds = _parse_timestamp(value.get("start", 0) if isinstance(value, dict) else 0)
                end_seconds = _parse_timestamp(value.get("end", 0) if isinstance(value, dict) else 0)
                
                # If timestamps are missing or zero, try to get from original ASR segment
                matched_original = None
                if (start_seconds == 0 or end_seconds == 0) and original_segments:
                    # Try to match with original segment by text similarity or order
                    # First, try exact text match
                    if text in original_segment_map:
                        matched_original = original_segment_map[text]
                    else:
                        # Try to find by position (if segments are in same order)
                        # This is a fallback - not perfect but better than 00:00:00
                        if len(formatted_segments) < len(original_segments):
                            matched_original = original_segments[len(formatted_segments)]
                    
                    if matched_original and isinstance(matched_original, dict):
                        # Use original timestamps if user didn't provide them
                        if start_seconds == 0:
                            orig_start = matched_original.get("start", 0)
                            if isinstance(orig_start, (int, float)):
                                start_seconds = float(orig_start)
                            elif isinstance(orig_start, str):
                                start_seconds = _parse_timestamp(orig_start)
                        
                        if end_seconds == 0:
                            orig_end = matched_original.get("end", 0)
                            if isinstance(orig_end, (int, float)):
                                end_seconds = float(orig_end)
                            elif isinstance(orig_end, str):
                                end_seconds = _parse_timestamp(orig_end)
                
                # Convert to HH:MM:SS format
                start_time = _seconds_to_timestamp(start_seconds) if start_seconds > 0 else "00:00:00"
                end_time = _seconds_to_timestamp(end_seconds) if end_seconds > 0 else "00:00:00"
                
                # Extract speaker (default to "Speaker A" if not specified)
                speaker = value.get("speaker", "Speaker A") if isinstance(value, dict) else "Speaker A"
                if not speaker or speaker == "":
                    speaker = "Speaker A"
                
                # Extract language - try multiple sources in order:
                # 1. From annotation value (if annotator set it in Label Studio)
                # 2. From AssemblyAI transcript metadata (language_code)
                # 3. From task metadata/CSV (Language, language, lang fields)
                language = ""
                
                # Check result items for this specific region (matching start/end times) to find language
                # Label Studio stores perRegion fields as separate result items with same start/end
                for result_item in result_items:
                    item_value = result_item.get("value", {})
                    if isinstance(item_value, dict):
                        # Only check language field if it matches this segment's timing
                        item_start = _parse_timestamp(item_value.get("start", 0))
                        item_end = _parse_timestamp(item_value.get("end", 0))
                        if abs(item_start - start_seconds) < 0.1 and abs(item_end - end_seconds) < 0.1:
                            # Check if this is a language textarea result
                            if result_item.get("from_name") == "language":
                                text_data = item_value.get("text", [])
                                if isinstance(text_data, list) and text_data:
                                    language = text_data[0] if text_data else ""
                                elif isinstance(text_data, str):
                                    language = text_data
                                if language:
                                    break
                            # Also check if it's in the value dict directly (legacy format)
                            elif item_value.get("language"):
                                language = item_value.get("language", "")
                                if language:
                                    break
                
                # If language not in annotation, try to get from AssemblyAI transcript metadata
                if not language and full_task_data:
                    task_data_obj = full_task_data.get("data", {})
                    if isinstance(task_data_obj, dict):
                        # Check AssemblyAI transcript_metadata for language_code
                        transcript_metadata = task_data_obj.get("transcript_metadata", {})
                        if isinstance(transcript_metadata, dict):
                            # AssemblyAI returns language_code in transcript result
                            language = transcript_metadata.get("language_code", "")
                        
                        # If still not found, try CSV metadata fields
                        if not language:
                            language = (
                                task_data_obj.get("Language") or
                                task_data_obj.get("language") or
                                task_data_obj.get("lang") or
                                ""
                            )
                
                # Also check if language is in the original segment (some ASR providers include it per-segment)
                if not language and matched_original and isinstance(matched_original, dict):
                    language = matched_original.get("language", "") or matched_original.get("language_code", "")
                
                # Convert language code to full name if it's a code
                if language:
                    language = _get_language_full_name(language)
                
                # Extract end_of_speech - try multiple sources:
                # 1. From annotation value (if annotator set it in Label Studio)
                # 2. From AssemblyAI utterances (if this segment ends at an utterance boundary)
                # 3. Default to false
                end_of_speech = False
                
                # Check result items for this specific region (matching start/end times) to find end_of_speech
                # Label Studio stores perRegion fields as separate result items with same start/end
                for result_item in result_items:
                    item_value = result_item.get("value", {})
                    if isinstance(item_value, dict):
                        # Only check end_of_speech field if it matches this segment's timing
                        item_start = _parse_timestamp(item_value.get("start", 0))
                        item_end = _parse_timestamp(item_value.get("end", 0))
                        if abs(item_start - start_seconds) < 0.1 and abs(item_end - end_seconds) < 0.1:
                            # Check if this is an end_of_speech choices result
                            if result_item.get("from_name") == "end_of_speech":
                                choices = item_value.get("choices", [])
                                if choices and ("true" in choices or True in choices):
                                    end_of_speech = True
                                    break
                            # Also check if it's in the value dict directly (legacy format)
                            elif item_value.get("end_of_speech") is True:
                                end_of_speech = True
                                break
                
                # If not set in annotation, check if this segment aligns with an utterance end from AssemblyAI
                if not end_of_speech and full_task_data:
                    task_data_obj = full_task_data.get("data", {})
                    if isinstance(task_data_obj, dict):
                        transcript_metadata = task_data_obj.get("transcript_metadata", {})
                        if isinstance(transcript_metadata, dict):
                            # Check if AssemblyAI has utterances data
                            utterances = transcript_metadata.get("utterances", [])
                            if isinstance(utterances, list) and utterances:
                                # Check if this segment's end time matches an utterance end time
                                # Utterances have start/end in milliseconds, convert to seconds
                                for utterance in utterances:
                                    utt_end_ms = utterance.get("end", 0)
                                    utt_end_sec = utt_end_ms / 1000.0 if utt_end_ms else 0
                                    # Check if segment end is close to utterance end (within 0.1 seconds)
                                    if utt_end_sec > 0 and abs(end_seconds - utt_end_sec) < 0.1:
                                        end_of_speech = True
                                        break
                
                # Build segment dictionary dynamically based on output_fields configuration
                segment = {}
                
                # Always include core fields if they're in output_fields
                if "start" in output_fields:
                    segment["start"] = start_time
                if "end" in output_fields:
                    segment["end"] = end_time
                if "speaker" in output_fields:
                    segment["speaker"] = speaker
                if "text" in output_fields:
                    segment["text"] = text
                if "language" in output_fields:
                    segment["language"] = language
                if "end_of_speech" in output_fields:
                    segment["end_of_speech"] = end_of_speech
                
                # Extract any additional custom fields from Label Studio annotations
                for result_item in result_items:
                    item_value = result_item.get("value", {})
                    if isinstance(item_value, dict):
                        # Only check fields that match this segment's timing
                        item_start = _parse_timestamp(item_value.get("start", 0))
                        item_end = _parse_timestamp(item_value.get("end", 0))
                        if abs(item_start - start_seconds) < 0.1 and abs(item_end - end_seconds) < 0.1:
                            field_name = result_item.get("from_name", "")
                            # If this field is in output_fields and not already processed, extract it
                            if field_name in output_fields and field_name not in segment and field_name not in skip_fields:
                                # Handle different field types
                                if result_item.get("type") == "textarea":
                                    text_data = item_value.get("text", [])
                                    if isinstance(text_data, list) and text_data:
                                        # Join all text elements (e.g., main text + QC notes)
                                        segment[field_name] = " ".join(str(t) for t in text_data if t and str(t).strip())
                                    elif isinstance(text_data, str):
                                        segment[field_name] = text_data
                                elif result_item.get("type") == "choices":
                                    choices = item_value.get("choices", [])
                                    segment[field_name] = choices[0] if choices else None
                                elif result_item.get("type") == "number":
                                    segment[field_name] = item_value.get("number")
                                elif result_item.get("type") == "rating":
                                    segment[field_name] = item_value.get("rating")
                                else:
                                    # For other types, try to get value directly
                                    segment[field_name] = item_value.get("value", item_value)
                
                formatted_segments.append(segment)
                full_transcript.append(text)

            # Final safety check - ensure task_id is still valid before creating filename
            if not task_id or str(task_id).strip().lower() in ("unknown", "none", ""):
                logger.error(
                    "Task ID became invalid before upload: %s; skipping annotation",
                    task_id,
                    extra={"task_id_type": type(task_id).__name__},
                )
                skipped_count += 1
                continue
            
            # Final validation right before creating filename - should never be needed but extra safety
            task_id_final = str(task_id).strip()
            if not task_id_final or task_id_final.lower() in ("unknown", "none", ""):
                logger.error(
                    "CRITICAL: Task ID validation failed right before upload: '%s' (type: %s); skipping",
                    task_id_final,
                    type(task_id).__name__,
                    extra={"original_task_id": task_id, "annotation_keys": list(annotation.keys())[:10]},
                )
                skipped_count += 1
                continue
            
            # Store final corrected transcript in S3
            # Store unified annotation payload (raw LS data + derived transcript segments)
            transcript_data = formatted_segments
            combined_payload = {
                "task_id": task_id_final,  # Use validated task_id_final
                "audio_url": audio_url,
                "transcript_segments": transcript_data,
                "full_transcript": " ".join(full_transcript),
                "label_studio_annotation": annotation,
            }
            annotation_key = f"{s3_prefix}/annotations/task_{task_id_final}_annotation.json"
            logger.info(
                "Saving task %s unified annotation (%s segment(s)) to %s",
                task_id_final,
                len(formatted_segments),
                annotation_key,
            )
            annotation_s3_url = self.s3_uploader.upload_bytes(
                json.dumps(combined_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                annotation_key,
                metadata={"task_id": task_id, "type": "annotation"},
            )
            transcript_urls.append(annotation_s3_url)
            processed_count += 1
            logger.debug("Task %s annotation saved", task_id)
            # Extract metadata from task data if available
            metadata = {}
            if full_task_data:
                task_metadata = full_task_data.get("data", {}).get("metadata", {})
                if task_metadata and isinstance(task_metadata, dict):
                    metadata = task_metadata.copy()
                # Also check preprocessing metadata
                preprocessing_meta = full_task_data.get("data", {}).get("preprocessing", {})
                if preprocessing_meta and isinstance(preprocessing_meta, dict):
                    # Include preprocessing metadata fields
                    if "metadata" in preprocessing_meta:
                        prep_meta = preprocessing_meta["metadata"]
                        if isinstance(prep_meta, dict):
                            metadata.update(prep_meta)
                        elif isinstance(prep_meta, (list, tuple)):
                            # If metadata is a list, convert to dict if possible
                            for item in prep_meta:
                                if isinstance(item, dict):
                                    metadata.update(item)
                                elif isinstance(item, (list, tuple)) and len(item) == 2:
                                    # Handle key-value pairs
                                    metadata[item[0]] = item[1]
            
            transcript_records.append(
                {
                    "task_id": task_id,
                    "audio_url": audio_url,
                    "transcript_segments": transcript_data,
                    "full_transcript": " ".join(full_transcript),
                    "annotation_s3_url": annotation_s3_url,
                    "segment_count": len(formatted_segments),
                    "metadata": json.dumps(metadata) if metadata else "",  # Store as JSON string for CSV
                }
            )

            # Extract and store audio segments if enabled
            task_segment_count = 0
            if extract_audio_segments and audio_url and formatted_segments:
                audio_segments = []
                for seg in formatted_segments:
                    start_sec = _parse_timestamp(seg["start"])
                    end_sec = _parse_timestamp(seg["end"])
                    audio_segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": seg["text"],
                        "speaker": seg["speaker"],
                    })

                if audio_segments:
                    try:
                        segments_dir = output_dir / f"segments_{task_id}"
                        segments = await self.audio_segmenter.extract_segments(
                            audio_url,
                            audio_segments,
                            segments_dir,
                            segment_prefix=f"task_{task_id}",
                        )

                        for seg in segments:
                            segment_path = Path(seg["file_path"])
                            segment_key = f"{s3_prefix}/annotations/audio_segments/task_{task_id}/{seg['filename']}"
                            segment_s3_url = self.s3_uploader.upload_file(
                                segment_path,
                                segment_key,
                                metadata={
                                    "task_id": task_id,
                                    "segment_id": str(seg["segment_id"]),
                                    "start": str(seg["start"]),
                                    "end": str(seg["end"]),
                                    "type": "audio_segment",
                                },
                            )
                            segment_urls.append({
                                "segment_id": seg["segment_id"],
                                "s3_url": segment_s3_url,
                                "start": seg["start"],
                                "end": seg["end"],
                                "text": seg.get("text", ""),
                            })
                        task_segment_count = len(segments)
                    except Exception as exc:
                        logger.warning("Failed to extract segments for task %s: %s", task_id, exc)

            metadata = {
                "task_id": task_id,
                "transcript_s3_url": annotation_s3_url,
                "segment_count": task_segment_count,
                "status": "completed",
            }

            if self.qc_updater:
                self.qc_updater.update_task_status(
                    task_id,
                    "completed",
                    transcript_path=annotation_s3_url,
                    qc_notes="Exported and stored in S3",
                )

            if self.qc_updater_callback:
                self.qc_updater_callback(task_id, metadata)

        logger.info(
            "Export summary: processed=%s skipped=%s destination=%s/annotations/",
            processed_count,
            skipped_count,
            s3_prefix,
        )

        # Ensure all annotations have task_id at top level for upload_deliverables
        # Extract task_id from raw Label Studio annotations if missing
        # Skip annotations where task_id cannot be extracted to avoid "unknown" files
        annotations_with_task_id = []
        for annotation in annotations:
            annotation_copy = annotation.copy()
            task_id = None
            
            # Check if task_id already exists
            if "task_id" in annotation_copy:
                task_id = annotation_copy.get("task_id")
            else:
                # Try to extract task_id using same logic as processing loop
                task_data = annotation_copy.get("task", {})
                if isinstance(task_data, dict) and task_data:
                    task_id = task_data.get("id")
                if not task_id:
                    task_id = annotation_copy.get("id")
                if not task_id and isinstance(task_data, (int, str)):
                    task_id = task_data
                if not task_id:
                    task_id = annotation_copy.get("task_id") or (annotation_copy.get("task", {}) or {}).get("id")
            
            # Only add annotation if we successfully extracted a valid task_id
            task_id_str = str(task_id).strip() if task_id else None
            if task_id_str and task_id_str.lower() != "unknown" and task_id_str != "None" and task_id_str:
                annotation_copy["task_id"] = task_id_str
                annotations_with_task_id.append(annotation_copy)
            else:
                # Log warning for annotations that couldn't be processed
                logger.warning(
                    "Skipping annotation - invalid or missing task_id: %s",
                    task_id_str,
                    extra={"annotation_keys": list(annotation_copy.keys())},
                )

        return {
            "project_id": project_id,
            "task_count": len(annotations),
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "transcript_urls": transcript_urls,
            "segment_urls": segment_urls,
            "s3_prefix": s3_prefix,
            "total_segments": len(segment_urls),
            "transcripts": transcript_records,
            "annotations": annotations_with_task_id,  # Use annotations with task_id extracted
        }


