"""Post-processing pipeline: merge labels, validate, export, and upload."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.integrations import S3Uploader


logger = logging.getLogger(__name__)


class PostProcessingPipeline:
    """Clean transcripts, validate, and package final outputs."""

    def __init__(self, *, s3_uploader: S3Uploader) -> None:
        self.s3 = s3_uploader

    def cleanup_transcript(self, text: str) -> str:
        import re
        import unicodedata

        text = re.sub(r"([.!?])\\1+", r"\\1", text)
        text = re.sub(r"\\s+([,.!?;:])", r"\\1", text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def build_final_csv(
        self,
        entries: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        if not entries:
            # Create empty CSV with headers if no entries (annotations not ready yet)
            logger.warning("No annotations found. Creating empty CSV with headers.")
            # Use default headers based on transcript structure
            fieldnames = [
                "task_id",
                "audio_url",
                "transcript_s3_url",
                "segment_count",
                "status",
            ]
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return output_path

        fieldnames = sorted({key for entry in entries for key in entry.keys()})
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
        return output_path

    def build_additional_outputs(
        self,
        *,
        transcripts: List[Dict[str, Any]],
        output_dir: Path,
        formats: List[str],
    ) -> Dict[str, Path]:
        """Build optional TXT/SRT outputs from transcript records."""
        outputs: Dict[str, Path] = {}
        normalized_formats = {fmt.lower() for fmt in formats}

        # Single merged TXT file with all transcripts
        if "txt" in normalized_formats and transcripts:
            txt_path = output_dir / "transcripts.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for entry in transcripts:
                    task_id = entry.get("task_id", "unknown")
                    full_text = entry.get("full_transcript") or ""
                    if not full_text:
                        # Fallback: join segment texts if available
                        segments = entry.get("transcript_segments") or []
                        full_text = " ".join(
                            seg.get("text", "") for seg in segments if seg.get("text")
                        )
                    if not full_text:
                        continue
                    f.write(f"[task {task_id}]\n{full_text}\n\n")
            outputs["transcripts.txt"] = txt_path

        # Per-task SRT files built from transcript segments
        if "srt" in normalized_formats and transcripts:
            for entry in transcripts:
                segments = entry.get("transcript_segments") or []
                if not segments:
                    continue
                task_id = entry.get("task_id", "unknown")
                srt_path = output_dir / f"task_{task_id}.srt"
                with open(srt_path, "w", encoding="utf-8") as f:
                    idx = 1
                    for seg in segments:
                        text = seg.get("text")
                        if not text:
                            continue
                        start = str(seg.get("start", "00:00:00"))
                        end = str(seg.get("end", "00:00:00"))
                        # Convert HH:MM:SS → HH:MM:SS,000
                        if "," not in start:
                            start = f"{start},000"
                        if "," not in end:
                            end = f"{end},000"
                        f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
                        idx += 1
                outputs.setdefault("srt_files", srt_path.parent)

        return outputs

    def validate_schema(
        self,
        *,
        items: List[Dict[str, Any]],
        schema_id: str,
    ) -> None:
        """
        Optionally validate final outputs against a JSON schema.

        Convention:
          - Schemas live under config/schemas/{schema_id}.json
          - If schema file or jsonschema lib is missing, we log and skip (no hard failure)
        """
        if not items:
            return

        schema_dir = Path("config") / "schemas"
        schema_path = schema_dir / f"{schema_id}.json"

        if not schema_path.exists():
            logger.info(
                "Schema validation skipped: schema file not found at %s (id=%s)",
                schema_path,
                schema_id,
            )
            return

        try:
            import jsonschema  # type: ignore[import]
        except ImportError:
            logger.info(
                "Schema validation skipped: 'jsonschema' package not installed. "
                "Add it to requirements.txt to enable JSON schema checks."
            )
            return

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Schema validation skipped: failed to load %s: %s", schema_path, exc)
            return

        for idx, item in enumerate(items, start=1):
            try:
                jsonschema.validate(instance=item, schema=schema)
            except Exception as exc:
                # We don't fail the pipeline; just log the error for now.
                logger.warning(
                    "Schema validation error for item #%s (schema_id=%s): %s",
                    idx,
                    schema_id,
                    exc,
                )

    def upload_deliverables(
        self,
        *,
        s3_prefix: str,
        transcripts: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        final_csv: Optional[Path],
        extra_files: Optional[Dict[str, Path]] = None,
        bucket_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        transcript_urls = []
        annotation_urls = []
        extra_urls: Dict[str, Any] = {}

        # Structure: {csv_name}/annotations/

        for transcript in transcripts:
            task_id = transcript.get("task_id", "unknown")
            payload = json.dumps(transcript, ensure_ascii=False, indent=2)
            key = f"{s3_prefix}/annotations/task_{task_id}_final.json"
            if bucket_override:
                url = self.s3.upload_bytes_to_bucket(
                    bucket_override,
                    payload.encode("utf-8"),
                    key,
                    metadata={"task_id": task_id},
                )
            else:
                url = self.s3.upload_bytes(
                    payload.encode("utf-8"),
                    key,
                    metadata={"task_id": task_id},
                )
            transcript_urls.append(url)

        # Skip uploading raw annotations - they're already uploaded by workflow.py
        # as processed annotation files (task_{task_id}_annotation.json) which include:
        # - processed transcript_segments
        # - full_transcript
        # - raw label_studio_annotation (nested inside)
        # Uploading raw annotations here would overwrite the better processed files
        # annotation_urls will remain empty - processed annotations are already in S3

        csv_url: Optional[str] = None
        if final_csv is not None:
            csv_key = f"{s3_prefix}/annotations/final_report.csv"
            if bucket_override:
                csv_url = self.s3.upload_file_to_bucket(bucket_override, final_csv, csv_key)
            else:
                csv_url = self.s3.upload_file(final_csv, csv_key)

        if extra_files:
            for name, path in extra_files.items():
                # If "srt_files" pseudo-key, upload all SRTs in that directory
                if name == "srt_files" and path.is_dir():
                    urls = []
                    for srt_file in path.glob("*.srt"):
                        key = f"{s3_prefix}/annotations/srt/{srt_file.name}"
                        if bucket_override:
                            url = self.s3.upload_file_to_bucket(bucket_override, srt_file, key)
                        else:
                            url = self.s3.upload_file(srt_file, key)
                        urls.append(url)
                    extra_urls["srt"] = urls
                    continue

                key = f"{s3_prefix}/annotations/{name}"
                if bucket_override:
                    url = self.s3.upload_file_to_bucket(bucket_override, path, key)
                else:
                    url = self.s3.upload_file(path, key)
                extra_urls[name] = url

        return {
            "transcripts": transcript_urls,
            "annotations": annotation_urls,
            "final_csv": csv_url,
            "extra": extra_urls,
        }


