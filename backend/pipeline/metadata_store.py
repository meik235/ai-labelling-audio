"""S3-backed project metadata store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.integrations import S3Uploader


@dataclass
class ProjectMetadata:
    """Lightweight manifest for orchestrated LS projects."""

    project_id: int
    project_title: str
    csv_filename: str
    preprocessing_s3_prefix: str
    export_s3_prefix: Optional[str]
    export_output_dir: str
    postprocessing: Optional[Dict[str, Any]]
    ml_backend_url: Optional[str]
    webhook_url: Optional[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata_key: Optional[str] = None


class ProjectMetadataStore:
    """Persist/retrieve pipeline metadata to S3 for automation hooks."""

    def __init__(
        self,
        uploader: S3Uploader,
        *,
        base_prefix: str = "metadata/projects",
    ) -> None:
        self.uploader = uploader
        self.base_prefix = base_prefix.strip("/")

    def _key(self, project_id: int) -> str:
        return f"{self.base_prefix}/{project_id}.json"

    def save(self, metadata: ProjectMetadata) -> str:
        key = metadata.metadata_key or self._key(metadata.project_id)
        metadata.metadata_key = key
        payload = json.dumps(asdict(metadata), ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        self.uploader.upload_bytes(
            payload, key, metadata={"project_id": str(metadata.project_id)}
        )
        return key

    def load(self, project_id: int) -> ProjectMetadata:
        key = self._key(project_id)
        payload = self.uploader.download_bytes(key)
        data = json.loads(payload.decode("utf-8"))
        
        # Filter out old/unknown fields that don't exist in current ProjectMetadata schema
        # This handles schema migrations gracefully by only keeping valid fields
        valid_fields = {
            "project_id", "project_title", "csv_filename",
            "preprocessing_s3_prefix", "export_s3_prefix", "export_output_dir",
            "postprocessing", "ml_backend_url", "webhook_url",
            "created_at", "metadata_key"
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        metadata = ProjectMetadata(**filtered_data)
        metadata.metadata_key = key
        return metadata

