"""Label Studio API client for task creation and export."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from httpx import HTTPStatusError


logger = logging.getLogger(__name__)


class LabelStudioClient:
    """Client for interacting with Label Studio API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        verify_ssl: bool = True,
    ) -> None:
        """
        Initialize Label Studio client.

        Args:
            base_url: Label Studio server URL (e.g., "http://localhost:8080")
            api_key: Label Studio API token
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (set False for HTTP/localhost)
        """
        if not api_key or not api_key.strip():
            raise ValueError(
                "Label Studio API key is empty. "
                "Set LABEL_STUDIO_API_KEY in your .env file. "
                "Get your token from Label Studio UI → Account & Settings → Access Token"
            )
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self.verify_ssl = verify_ssl
        
        self.is_pat_token = self.api_key.startswith("eyJ")
        self._access_token: Optional[str] = None
        
        self._headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.base_url.startswith("http://"):
            self.verify_ssl = False


    async def _get_access_token(self, force_refresh: bool = False) -> str:
        """
        Get access token, exchanging PAT if necessary.
        
        Args:
            force_refresh: Force refresh even if token is cached
        
        Returns:
            Access token to use for API calls
        """
        if not self.is_pat_token:
            return self.api_key
        
        if self._access_token and not force_refresh:
            return self._access_token
        url = f"{self.base_url}/api/token/refresh"
        payload = {"refresh": self.api_key}
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 401:
                    raise RuntimeError(
                        f"Failed to exchange PAT token for access token. "
                        f"The PAT token may be invalid or expired.\n"
                        f"To fix this:\n"
                        f"  1. Open Label Studio UI at {self.base_url}\n"
                        f"  2. Go to Account & Settings → Access Token\n"
                        f"  3. Create a new Personal Access Token or use a Legacy Token"
                    )
                response.raise_for_status()
                data = response.json()
                self._access_token = data.get("access")
                if not self._access_token:
                    raise RuntimeError("Label Studio did not return an access token")
                return self._access_token
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Failed to connect to Label Studio at {self.base_url}. "
                f"Is Label Studio running? Error: {e}"
            ) from e

    async def _update_headers(self) -> None:
        """Update headers with current access token."""
        token = await self._get_access_token()
        auth_type = "Bearer" if self.is_pat_token else "Token"
        self._headers["Authorization"] = f"{auth_type} {token}"

    async def verify_connection(self) -> Dict[str, Any]:
        """
        Verify Label Studio connection and API token validity.
        
        Returns:
            User info if token is valid (or empty dict if endpoint not available)
            
        Raises:
            ConnectionError: If cannot connect to Label Studio
            RuntimeError: If token is invalid
        """
        # Update headers with correct token (exchange PAT if needed)
        await self._update_headers()
        
        endpoints_to_try = [
            "/api/users/me",
            "/api/current-user",
            "/api/user",
            "/api/projects/",
        ]
        
        for endpoint in endpoints_to_try:
            url = f"{self.base_url}{endpoint}"
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                    response = await client.get(url, headers=self._headers)
                    
                    if response.status_code == 401:
                        raise RuntimeError(
                            f"Invalid Label Studio API token.\n"
                            f"To get your API token:\n"
                            f"  1. Open Label Studio UI at {self.base_url}\n"
                            f"  2. Go to Account & Settings → Access Token\n"
                            f"  3. Copy the token and set it as LABEL_STUDIO_API_KEY in your .env file\n"
                            f"Current token (first 10 chars): {self.api_key[:10]}..."
                        )
                    
                    if response.status_code in (200, 403):
                        try:
                            return response.json()
                        except:
                            return {"status": "connected"}
                    
                    if response.status_code == 404:
                        continue
                    
                    response.raise_for_status()
                    
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Failed to connect to Label Studio at {self.base_url}. "
                    f"Is Label Studio running? Error: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise RuntimeError(
                        f"Invalid Label Studio API token.\n"
                        f"To get your API token:\n"
                        f"  1. Open Label Studio UI at {self.base_url}\n"
                        f"  2. Go to Account & Settings → Access Token\n"
                        f"  3. Copy the token and set it as LABEL_STUDIO_API_KEY in your .env file\n"
                        f"Current token (first 10 chars): {self.api_key[:10]}..."
                    ) from e
                if e.response.status_code == 404:
                    continue
                if endpoint == endpoints_to_try[-1]:
                    raise RuntimeError(
                        f"Label Studio API error: {e.response.status_code} - {e.response.text}"
                    ) from e
        
        return {"status": "connected", "note": "User endpoint not available"}


    async def create_project(
        self,
        *,
        title: str,
        description: Optional[str] = None,
        label_config: str,
        **extra_fields: Any,
    ) -> Dict[str, Any]:
        """Create a Label Studio project."""
        # Update headers with correct token (exchange PAT if needed)
        await self._update_headers()
        
        payload: Dict[str, Any] = {
            "title": title,
            "label_config": label_config,
        }
        if description:
            payload["description"] = description
        payload.update(extra_fields)

        url = f"{self.base_url}/api/projects/"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                response = await client.post(url, json=payload, headers=self._headers)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Failed to connect to Label Studio at {self.base_url}. "
                f"Is Label Studio running? Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    f"Label Studio authentication failed (401). Invalid API token.\n"
                    f"To get your API token:\n"
                    f"  1. Open Label Studio UI at {self.base_url}\n"
                    f"  2. Go to Account & Settings → Access Token\n"
                    f"  3. Copy the token and set it as LABEL_STUDIO_API_KEY in your .env file\n"
                    f"Current token (first 10 chars): {self.api_key[:10]}..."
                ) from e
            raise RuntimeError(
                f"Label Studio API error: {e.response.status_code} - {e.response.text}"
            ) from e

    async def get_project(self, project_id: int) -> Dict[str, Any]:
        """Get a project by ID."""
        await self._update_headers()
        
        url = f"{self.base_url}/api/projects/{project_id}/"
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(url, headers=self._headers)
            response.raise_for_status()
            return response.json()

    async def update_project(
        self,
        project_id: int,
        *,
        label_config: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update project metadata such as label_config/title/description."""
        await self._update_headers()

        payload: Dict[str, Any] = {}
        if label_config is not None:
            payload["label_config"] = label_config
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description

        if not payload:
            raise ValueError("update_project requires at least one field to change")

        url = f"{self.base_url}/api/projects/{project_id}/"
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.patch(url, json=payload, headers=self._headers)
            try:
                response.raise_for_status()
            except HTTPStatusError as exc:
                message = exc.response.text
                try:
                    message = json.dumps(exc.response.json(), indent=2)
                except Exception:
                    pass
                raise HTTPStatusError(
                    f"Label Studio project update failed ({exc.response.status_code}): {message}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            return response.json()

    async def find_project_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Find a project by exact title match."""
        await self._update_headers()
        
        url = f"{self.base_url}/api/projects/"
        params = {"page_size": 1000}  # Get all projects
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(url, headers=self._headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Search through results
            projects = data.get("results", [])
            for project in projects:
                if project.get("title") == title:
                    return project
            
            return None

    def _normalize_task_payload(
        self,
        *,
        audio_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        raw_task: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize a task payload into Label Studio's expected structure.

        LS wants:
            {
              "data": {...task fields...},
              "annotations": [...],   # optional
              "predictions": [...],   # optional
            }
        """
        if raw_task is not None:
            task_body = dict(raw_task)
        else:
            task_body = {}

        if metadata:
            task_body.update(metadata)

        if audio_url:
            task_body["audio"] = audio_url
        elif not task_body.get("audio"):
            raise ValueError("Task payload missing required 'audio' field")

        reserved = {"annotations", "predictions", "data"}
        data_payload = {k: v for k, v in task_body.items() if k not in reserved}

        def clean_value(v: Any) -> Any:
            """Recursively clean values to ensure JSON serializability."""
            if v is None:
                return None  # Will be filtered out at top level only
            elif isinstance(v, (str, int, float, bool)):
                return v
            elif isinstance(v, dict):
                cleaned = {}
                for k, val in v.items():
                    cleaned_val = clean_value(val)
                    if cleaned_val is not None:
                        cleaned[k] = cleaned_val
                return cleaned
            elif isinstance(v, list):
                cleaned = [clean_value(item) for item in v if clean_value(item) is not None]
                return cleaned
            else:
                try:
                    json.dumps(v)
                    return v
                except (TypeError, ValueError):
                    return str(v)
        
        cleaned_data: Dict[str, Any] = {}
        for key, value in data_payload.items():
            cleaned_val = clean_value(value)
            if cleaned_val is not None:
                cleaned_data[key] = cleaned_val

        payload: Dict[str, Any] = {"data": cleaned_data}

        annotations_payload = annotations or task_body.get("annotations")
        predictions_payload = predictions or task_body.get("predictions")

        if annotations_payload and isinstance(annotations_payload, list) and len(annotations_payload) > 0:
            payload["annotations"] = annotations_payload
        if predictions_payload and isinstance(predictions_payload, list) and len(predictions_payload) > 0:
            payload["predictions"] = predictions_payload

        return payload

    async def create_task(
        self,
        project_id: int,
        audio_url: str,
        metadata: Optional[Dict[str, Any]] = None,
        pre_annotation: Optional[List[Dict[str, Any]]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new task in Label Studio project.

        Args:
            project_id: Label Studio project ID
            audio_url: URL to audio file
            metadata: Additional metadata to include in task data
            pre_annotation: Optional pre-annotation to include (list of annotation dicts)

        Returns:
            Created task data
        """
        await self._update_headers()
        
        payload = self._normalize_task_payload(
            audio_url=audio_url,
            metadata=metadata,
            annotations=pre_annotation,
            predictions=predictions,
        )

        url = f"{self.base_url}/api/tasks"
        params = {"project": project_id}

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    async def update_task(
        self,
        task_id: int,
        audio_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing task in Label Studio.

        Args:
            task_id: Label Studio task ID to update
            audio_url: Optional new audio URL (if updating audio)
            metadata: Optional metadata to update
            predictions: Optional predictions to update

        Returns:
            Updated task data
        """
        await self._update_headers()
        
        # Fetch existing task first to preserve existing data
        existing_task = await self.get_task(task_id)
        existing_data = existing_task.get("data", {})
        
        # Merge with new data
        if metadata:
            existing_data.update(metadata)
        if audio_url:
            existing_data["audio"] = audio_url
        
        payload = self._normalize_task_payload(
            audio_url=existing_data.get("audio"),
            metadata=existing_data,
            predictions=predictions,
        )
        
        # For updates, we only send data and predictions, not annotations
        # Annotations are managed separately in Label Studio
        update_payload = {
            "data": payload.get("data", {}),
        }
        if predictions:
            update_payload["predictions"] = predictions

        url = f"{self.base_url}/api/tasks/{task_id}"
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.patch(
                url,
                json=update_payload,
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    async def get_task(self, task_id: int) -> Dict[str, Any]:
        """
        Get a specific task by ID.

        Args:
            task_id: Label Studio task ID

        Returns:
            Task data
        """
        await self._update_headers()
        
        url = f"{self.base_url}/api/tasks/{task_id}"
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(
                url,
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    async def create_tasks_batch(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Create multiple tasks in batch.

        Args:
            project_id: Label Studio project ID
            tasks: List of task data dicts, each with task fields

        Returns:
            List of created task data
        """
        await self._update_headers()
        
        payload = [
            self._normalize_task_payload(raw_task=task)
            for task in tasks
        ]

        debug_enabled = os.getenv("LABELSTUDIO_DEBUG_PAYLOAD")
        if debug_enabled:
            try:
                Path(".tmp").mkdir(exist_ok=True)
                Path(".tmp/ls_bulk_payload.json").write_text(
                    json.dumps(payload[:1], indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("Failed to persist payload sample", exc_info=exc)

        url = f"{self.base_url}/api/tasks"
        params = {"project": project_id}

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    async def bulk_upload_tasks(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Upload tasks via /tasks/bulk endpoint.
        
        Args:
            project_id: Label Studio project ID
            tasks: List of task data dicts. Each task should be a dict with task data.
                  If tasks already have "data" key, they're used as-is.
                  Otherwise, each task is wrapped in {"data": task}.
        """
        await self._update_headers()
        
        url = f"{self.base_url}/api/projects/{project_id}/tasks/bulk/"
        
        payload = [
            self._normalize_task_payload(raw_task=task)
            for task in tasks
        ]

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._headers,
            )
            if response.status_code == 400:
                try:
                    error_json = response.json()
                    error_detail = json.dumps(error_json, indent=2)
                    if "validation_errors" in error_json:
                        validation_errors = error_json.get("validation_errors", {})
                        logger.error(
                            "[LabelStudio] Validation errors",
                            extra={"errors": validation_errors},
                        )
                        error_detail = (
                            f"Validation errors: {json.dumps(validation_errors, indent=2)}\n\nFull response: {error_detail}"
                        )
                except Exception:
                    error_detail = response.text
                
                try:
                    Path(".tmp").mkdir(exist_ok=True)
                    Path(".tmp/ls_bulk_payload.json").write_text(
                        json.dumps(payload[:1] if payload else [], indent=2),
                        encoding="utf-8",
                    )
                    logger.debug("Payload sample saved to .tmp/ls_bulk_payload.json")
                except Exception as exc:
                    logger.debug("Failed to persist payload sample", exc_info=exc)
                
                logger.error("[LabelStudio] Bulk upload error: %s", error_detail)
                
                raise RuntimeError(
                    f"Label Studio bulk upload failed (400 Bad Request).\n"
                    f"Error: {error_detail[:2000]}\n"
                    f"See .tmp/ls_bulk_payload.json for payload sample"
                )
            response.raise_for_status()
            return response.json()

    async def export_project(
        self,
        project_id: int,
        export_type: str = "JSON",
        download_all: bool = True,
    ) -> bytes:
        """
        Export project annotations from Label Studio.

        Args:
            project_id: Label Studio project ID
            export_type: Export format (JSON, JSON_MIN, CSV, TSV, CONLL2003, COCO, etc.)
            download_all: Include all tasks or only completed ones

        Returns:
            Exported data as bytes
        """
        await self._update_headers()
        
        url = f"{self.base_url}/api/projects/{project_id}/export"
        params = {
            "exportType": export_type,
            "download_all": str(download_all).lower(),
        }

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(
                url,
                headers=self._headers,
                params=params,
            )
            response.raise_for_status()
            return response.content

    async def get_project_tasks(
        self,
        project_id: int,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Get tasks from a project.

        Args:
            project_id: Label Studio project ID
            page: Page number
            page_size: Number of tasks per page

        Returns:
            Tasks data with pagination info
        """
        await self._update_headers()
        
        url = f"{self.base_url}/api/tasks"
        params = {
            "project": project_id,
            "page": page,
            "page_size": page_size,
        }

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(
                url,
                headers=self._headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    async def get_task_annotations(
        self,
        task_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Get annotations for a specific task.

        Args:
            task_id: Label Studio task ID

        Returns:
            List of annotations
        """
        await self._update_headers()
        
        url = f"{self.base_url}/api/tasks/{task_id}/annotations"

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(
                url,
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()


    async def register_ml_backend(
        self,
        *,
        title: str,
        url: str,
        description: Optional[str] = None,
        is_interactive: bool = False,
    ) -> Dict[str, Any]:
        """Register ML backend in Label Studio."""
        await self._update_headers()
        
        payload: Dict[str, Any] = {
            "title": title,
            "url": url,
            "is_interactive": is_interactive,
        }
        if description:
            payload["description"] = description

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.post(
                f"{self.base_url}/api/ml",
                json=payload,
                headers=self._headers,
            )
            if response.status_code == 400:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", error_json.get("error", error_detail))
                except:
                    pass
                raise RuntimeError(
                    f"Label Studio ML backend registration failed (400 Bad Request).\n"
                    f"Error: {error_detail}\n"
                    f"Payload: {json.dumps(payload, indent=2)}"
                )
            response.raise_for_status()
            return response.json()

    async def attach_ml_backend_to_project(
        self,
        project_id: int,
        ml_backend_id: int,
    ) -> Dict[str, Any]:
        """Attach an ML backend to a project."""
        await self._update_headers()

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/projects/{project_id}/ml/",
                    json={"model": ml_backend_id},
                    headers=self._headers,
                )
                response.raise_for_status()
                return response.json() if response.content else {"status": "attached"}
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (404, 405):
                    raise

            payload = {"ml_backends": [ml_backend_id]}
            response = await client.patch(
                f"{self.base_url}/api/projects/{project_id}/",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    async def list_ml_backends(self) -> List[Dict[str, Any]]:
        """List registered ML backends."""
        await self._update_headers()

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(
                f"{self.base_url}/api/ml",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    async def configure_cloud_storage(
        self,
        project_id: int,
        *,
        storage_type: str = "s3",
        bucket: str,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        use_blob_urls: bool = True,
        title: Optional[str] = None,
        is_source: bool = True,
        is_target: bool = False,
    ) -> Dict[str, Any]:
        """
        Configure cloud storage connection for a Label Studio project.
        
        Args:
            project_id: Label Studio project ID
            storage_type: Storage type ('s3', 'gcs', 'azure')
            bucket: S3 bucket name (or GCS bucket / Azure container)
            prefix: S3 prefix/folder path within bucket (optional)
            region: AWS region (optional, defaults to bucket region)
            aws_access_key_id: AWS access key ID (optional, uses IAM role if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            use_blob_urls: Whether to use direct blob URLs for tasks
            title: Storage connection title (optional)
            is_source: Whether this is source storage (for importing tasks) - default True
            is_target: Whether this is target storage (for exporting annotations) - default False
            
        Returns:
            Storage configuration response
        """
        await self._update_headers()
        
        payload: Dict[str, Any] = {
            "project": project_id,
            "storage_type": storage_type.upper(),
            "bucket": bucket,
            "use_blob_urls": use_blob_urls,
        }
        
        if is_source or is_target:
            payload["can_sync"] = is_source
            payload["can_save"] = is_target
        else:
            payload["can_sync"] = True
            payload["can_save"] = False
        
        if prefix:
            payload["prefix"] = prefix.rstrip("/")
        if region:
            payload["region"] = region
        if aws_access_key_id:
            payload["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            payload["aws_secret_access_key"] = aws_secret_access_key
        if title:
            payload["title"] = title
        else:
            title_parts = [bucket]
            if prefix:
                title_parts.append(prefix)
            storage_role = "Source" if is_source and not is_target else ("Target" if is_target and not is_source else "Source+Target")
            payload["title"] = f"{storage_role} {storage_type.upper()}: {'/'.join(title_parts)}"
        
        endpoints_to_try = [
            f"{self.base_url}/api/storages/{storage_type.lower()}/",
            f"{self.base_url}/api/storages/{storage_type.lower()}",
            f"{self.base_url}/api/storages/",
        ]
        
        response = None
        last_error = None
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            for url in endpoints_to_try:
                try:
                    logger.debug(
                        "[LabelStudio][Storage] Trying endpoint",
                        extra={
                            "url": url,
                            "payload": {
                                k: v
                                for k, v in payload.items()
                                if k not in ["aws_access_key_id", "aws_secret_access_key"]
                            },
                        },
                    )
                    response = await client.post(url, json=payload, headers=self._headers)
                    logger.debug(
                        "[LabelStudio][Storage] Response",
                        extra={"url": url, "status": response.status_code},
                    )
                    if response.status_code == 200 or response.status_code == 201:
                        logger.info("[LabelStudio][Storage] Storage created", extra={"url": url})
                        return response.json()
                    if response.status_code == 401:
                        logger.error("[LabelStudio][Storage] Authentication failed (401)")
                        error_text = response.text[:500]
                        raise RuntimeError(
                            f"Label Studio authentication failed (401). Invalid API token.\n"
                            f"Response: {error_text}\n"
                            f"Check your LABEL_STUDIO_API_KEY in .env file"
                        )
                    if response.status_code == 400:
                        error_detail = response.text
                        try:
                            error_json = response.json()
                            error_detail = error_json.get("detail", error_json.get("error", error_detail))
                        except:
                            pass
                        logger.error(
                            "[LabelStudio][Storage] Bad Request",
                            extra={"error": error_detail[:300]},
                        )
                        raise RuntimeError(
                            f"Label Studio storage configuration failed (400 Bad Request).\n"
                            f"Error: {error_detail}\n"
                            f"Payload: {json.dumps({k: v for k, v in payload.items() if k not in ['aws_access_key_id', 'aws_secret_access_key']}, indent=2)}"
                        )
                    if response.status_code != 404:
                        logger.warning(
                            "[LabelStudio][Storage] Unexpected status",
                            extra={
                                "status": response.status_code,
                                "response": response.text[:300],
                                "url": url,
                            },
                        )
                    logger.debug(
                        "[LabelStudio][Storage] Endpoint fallback",
                        extra={"status": response.status_code, "url": url},
                    )
                except RuntimeError:
                    raise
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 401:
                        error_text = e.response.text[:500]
                        raise RuntimeError(
                            f"Label Studio authentication failed (401). Invalid API token.\n"
                            f"Response: {error_text}"
                        ) from e
                    if e.response.status_code == 400:
                        error_detail = e.response.text
                        try:
                            error_json = e.response.json()
                            error_detail = error_json.get("detail", error_json.get("error", error_detail))
                        except:
                            pass
                        raise RuntimeError(
                            f"Label Studio storage configuration failed (400 Bad Request).\n"
                            f"Error: {error_detail}"
                        ) from e
                    last_error = e
                    logger.warning(
                        "[LabelStudio][Storage] HTTP error",
                        extra={
                            "status": e.response.status_code,
                            "response": e.response.text[:200],
                        },
                    )
                    continue
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "[LabelStudio][Storage] Exception",
                        extra={"exception": type(e).__name__, "message": str(e)},
                    )
                    continue
        
        if response is None:
            raise RuntimeError(
                f"Failed to connect to Label Studio storage API. All endpoints failed. "
                f"Last error: {last_error}"
            )
        
        if response.status_code == 404:
            logger.warning("[LabelStudio][Storage] All endpoints returned 404")
            logger.warning("[LabelStudio][Storage] Storage API likely unsupported by LS version")
            raise RuntimeError(
                f"Label Studio storage API not available (404). "
                f"Your Label Studio version (1.21.0) may not support cloud storage API. "
                f"Please configure cloud storage manually at: "
                f"{self.base_url}/projects/{project_id}/settings/storage"
            )
        
        if response.status_code == 401:
            raise RuntimeError(
                f"Label Studio authentication failed (401). Invalid API token. "
                f"Check your LABEL_STUDIO_API_KEY in .env file"
            )
        
        try:
            if response.status_code == 400:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", error_json.get("error", error_detail))
                except:
                    pass
                raise RuntimeError(
                    f"Label Studio storage configuration failed (400 Bad Request).\n"
                    f"Error: {error_detail}\n"
                    f"Payload: {json.dumps(payload, indent=2)}"
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    f"Label Studio authentication failed (401). Invalid API token."
                ) from e
            error_text = e.response.text
            logger.error(
                "[LabelStudio][Storage] API error",
                extra={"status": e.response.status_code, "response": error_text[:500]},
            )
            raise RuntimeError(
                f"Label Studio storage API error: {e.response.status_code} - {error_text[:500]}"
            ) from e


    async def list_cloud_storages(
        self,
        *,
        storage_type: str = "s3",
        project_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List existing storages of a given type (optionally filtered by project)."""
        await self._update_headers()

        params = {}
        if project_id:
            params["project"] = project_id

        url = f"{self.base_url}/api/storages/{storage_type.lower()}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                response = await client.get(url, headers=self._headers, params=params or None)
                if response.status_code == 404:
                    return []
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            raise

    async def find_cloud_storage(
        self,
        *,
        project_id: int,
        storage_type: str,
        bucket: str,
        prefix: Optional[str],
        use_blob_urls: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        """Find an existing storage connection matching bucket/prefix/blob usage.
        
        Simplified matching: matches by project + bucket + storage_type only.
        Prefix and use_blob_urls differences are ignored to prevent duplicates.
        """
        storage_type_normalized = storage_type.upper()
        storages = await self.list_cloud_storages(storage_type=storage_type, project_id=project_id)
        
        if not storages:
            all_storages = await self.list_cloud_storages(storage_type=storage_type, project_id=None)
            storages = [s for s in all_storages if self._extract_project_id(s) == project_id]
        
        if not storages:
            for alt_type in ["s3", "S3", "gcs", "GCS", "azure", "AZURE"]:
                if alt_type.upper() == storage_type_normalized:
                    continue
                alt_storages = await self.list_cloud_storages(storage_type=alt_type, project_id=project_id)
                if not alt_storages:
                    alt_storages = await self.list_cloud_storages(storage_type=alt_type, project_id=None)
                    alt_storages = [s for s in alt_storages if self._extract_project_id(s) == project_id]
                storages.extend(alt_storages)
        
        logger.debug(
            "[LabelStudio][Storage] Searching existing storage",
            extra={"project_id": project_id, "storage_type": storage_type, "bucket": bucket},
        )
        logger.debug(
            "[LabelStudio][Storage] Candidate storages",
            extra={"count": len(storages)},
        )
        
        for storage in storages:
            storage_project_id = self._extract_project_id(storage)
            storage_bucket = storage.get("bucket")
            storage_type_actual = (storage.get("storage_type") or "").upper()
            
            logger.debug(
                "[LabelStudio][Storage] Inspecting storage",
                extra={
                    "storage_id": storage.get("id"),
                    "project_id": storage_project_id,
                    "bucket": storage_bucket,
                    "storage_type": storage_type_actual,
                },
            )
            
            if storage_project_id != project_id:
                logger.debug(
                    "[LabelStudio][Storage] Project mismatch",
                    extra={"storage_project_id": storage_project_id, "desired_project_id": project_id},
                )
                continue
            
            if storage_bucket != bucket:
                logger.debug(
                    "[LabelStudio][Storage] Bucket mismatch",
                    extra={"storage_bucket": storage_bucket, "desired_bucket": bucket},
                )
                continue
            
            if storage_type_actual and storage_type_actual != storage_type_normalized:
                logger.debug(
                    "[LabelStudio][Storage] Storage type differs but acceptable",
                    extra={"storage_type": storage_type_actual, "expected": storage_type_normalized},
                )
            
            logger.debug(
                "[LabelStudio][Storage] Match found",
                extra={
                    "project_id": project_id,
                    "bucket": bucket,
                    "storage_id": storage.get("id"),
                    "existing_prefix": storage.get("prefix"),
                    "desired_prefix": prefix,
                    "existing_use_blob": storage.get("use_blob_urls"),
                    "desired_use_blob": use_blob_urls,
                },
            )
            return storage
        
        logger.debug("[LabelStudio][Storage] No matching storage found")
        return None
    
    def _extract_project_id(self, obj: Dict[str, Any]) -> Optional[int]:
        """Extract project ID from storage/webhook object, handling both int and nested dict formats."""
        project = obj.get("project")
        if isinstance(project, dict):
            return project.get("id")
        return project

    async def find_ml_backend(self, *, url: str) -> Optional[Dict[str, Any]]:
        """Find an existing ML backend by URL."""
        backends = await self.list_ml_backends()
        for backend in backends:
            if backend.get("url") == url:
                return backend
        return None


    async def register_webhook(
        self,
        project_id: int,
        target_url: str,
        events: Optional[List[str]] = None,
        *,
        is_active: bool = True,
    ) -> Dict[str, Any]:
        """Register a webhook for a project."""
        await self._update_headers()
        
        # Label Studio webhook events format
        default_events = [
            "TASK_CREATED",
            "ANNOTATION_CREATED",
            "ANNOTATION_UPDATED",
            "TASK_COMPLETED",
        ]
        webhook_events = events or default_events
        
        payload = {
            "project": project_id,
            "url": target_url,
            "events": webhook_events,
            "is_active": is_active,
        }

        url = f"{self.base_url}/api/webhooks/"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._headers,
                )
                if response.status_code == 400:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("detail", error_json.get("error", error_json.get("validation_errors", error_detail)))
                    except:
                        pass
                    raise RuntimeError(
                        f"Label Studio webhook registration failed (400 Bad Request).\n"
                        f"Error: {error_detail}\n"
                        f"Payload: {json.dumps(payload, indent=2)}\n"
                        f"URL: {url}"
                    )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    f"Label Studio authentication failed (401). Invalid API token."
                ) from e
            raise RuntimeError(
                f"Label Studio webhook API error: {e.response.status_code} - {e.response.text}"
            ) from e

    async def list_webhooks(self, *, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List configured webhooks (optionally filtered by project)."""
        await self._update_headers()

        params = {}
        if project_id:
            params["project"] = project_id

        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
            response = await client.get(
                f"{self.base_url}/api/webhooks/",
                headers=self._headers,
                params=params or None,
            )
            response.raise_for_status()
            return response.json()

    async def find_webhook(
        self,
        *,
        project_id: int,
        target_url: str,
        events: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Locate an existing webhook for a project.
        
        Matches by project_id and URL. Events are checked if provided,
        but a match is still returned if URL and project match (events can be updated).
        """
        existing = await self.list_webhooks(project_id=project_id)
        events_set = set(events or []) if events else None

        logger.debug(
            "[LabelStudio][Webhook] Searching",
            extra={"project_id": project_id, "url": target_url, "events": events},
        )
        logger.debug(
            "[LabelStudio][Webhook] Existing count",
            extra={"count": len(existing)},
        )

        for hook in existing:
            hook_project_id = self._extract_project_id(hook)
            
            logger.debug(
                "[LabelStudio][Webhook] Inspecting webhook",
                extra={
                    "webhook_id": hook.get("id"),
                    "project_id": hook_project_id,
                    "url": hook.get("url"),
                },
            )
            
            if hook_project_id != project_id:
                logger.debug(
                    "[LabelStudio][Webhook] Project mismatch",
                    extra={"webhook_project": hook_project_id, "desired_project": project_id},
                )
                continue
            if hook.get("url") != target_url:
                logger.debug(
                    "[LabelStudio][Webhook] URL mismatch",
                    extra={"webhook_url": hook.get("url"), "desired_url": target_url},
                )
                continue
            
            if events_set:
                hook_events = set(hook.get("events") or [])
                if hook_events == events_set:
                    logger.debug("[LabelStudio][Webhook] Exact match (events aligned)")
                    return hook
                logger.debug(
                    "[LabelStudio][Webhook] Match found (events differ)",
                    extra={"existing_events": list(hook_events), "desired_events": list(events_set)},
                )
                return hook
            
            logger.debug("[LabelStudio][Webhook] Match found (URL + project)")
            return hook
        logger.debug("[LabelStudio][Webhook] No matching webhook")
        return None

