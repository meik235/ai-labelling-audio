"""FastAPI app bridging Label Studio and AssemblyAI."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from backend.integrations import AssemblyAIClient, AssemblyAIError, S3Uploader
from .config import get_config, get_value
from .preprocessing import AudioPreprocessor
from .storage import StorageError, fetch_audio_bytes

app = FastAPI(title="Label Studio ML Backend", version="0.1.0")
logger = logging.getLogger(__name__)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Simple landing endpoint for sanity probes."""
    return {
        "service": app.title,
        "version": app.version,
        "endpoints": {
            "health": "/health",
            "setup": "/setup",
            "predict": "/predict",
        },
        "docs": "/docs",
    }


class LabelStudioTask(BaseModel):
    """Subset of the Label Studio task payload we care about."""

    id: Optional[Any] = None
    data: Dict[str, Any]


class PredictRequest(BaseModel):
    """Incoming request from Label Studio."""

    task: Optional[LabelStudioTask] = None
    tasks: Optional[List[LabelStudioTask]] = None
    params: Dict[str, Any] = Field(default_factory=dict)


def _select_task(payload: PredictRequest) -> LabelStudioTask:
    if payload.task:
        return payload.task
    if payload.tasks:
        return payload.tasks[0]
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Expected `task` or `tasks` in payload.",
    )


def _extract_audio_url(task: LabelStudioTask, settings_field: str) -> str:
    candidates = [
        settings_field,
        "audio_url",
        "audio",
        "url",
    ]
    for key in candidates:
        value = task.data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Could not locate audio URL in task data: tried {candidates}",
    )


def _guess_filename(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if not parsed.path:
        return None
    name = os.path.basename(parsed.path)
    return name or None


@lru_cache()
def _get_s3_uploader() -> Optional[S3Uploader]:
    s3_cfg = get_value("integrations", "s3")
    if not isinstance(s3_cfg, dict):
        return None

    bucket = s3_cfg.get("bucket")
    access_key = s3_cfg.get("accessKeyId")
    secret_key = s3_cfg.get("secretAccessKey")
    region = s3_cfg.get("region")

    if not all([bucket, access_key, secret_key, region]):
        return None

    try:
        return S3Uploader(
            bucket_name=bucket,
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=region,
        )
    except Exception as exc:
        logger.warning("Failed to initialize S3 uploader for preannotations: %s", exc)
        return None


def _extract_prefix_from_audio_url(audio_url: str) -> Optional[str]:
    """Infer the CSV-level prefix (e.g., user_responses) from the audio URL."""
    parsed = urlparse(audio_url)
    path = parsed.path.lstrip("/")
    for marker in ("/ml-ready/", "/preprocessed/", "/raw/"):
        if marker in path:
            return path.split(marker)[0]
    return None


async def _store_preannotation(
    *,
    task_id: Any,
    audio_url: str,
    prediction: Dict[str, Any],
) -> None:
    """Persist the ML prediction JSON under ml-ready/preannotations/."""
    uploader = _get_s3_uploader()
    if not uploader:
        return

    prefix = _extract_prefix_from_audio_url(audio_url)
    if not prefix:
        return

    key = f"{prefix}/ml-ready/preannotations/task_{task_id}.json"
    try:
        payload = json.dumps(prediction, ensure_ascii=False, indent=2).encode("utf-8")
        uploader.upload_bytes(
            payload,
            key,
            metadata={
                "task_id": str(task_id),
                "type": "preannotation",
            },
        )
    except Exception as exc:
        logger.warning("Failed to store preannotation for task %s: %s", task_id, exc)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Basic readiness probe."""
    return {"status": "ok"}


@app.get("/setup")
async def setup_get() -> Dict[str, str]:
    """
    Optional setup endpoint for Label Studio compatibility.

    LS probes /setup to check if a backend needs configuration.
    """
    return {"message": "No setup required"}


@app.post("/setup")
async def setup_post(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Optional setup handler for POST requests from Label Studio.

    Payload is ignored; we just acknowledge the call.
    """
    return {
        "message": "Setup acknowledged",
        "received": payload or {},
    }


@app.post("/predict")
async def predict(payload: PredictRequest) -> Dict[str, Any]:
    config = get_config()

    incoming_field = get_value("runtime", "incomingAudioField", config=config)
    if not incoming_field:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="runtime.incomingAudioField is missing in modules.yaml.",
        )

    task = _select_task(payload)
    audio_url = _extract_audio_url(task, incoming_field)
    filename = _guess_filename(audio_url)
    language_code = payload.params.get("language")

    try:
        audio_bytes = await fetch_audio_bytes(audio_url)
    except StorageError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    enable_preprocessing = payload.params.get("enable_preprocessing", True)
    preprocessing_metadata = {}
    if enable_preprocessing:
        try:
            preprocessor = AudioPreprocessor(
                enable_noise_check=True,
                enable_segmentation=False,
            )
            audio_bytes, preprocessing_metadata = await preprocessor.preprocess_audio(
                audio_bytes, filename=filename
            )
        except Exception as e:
            # Log but don't fail if preprocessing fails
            preprocessing_metadata = {"error": str(e)}

    assembly_cfg = get_value("integrations", "assemblyAI", config=config)
    if not isinstance(assembly_cfg, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="integrations.assemblyAI section missing in modules.yaml.",
        )

    required_fields = ("apiKey", "baseUrl", "pollInterval", "autoChapters", "speakerLabels")
    missing = [field for field in required_fields if not assembly_cfg.get(field) and assembly_cfg.get(field) not in (False, 0)]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing integrations.assemblyAI fields: {', '.join(missing)}",
        )

    client = AssemblyAIClient(
        api_key=str(assembly_cfg["apiKey"]),
        base_url=str(assembly_cfg["baseUrl"]),
        poll_interval=float(assembly_cfg["pollInterval"]),
        auto_chapters=bool(assembly_cfg["autoChapters"]),
        speaker_labels=bool(assembly_cfg["speakerLabels"]),
    )

    try:
        transcript = await client.transcribe(
            audio_bytes,
            filename=filename,
            language_code=language_code,
        )
    except AssemblyAIError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    transcript_text = (transcript.get("text") or "").strip()
    if not transcript_text:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AssemblyAI returned no transcript text.",
        )

    from_name = payload.params.get("from_name", "transcription")
    to_name = payload.params.get("to_name", incoming_field)
    type_name = payload.params.get("type", "textarea")

    result = {
        "result": [
            {
                "id": "transcript",
                "from_name": from_name,
                "to_name": to_name,
                "type": type_name,
                "value": {"text": [transcript_text]},
            }
        ],
        "score": 1.0,
        "model_version": transcript.get("id"),
        "id": task.id or transcript.get("id"),
    }

    if transcript.get("chapters"):
        result["meta"] = {"chapters": transcript["chapters"]}

    # Include preprocessing metadata if available
    if preprocessing_metadata:
        if "meta" not in result:
            result["meta"] = {}
        result["meta"]["preprocessing"] = preprocessing_metadata

    # Store ML pre-annotation in S3 (best-effort)
    try:
        await _store_preannotation(
            task_id=task.id or transcript.get("id"),
            audio_url=audio_url,
            prediction=result,
        )
    except Exception as exc:
        logger.warning("Unable to persist preannotation for task %s: %s", task.id, exc)

    return {"predictions": [result]}


