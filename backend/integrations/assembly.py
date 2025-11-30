"""AssemblyAI transcription client."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, List

import httpx


logger = logging.getLogger(__name__)


class AssemblyAIError(RuntimeError):
    """Raised when the AssemblyAI API responds with an error."""


SUPPORTED_LANGUAGES = {
    # From https://www.assemblyai.com/docs/concepts/supported-languages
    "en",
    "en_us",
    "en_uk",
    "en_au",
    "en_za",
    "hi",
    "bn",
    "es",
    "es_419",
    "es_es",
    "fr",
    "fr_ca",
    "de",
    "pt",
    "pt_br",
    "it",
    "nl",
    "ja",
    "ko",
    "zh",
    "zh_cn",
    "zh_tw",
    "pl",
    "ru",
    "sv",
    "tr",
    "uk",
    "vi",
}

LANGUAGE_OVERRIDES = {
    "auto": None,  # Use auto-detection for all languages
    "hinglish": None,  # Use auto-detection for Hinglish (Hindi-English code-switching)
    "hi-en": None,  # Use auto-detection for Hindi-English mix
    "english": "en",
    "en-in": "en",
    "en-gb": "en_uk",
    "en-us": "en_us",
}


class AssemblyAIClient:
    """Minimal async client wrapper for AssemblyAI's transcription API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.assemblyai.com/v2",
        poll_interval: float = 3.0,
        auto_chapters: bool = False,
        speaker_labels: bool = True,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.auto_chapters = auto_chapters
        self.speaker_labels = speaker_labels
        self._timeout = httpx.Timeout(600.0, connect=30.0)
        self._max_poll_time = 3600.0

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "authorization": self.api_key,
        }

    def resolve_language(self, language: Optional[str]) -> Dict[str, Any]:
        """
        Decide how to pass language hints to AssemblyAI.
        
        Returns:
            {
                "language_code": Optional[str],
                "language_detection": bool,
                "requested": Optional[str],
                "resolved": Optional[str],
                "note": str,
            }
        """
        if not language:
            return {
                "language_code": "en",
                "language_detection": False,
                "requested": None,
                "resolved": "en",
                "note": "default_en",
            }
        lang = language.strip().lower()
        if lang in LANGUAGE_OVERRIDES:
            resolved = LANGUAGE_OVERRIDES[lang]
            if resolved is None:
                return {
                    "language_code": None,
                    "language_detection": True,
                    "requested": language,
                    "resolved": None,
                    "note": "override_auto_detect",
                }
            return {
                "language_code": resolved,
                "language_detection": False,
                "requested": language,
                "resolved": resolved,
                "note": "override",
            }
        if lang in SUPPORTED_LANGUAGES:
            return {
                "language_code": lang,
                "language_detection": False,
                "requested": language,
                "resolved": lang,
                "note": "supported",
            }
        logger.warning(
            "AssemblyAI does not support requested language '%s'; enabling auto-detection.",
            language,
        )
        return {
            "language_code": None,
            "language_detection": True,
            "requested": language,
            "resolved": None,
            "note": "auto_detect",
        }

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: Optional[str] = None,
        language_code: Optional[str] = None,
        language_detection: bool = False,
        sentiment_analysis: bool = False,
    ) -> Dict[str, Any]:
        """Upload audio and return the completed transcript payload."""
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self._timeout,
        ) as client:
            upload_url = await self._upload_audio(client, audio_bytes)
            transcript_id = await self._create_transcript(
                client,
                upload_url,
                filename=filename,
                language_code=language_code,
                language_detection=language_detection,
                sentiment_analysis=sentiment_analysis,
            )
            transcript = await self._wait_for_completion(client, transcript_id)

        segments: List[Dict[str, Any]] = []
        utterances = transcript.get("utterances")
        if isinstance(utterances, list) and utterances:
            segments = [
                {
                    "start": float(item.get("start", 0)) / 1000.0,
                    "end": float(item.get("end", 0)) / 1000.0,
                    "text": (item.get("text") or "").strip(),
                    "speaker": item.get("speaker"),
                }
                for item in utterances
                if item.get("text")
            ]
        elif isinstance(transcript.get("words"), list):
            running: List[Dict[str, Any]] = []
            for word in transcript["words"]:
                running.append(
                    {
                        "start": float(word.get("start", 0)) / 1000.0,
                        "end": float(word.get("end", 0)) / 1000.0,
                        "text": (word.get("text") or "").strip(),
                        "speaker": word.get("speaker"),
                    }
                )
                if len(running) >= 25:
                    merged_text = " ".join(seg["text"] for seg in running if seg["text"])
                    if merged_text:
                        segments.append(
                            {
                                "start": running[0]["start"],
                                "end": running[-1]["end"],
                                "text": merged_text,
                                "speaker": running[0].get("speaker"),
                            }
                        )
                    running = []
            if running:
                merged_text = " ".join(seg["text"] for seg in running if seg["text"])
                if merged_text:
                    segments.append(
                        {
                            "start": running[0]["start"],
                            "end": running[-1]["end"],
                            "text": merged_text,
                            "speaker": running[0].get("speaker"),
                        }
                    )
        transcript["segments"] = segments
        return transcript

    async def _upload_audio(self, client: httpx.AsyncClient, audio_bytes: bytes) -> str:
        response = await client.post(
            "/upload",
            content=audio_bytes,
            headers={
                **self._headers,
                "content-type": "application/octet-stream",
            },
        )
        self._raise_for_status(response, "upload audio")
        payload = response.json()
        upload_url = payload.get("upload_url")
        if not upload_url:
            raise AssemblyAIError("Upload succeeded but no upload_url was returned.")
        return upload_url

    async def _create_transcript(
        self,
        client: httpx.AsyncClient,
        upload_url: str,
        *,
        filename: Optional[str],
        language_code: Optional[str],
        language_detection: bool = False,
        sentiment_analysis: bool = False,
    ) -> str:
        body: Dict[str, Any] = {"audio_url": upload_url}
        if language_code:
            body["language_code"] = language_code
        if language_detection:
            body["language_detection"] = True
        if self.auto_chapters:
            body["auto_chapters"] = True
        if self.speaker_labels:
            body["speaker_labels"] = True
        if sentiment_analysis:
            body["sentiment_analysis"] = True

        response = await client.post("/transcript", json=body)
        self._raise_for_status(response, "create transcript")
        payload = response.json()
        transcript_id = payload.get("id")
        if not transcript_id:
            raise AssemblyAIError("Transcript creation succeeded but no id was returned.")
        return transcript_id

    async def _wait_for_completion(
        self,
        client: httpx.AsyncClient,
        transcript_id: str,
    ) -> Dict[str, Any]:
        import time
        poll_count = 0
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > self._max_poll_time:
                raise AssemblyAIError(
                    f"Transcript {transcript_id} timed out after {self._max_poll_time}s. "
                    "Long audio files may take longer to process."
                )
            
            response = await client.get(f"/transcript/{transcript_id}")
            self._raise_for_status(response, "poll transcript")
            data = response.json()
            status = data.get("status")
            
            if status == "completed":
                if poll_count > 0:
                    logger.info(
                        "Transcript %s completed after %s poll(s) (%.1f seconds)",
                        transcript_id, poll_count, elapsed_time
                    )
                return data
            
            if status == "error":
                if poll_count > 0:
                    logger.error("Transcript %s failed after %s poll(s)", transcript_id, poll_count)
                raise AssemblyAIError(f"Transcript failed: {data.get('error')}")
            
            if poll_count == 0:
                logger.info("Polling transcript %s (AssemblyAI)", transcript_id)
            elif poll_count % 20 == 0:
                logger.info(
                    "Still processing transcript %s (poll %s, %.1f seconds elapsed)",
                    transcript_id, poll_count, elapsed_time
                )
            
            poll_count += 1
            await asyncio.sleep(self.poll_interval)

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = None
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise AssemblyAIError(f"Failed to {operation}: {detail}") from exc


