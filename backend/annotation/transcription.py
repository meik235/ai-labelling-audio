"""Transcription runner orchestrating ASR per config."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from backend.integrations import AssemblyAIClient, AssemblyAIError
from backend.pipeline.config import TranscriptionConfig
from backend.storage import fetch_audio_bytes


logger = logging.getLogger(__name__)


class TranscriptionRunner:
    """Runs ASR on tasks based on config (currently AssemblyAI support)."""

    def __init__(
        self,
        assemblyai_client: Optional[AssemblyAIClient] = None,
    ) -> None:
        self.assemblyai = assemblyai_client

    async def run(
        self,
        tasks: List[Dict[str, Any]],
        config: TranscriptionConfig,
    ) -> List[Dict[str, Any]]:
        if not tasks:
            return tasks

        if config.asr.model.lower() != "assemblyai":
            # Future: plug other providers here
            return tasks

        if not self.assemblyai:
            raise RuntimeError("AssemblyAI client not configured for transcription runner.")

        updated: List[Dict[str, Any]] = []
        transcripts_generated = 0
        transcripts_failed = 0
        for task in tasks:
            if task.get("transcription"):
                updated.append(task)
                continue

            # Use transcription_audio if available (cleaned audio when cleaning is enabled),
            # otherwise fall back to ml_ready_audio or original audio
            audio_url = task.get("transcription_audio") or task.get("ml_ready_audio") or task.get("audio")
            if not audio_url:
                updated.append(task)
                continue

            try:
                audio_bytes = await self._download_audio(audio_url)
                # Resolve language code using AssemblyAI client's resolver
                # This handles special cases like "hinglish" -> auto-detection
                lang_resolution = self.assemblyai.resolve_language(config.asr.language)
                transcript_result = await self.assemblyai.transcribe(
                    audio_bytes,
                    filename=urlparse(audio_url).path.split("/")[-1],
                    language_code=lang_resolution.get("language_code"),
                    language_detection=lang_resolution.get("language_detection", False),
                    sentiment_analysis=config.sentiment_enabled,
                )
                generated_text = (transcript_result.get("text") or "").strip()
                audio_duration = transcript_result.get("audio_duration", 0)
                
                # Only fallback if transcription is truly empty AND duration is suspiciously short
                # This prevents unnecessary double transcription when processed audio is valid
                # Check if we're using processed audio and it's likely over-trimmed
                is_processed_audio = (
                    audio_url == task.get("transcription_audio") and 
                    task.get("transcription_audio") != task.get("audio")
                )
                
                if (
                    not generated_text 
                    and audio_duration < 5.0 
                    and is_processed_audio
                    and task.get("audio") 
                    and task.get("audio") != audio_url
                ):
                    logger.warning(
                        "Transcription returned empty text from processed audio (duration=%s), trying original audio",
                        audio_duration,
                        extra={"processed_url": audio_url, "original_url": task.get("audio")},
                    )
                    try:
                        original_audio_bytes = await self._download_audio(task["audio"])
                        transcript_result = await self.assemblyai.transcribe(
                            original_audio_bytes,
                            filename=urlparse(task["audio"]).path.split("/")[-1],
                            language_code=lang_resolution.get("language_code"),
                            language_detection=lang_resolution.get("language_detection", False),
                            sentiment_analysis=config.sentiment_enabled,
                        )
                        generated_text = (transcript_result.get("text") or "").strip()
                        if generated_text:
                            logger.info(
                                "Fallback to original audio succeeded",
                                extra={"text_length": len(generated_text)},
                            )
                        else:
                            logger.warning(
                                "Fallback to original audio also returned empty text",
                                extra={"original_url": task.get("audio")},
                            )
                    except Exception as fallback_error:
                        logger.warning(
                            "Fallback to original audio failed: %s",
                            fallback_error,
                            extra={"original_url": task.get("audio")},
                        )
                
                preview = generated_text[:120] + ("..." if len(generated_text) > 120 else "")
                logger.debug(
                    "AssemblyAI transcription generated",
                    extra={"audio_url": audio_url, "preview": preview, "text_length": len(generated_text)},
                )
                enriched = dict(task)
                enriched["transcription"] = generated_text
                asr_block = dict(enriched.get("asr") or {})
                asr_block["transcription"] = generated_text
                enriched["asr"] = asr_block
                enriched["transcript_metadata"] = transcript_result
                if config.sentiment_enabled:
                    sentiments = transcript_result.get("sentiment_analysis_results") or []
                    if sentiments:
                        enriched["emotions"] = sentiments
                segments = self._normalize_segments(transcript_result.get("segments") or [])
                
                # Determine speaker count from AssemblyAI results
                speaker_count = 0
                if segments:
                    enriched["segments"] = segments
                    diarization_block = dict(enriched.get("diarization") or {})
                    existing_segments = diarization_block.get("segments") or []
                    if not existing_segments:
                        diarization_block["segments"] = segments
                        diarization_block["source"] = "assemblyai"
                        enriched["diarization"] = diarization_block
                    else:
                        logger.debug(
                            "Preserving existing diarization segments from preprocessing",
                            extra={"existing_count": len(existing_segments)},
                        )
                    
                    speaker_count = len({seg.get("speaker") for seg in segments if seg.get("speaker")})
                    diarization_block["speaker_count"] = speaker_count
                    enriched["diarization"] = diarization_block
                else:
                    speaker_count = 1
                
                # Detect audio channels (mono/stereo) and update channel name
                current_channel = enriched.get("channel", "audio")
                audio_channels = None
                
                # Check AssemblyAI transcript metadata for channel information
                if transcript_result:
                    # AssemblyAI may return channel info in various fields
                    audio_channels = (
                        transcript_result.get("audio_channels") or
                        transcript_result.get("channels") or
                        transcript_result.get("channel_count")
                    )
                
                # Fallback to preprocessing metadata if available
                if audio_channels is None:
                    preprocessing = enriched.get("preprocessing", {})
                    validation = preprocessing.get("validation", {})
                    audio_channels = validation.get("channels")
                
                # Check if channel_info exists (from stereo split in preprocessing)
                channel_info = enriched.get("channel_info") or {}
                channel_index = channel_info.get("index", 0)
                channel_count = channel_info.get("count", audio_channels or 1)
                
                # Update channel name based on audio channels (mono/stereo)
                # If channel name is already set to channel_1, channel_2, etc., keep it
                if current_channel.startswith("channel_"):
                    # Channel name already set correctly from preprocessing
                    pass
                elif current_channel == "audio" or current_channel.startswith("audio"):
                    # Determine channel name based on detected channels
                    if channel_count == 1 or audio_channels == 1:
                        # Mono audio
                        enriched["channel"] = "channel_1"
                    elif channel_count > 1 or (audio_channels and audio_channels > 1):
                        # Stereo or multi-channel audio
                        # Use channel index from preprocessing if available, otherwise default to channel_1
                        if channel_index is not None:
                            enriched["channel"] = f"channel_{channel_index + 1}"
                        else:
                            enriched["channel"] = "channel_1"
                            logger.debug(
                                "Multi-channel audio detected, using channel_1",
                                extra={"audio_channels": audio_channels, "channel_count": channel_count},
                            )
                    else:
                        # Unknown channel count, default based on speaker count
                        if speaker_count > 1:
                            enriched["channel"] = "speaker_mix"
                        else:
                            enriched["channel"] = "speaker_single"
                elif current_channel in ("speaker_single", "speaker_mix"):
                    # If channel was set based on speaker count, update if we have channel info
                    if channel_count == 1 or audio_channels == 1:
                        enriched["channel"] = "channel_1"
                    elif channel_count > 1 or (audio_channels and audio_channels > 1):
                        if channel_index is not None:
                            enriched["channel"] = f"channel_{channel_index + 1}"
                        else:
                            enriched["channel"] = "channel_1"
                
                updated.append(enriched)
                transcripts_generated += 1
            except (AssemblyAIError, httpx.HTTPError, RuntimeError) as exc:
                logger.warning("Transcription failed for %s: %s", audio_url, exc)
                updated.append(task)
                transcripts_failed += 1

        logger.info(
            "Transcription pass complete",
            extra={
                "generated": transcripts_generated,
                "failed": transcripts_failed,
                "skipped": len(tasks) - transcripts_generated - transcripts_failed,
            },
        )

        return updated

    def _normalize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for segment in segments:
            text = (segment.get("text") or "").strip()
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            if not text or end <= start:
                continue
            entry = {
                "start": start,
                "end": end,
                "text": text,
            }
            speaker_name = self._normalize_speaker_name(segment.get("speaker") or segment.get("label"))
            if speaker_name:
                entry["speaker"] = speaker_name
            normalized.append(entry)
        return normalized

    @staticmethod
    def _normalize_speaker_name(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        value = str(raw).strip()
        if not value:
            return None
        lower = value.lower()
        if lower.startswith("speaker"):
            suffix = value.split()[-1]
            if len(suffix) == 1 and suffix.isalpha():
                return f"Speaker {suffix.upper()}"
            return f"Speaker {suffix}" if suffix.isdigit() else value.title()
        if len(value) == 1 and value.isalpha():
            return f"Speaker {value.upper()}"
        if value.isdigit():
            idx = int(value)
            letter = chr(ord("A") + idx)
            return f"Speaker {letter}"
        if value.upper().startswith("SPK"):
            suffix = value[3:]
            return f"Speaker {suffix}".strip()
        return value.title()

    async def _download_audio(self, url: str) -> bytes:
        timeout = httpx.Timeout(120.0, connect=10.0)
        return await fetch_audio_bytes(url, timeout=timeout)

