"""Extract audio segments from transcript timestamps."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import httpx

from .storage import fetch_audio_bytes


class AudioSegmenter:
    """Extract audio segments based on transcript timestamps."""

    async def extract_segments(
        self,
        audio_url: str,
        transcript_segments: List[Dict[str, Any]],
        output_dir: str | Path,
        segment_prefix: str = "segment",
    ) -> List[Dict[str, Any]]:
        """
        Extract audio segments from transcript timestamps.

        Args:
            audio_url: URL to original audio file
            transcript_segments: List of segments with start/end times
                Format: [{"start": 0.0, "end": 5.5, "text": "...", ...}, ...]
            output_dir: Directory to save extracted segments
            segment_prefix: Prefix for segment filenames

        Returns:
            List of segment metadata with file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download audio
        audio_bytes = await fetch_audio_bytes(audio_url)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_audio_path = tmp.name

        try:
            segments = []
            for i, seg in enumerate(transcript_segments):
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)

                # Convert to seconds if in milliseconds
                if start > 1000:
                    start = start / 1000.0
                if end > 1000:
                    end = end / 1000.0

                duration = end - start
                if duration <= 0:
                    continue

                # Extract segment
                segment_filename = f"{segment_prefix}_{i:04d}.wav"
                segment_path = output_dir / segment_filename

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-i",
                    tmp_audio_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(segment_path),
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                segments.append(
                    {
                        "segment_id": i,
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "text": seg.get("text", ""),
                        "speaker": seg.get("speaker"),
                        "file_path": str(segment_path),
                        "filename": segment_filename,
                    }
                )

            return segments

        finally:
            # Cleanup temp file
            Path(tmp_audio_path).unlink()

