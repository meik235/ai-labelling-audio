"""Audio pre-processing utilities for ML Backend."""

from __future__ import annotations

import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.pipeline.config import ClearAudioConfig


logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio pre-processing: noise check, segmentation."""

    def __init__(
        self,
        enable_noise_check: bool = True,
        enable_segmentation: bool = False,
        chunk_duration_sec: int = 30,
        chunk_overlap_sec: float = 0.0,
        clear_config: Optional[ClearAudioConfig] = None,
    ) -> None:
        """
        Initialize audio preprocessor.

        Args:
            enable_noise_check: Check and report noise levels
            enable_segmentation: Split audio into chunks
            chunk_duration_sec: Chunk duration if segmentation enabled
        """
        self.enable_noise_check = enable_noise_check
        self.enable_segmentation = enable_segmentation
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_overlap_sec = chunk_overlap_sec
        self.clear_config = clear_config or ClearAudioConfig()

    async def preprocess_audio(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        original_duration: Optional[float] = None,
    ) -> tuple[bytes, dict]:
        """
        Pre-process audio bytes.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Optional filename for format detection

        Returns:
            Tuple of (processed_audio_bytes, metadata_dict)
        """
        metadata = {
            "original_size_bytes": len(audio_bytes),
            "preprocessing_applied": [],
        }

        # Write to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input_path = tmp_input.name

        try:
            processed_path = tmp_input_path

            processed_path = await self._standardize_audio(processed_path)
            metadata["preprocessing_applied"].append(
                f"standardization:{self.clear_config.target_format}@{self.clear_config.target_sample_rate}"
            )

            processed_path = await self._maybe_normalize(processed_path, metadata)
            processed_path = await self._maybe_denoise(processed_path, metadata)
            processed_path = await self._maybe_trim(processed_path, metadata)

            if self.enable_noise_check:
                noise_info = await self._check_noise(processed_path)
                metadata["noise"] = noise_info
                metadata["preprocessing_applied"].append("noise_check")

            if self.enable_segmentation:
                metadata["chunking_enabled"] = True
                metadata["preprocessing_applied"].append("chunking_marked")

            with open(processed_path, "rb") as f:
                processed_bytes = f.read()

            metadata["processed_size_bytes"] = len(processed_bytes)

            return processed_bytes, metadata

        finally:
            for path in [tmp_input_path, processed_path]:
                if path != tmp_input_path and Path(path).exists():
                    Path(path).unlink()
            if Path(tmp_input_path).exists():
                Path(tmp_input_path).unlink()

    async def _standardize_audio(self, input_path: str) -> str:
        """Convert to WAV, mono, 16kHz."""
        output_path = input_path.replace(".wav", "_standardized.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    async def _maybe_normalize(self, input_path: str, metadata: dict) -> str:
        if not self.clear_config.normalization_enabled:
            return input_path

        mode = (self.clear_config.normalization_mode or "loudnorm").lower()
        if mode == "dynaudnorm":
            try:
                return await self._run_dynaudnorm(input_path, metadata)
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr if exc.stderr else "No stderr available"
                logger.warning(
                    "dynaudnorm failed (exit=%s, error=%s). Skipping normalization.",
                    exc.returncode,
                    stderr[:300],
                )
                return input_path

        output_path = input_path.replace(".wav", "_norm.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter:a",
            f"loudnorm=I={self.clear_config.normalization_level_db}",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            metadata["preprocessing_applied"].append("normalization")
            Path(input_path).unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr if exc.stderr else "No stderr available"
            logger.warning(
                "loudnorm failed (exit=%s, error=%s). Falling back to dynaudnorm.",
                exc.returncode,
                stderr[:300],
            )
            Path(output_path).unlink(missing_ok=True)
            
            try:
                return await self._run_dynaudnorm(input_path, metadata)
            except subprocess.CalledProcessError as fallback_exc:
                fallback_stderr = fallback_exc.stderr if fallback_exc.stderr else str(fallback_exc)
                logger.error(
                    "dynaudnorm fallback failed (exit=%s, error=%s). Using original audio.",
                    fallback_exc.returncode,
                    fallback_stderr[:200],
                )
                return input_path

    async def _run_dynaudnorm(self, input_path: str, metadata: dict) -> str:
        output_path = input_path.replace(".wav", "_dynorm.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter:a",
            "dynaudnorm",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        metadata["preprocessing_applied"].append("normalization (dynaudnorm)")
        Path(input_path).unlink(missing_ok=True)
        return output_path

    async def _maybe_denoise(self, input_path: str, metadata: dict) -> str:
        if not self.clear_config.denoise_enabled:
            return input_path

        intensity_map = {
            "low": "-20",
            "medium": "-30",
            "high": "-40",
        }
        noise_floor = intensity_map.get(self.clear_config.denoise_intensity, "-25")
        output_path = input_path.replace(".wav", "_denoise.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            f"afftdn=nf={noise_floor}",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            metadata["preprocessing_applied"].append("denoise")
            Path(input_path).unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as exc:
            logger.warning("Denoise failed: %s", exc)
            Path(output_path).unlink(missing_ok=True)
            return input_path

    async def _maybe_trim(self, input_path: str, metadata: dict) -> str:
        if not self.clear_config.trim_enabled:
            return input_path

        output_path = input_path.replace(".wav", "_trim.wav")
        threshold = self.clear_config.trim_silence_threshold
        duration = self.clear_config.trim_min_silence_duration
        filter_args = (
            f"silenceremove=start_periods=1:start_threshold={threshold}dB:"
            f"start_duration={duration}:stop_periods=1:stop_threshold={threshold}dB:"
            f"stop_duration={duration}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            filter_args,
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            metadata["preprocessing_applied"].append("trim")
            Path(input_path).unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as exc:
            logger.warning("Trim failed: %s", exc)
            Path(output_path).unlink(missing_ok=True)
            return input_path

    async def _check_noise(self, audio_path: str) -> dict:
        """Check noise levels in audio."""
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(audio_path, sr=None)

            signal_power = np.mean(y ** 2)
            noise_floor = np.percentile(y ** 2, 10)
            snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))

            energy = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)

            if snr_db > 20:
                noise_level = "low"
            elif snr_db > 10:
                noise_level = "medium"
            else:
                noise_level = "high"

            return {
                "snr_db": float(snr_db),
                "noise_level": noise_level,
                "energy_mean": float(energy_mean),
                "energy_std": float(energy_std),
            }
        except ImportError:
            return {"error": "librosa not available"}

