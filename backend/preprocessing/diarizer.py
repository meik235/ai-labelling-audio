"""Lightweight speaker diarization heuristics for mix tracks."""

from __future__ import annotations

import io
import logging
import math
import tempfile
import os
import warnings
from functools import lru_cache, wraps
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import inspect

# Suppress pyannote.audio and dependency warnings globally
# These are library deprecation warnings that don't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import librosa  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError("librosa is required for heuristic diarization") from exc

try:
    from huggingface_hub import HfFolder, login, hf_hub_download  # type: ignore
except ImportError:  # pragma: no cover - optional
    HfFolder = None  # type: ignore
    login = None  # type: ignore
    hf_hub_download = None  # type: ignore

logger = logging.getLogger(__name__)

def _patch_hf_hub_download_early():
    """Patch hf_hub_download and snapshot_download to forward use_auth_token -> token.
    
    Also patches snapshot_download to handle positional argument issues.
    
    Returns the patched hf_hub_download function if successful, None otherwise.
    """
    try:
        import importlib
        import sys
        
        hf_module = importlib.import_module("huggingface_hub")
        
        funcs_to_patch = ["hf_hub_download", "snapshot_download"]
        patched_hf_hub_download = None
        
        for func_name in funcs_to_patch:
            if not hasattr(hf_module, func_name):
                continue
                
            original = getattr(hf_module, func_name)
            
            sig = inspect.signature(original)
            if "use_auth_token" in sig.parameters:
                logger.debug(f"{func_name} already supports use_auth_token")
                if func_name == "hf_hub_download":
                    patched_hf_hub_download = original
                continue
            
            def _make_patched_func(orig_func, name):
                @wraps(orig_func)
                def _patched_func(*args, **kwargs):
                    if "use_auth_token" in kwargs:
                        logger.debug(
                            f"Intercepted {name} call with use_auth_token, forwarding to token",
                            extra={"has_token": "token" in kwargs}
                        )
                        if "token" not in kwargs:
                            kwargs["token"] = kwargs.pop("use_auth_token")
                        else:
                            kwargs.pop("use_auth_token")
                    
                    if name == "snapshot_download" and len(args) > 1:
                        try:
                            sig = inspect.signature(orig_func)
                            params = list(sig.parameters.keys())
                            
                            if params and params[0] == "repo_id":
                                repo_id = args[0]
                                new_kwargs = dict(kwargs)
                                logger.debug(
                                    f"snapshot_download called with {len(args)} positional args, "
                                    f"converting to repo_id={repo_id} + kwargs"
                                )
                                return orig_func(repo_id, **new_kwargs)
                        except Exception as patch_exc:
                            logger.debug(f"Error in snapshot_download patch: {patch_exc}")
                    
                    return orig_func(*args, **kwargs)
                return _patched_func
            
            patched_func = _make_patched_func(original, func_name)
            
            if func_name == "hf_hub_download":
                patched_hf_hub_download = patched_func
            
            setattr(hf_module, func_name, patched_func)
            hf_module.__dict__[func_name] = patched_func
            
            try:
                file_download = importlib.import_module("huggingface_hub.file_download")
                if hasattr(file_download, func_name):
                    setattr(file_download, func_name, patched_func)
                    file_download.__dict__[func_name] = patched_func
            except Exception:
                pass
                
            if "huggingface_hub" in sys.modules:
                setattr(sys.modules["huggingface_hub"], func_name, patched_func)
                sys.modules["huggingface_hub"].__dict__[func_name] = patched_func
                
            logger.info(f"Patched {func_name}: use_auth_token -> token")

        return patched_hf_hub_download
    except Exception as exc:
        logger.warning("Failed to patch huggingface_hub functions", exc_info=exc)
        return None

_patch_hf_hub_download_early()

try:
    import huggingface_hub
    test_sig = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" not in test_sig.parameters:
        logger.debug("Verified: hf_hub_download patch is active (use_auth_token not in signature)")
    else:
        logger.warning("Patch verification failed: use_auth_token still in signature")
except Exception as exc:
    logger.debug("Could not verify patch", exc_info=exc)

try:
    from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore

    PYANNOTE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    PyannotePipeline = None  # type: ignore
    PYANNOTE_AVAILABLE = False


def _detect_stereo_channels(audio_bytes: bytes) -> Optional[Tuple[float, int]]:
    """Detect if audio is stereo (2 channels) and return duration and channel count.
    
    Returns:
        (duration, num_channels) if stereo detected, None otherwise
    """
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        return None
    
    try:
        buffer = io.BytesIO(audio_bytes)
        data, sr = sf.read(buffer, always_2d=True)
        num_channels = data.shape[1] if len(data.shape) > 1 else 1
        duration = len(data) / sr if sr > 0 else 0.0
        
        if num_channels == 2:
            return (duration, num_channels)
    except Exception as exc:
        logger.debug("Failed to detect stereo channels: %s", exc)
    
    return None


def _create_stereo_diarization_segments(duration: float) -> List[dict]:
    """Create diarization segments for stereo audio where each channel is a speaker.
    
    For stereo audio, channel 0 (left) = Speaker one, channel 1 (right) = Speaker two.
    Both speakers are active for the full duration.
    """
    return [
        {
            "start": 0.0,
            "end": duration,
            "label": "Speaker one",
            "speaker": "Speaker one",
            "speaker_index": 0,
            "source": "stereo_channel",
        },
        {
            "start": 0.0,
            "end": duration,
            "label": "Speaker two",
            "speaker": "Speaker two",
            "speaker_index": 1,
            "source": "stereo_channel",
        },
    ]


def diarize_audio_heuristic(
    audio_bytes: bytes,
    *,
    sample_rate: int = 16000,
    min_speakers: int = 2,
    max_speakers: int = 4,
    window_sec: float = 0.5,
) -> List[dict]:
    """Return coarse speaker segments using MFCC + clustering heuristics."""
    if not audio_bytes:
        return []

    # Check if audio is stereo (2 channels) - treat each channel as a separate speaker
    stereo_info = _detect_stereo_channels(audio_bytes)
    if stereo_info:
        duration, num_channels = stereo_info
        logger.info(
            "Detected stereo audio with %s channels; treating each channel as a separate speaker",
            num_channels,
            extra={"duration": duration, "channels": num_channels},
        )
        return _create_stereo_diarization_segments(duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        signal, sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
        audio_duration = len(signal) / sr
    finally:
        tmp_path.unlink(missing_ok=True)

    if not signal.size:
        return []

    if audio_duration < 2.0:
        return [{
            "start": 0.0,
            "end": audio_duration,
            "label": "Speaker one",
            "speaker_index": 0,
        }]

    hop_length = max(1, int(sr * window_sec))
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, hop_length=hop_length, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate([mfcc, delta, delta2], axis=0).T
    if feats.shape[0] > 1:
        feats = StandardScaler().fit_transform(feats)
    
    target_speakers = max(2, min_speakers) if min_speakers >= 2 else min_speakers
    
    if feats.shape[0] < target_speakers:
        logger.info(
            "Not enough frames for clustering; falling back to time-based split",
            extra={"target_speakers": target_speakers, "frame_count": feats.shape[0]},
        )
        return _time_based_split(audio_duration, target_speakers)
    
    best_labels, best_k = _select_cluster_count(feats, target_speakers, max_speakers)
    
    # Ensure best_k is valid
    if best_k < 1:
        logger.warning("Invalid cluster count (%s), falling back to time-based split", best_k)
        return _time_based_split(audio_duration, target_speakers)
    
    # Ensure best_labels is valid
    if best_labels is None or len(best_labels) == 0:
        logger.warning("No labels generated from clustering, falling back to time-based split")
        return _time_based_split(audio_duration, target_speakers)
    
    if best_k == 1 and target_speakers >= 2:
        logger.warning(
            "Clustering favored a single speaker; forcing %s speakers via k-means",
            target_speakers,
        )
        try:
            forced_labels, _ = _kmeans(feats, target_speakers)
            best_labels = forced_labels
            best_k = target_speakers
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Forced multi-speaker clustering failed; splitting by time",
                extra={"error": str(exc)},
            )
            return _time_based_split(audio_duration, target_speakers)
    
    # Ensure frame_duration is valid
    if sr <= 0:
        logger.warning("Invalid sample rate (%s), falling back to time-based split", sr)
        return _time_based_split(audio_duration, target_speakers)
    
    frame_duration = hop_length / sr
    if frame_duration <= 0:
        logger.warning("Invalid frame duration (%s), falling back to time-based split", frame_duration)
        return _time_based_split(audio_duration, target_speakers)
    
    try:
        segments = _labels_to_segments(best_labels, frame_duration, best_k)
    except Exception as exc:
        logger.warning(
            "Failed to convert labels to segments, falling back to time-based split",
            extra={"error": str(exc), "best_k": best_k, "frame_duration": frame_duration, "labels_count": len(best_labels)},
            exc_info=True,
        )
        return _time_based_split(audio_duration, target_speakers)
    
    if len(segments) == 1 and target_speakers >= 2:
        logger.info(
            "Only one diarization segment found; splitting into %s segments by time",
            target_speakers,
        )
        return _time_based_split(audio_duration, target_speakers)
    
    if len(segments) > 1:
        merged_segments = []
        for i, seg in enumerate(segments):
            seg_duration = seg["end"] - seg["start"]
            if seg_duration < 0.5 and merged_segments and len(segments) - i > 1:
                merged_segments[-1]["end"] = seg["end"]
            else:
                merged_segments.append(seg)
        
        if len(merged_segments) == 1 and target_speakers >= 2:
            logger.warning("Merging resulted in a single segment, forcing time-based split")
            return _time_based_split(audio_duration, target_speakers)
        
        segments = merged_segments
    
    if len(segments) == 1 and target_speakers >= 2:
        logger.warning("Final check: only one segment detected; forcing time-based split")
        return _time_based_split(audio_duration, target_speakers)
    
    segments = _ensure_full_audio_coverage(segments, audio_duration)
    return segments


def diarize_audio_pyannote(
    audio_bytes: bytes,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[dict]:
    """High-quality diarization using pyannote.audio pipeline."""
    if not audio_bytes:
        return []

    # Check if audio is stereo (2 channels) - treat each channel as a separate speaker
    stereo_info = _detect_stereo_channels(audio_bytes)
    if stereo_info:
        duration, num_channels = stereo_info
        logger.info(
            "Detected stereo audio with %s channels; treating each channel as a separate speaker (skipping pyannote)",
            num_channels,
            extra={"duration": duration, "channels": num_channels},
        )
        return _create_stereo_diarization_segments(duration)

    if not PYANNOTE_AVAILABLE or PyannotePipeline is None:
        raise RuntimeError(
            "pyannote.audio is not installed. Install project extras or set diarization model to 'heuristic'."
        )

    token = _resolve_hf_token(hf_token)
    if not token:
        raise RuntimeError(
            "HuggingFace token is required for pyannote diarization. "
            "Set HUGGINGFACE_TOKEN environment variable or provide via config."
        )

    logger.info("Initializing Pyannote diarization pipeline...")
    pipeline = _load_pyannote_pipeline(token)
    logger.info("Pyannote pipeline loaded successfully")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        import librosa
        signal, sr = librosa.load(tmp_path, sr=None, mono=True)
        audio_duration = len(signal) / sr
        
        logger.info(
            "Running Pyannote diarization (this may take several minutes for long audio files)",
            extra={"audio_duration": round(audio_duration, 2), "min_speakers": min_speakers, "max_speakers": max_speakers},
        )
        # Pyannote diarization can take a long time - log progress periodically
        import threading
        import time
        progress_logged = threading.Event()
        
        def log_progress():
            start_time = time.time()
            while not progress_logged.is_set():
                elapsed = time.time() - start_time
                if elapsed > 30:  # Log every 30 seconds
                    logger.info(
                        f"Pyannote diarization still running... ({int(elapsed)}s elapsed)",
                        extra={"audio_duration": round(audio_duration, 2)},
                    )
                    start_time = time.time()  # Reset to avoid spam
                time.sleep(10)  # Check every 10 seconds
        
        progress_thread = threading.Thread(target=log_progress, daemon=True)
        progress_thread.start()
        
        try:
            diarization = pipeline(
                tmp_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        finally:
            progress_logged.set()
        
        logger.info("Pyannote diarization completed")
    finally:
        tmp_path.unlink(missing_ok=True)

    segments: List[dict] = []
    speaker_map: Dict[str, Tuple[str, int]] = {}
    next_idx = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.end <= turn.start:
            continue

        label_key = speaker or f"speaker_{next_idx:02d}"
        if label_key not in speaker_map:
            speaker_name = _ls_speaker_label(next_idx)
            speaker_map[label_key] = (speaker_name, next_idx)
            next_idx += 1

        speaker_name, speaker_idx = speaker_map[label_key]
        segments.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "label": speaker_name,
                "speaker": speaker_name,
                "speaker_index": speaker_idx,
                "source": "pyannote",
            }
        )

    if not segments:
        raise RuntimeError("pyannote returned zero segments")

    segments = _ensure_full_audio_coverage(segments, audio_duration)
    return segments


def _time_based_split(duration: float, expected_speakers: int, max_chunk_duration: float = 10.0) -> List[dict]:
    """Split audio by time into multiple speaker segments for fallback scenarios.

    We create many short alternating segments so the user can still make fine edits.
    """
    if duration <= 0:
        return []
    
    expected_speakers = max(1, expected_speakers)
    # Create several short chunks (at least 4 per speaker or every `max_chunk_duration` seconds)
    min_chunks = expected_speakers * 4
    chunks_from_duration = max(1, int(math.ceil(duration / max_chunk_duration)))
    total_chunks = max(min_chunks, chunks_from_duration)
    segment_duration = duration / total_chunks
    
    segments: List[dict] = []
    for i in range(total_chunks):
        start = i * segment_duration
        end = duration if i == total_chunks - 1 else (i + 1) * segment_duration
        speaker_idx = i % expected_speakers
        speaker_label = _ls_speaker_label(speaker_idx)
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "label": speaker_label,
                "speaker_index": speaker_idx,
            }
        )
    
    return segments


def _select_cluster_count(
    feats: np.ndarray,
    min_k: int,
    max_k: int,
) -> Tuple[np.ndarray, int]:
    best_score = -math.inf
    best_labels: np.ndarray | None = None
    best_k = min_k
    max_k = max(min_k, min(max_k, feats.shape[0]))

    # If min_k > 1, we want to find multiple speakers, so don't return k=1 immediately
    for k in range(max(1, min_k), max_k + 1):
        labels, _ = _kmeans(feats, k)
        if k == 1 and min_k == 1:
            # Only return k=1 immediately if min_k is also 1
            return labels, k
        score = _silhouette_score(feats, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    assert best_labels is not None
    return best_labels, best_k


def _kmeans(points: np.ndarray, k: int, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(points), size=k, replace=False)
    centers = points[indices]

    for _ in range(max_iter):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array(
            [points[labels == idx].mean(axis=0) if np.any(labels == idx) else centers[idx] for idx in range(k)]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers


def _silhouette_score(points: np.ndarray, labels: np.ndarray) -> float:
    n = len(points)
    if n <= 1:
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores: List[float] = []

    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]

        if np.sum(same_cluster) > 1:
            a = np.mean(distances[i, same_cluster][distances[i, same_cluster] > 0])
        else:
            a = 0.0

        b_values: List[float] = []
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            mask = labels == lbl
            if np.any(mask):
                b_values.append(np.mean(distances[i, mask]))
        b = min(b_values) if b_values else 0.0

        if max(a, b) == 0:
            score = 0.0
        else:
            score = (b - a) / max(a, b)
        scores.append(score)

    return float(np.mean(scores))


def _labels_to_segments(labels: Sequence[int], frame_duration: float, num_speakers: int) -> List[dict]:
    # Handle numpy arrays and empty sequences properly
    if labels is None or len(labels) == 0:
        return []

    segments: List[dict] = []
    start = 0.0
    current_label = labels[0]
    min_segment_duration = 0.3  # Minimum segment duration in seconds

    for idx, label in enumerate(labels):
        if label != current_label:
            segment_duration = idx * frame_duration - start
            if segment_duration >= min_segment_duration:
                speaker_label = _ls_speaker_label(int(current_label))
                segments.append(
                    {
                        "start": round(start, 3),
                        "end": round(idx * frame_duration, 3),
                        "label": speaker_label,
                        "speaker_index": int(current_label),
                    }
                )
            current_label = label
            start = idx * frame_duration

    final_duration = len(labels) * frame_duration - start
    if final_duration >= min_segment_duration:
        speaker_label = _ls_speaker_label(int(current_label))
        segments.append(
            {
                "start": round(start, 3),
                "end": round(len(labels) * frame_duration, 3),
                "label": speaker_label,
                "speaker_index": int(current_label),
            }
        )
    
    if len(segments) == 1 and num_speakers > 1:
        total_duration = segments[0]["end"] - segments[0]["start"]
        offset = segments[0]["start"]
        split_segments = _time_based_split(total_duration, num_speakers)
        for seg in split_segments:
            seg["start"] = round(seg["start"] + offset, 3)
            seg["end"] = round(seg["end"] + offset, 3)
        return split_segments
    
    return segments


def _ensure_full_audio_coverage(segments: List[dict], audio_duration: float) -> List[dict]:
    """Ensure diarization segments cover the entire audio duration.
    
    Fills gaps at the beginning, middle, and end by extending adjacent segments
    or creating new segments for silence/non-speech regions.
    """
    if not segments or audio_duration <= 0:
        return segments
    
    segments = sorted(segments, key=lambda s: s["start"])
    covered_segments: List[dict] = []
    
    first_segment_start = segments[0]["start"]
    last_segment_end = segments[-1]["end"]
    
    if first_segment_start > 0:
        first_speaker = segments[0].get("speaker_index", 0)
        first_label = segments[0].get("label", _ls_speaker_label(first_speaker))
        covered_segments.append({
            "start": 0.0,
            "end": first_segment_start,
            "label": first_label,
            "speaker": first_label,
            "speaker_index": first_speaker,
            "source": segments[0].get("source", "pyannote"),
        })
    
    for i, seg in enumerate(segments):
        if i > 0:
            prev_end = covered_segments[-1]["end"]
            gap = seg["start"] - prev_end
            if gap > 0.1:
                prev_speaker = covered_segments[-1].get("speaker_index", 0)
                prev_label = covered_segments[-1].get("label", _ls_speaker_label(prev_speaker))
                covered_segments[-1]["end"] = seg["start"]
        
        covered_segments.append(seg.copy())
    
    if last_segment_end < audio_duration:
        last_speaker = covered_segments[-1].get("speaker_index", 0)
        last_label = covered_segments[-1].get("label", _ls_speaker_label(last_speaker))
        covered_segments[-1]["end"] = audio_duration
    
    return covered_segments


def _ls_speaker_label(label_idx: int) -> str:
    if label_idx == 0:
        return "Speaker one"
    if label_idx == 1:
        return "Speaker two"
    return _speaker_name(label_idx)


def _speaker_name(label_idx: int) -> str:
    return f"Speaker {label_idx + 1}"


def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    """Resolve HuggingFace token from explicit parameter or HUGGINGFACE_TOKEN environment variable.
    
    Checks in order:
    1. Explicit parameter
    2. HUGGINGFACE_TOKEN env var
    """
    if explicit and explicit.strip():
        return explicit.strip()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token and token.strip():
        return token.strip()
    return None


@lru_cache(maxsize=2)
def _load_pyannote_pipeline(token: str) -> "PyannotePipeline":
    """Load pyannote pipeline using environment variable authentication."""
    if not PYANNOTE_AVAILABLE or PyannotePipeline is None:
        raise RuntimeError("pyannote.audio is not installed")
    if not token:
        raise RuntimeError("HuggingFace token required to load pyannote pipeline")

    import os
    # Save original values
    original_huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    original_hf_token = os.environ.get("HF_TOKEN")  # May exist from previous runs
    
    # Suppress deprecation warnings from pyannote.audio and its dependencies
    # These are library warnings, not errors, and don't affect functionality
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
        warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
        warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
        warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
        warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
        warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
        
        try:
            # Remove HF_TOKEN if it exists to avoid huggingface_hub warnings
            # We only use HUGGINGFACE_TOKEN
            if "HF_TOKEN" in os.environ:
                os.environ.pop("HF_TOKEN", None)
            
            # Set HUGGINGFACE_TOKEN
            os.environ["HUGGINGFACE_TOKEN"] = token
            
            if login is not None:
                try:
                    login(token=token, add_to_git_credential=False)
                except Exception:
                    pass
            
            if HfFolder is not None:
                try:
                    HfFolder.save_token(token)
                except Exception:
                    pass
            
            logger.info("Loading pyannote speaker diarization pipeline (this may take 1-2 minutes on first run)...")
            
            try:
                pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                logger.info("Pyannote pipeline loaded successfully")
                return pipeline
            except Exception as e:
                logger.debug("Loading without token parameter failed, trying with token parameter", exc_info=e)
                try:
                    return PyannotePipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=token,
                    )
                except TypeError:
                    raise e
        finally:
            # Restore original environment variables
            if original_huggingface_token:
                os.environ["HUGGINGFACE_TOKEN"] = original_huggingface_token
            elif "HUGGINGFACE_TOKEN" in os.environ:
                os.environ.pop("HUGGINGFACE_TOKEN", None)
            
            # Restore HF_TOKEN if it was originally set (though we prefer it not to be)
            if original_hf_token:
                os.environ["HF_TOKEN"] = original_hf_token
            elif "HF_TOKEN" in os.environ:
                # Don't restore if it wasn't originally there - keep it removed
                pass


diarize_audio = diarize_audio_heuristic

