"""Media analysis helpers for VSort.

Handles image, video, and GIF frame extraction for vision-based sorting.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"}
GIF_EXTENSION = {".gif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".avif"}
AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".aiff", ".alac"}


def extract_video_frames(video_path: Path, num_frames: int = 3) -> List[Path]:
    """Extract N frames from a video using ffmpeg at strategic positions.

    Grabs frames at 10%, 50%, and 90% of the video duration.
    Returns list of temporary image file paths.
    """
    if not video_path.exists():
        logger.warning("Video file not found: %s", video_path)
        return []

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning("ffmpeg not found; cannot extract video frames from %s", video_path)
        return []

    # Get video duration using ffprobe
    duration = _get_video_duration(video_path)
    if duration is None or duration <= 0:
        logger.warning("Could not determine duration for %s", video_path)
        return []

    # Calculate timestamps at strategic positions
    positions = [0.1, 0.5, 0.9]
    if num_frames < 3:
        positions = positions[:num_frames]
    elif num_frames > 3:
        # Add more evenly spaced positions
        step = 1.0 / (num_frames + 1)
        positions = [step * (i + 1) for i in range(num_frames)]

    timestamps = [duration * p for p in positions]

    # Create temp directory for frames
    tmp_dir = tempfile.mkdtemp(prefix="ai_sort_video_")
    frame_paths: List[Path] = []

    for i, ts in enumerate(timestamps):
        out_path = Path(tmp_dir) / f"frame_{i:03d}.jpg"
        try:
            cmd = [
                ffmpeg_path,
                "-ss", f"{ts:.2f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                str(out_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and out_path.exists():
                frame_paths.append(out_path)
            else:
                logger.warning("ffmpeg frame extraction failed at %.2fs for %s: %s",
                               ts, video_path, result.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out extracting frame from %s", video_path)
        except Exception as exc:
            logger.warning("Error extracting frame from %s: %s", video_path, exc)

    return frame_paths


def _get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        # Try ffmpeg itself as a fallback (parse stderr)
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return None
        try:
            cmd = [ffmpeg_path, "-i", str(video_path), "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            # Parse "Duration: HH:MM:SS.mm" from stderr
            import re
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
            if match:
                h, m, s = match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass
        return None

    try:
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception as exc:
        logger.debug("ffprobe failed for %s: %s", video_path, exc)

    return None


def get_video_duration_str(video_path: Path) -> str:
    """Return a human-readable duration string for a video, or empty string."""
    dur = _get_video_duration(video_path)
    if dur is None:
        return ""
    mins, secs = divmod(int(dur), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h{mins}m{secs}s"
    return f"{mins}m{secs}s"


def extract_gif_frames(gif_path: Path, num_frames: int = 3) -> List[Path]:
    """Extract frames from a GIF using Pillow.

    Grabs frames at evenly spaced positions.
    Returns list of temporary image file paths.
    """
    if not gif_path.exists():
        logger.warning("GIF file not found: %s", gif_path)
        return []

    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; cannot extract GIF frames from %s", gif_path)
        return []

    tmp_dir = tempfile.mkdtemp(prefix="ai_sort_gif_")
    frame_paths: List[Path] = []

    try:
        img = Image.open(gif_path)
        n_frames = getattr(img, "n_frames", 1)

        if n_frames <= 1:
            # Single-frame GIF (basically an image)
            out_path = Path(tmp_dir) / "frame_000.jpg"
            img.save(out_path, "JPEG")
            frame_paths.append(out_path)
        else:
            # Select evenly spaced frame indices
            indices = [int(i * (n_frames - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [0]
            for i, idx in enumerate(indices):
                try:
                    img.seek(idx)
                    frame = img.convert("RGB")
                    out_path = Path(tmp_dir) / f"frame_{i:03d}.jpg"
                    frame.save(out_path, "JPEG")
                    frame_paths.append(out_path)
                except Exception as exc:
                    logger.warning("Failed to extract GIF frame %d from %s: %s", idx, gif_path, exc)
    except Exception as exc:
        logger.warning("Failed to open GIF %s: %s", gif_path, exc)

    return frame_paths


def resize_image_for_vision(image_path: Path, max_size: int = 512) -> bytes:
    """Resize an image and return as JPEG bytes for base64 encoding.

    Maintains aspect ratio, max dimension = max_size.
    Returns JPEG bytes ready for base64.b64encode().
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for image resizing. Install with: pip install Pillow")

    img = Image.open(image_path)

    # Convert to RGB if necessary (e.g. RGBA, P mode)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def get_image_metadata(path: Path) -> str:
    """Extract EXIF/metadata from image files using Pillow.
    Returns a short summary string.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return f"[image, no metadata available]"

    parts: List[str] = []

    try:
        img = Image.open(path)
        parts.append(f"{img.width}x{img.height}")
        parts.append(f"mode={img.mode}")

        # Try to get EXIF data
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ("CameraMake", "Make"):
                    parts.append(f"make={value}")
                elif tag in ("CameraModel", "Model"):
                    parts.append(f"model={value}")
                elif tag == "DateTime":
                    parts.append(f"date={value}")
                elif tag in ("GPSInfo",):
                    # Skip GPS for privacy
                    parts.append("has-gps")
                # Only include a few key fields
                if len(parts) > 6:
                    break
    except Exception as exc:
        logger.debug("Could not read image metadata for %s: %s", path, exc)
        return f"[image, metadata unavailable]"

    return " ".join(parts) if parts else "[image]"


def get_audio_metadata(path: Path) -> str:
    """Extract metadata from audio files using mutagen.
    Returns a short summary string (artist, album, genre, duration).
    If mutagen is not available, return basic info from filename/size.
    """
    try:
        import mutagen
    except ImportError:
        # Fallback: basic info
        size = path.stat().st_size if path.exists() else 0
        return f"[audio, {size} bytes, no metadata]"

    parts: List[str] = []

    try:
        mf = mutagen.File(str(path))
        if mf is None:
            return f"[audio, unrecognized format]"

        # Duration
        if hasattr(mf, "info") and mf.info is not None:
            duration = getattr(mf.info, "length", None)
            if duration is not None:
                mins, secs = divmod(int(duration), 60)
                parts.append(f"{mins}m{secs}s")

        # Tags
        tags = getattr(mf, "tags", None)
        if tags:
            # Try common tag keys (vary by format)
            for key_candidates in [
                ("artist", ["TPE1", "\xa9ART", "ARTIST", "author"]),
                ("album", ["TALB", "\xa9alb", "ALBUM", "album"]),
                ("genre", ["TCON", "\xa9gen", "GENRE", "genre"]),
                ("title", ["TIT2", "\xa9nam", "TITLE", "title"]),
            ]:
                label, keys = key_candidates
                for k in keys:
                    val = tags.get(k)
                    if val:
                        text = str(val[0]) if isinstance(val, list) else str(val)
                        if text.strip():
                            parts.append(f"{label}={text.strip()}")
                        break

    except Exception as exc:
        logger.debug("Could not read audio metadata for %s: %s", path, exc)
        size = path.stat().st_size if path.exists() else 0
        return f"[audio, {size} bytes, metadata read error]"

    return " ".join(parts) if parts else "[audio]"
