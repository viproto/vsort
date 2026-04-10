"""File-move parser and executor for VSort.

Takes the SLM's {filename: category} mapping, creates category
subdirectories inside each target directory, moves files, and
validates the results.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class MoveResult:
    """Outcome of a single file move."""
    src: str
    dst: str
    category: str
    success: bool
    error: Optional[str] = None


@dataclass
class DirectoryResult:
    """Outcome of sorting one directory."""
    directory: str
    moves: List[MoveResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.moves)

    @property
    def succeeded(self) -> int:
        return sum(1 for m in self.moves if m.success)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded


# ── File classification ───────────────────────────────────────────────

# Extensions commonly indicating text / readable files
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h",
    ".java", ".go", ".rs", ".rb", ".php", ".sh", ".bash",
    ".html", ".htm", ".css", ".scss", ".less",
    ".ini", ".cfg", ".conf", ".toml", ".env",
    ".log", ".sql", ".r", ".tex", ".bib",
    ".rst", ".adoc", ".org",
}

# Size limit for reading text content (bytes)
DEFAULT_TEXT_SIZE_LIMIT = 50 * 1024  # 50 KB

# Maximum characters of content to include in the SLM prompt
DEFAULT_SAMPLE_CHARS = 2000


def _is_text_file(path: Path) -> bool:
    """Heuristic: is this file likely human-readable text?"""
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    # Fallback: try to read a small chunk and check for null bytes
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"\x00" not in chunk
    except OSError:
        return False


def _is_image_file(path: Path) -> bool:
    """Check if a file is an image by extension."""
    from media import IMAGE_EXTENSIONS
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_video_file(path: Path) -> bool:
    """Check if a file is a video by extension."""
    from media import VIDEO_EXTENSIONS
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _is_gif_file(path: Path) -> bool:
    """Check if a file is a GIF."""
    from media import GIF_EXTENSION
    return path.suffix.lower() in GIF_EXTENSION


def _is_audio_file(path: Path) -> bool:
    """Check if a file is an audio file by extension."""
    from media import AUDIO_EXTENSIONS
    return path.suffix.lower() in AUDIO_EXTENSIONS


def _read_text_sample(path: Path, size_limit: int, sample_chars: int) -> str:
    """Read a sample of a text file's content.

    Returns up to *sample_chars* characters from the beginning of the
    file if it is under *size_limit* bytes.  Otherwise returns an
    empty string.
    """
    try:
        file_size = path.stat().st_size
    except OSError:
        return ""

    if file_size > size_limit:
        return f"[file too large: {file_size} bytes]"

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return f"[unreadable: {exc}]"

    if len(text) > sample_chars:
        return text[:sample_chars] + "\n[...truncated]"
    return text


def _get_audio_description(path: Path) -> str:
    """Get a description string for an audio file using metadata."""
    try:
        from media import get_audio_metadata
        meta = get_audio_metadata(path)
        if meta and meta != "[audio]":
            return f"Audio metadata: {meta}"
    except Exception as exc:
        logger.debug("Audio metadata extraction failed for %s: %s", path, exc)
    return "[audio file]"


def _get_video_description(path: Path) -> str:
    """Get a description string for a video file including duration."""
    parts = ["video file"]
    try:
        from media import get_video_duration_str
        dur = get_video_duration_str(path)
        if dur:
            parts.append(f"duration={dur}")
    except Exception:
        pass
    return ", ".join(parts)


def _get_image_description(path: Path) -> str:
    """Get a description string for an image file using metadata."""
    try:
        from media import get_image_metadata
        meta = get_image_metadata(path)
        if meta and meta != "[image]":
            return f"Image info: {meta}"
    except Exception as exc:
        logger.debug("Image metadata extraction failed for %s: %s", path, exc)
    return "[image file]"


def build_file_descriptions(
    directory: Path,
    size_limit: int = DEFAULT_TEXT_SIZE_LIMIT,
    sample_chars: int = DEFAULT_SAMPLE_CHARS,
    sort_mode: str = "normal",
    mmproj_path: Optional[str] = None,
    only_files: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """Build a text summary of all files in *directory* for the SLM.

    Returns a tuple of (descriptions_text, vision_candidates) where
    vision_candidates is a list of filenames that should be analyzed
    with vision if mmproj is available.

    For each file: name, size, and (for small text files) a content
    sample. Binary files only get name/size/extension.

    If *only_files* is provided, only those filenames are included
    (used for batching large directories).

    Sort modes:
      "rapid": only filenames + extensions + sizes, NO content preview
      "normal": current behavior (truncated content for small text files)
      "full": send full content for text files (up to larger limit)
    """
    lines: List[str] = []
    vision_candidates: List[str] = []

    # Convert only_files to a set for O(1) lookups
    only_files_set: Optional[Set[str]] = set(only_files) if only_files else None

    # Vision is available if mmproj is configured and the file exists
    vision_available = bool(mmproj_path) and Path(mmproj_path).exists() if mmproj_path else False

    try:
        items = sorted(directory.iterdir())
    except PermissionError:
        logger.warning("Cannot list %s", directory)
        return f"[cannot list directory: {directory}]", vision_candidates

    for item in items:
        if item.is_dir():
            continue
        if item.name.startswith("."):
            continue  # skip hidden files

        # If only_files filter is active, skip files not in the list
        if only_files_set is not None and item.name not in only_files_set:
            continue

        try:
            size = item.stat().st_size
        except OSError:
            size = 0

        line = f"- {item.name} (size: {size} bytes)"

        is_image = _is_image_file(item) or _is_gif_file(item)
        is_video = _is_video_file(item)
        is_audio = _is_audio_file(item)

        # Mark vision candidates
        if (is_image or is_video) and vision_available:
            line += " [VISION_CANDIDATE]"
            vision_candidates.append(item.name)

        if sort_mode == "rapid":
            # Rapid mode: just filename, extension, size — no content
            if is_image:
                line += f"\n  {_get_image_description(item)}"
            elif is_video:
                line += f"\n  {_get_video_description(item)}"
            elif is_audio:
                line += f"\n  {_get_audio_description(item)}"
            elif _is_text_file(item):
                ext = item.suffix or "none"
                line += f"\n  [text file, extension: {ext}]"
            else:
                line += f"\n  [binary file, extension: {item.suffix or 'none'}]"

        elif sort_mode == "full":
            # Full mode: larger limits, more content
            if is_image:
                line += f"\n  {_get_image_description(item)}"
            elif is_video:
                line += f"\n  {_get_video_description(item)}"
            elif is_audio:
                line += f"\n  {_get_audio_description(item)}"
            elif _is_text_file(item):
                # Use the passed-in (full-mode) limits
                sample = _read_text_sample(item, size_limit, sample_chars)
                if sample:
                    line += f"\n  Content:\n  {sample.replace(chr(10), chr(10) + '  ')}"
            else:
                line += f"\n  [binary file, extension: {item.suffix or 'none'}]"

        else:
            # Normal mode (default)
            if is_image:
                line += f"\n  {_get_image_description(item)}"
            elif is_video:
                line += f"\n  {_get_video_description(item)}"
            elif is_audio:
                line += f"\n  {_get_audio_description(item)}"
            elif _is_text_file(item):
                sample = _read_text_sample(item, size_limit, sample_chars)
                if sample:
                    line += f"\n  Content preview:\n  {sample.replace(chr(10), chr(10) + '  ')}"
            else:
                line += f"\n  [binary file, extension: {item.suffix or 'none'}]"

        lines.append(line)

    if not lines:
        return "[no files found in directory]", vision_candidates

    return "\n".join(lines), vision_candidates


# ── Move execution ────────────────────────────────────────────────────


def _fuzzy_match_filename(directory: Path, candidate: str) -> Optional[Path]:
    """Try to find the real file that the SLM meant when it mangled the name.

    The SLM often: drops extensions, replaces underscores with hyphens,
    truncates long names, changes case. We try several strategies to
    recover the actual file.
    """
    # 1. Exact match (fast path)
    exact = directory / candidate
    if exact.exists() and exact.is_file():
        return exact

    # Build a list of actual files once (cached by caller, but we do it here
    # for the standalone case — the caller should pre-build if calling in a loop)
    actual_files = {
        f.name: f
        for f in directory.iterdir()
        if f.is_file() and not f.name.startswith(".")
    }

    # 2. Case-insensitive match
    lower_candidate = candidate.lower()
    for name, path in actual_files.items():
        if name.lower() == lower_candidate:
            return path

    # 3. Normalize: replace hyphens with underscores and vice-versa
    norm_candidate = lower_candidate.replace("-", "_")
    for name, path in actual_files.items():
        if name.lower().replace("-", "_") == norm_candidate:
            return path

    # 4. Candidate missing extension or has extension mangled into name
    #    e.g. "Illu-PNG" -> "Illu.png", "03222-mp4" -> "03222.mp4"
    base = candidate.rsplit(".", 1)[0] if "." in candidate else candidate
    base_lower = base.lower().replace("-", "_")
    for name, path in actual_files.items():
        name_lower = name.lower().replace("-", "_")
        # Check if actual filename (without ext) matches candidate base
        name_base = name_lower.rsplit(".", 1)[0]
        if name_base == base_lower or name_base == lower_candidate.replace("-", "_"):
            return path

    # 4b. SLM mangled extension into name: "Illu-PNG" -> "Illu.png"
    #     Try stripping known extensions from the candidate suffix
    ext_map = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpeg", "mp4": ".mp4",
               "mp3": ".mp3", "gif": ".gif", "webp": ".webp", "zip": ".zip",
               "exe": ".exe", "pdf": ".pdf", "iso": ".iso", "json": ".json"}
    for ext_suffix, real_ext in ext_map.items():
        # Candidate ends with -EXT or _EXT
        for sep in ("-", "_"):
            suffix = sep + ext_suffix
            if lower_candidate.endswith(suffix):
                candidate_base = lower_candidate[:-len(suffix)].replace("-", "_")
                for name, path in actual_files.items():
                    name_base = name.lower().replace("-", "_").rsplit(".", 1)[0]
                    name_ext = name.lower().rsplit(".", 1)[-1] if "." in name.lower() else ""
                    if name_base == candidate_base and name_ext == ext_suffix:
                        return path

    # 5. Prefix match — SLM may have truncated a long filename
    prefix = lower_candidate.replace("-", "_")[:20]
    if len(prefix) >= 10:  # only try if prefix is long enough to be unique
        matches = []
        for name, path in actual_files.items():
            name_norm = name.lower().replace("-", "_")
            if name_norm.startswith(prefix):
                matches.append(path)
        if len(matches) == 1:
            return matches[0]

    # 6. Similarity match using longest common subsequence ratio
    best_path = None
    best_score = 0.0
    for name, path in actual_files.items():
        score = _name_similarity(candidate, name)
        if score > best_score:
            best_score = score
            best_path = path
    if best_score >= 0.6 and best_path is not None:
        return best_path

    return None


def _name_similarity(a: str, b: str) -> float:
    """Compute a simple similarity score between two filenames.

    Uses the ratio of the longest common prefix + suffix length
    to the average length. Quick and dirty but good enough for
    SLM filename mangling.
    """
    a_low = a.lower().replace("-", "_")
    b_low = b.lower().replace("-", "_")

    # Strip extensions for comparison
    a_base = a_low.rsplit(".", 1)[0] if "." in a_low else a_low
    b_base = b_low.rsplit(".", 1)[0] if "." in b_low else b_low

    if not a_base or not b_base:
        return 0.0

    # Simple character-level overlap
    common = sum(1 for ca, cb in zip(a_base, b_base) if ca == cb)
    max_len = max(len(a_base), len(b_base))
    if max_len == 0:
        return 0.0

    # Penalise length difference
    len_penalty = min(len(a_base), len(b_base)) / max_len

    return (common / max_len) * len_penalty


def execute_sorts(
    directory: Path,
    sorts: Dict[str, str],
    all_filenames: Optional[Set[str]] = None,
) -> DirectoryResult:
    """Move files in *directory* into category subdirectories.

    *sorts* maps filenames (relative to *directory*) to category names.
    Categories become subdirectories of *directory*.

    If *all_filenames* is provided, any files NOT in *sorts* will be
    logged as unaccounted. The function also attempts fuzzy matching
    when the SLM returns a mangled filename that doesn't exactly exist.

    Returns a DirectoryResult with details of every attempted move.
    """
    result = DirectoryResult(directory=str(directory))

    # Pre-build the actual file set for fuzzy matching
    actual_files = {
        f.name: f
        for f in directory.iterdir()
        if f.is_file() and not f.name.startswith(".")
    }

    for filename, category in sorts.items():
        src = directory / filename

        # Validate source exists — try fuzzy match if not
        if not src.exists() or src.is_dir():
            matched = _fuzzy_match_filename(directory, filename)
            if matched is not None:
                logger.info(
                    "Fuzzy matched SLM name %r -> actual %r",
                    filename, matched.name,
                )
                src = matched
            else:
                result.moves.append(MoveResult(
                    src=str(src),
                    dst="",
                    category=category,
                    success=False,
                    error="source file does not exist",
                ))
                logger.warning("Source file missing (no fuzzy match): %s", src)
                continue

        # Validate source is a file (not a directory)
        if src.is_dir():
            result.moves.append(MoveResult(
                src=str(src),
                dst="",
                category=category,
                success=False,
                error="source is a directory, not a file",
            ))
            continue

        # Sanitise category name
        safe_category = _sanitize_category(category)
        if not safe_category:
            result.moves.append(MoveResult(
                src=str(src),
                dst="",
                category=category,
                success=False,
                error="invalid category name after sanitisation",
            ))
            continue

        # Create category directory
        cat_dir = directory / safe_category
        try:
            cat_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            result.moves.append(MoveResult(
                src=str(src),
                dst=str(cat_dir / filename),
                category=category,
                success=False,
                error=f"cannot create category dir: {exc}",
            ))
            continue

        # Move file
        dst = cat_dir / src.name
        try:
            shutil.move(str(src), str(dst))
            result.moves.append(MoveResult(
                src=str(src),
                dst=str(dst),
                category=category,
                success=True,
            ))
            logger.info("Moved %s -> %s", src, dst)
        except OSError as exc:
            result.moves.append(MoveResult(
                src=str(src),
                dst=str(dst),
                category=category,
                success=False,
                error=str(exc),
            ))
            logger.error("Failed to move %s: %s", src, exc)

    # Log unaccounted files (real files that the SLM didn't categorise)
    sorted_sources = {Path(m.src).name for m in result.moves if m.success}
    unaccounted = [name for name in actual_files if name not in sorted_sources]
    if unaccounted:
        logger.warning(
            "%d file(s) unaccounted (not returned by SLM): %s",
            len(unaccounted),
            ", ".join(unaccounted[:20]),
        )
        for name in unaccounted:
            result.moves.append(MoveResult(
                src=str(directory / name),
                dst="",
                category="(unaccounted)",
                success=False,
                error="not returned by SLM — skipped",
            ))

    return result


def _sanitize_category(name: str) -> str:
    """Sanitise a category name for use as a directory name.

    Strips leading/trailing whitespace, collapses internal whitespace,
    removes characters that are problematic on any OS.
    """
    import re
    name = name.strip()
    name = re.sub(r"\s+", "-", name)  # spaces -> hyphens
    # Remove anything not alphanumeric, hyphen, or underscore
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name


# ── Validation & display ──────────────────────────────────────────────


def validate_results(results: List[DirectoryResult]) -> bool:
    """Return True if all moves succeeded and all expected files are in place."""
    all_ok = True
    for dr in results:
        for move in dr.moves:
            if move.success:
                dst_path = Path(move.dst)
                if not dst_path.exists():
                    logger.error("Validation fail: %s does not exist", dst_path)
                    all_ok = False
            else:
                all_ok = False
    return all_ok


def display_results(results: List[DirectoryResult]) -> None:
    """Print a rich table summarising the sort results."""
    for dr in results:
        table = Table(
            title=f"Sort Results: {dr.directory}",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("Destination", style="green")
        table.add_column("Status", justify="center")

        for move in dr.moves:
            status = "[green]✓[/]" if move.success else f"[red]✗ {move.error or ''}[/]"
            table.add_row(
                Path(move.src).name,
                move.category,
                move.dst if move.success else "—",
                status,
            )

        console.print(table)
        console.print(
            f"  [bold]{dr.succeeded}/{dr.total}[/] files moved successfully."
            + (f"  [red]{dr.failed} failed.[/]" if dr.failed else "")
            + "\n"
        )
