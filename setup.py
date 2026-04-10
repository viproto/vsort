"""First-time setup for VSort.

Downloads the llama-server binary from llama.cpp GitHub releases and
the GGUF model (plus mmproj for vision support) from HuggingFace.
Verifies the setup is functional.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from config import (
    AppConfig, CONFIG_DIR, LLAMA_CPP_REPO, MMPROJ_FILENAME,
    get_model_repo_id, get_model_filename_glob,
)

logger = logging.getLogger(__name__)
console = Console()

# ── Platform detection ────────────────────────────────────────────────


def _detect_release_asset_name(tag: str) -> Tuple[str, str]:
    """Return (asset_filename, archive_format) for the current OS/arch.

    llama.cpp releases use archives with this naming convention:
      llama-b{tag}-bin-ubuntu-x64.tar.gz
      llama-b{tag}-bin-ubuntu-arm64.tar.gz
      llama-b{tag}-bin-macos-arm64.tar.gz
      llama-b{tag}-bin-macos-x64.tar.gz
      llama-b{tag}-bin-win-cpu-x64.zip
      llama-b{tag}-bin-win-cpu-arm64.zip

    Returns (asset_filename, fmt) where fmt is "tar.gz" or "zip".
    """
    syst = platform.system().lower()
    arch = platform.machine().lower()

    if syst == "linux":
        if arch in ("x86_64", "amd64"):
            return f"llama-{tag}-bin-ubuntu-x64.tar.gz", "tar.gz"
        elif arch in ("aarch64", "arm64"):
            return f"llama-{tag}-bin-ubuntu-arm64.tar.gz", "tar.gz"
    elif syst == "darwin":
        if arch in ("arm64",):
            return f"llama-{tag}-bin-macos-arm64.tar.gz", "tar.gz"
        elif arch in ("x86_64", "amd64"):
            return f"llama-{tag}-bin-macos-x64.tar.gz", "tar.gz"
    elif syst == "windows":
        if arch in ("x86_64", "amd64"):
            return f"llama-{tag}-bin-win-cpu-x64.zip", "zip"
        elif arch in ("arm64",):
            return f"llama-{tag}-bin-win-cpu-arm64.zip", "zip"

    raise RuntimeError(
        f"Unsupported platform: system={syst}, arch={arch}. "
        "Please download llama-server manually from "
        "https://github.com/ggml-org/llama.cpp/releases"
    )


def _server_binary_name() -> str:
    """Return the llama-server binary filename for the current OS."""
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def _latest_release_tag() -> str:
    """Fetch the latest release tag from the llama.cpp GitHub repo."""
    api_url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
    try:
        resp = requests.get(api_url, timeout=15)
        resp.raise_for_status()
        tag: str = resp.json()["tag_name"]
        logger.info("Latest llama.cpp release tag: %s", tag)
        return tag
    except Exception as exc:
        logger.warning("Could not fetch latest tag (%s); using default.", exc)
        from config import LLAMA_CPP_RELEASE_TAG
        return LLAMA_CPP_RELEASE_TAG


# ── Download helpers ───────────────────────────────────────────────────


def _download_file(url: str, dest: Path, description: str = "Downloading") -> None:
    """Stream-download *url* to *dest* with a Rich progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress,
        ):
            task = progress.add_task(description, total=total or None)
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    progress.update(task, advance=len(chunk))

        tmp.replace(dest)
        logger.info("Downloaded %s -> %s", url, dest)

    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _make_executable(path: Path) -> None:
    """Add executable permission on Unix-like systems."""
    if platform.system() != "Windows":
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _extract_archive(archive_path: Path, dest_dir: Path, fmt: str) -> Path:
    """Extract the llama-server binary from a downloaded archive.

    Returns the path to the extracted llama-server binary.
    Also extracts all shared libraries (.so/.dylib/.dll) needed by it.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary_name = _server_binary_name()

    # Extract full archive into a staging subdir first
    staging = dest_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(staging, filter="data")
        elif fmt == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(staging)
        else:
            raise ValueError(f"Unknown archive format: {fmt}")

        # Walk the staging tree to find the server binary and shared libs
        server_src = None
        lib_srcs = []

        for root, dirs, files in os.walk(staging):
            for fname in files:
                fpath = Path(root) / fname
                if fname == binary_name:
                    server_src = fpath
                elif fname.endswith(".so") or fname.endswith(".dylib") or \
                     fname.endswith(".dll") or (".so." in fname):
                    lib_srcs.append(fpath)

        if server_src is None:
            # List everything for debug
            all_files = []
            for root, dirs, files in os.walk(staging):
                for fname in files:
                    all_files.append(str(Path(root) / fname))
            raise FileNotFoundError(
                f"{binary_name} not found in archive. "
                f"Files found: {all_files[:30]}"
            )

        # Move server binary + shared libs to dest_dir (flat)
        shutil.move(str(server_src), str(dest_dir / binary_name))
        for lib_src in lib_srcs:
            shutil.move(str(lib_src), str(dest_dir / lib_src.name))

        server_path = dest_dir / binary_name

    finally:
        # Clean up staging dir
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    if not server_path.exists():
        raise FileNotFoundError(f"Extraction failed: {server_path} not found")

    _make_executable(server_path)
    # Also make shared libs executable on Unix
    if platform.system() != "Windows":
        for child in dest_dir.iterdir():
            if child.suffix in (".so", ".dylib") or ".so." in child.name:
                _make_executable(child)

    return server_path


# ── Public API ────────────────────────────────────────────────────────


def download_llama_server(cfg: AppConfig) -> Path:
    """Download the llama-server binary; return its path on disk.

    Downloads the appropriate release archive for the current platform,
    extracts llama-server and its shared libraries into the config dir,
    then cleans up the archive.
    """
    binary_name = _server_binary_name()
    server_dir = CONFIG_DIR / "llama-server"
    expected = server_dir / binary_name

    # Check if already extracted and usable
    if expected.exists() and os.access(expected, os.X_OK if platform.system() != "Windows" else os.R_OK):
        logger.info("llama-server already exists at %s", expected)
        cfg.llama_server_path = str(expected)
        return expected

    tag = _latest_release_tag()
    cfg.llama_cpp_tag = tag

    asset_name, fmt = _detect_release_asset_name(tag)
    url = (
        f"https://github.com/{LLAMA_CPP_REPO}/releases/"
        f"download/{tag}/{asset_name}"
    )

    archive_dest = CONFIG_DIR / asset_name
    console.print(f"\n[bold cyan]Downloading llama.cpp ({tag}, {asset_name})...[/]")

    try:
        _download_file(url, archive_dest, description="llama.cpp archive")
    except requests.HTTPError as exc:
        logger.error("Download failed: %s", exc)
        console.print(f"[red]✗ Failed to download archive: {exc}[/]")
        raise

    # Extract the server binary + shared libraries
    console.print("[dim]Extracting llama-server...[/]")
    try:
        server_path = _extract_archive(archive_dest, server_dir, fmt)
    except Exception as exc:
        console.print(f"[red]✗ Extraction failed: {exc}[/]")
        raise
    finally:
        # Clean up the archive regardless of extraction result
        if archive_dest.exists():
            archive_dest.unlink(missing_ok=True)
            logger.info("Cleaned up archive: %s", archive_dest)

    cfg.llama_server_path = str(server_path)
    console.print(f"[green]✓ llama-server extracted to {server_path}[/]")
    return server_path


def _resolve_model_filename(cfg: AppConfig) -> str:
    """Resolve the glob pattern to an exact filename in the HF repo.

    hf_hub_download requires an exact filename, not a glob.
    We list the repo files and find the matching one.
    Uses cfg.model_variant and cfg.quantization to determine the repo and filename.
    """
    repo_id = get_model_repo_id(cfg.model_variant)
    filename_glob = get_model_filename_glob(cfg.model_variant, cfg.quantization)

    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(repo_id))
    except ImportError:
        from huggingface_hub import HfApi
        api = HfApi()
        files = list(api.list_repo_files(repo_id))

    glob_pat = filename_glob.strip("*")
    for f in files:
        if glob_pat in f and f.endswith(".gguf"):
            logger.info("Resolved model filename: %s", f)
            return f

    # Fallback: try fnmatch
    for f in files:
        if fnmatch.fnmatch(f, filename_glob):
            logger.info("Resolved model filename (fnmatch): %s", f)
            return f

    raise FileNotFoundError(
        f"No file matching '{filename_glob}' found in {repo_id}. "
        "Available files: " + ", ".join(files[:20])
    )


def download_model(cfg: AppConfig) -> Path:
    """Download the GGUF model via huggingface_hub; return local path.

    Skips download if the model file already exists locally.
    Uses cfg.model_variant and cfg.quantization to determine which repo
    and filename to download.
    """
    if cfg.model_path and Path(cfg.model_path).exists():
        logger.info("Model already exists at %s", cfg.model_path)
        return Path(cfg.model_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]huggingface-hub is not installed. Run: pip install huggingface-hub[/]")
        raise

    # Resolve the glob to an exact filename
    exact_filename = _resolve_model_filename(cfg)
    repo_id = get_model_repo_id(cfg.model_variant)

    console.print(f"\n[bold cyan]Downloading model {repo_id} ({exact_filename})...[/]")
    console.print("[dim]This may take a while depending on your connection.[/]")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=exact_filename,
            local_dir=str(CONFIG_DIR / "models"),
        )
    except Exception as exc:
        logger.error("Model download failed: %s", exc)
        console.print(f"[red]✗ Model download failed: {exc}[/]")
        raise

    resolved = Path(local_path)
    cfg.model_path = str(resolved)
    console.print(f"[green]✓ Model downloaded to {resolved}[/]")
    return resolved


def download_mmproj(cfg: AppConfig) -> Path:
    """Download the mmproj (multimodal projector) file for vision support.

    Downloads mmproj-F16.gguf from the same HF repo as the model
    (determined by cfg.model_variant).
    Skips if the file already exists locally.
    """
    models_dir = CONFIG_DIR / "models"
    expected = models_dir / MMPROJ_FILENAME
    repo_id = get_model_repo_id(cfg.model_variant)

    if expected.exists():
        logger.info("mmproj already exists at %s", expected)
        cfg.mmproj_path = str(expected)
        return expected

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]huggingface-hub is not installed. Run: pip install huggingface-hub[/]")
        raise

    console.print(f"\n[bold cyan]Downloading mmproj ({MMPROJ_FILENAME}) for vision support from {repo_id}...[/]")
    console.print("[dim]This enables image/video analysis capabilities.[/]")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=MMPROJ_FILENAME,
            local_dir=str(models_dir),
        )
    except Exception as exc:
        logger.warning("mmproj download failed: %s", exc)
        console.print(f"[yellow]⚠ mmproj download failed: {exc}[/]")
        console.print("[dim]Vision features will be unavailable. You can retry later.[/]")
        # Don't raise — mmproj is optional
        return Path("")

    resolved = Path(local_path)
    cfg.mmproj_path = str(resolved)
    console.print(f"[green]✓ mmproj downloaded to {resolved}[/]")
    return resolved


def verify_setup(cfg: AppConfig) -> bool:
    """Return True if llama-server binary and model file exist."""
    server_ok = bool(cfg.llama_server_path) and Path(cfg.llama_server_path).exists()
    model_ok = bool(cfg.model_path) and Path(cfg.model_path).exists()

    if not server_ok:
        logger.warning("llama-server binary missing or not configured.")
    if not model_ok:
        logger.warning("Model file missing or not configured.")

    # mmproj is optional, just log
    if cfg.mmproj_path and not Path(cfg.mmproj_path).exists():
        logger.warning("mmproj configured but file missing. Vision features may not work.")
        cfg.mmproj_path = None

    return server_ok and model_ok


def run_setup(cfg: AppConfig) -> None:
    """Execute the full first-time setup flow.

    Downloads llama-server, downloads the model, downloads mmproj,
    updates config.
    """
    console.print("\n[bold magenta]═══ First-Time Setup ═══[/]\n")

    with console.status("[bold cyan]Checking for existing setup..."):
        if verify_setup(cfg):
            console.print("[green]✓ Setup already complete — skipping download.[/]")
            cfg.setup_complete = True
            cfg.save()
            return

    # Download llama-server
    try:
        download_llama_server(cfg)
    except Exception as exc:
        console.print(f"[red]✗ Failed to download llama-server: {exc}[/]")
        logger.error("llama-server download error: %s", exc, exc_info=True)
        raise

    # Download model
    try:
        download_model(cfg)
    except Exception as exc:
        console.print(f"[red]✗ Failed to download model: {exc}[/]")
        logger.error("Model download error: %s", exc, exc_info=True)
        raise

    # Download mmproj (optional — failures are non-fatal)
    try:
        download_mmproj(cfg)
    except Exception as exc:
        logger.warning("mmproj download error (non-fatal): %s", exc)
        console.print("[yellow]⚠ mmproj download failed; vision features disabled.[/]")

    # Final check
    if verify_setup(cfg):
        cfg.setup_complete = True
        cfg.save()
        console.print("\n[bold green]✓ Setup complete![/]")
    else:
        console.print("\n[bold red]✗ Setup verification failed. Check logs.[/]")
        cfg.setup_complete = False
        cfg.save()
        raise RuntimeError("Setup verification failed after downloads.")
