"""OS-level scheduled task management for VSort.

Creates and removes scheduled tasks using `cron` on Linux/macOS
and `schtasks` on Windows.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm

from config import AppConfig, SortDirectory

logger = logging.getLogger(__name__)
console = Console()

TASK_NAME = "VSort"


# ── Cron helpers (Linux / macOS) ─────────────────────────────────────


def _cron_interval_expr(interval: str) -> str:
    """Map a human interval name to a cron expression."""
    mapping = {
        "hourly": "0 * * * *",
        "daily": "0 3 * * *",       # 3 AM daily
        "weekly": "0 3 * * 0",      # 3 AM every Sunday
        "monthly": "0 3 1 * *",     # 3 AM on the 1st of each month
    }
    return mapping.get(interval.lower(), "0 3 * * *")


def _cron_entry(interval: str, python_path: str, script_path: str) -> str:
    """Build a full crontab line."""
    expr = _cron_interval_expr(interval)
    return f"{expr} {python_path} {script_path} --sort >/dev/null 2>&1\n"


def _cron_install(interval: str) -> bool:
    """Install a cron entry for the sorter. Returns True on success."""
    python_path = sys.executable
    script_path = str(Path(__file__).parent / "vsort.py")
    entry = _cron_entry(interval, python_path, script_path)

    # Read existing crontab
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        existing = result.stdout if result.returncode == 0 else ""
    except FileNotFoundError:
        console.print("[red]crontab not found on this system.[/]")
        return False

    # Avoid duplicates (check both new and legacy names)
    if "VSort" in existing or "AIFileSorter" in existing or script_path in existing:
        logger.info("Cron entry already exists; skipping.")
        return True

    new_crontab = existing + f"# VSort\n{entry}"

    try:
        proc = subprocess.run(
            ["crontab", "-"],
            input=new_crontab,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            logger.error("crontab install failed: %s", proc.stderr)
            console.print(f"[red]Failed to install cron: {proc.stderr}[/]")
            return False
    except Exception as exc:
        logger.error("crontab install error: %s", exc)
        return False

    logger.info("Cron entry installed.")
    return True


def _cron_remove() -> bool:
    """Remove the VSort cron entry (also removes legacy AIFileSorter entries)."""
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        existing = result.stdout if result.returncode == 0 else ""
    except FileNotFoundError:
        return False

    lines = existing.splitlines(keepends=True)
    filtered = [
        ln for ln in lines
        if "VSort" not in ln
        and "AIFileSorter" not in ln
        and "vsort.py" not in ln
        and "sorter.py" not in ln
    ]
    new_crontab = "".join(filtered)

    try:
        subprocess.run(
            ["crontab", "-"],
            input=new_crontab,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        logger.error("crontab remove error: %s", exc)
        return False

    return True


def _cron_exists() -> bool:
    """Check if the VSort cron entry is present (also matches legacy name)."""
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return ("vsort.py" in result.stdout or "VSort" in result.stdout
                or "sorter.py" in result.stdout or "AIFileSorter" in result.stdout)
    except FileNotFoundError:
        return False


# ── Schtasks helpers (Windows) ────────────────────────────────────────


def _schtasks_interval_flag(interval: str) -> str:
    """Map interval to schtasks /SC value."""
    mapping = {
        "hourly": "HOURLY",
        "daily": "DAILY",
        "weekly": "WEEKLY",
        "monthly": "MONTHLY",
    }
    return mapping.get(interval.lower(), "DAILY")


def _schtasks_install(interval: str) -> bool:
    """Install a Windows scheduled task. Returns True on success."""
    python_path = sys.executable
    script_path = str(Path(__file__).parent / "vsort.py")
    sc_flag = _schtasks_interval_flag(interval)

    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", f'"{python_path}" "{script_path}" --sort',
        "/SC", sc_flag,
        "/ST", "03:00",
        "/F",   # force overwrite
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error("schtasks create failed: %s", result.stderr)
            console.print(f"[red]schtasks error: {result.stderr}[/]")
            return False
    except FileNotFoundError:
        console.print("[red]schtasks not found.[/]")
        return False
    except Exception as exc:
        logger.error("schtasks error: %s", exc)
        return False

    logger.info("schtasks entry installed.")
    return True


def _schtasks_remove() -> bool:
    """Remove the Windows scheduled task."""
    try:
        result = subprocess.run(
            ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _schtasks_exists() -> bool:
    """Check if the scheduled task exists on Windows."""
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── Unified public API ────────────────────────────────────────────────


def _is_windows() -> bool:
    return platform.system() == "Windows"


def install_scheduled_task(interval: str) -> bool:
    """Install an OS-level scheduled task for periodic sorting."""
    if _is_windows():
        return _schtasks_install(interval)
    else:
        return _cron_install(interval)


def remove_scheduled_task() -> bool:
    """Remove the OS-level scheduled task."""
    if _is_windows():
        return _schtasks_remove()
    else:
        return _cron_remove()


def scheduled_task_exists() -> bool:
    """Return True if the scheduled task is currently installed."""
    if _is_windows():
        return _schtasks_exists()
    else:
        return _cron_exists()


def setup_scheduling(cfg: AppConfig) -> None:
    """Set up scheduled tasks for all periodic directories in config."""
    dirs = cfg.get_sort_directories()
    periodic = [d for d in dirs if d.schedule == "periodic"]
    if not periodic:
        logger.info("No periodic schedules to set up.")
        return

    # All periodic dirs share the same interval (use the first one)
    interval = periodic[0].schedule_interval
    console.print(f"\n[bold cyan]Setting up scheduled task ({interval})...[/]")

    if install_scheduled_task(interval):
        console.print("[green]✓ Scheduled task installed.[/]")
    else:
        console.print("[red]✗ Failed to install scheduled task.[/]")


def verify_scheduling(cfg: AppConfig) -> None:
    """Verify scheduled task exists if config says periodic; install if missing."""
    dirs = cfg.get_sort_directories()
    periodic = [d for d in dirs if d.schedule == "periodic"]
    if not periodic:
        return

    if scheduled_task_exists():
        console.print("[green]✓ Scheduled task is active.[/]")
    else:
        console.print("[yellow]⚠ Scheduled task not found. Reinstalling...[/]")
        interval = periodic[0].schedule_interval
        if install_scheduled_task(interval):
            console.print("[green]✓ Reinstalled scheduled task.[/]")
        else:
            console.print("[red]✗ Could not reinstall scheduled task.[/]")
