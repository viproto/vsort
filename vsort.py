#!/usr/bin/env python3
"""VSort — Main entry point.

Orchestrates the full flow:
  1. Onboarding (first-run only)
  2. First-time setup (download llama-server + model)
  3. SLM prompting (analyse files, get category mappings)
  4. Parsing & moving files
  5. Display results & verify schedule

Usage:
  python vsort.py            # interactive onboarding + sort
  python vsort.py --sort     # non-interactive sort (for scheduled runs)
  python vsort.py --reset    # delete config and start over
  python vsort.py --yolo     # max context, no batching
  python vsort.py --think    # show SLM reasoning
  python vsort.py --dry-run  # preview categorization, don't move files
  python vsort.py --undo     # reverse the last sort
  python vsort.py --rename   # suggest better filenames during sort
"""

from __future__ import annotations

import argparse
import atexit
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# Local modules — imported after argparse so --help stays fast
from config import AppConfig
from onboarding import run_onboarding
from setup import run_setup, verify_setup
from slm import LlamaServer
from parser import (
    DirectoryResult,
    build_file_descriptions,
    display_preview,
    display_results,
    execute_sorts,
    load_latest_manifest,
    save_manifest,
    validate_results,
)
from scheduler import setup_scheduling, verify_scheduling

console = Console()
logger = logging.getLogger(__name__)


# ── Logging ───────────────────────────────────────────────────────────


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )


# ── Argument parsing ──────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsort",
        description="Sort files in directories by content using a local SLM.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Non-interactive sort mode (for scheduled/cron runs).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete saved config and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="YOLO mode: max context (131072), no batching, send all files in one go. "
             "May use significant RAM and be slower but gives the SLM full visibility.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Show the SLM's chain-of-thought reasoning as it sorts. "
             "Lets you see WHY it puts each file where it does.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview categorization without moving files. Shows what would "
             "happen, then exits — nothing is changed on disk.",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the last sort: reverse all file moves from the most recent run.",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Ask the AI to suggest better filenames when sorting. "
             "e.g. IMG_20250410_092340.jpg -> Beach-Sunset-2025-04-10.jpg",
    )
    return parser


# ── Exclusion helpers ──────────────────────────────────────────────────


def _is_excluded(filename: str, patterns: List[str]) -> bool:
    """Check if a filename matches any exclusion glob pattern."""
    import fnmatch
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False


# ── Core orchestration ────────────────────────────────────────────────


def _do_sorting(cfg: AppConfig, server: LlamaServer) -> List[DirectoryResult]:
    """Run the SLM-based sorting on all configured directories.

    Returns a list of DirectoryResult objects.
    """
    # Maximum files per SLM batch — too many and the model drops/truncates
    # In yolo mode, we skip batching entirely
    BATCH_SIZE = 25 if not cfg.yolo else 999999

    sort_dirs = cfg.get_sort_directories()
    if not sort_dirs:
        console.print("[yellow]No directories configured. Run without --sort to set up.[/]")
        return []

    all_results: List[DirectoryResult] = []

    for sd in sort_dirs:
        dir_path = Path(sd.path)
        if not dir_path.is_dir():
            console.print(f"[yellow]⚠ Directory not found: {dir_path} — skipping.[/]")
            continue

        console.print(f"\n[bold cyan]Sorting: {dir_path}[/]")

        # Collect the actual file list first (for batching and unaccounted tracking)
        all_filenames: Set[str] = set()
        for item in sorted(dir_path.iterdir()):
            if item.is_file() and not item.name.startswith("."):
                # Apply exclusion patterns
                if _is_excluded(item.name, cfg.exclusions):
                    logger.debug("Excluding: %s (matches exclusion pattern)", item.name)
                    continue
                all_filenames.add(item.name)

        if not all_filenames:
            console.print(f"  [dim]No files to sort in {dir_path}.[/]")
            continue

        total_files = len(all_filenames)
        num_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

        if cfg.yolo:
            console.print(
                f"  [bold yellow]YOLO: sending all {total_files} files in one pass.[/]"
            )
        elif num_batches > 1:
            console.print(
                f"  [dim]{total_files} files — splitting into {num_batches} batches of ≤{BATCH_SIZE}.[/]"
            )

        # Process in batches
        merged_sorts: Dict[str, str] = {}
        all_vision_candidates: List[str] = []

        for batch_idx in range(num_batches):
            file_list = sorted(all_filenames)[
                batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE
            ]

            # Build descriptions for just this batch
            descriptions, vision_candidates = build_file_descriptions(
                dir_path,
                size_limit=cfg.effective_text_size_limit(),
                sample_chars=cfg.effective_sample_chars(),
                sort_mode=cfg.sort_mode,
                mmproj_path=cfg.mmproj_path,
                only_files=file_list,
            )
            all_vision_candidates.extend(vision_candidates)

            if num_batches > 1:
                console.print(
                    f"  [dim]Batch {batch_idx + 1}/{num_batches} ({len(file_list)} files)...[/]"
                )

            logger.debug("Batch %d descriptions for %s:\n%s", batch_idx, dir_path, descriptions)

            # Ask the SLM for a categorisation (with retry logic)
            # Only pass vision_candidates on the last batch to avoid
            # redundant vision passes
            is_last = batch_idx == num_batches - 1
            vc = all_vision_candidates if is_last else None

            try:
                sorts = server.sort_files_with_retry(
                    descriptions,
                    str(dir_path),
                    max_retries=2,
                    vision_candidates=vc,
                )
            except Exception as exc:
                logger.error("SLM categorisation failed for %s batch %d: %s", dir_path, batch_idx, exc)
                console.print(f"[red]✗ SLM error for batch {batch_idx + 1}: {exc}[/]")
                continue

            if sorts:
                merged_sorts.update(sorts)
                logger.info("Batch %d: %d files categorised", batch_idx, len(sorts))

                # Show model reasoning if --think is active
                if cfg.think and server.last_reasoning:
                    console.print(Panel(
                        f"[dim]{server.last_reasoning[:3000]}[/]",
                        title=f"[bold blue]SLM Thinking (batch {batch_idx + 1})[/]",
                        border_style="blue",
                        padding=(1, 2),
                    ))

        if not merged_sorts:
            console.print(f"[yellow]No sort mapping returned for {dir_path}.[/]")
            continue

        logger.info("SLM sorts for %s: %d files categorised", dir_path, len(merged_sorts))
        console.print(f"  SLM categorised [bold]{len(merged_sorts)}[/] of {total_files} files.")

        # Dry-run: show preview, skip execution
        if cfg.dry_run:
            display_preview(dir_path, merged_sorts, renames=server.last_renames)
            console.print(f"  [bold yellow][DRY RUN] No files were moved.[/]")
            continue

        # Execute the moves (pass all_filenames for unaccounted tracking)
        result = execute_sorts(
            dir_path, merged_sorts,
            all_filenames=all_filenames,
            renames=server.last_renames,
        )
        all_results.append(result)

        # Save manifest for --undo (only for real sorts, not dry-run)
        if not cfg.dry_run:
            save_manifest(result, renames=server.last_renames)

    return all_results


def _interactive_sort(cfg: AppConfig) -> None:
    """Interactive flow: onboarding → setup → sort → display."""
    # ── Step 1: Onboarding ────────────────────────────────────────
    if not cfg.initialized:
        run_onboarding(cfg)
        # Re-load after onboarding saves
        cfg = AppConfig.load()
    else:
        console.print("[dim]Already onboarded — using saved config.[/]")

    # ── Step 2: First-time setup ───────────────────────────────────
    if not cfg.setup_complete or not verify_setup(cfg):
        run_setup(cfg)
        cfg = AppConfig.load()
    else:
        console.print("[dim]Setup already complete — skipping downloads.[/]")

    # ── Step 3-4: SLM sorting ─────────────────────────────────────
    with LlamaServer(cfg) as server:
        results = _do_sorting(cfg, server)

    # ── Step 5: Display & verify schedule ──────────────────────────
    if results:
        console.print("\n[bold magenta]═══ Results ═══[/]\n")
        display_results(results)

        if validate_results(results):
            console.print("[bold green]✓ All files sorted successfully.[/]")
        else:
            console.print("[bold yellow]⚠ Some files could not be moved. See table above.[/]")

    # Schedule setup (skip in dry-run)
    if cfg.dry_run:
        console.print("[dim]Dry-run — no scheduled task needed.[/]")
    else:
        dirs = cfg.get_sort_directories()
        if any(d.schedule == "periodic" for d in dirs):
            setup_scheduling(cfg)
            verify_scheduling(cfg)
        else:
            console.print("[dim]One-time sort — no scheduled task needed.[/]")

    if cfg.dry_run:
        console.print("\n[bold yellow]═══ DRY RUN COMPLETE ═══[/]\n"
                      "No files were moved. Run without --dry-run to apply changes.")
    else:
        console.print("\n[bold]Done.[/]")


def _noninteractive_sort(cfg: AppConfig) -> None:
    """Non-interactive sort for cron/schtasks runs."""
    if not cfg.initialized or not cfg.setup_complete:
        logger.error("Not configured. Run interactively first.")
        sys.exit(1)

    if not verify_setup(cfg):
        logger.error("Setup incomplete (missing binary or model). Run interactively.")
        sys.exit(1)

    with LlamaServer(cfg) as server:
        results = _do_sorting(cfg, server)

    if results:
        display_results(results)
        if not validate_results(results):
            logger.warning("Some moves failed.")
            sys.exit(2)


# ── Undo ───────────────────────────────────────────────────────────────


def _undo_last_sort() -> None:
    """Reverse the most recent sort using the saved manifest."""
    manifest = load_latest_manifest()
    if not manifest:
        console.print("[red]No manifest found — nothing to undo.[/]")
        return

    directory = manifest["directory"]
    moves = manifest["moves"]
    dir_path = Path(directory)

    if not dir_path.is_dir():
        console.print(f"[red]Directory no longer exists: {directory}[/]")
        return

    console.print(f"\n[bold cyan]Undoing last sort in {directory}[/]")
    console.print(f"  [dim]Manifest from {manifest['timestamp']} — {len(moves)} moves[/]\n")

    undone = 0
    failed = 0

    for entry in moves:
        dst_rel = entry["dst"]  # relative path like "Category/file.jpg"
        src_name = entry["src"]  # original filename like "file.jpg"
        dst_full = dir_path / dst_rel
        src_full = dir_path / src_name

        # If the file was renamed, restore original name
        if "renamed_to" in entry:
            src_full = dir_path / entry["original_name"]

        if not dst_full.exists():
            console.print(f"  [yellow]⚠ Already moved/missing: {dst_rel}[/]")
            failed += 1
            continue

        # Check for name collision at destination
        if src_full.exists():
            console.print(f"  [yellow]⚠ Name collision, skipping: {src_name}[/]")
            failed += 1
            continue

        try:
            shutil.move(str(dst_full), str(src_full))
            console.print(f"  [green]✓[/] {dst_rel} -> {src_name}")
            undone += 1
        except OSError as exc:
            console.print(f"  [red]✗ Move failed: {dst_rel}: {exc}[/]")
            failed += 1

    # Clean up empty category directories
    for entry in moves:
        dst_rel = entry["dst"]
        cat_dir = dir_path / Path(dst_rel).parts[0]
        if cat_dir.is_dir():
            try:
                # rmdir only works on empty dirs
                cat_dir.rmdir()
                console.print(f"  [dim]Removed empty category dir: {cat_dir.name}[/]")
            except OSError:
                pass  # not empty, that's fine

    console.print(f"\n  [bold green]{undone}[/] files restored, [bold red]{failed}[/] failed.")

    # Remove the used manifest
    from config import MANIFESTS_DIR
    manifests = sorted(MANIFESTS_DIR.glob("*.json"))
    if manifests:
        manifests[-1].unlink(missing_ok=True)
        logger.info("Removed manifest %s", manifests[-1])


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    # --reset: wipe config and exit
    if args.reset:
        from config import CONFIG_FILE, CONFIG_DIR
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
            console.print("[yellow]Config deleted.[/]")
        # Also try to remove the directory if empty
        try:
            CONFIG_DIR.rmdir()
        except OSError:
            pass
        return

    # Load config
    cfg = AppConfig.load()

    # Apply --yolo override
    if args.yolo:
        cfg.yolo = True
        cfg.ctx_size = 131072
        console.print("[bold yellow]YOLO mode: max context (131072), no batching. May use significant RAM.[/]")

    # Apply --think override
    if args.think:
        cfg.think = True
        console.print("[bold blue]Think mode: SLM reasoning will be displayed.[/]")

    # Apply --dry-run override
    if args.dry_run:
        cfg.dry_run = True
        console.print("[bold yellow]DRY RUN: files will NOT be moved.[/]")

    # Apply --rename override
    if args.rename:
        cfg.rename = True
        console.print("[bold cyan]Rename mode: AI will suggest better filenames.[/]")

    # --undo: reverse the last sort and exit
    if args.undo:
        _undo_last_sort()
        return

    if args.sort:
        _noninteractive_sort(cfg)
    else:
        _interactive_sort(cfg)


if __name__ == "__main__":
    main()
