"""Onboarding flow for VSort.

Scans the user's home directory for subdirectories, presents them for
interactive selection, and collects scheduling preferences.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from config import AppConfig, SortDirectory, SORT_MODES, MODEL_VARIANTS, QUANTIZATION_OPTIONS, SORT_STRATEGIES

logger = logging.getLogger(__name__)
console = Console()


# ── Directory scanning ────────────────────────────────────────────────


def _scan_directories(root: Path, max_depth: int = 2) -> List[Path]:
    """Return a sorted list of directories under *root* up to *max_depth*.

    Skips hidden directories, symlinks, and unreadable paths.
    """
    dirs: List[Path] = []
    try:
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                continue
            try:
                # quick readability check
                list(entry.iterdir())
            except PermissionError:
                continue
            dirs.append(entry)
            if max_depth > 1:
                try:
                    dirs.extend(_scan_directories(entry, max_depth - 1))
                except PermissionError:
                    pass
    except PermissionError:
        pass
    return sorted(dirs)


def scan_user_directories() -> List[Path]:
    """Scan the user's home directory for candidate folders."""
    home = Path.home()
    console.print(f"\n[bold cyan]Scanning directories under {home}...[/]")
    with console.status("[bold cyan]Scanning..."):
        dirs = _scan_directories(home, max_depth=2)
    console.print(f"  Found [bold]{len(dirs)}[/] directories.\n")
    return dirs


# ── Interactive selection ─────────────────────────────────────────────


def _pick_from_list(items: List[Path], title: str) -> List[Path]:
    """Present a numbered list of paths and let the user pick multiple."""
    if not items:
        console.print("[yellow]No directories found.[/]")
        return []

    table = Table(title=title, show_lines=False, padding=(0, 2))
    table.add_column("#", style="dim", justify="right")
    table.add_column("Path", style="cyan")

    for idx, p in enumerate(items, 1):
        table.add_row(str(idx), str(p))

    console.print(table)

    while True:
        raw = Prompt.ask(
            "\nEnter numbers of directories to sort (comma-separated, or 'all')",
            default="",
        )
        if raw.strip().lower() == "all":
            return items[:]
        if not raw.strip():
            console.print("[yellow]Please select at least one directory.[/]")
            continue
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if all(1 <= i <= len(items) for i in indices):
                return [items[i - 1] for i in indices]
            console.print(
                f"[red]Numbers must be between 1 and {len(items)}. Try again.[/]"
            )
        except ValueError:
            console.print("[red]Invalid input. Use comma-separated numbers.[/]")


def select_directories() -> List[Path]:
    """Full interactive flow: scan + select + optional custom path."""
    dirs = scan_user_directories()
    selected = _pick_from_list(dirs, "Select Directories to Sort")

    while Confirm.ask("Add a custom directory path?", default=False):
        custom = Prompt.ask("  Enter full directory path")
        p = Path(custom).expanduser().resolve()
        if p.is_dir():
            if p not in selected:
                selected.append(p)
                console.print(f"  [green]✓ Added {p}[/]")
            else:
                console.print("  [yellow]Already selected.[/]")
        else:
            console.print(f"  [red]Not a valid directory: {p}[/]")

    return selected


# ── Model variant selection ─────────────────────────────────────────────


def ask_model_variant() -> str:
    """Ask the user to pick a model variant (E2B or E4B).

    Returns the selected variant string.
    """
    console.print(Panel(
        "[bold]Model Variant[/]\n"
        "Choose the AI model size. Smaller is faster; larger is more accurate.",
        border_style="green",
    ))

    variant_choices = list(MODEL_VARIANTS.keys())
    for variant, info in MODEL_VARIANTS.items():
        console.print(f"  [bold green]{variant}[/]: {info['label']}")

    selected = Prompt.ask(
        "\nModel variant",
        choices=variant_choices,
        default="E2B",
    )

    return selected


# ── Quantization selection ─────────────────────────────────────────────


def ask_quantization() -> str:
    """Ask the user to pick a quantization format.

    Shows available quantization options with descriptions.
    Returns the selected quantization string.
    """
    console.print(Panel(
        "[bold]Quantization[/]\n"
        "Choose the model precision. Smaller files use less RAM but may be less accurate.\n"
        "Q4_K_M is recommended for most users.",
        border_style="yellow",
    ))

    quant_choices = list(QUANTIZATION_OPTIONS.keys())
    for quant, info in QUANTIZATION_OPTIONS.items():
        recommended = " [dim](recommended)[/]" if "recommended" in info["label"] else ""
        console.print(f"  [bold yellow]{quant}[/]: {info['label']}{recommended}")

    selected = Prompt.ask(
        "\nQuantization",
        choices=quant_choices,
        default="Q4_K_M",
    )

    return selected


# ── Schedule preferences ──────────────────────────────────────────────


SCHEDULE_OPTIONS = ["one-time", "periodic"]
PERIODIC_INTERVALS = ["hourly", "daily", "weekly", "monthly"]


def ask_schedule() -> Tuple[str, str]:
    """Ask the user for schedule preference.

    Returns (schedule_type, interval) where interval is only meaningful
    when schedule_type == "periodic".
    """
    console.print(Panel(
        "[bold]Schedule[/]\n"
        "Choose how often the sorter should run.",
        border_style="magenta",
    ))

    schedule = Prompt.ask(
        "Run type",
        choices=SCHEDULE_OPTIONS,
        default="one-time",
    )

    interval = "daily"
    if schedule == "periodic":
        interval = Prompt.ask(
            "How often?",
            choices=PERIODIC_INTERVALS,
            default="daily",
        )

    return schedule, interval


# ── Sort mode selection ────────────────────────────────────────────────


def ask_sort_mode() -> str:
    """Ask the user to pick a sort mode with descriptions.

    Returns the selected sort mode string.
    """
    console.print(Panel(
        "[bold]Sort Mode[/]\n"
        "Choose how much file content to analyze.\n"
        "Faster modes use less info; fuller modes are more accurate.",
        border_style="cyan",
    ))

    mode_choices = list(SORT_MODES.keys())
    for mode, desc in SORT_MODES.items():
        console.print(f"  [bold cyan]{mode}[/]: {desc}")

    selected = Prompt.ask(
        "\nSort mode",
        choices=mode_choices,
        default="normal",
    )

    return selected


# ── Sort strategy selection ────────────────────────────────────────────


def ask_sort_strategy() -> str:
    """Ask the user to pick a sort strategy (steering for the SLM).

    Returns the selected strategy string.
    """
    console.print(Panel(
        "[bold]Sort Strategy[/]\\n"
        "Choose how the AI should group your files.\\n"
        "Content-based: groups by topic/theme (e.g. 'Vacation-Photos').\\n"
        "Date-based: groups by time period (e.g. '2025-June').",
        border_style="green",
    ))

    strategy_choices = list(SORT_STRATEGIES.keys())
    for strategy, desc in SORT_STRATEGIES.items():
        console.print(f"  [bold green]{strategy}[/]: {desc}")

    selected = Prompt.ask(
        "\nSort strategy",
        choices=strategy_choices,
        default="auto",
    )

    return selected


# ── Onboarding orchestration ──────────────────────────────────────────


def run_onboarding(cfg: AppConfig) -> None:
    """Full onboarding: select directories, pick schedule, pick sort mode, update config."""
    console.print(Panel(
        "[bold magenta]═══ VSort — Onboarding ═══[/]\n\n"
        "Welcome! Let's set up your directories and preferences.",
        border_style="magenta",
    ))

    # Step 1: select directories
    selected = select_directories()
    if not selected:
        console.print("[red]No directories selected. Exiting.[/]")
        raise SystemExit(1)

    # Step 2: schedule
    schedule, interval = ask_schedule()

    # Step 3: sort mode
    sort_mode = ask_sort_mode()

    # Step 4: sort strategy (steering)
    sort_strategy = ask_sort_strategy()

    # Step 5: model variant
    model_variant = ask_model_variant()

    # Step 6: quantization
    quantization = ask_quantization()

    # Step 7: build SortDirectory list
    sort_dirs: List[SortDirectory] = []
    for p in selected:
        sd = SortDirectory(
            path=str(p),
            schedule=schedule,
            schedule_interval=interval,
        )
        sort_dirs.append(sd)
        console.print(f"  [green]✓[/] {p}  ({schedule})")

    cfg.set_sort_directories(sort_dirs)
    cfg.sort_mode = sort_mode
    cfg.sort_strategy = sort_strategy
    cfg.model_variant = model_variant
    cfg.quantization = quantization
    cfg.initialized = True
    cfg.save()

    console.print(f"\n[bold green]Onboarding complete! Sort mode: {sort_mode}, Strategy: {sort_strategy}, Model: {model_variant}, Quant: {quantization}. Preferences saved.[/]")
