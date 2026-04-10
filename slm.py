"""SLM (Small Language Model) interface for VSort.

Manages the llama-server subprocess lifecycle and provides helpers
to send prompts and parse structured JSON responses via the
OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil
import requests
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from config import AppConfig

logger = logging.getLogger(__name__)
console = Console()

# Maximum seconds to wait for llama-server to become healthy
STARTUP_TIMEOUT = 180
HEALTH_POLL_INTERVAL = 2


# ── Server management ─────────────────────────────────────────────────


class LlamaServer:
    """Manages a llama-server subprocess."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{cfg.host}:{cfg.port}"
        self.last_reasoning: str = ""  # populated by chat_completion()

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start llama-server and block until it responds to health checks."""
        if not self.cfg.llama_server_path or not Path(self.cfg.llama_server_path).exists():
            raise FileNotFoundError(
                f"llama-server not found at {self.cfg.llama_server_path}. Run setup first."
            )
        if not self.cfg.model_path or not Path(self.cfg.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {self.cfg.model_path}. Run setup first."
            )

        # Use a generous context size — Gemma 4 supports up to 131072.
        # 8192 is enough for most directory listings + model output.
        ctx_size = self.cfg.ctx_size

        cmd: List[str] = [
            str(self.cfg.llama_server_path),
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
            "--model", str(self.cfg.model_path),
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", "0",   # CPU by default; user can tweak
            "--threads", str(max(1, (psutil.cpu_count(logical=True) or 2) // 2)),
            "--metrics",              # expose /health endpoint
        ]

        # Add --mmproj argument if mmproj is configured and exists
        if self.cfg.mmproj_path and Path(self.cfg.mmproj_path).exists():
            cmd.extend(["--mmproj", str(self.cfg.mmproj_path)])
            logger.info("Vision support enabled with mmproj: %s", self.cfg.mmproj_path)

        # Set up environment: ensure shared libs are findable
        env = os.environ.copy()
        server_dir = Path(self.cfg.llama_server_path).parent
        if platform.system() != "Windows":
            existing = env.get("LD_LIBRARY_PATH", "")
            if str(server_dir) not in existing:
                env["LD_LIBRARY_PATH"] = f"{server_dir}:{existing}" if existing else str(server_dir)

        logger.info("Starting llama-server: %s", " ".join(cmd))
        console.print("[dim]Starting local SLM server...[/]")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                # New process group so we can kill the tree cleanly
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                if sys.platform == "win32" else 0,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start llama-server: {exc}") from exc

        self._wait_for_ready()

    def _wait_for_ready(self) -> None:
        """Poll the /health endpoint until the server is up or we time out."""
        spinner = Spinner("dots", text=Text("Waiting for SLM server to be ready..."))
        elapsed = 0.0

        with Live(spinner, console=console, transient=True):
            while elapsed < STARTUP_TIMEOUT:
                try:
                    resp = requests.get(f"{self.base_url}/health", timeout=3)
                    if resp.status_code == 200:
                        logger.info("llama-server is ready after %.1fs", elapsed)
                        console.print("[green]✓ SLM server ready.[/]")
                        return
                except requests.ConnectionError:
                    pass
                time.sleep(HEALTH_POLL_INTERVAL)
                elapsed += HEALTH_POLL_INTERVAL

        # Timed out
        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy within {STARTUP_TIMEOUT}s. "
            "Check the logs for errors."
        )

    def stop(self) -> None:
        """Terminate the llama-server process and all its children."""
        if self.process is None:
            return

        logger.info("Stopping llama-server (PID %d)...", self.process.pid)
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs([parent] + children, timeout=10)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        except Exception as exc:
            logger.warning("Error during llama-server shutdown: %s", exc)
        finally:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        console.print("[dim]SLM server stopped.[/]")

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ── prompting ──────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat-completion request and return the assistant content.

        Handles Gemma 4's thinking mode: if `content` is empty but
        `reasoning_content` is present, extracts JSON from reasoning.
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("llama-server is not running.")

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.base_url}/v1/chat/completions"

        try:
            resp = requests.post(url, json=payload, timeout=600)
        except requests.RequestException as exc:
            logger.error("SLM request failed: %s", exc)
            raise RuntimeError(f"SLM request failed: {exc}") from exc

        # Capture the response body even on HTTP errors for debugging
        if resp.status_code != 200:
            body = resp.text[:500]
            logger.error("SLM returned %d: %s", resp.status_code, body)
            raise RuntimeError(
                f"SLM request returned {resp.status_code}: {body}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("SLM returned non-JSON: %s", resp.text[:500])
            raise RuntimeError("SLM returned non-JSON response")

        try:
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected SLM response shape: %s", data)
            raise RuntimeError(f"Unexpected SLM response: {data}") from exc

        # Gemma 4 thinking mode: actual answer may be in reasoning_content
        # while content is empty, or the useful JSON may be in reasoning
        if not content.strip() and reasoning.strip():
            logger.info("SLM content empty, using reasoning_content")
            self.last_reasoning = "(reasoning was the primary output)"
            return reasoning

        # Store reasoning for --think display (this is the model's chain-of-thought)
        self.last_reasoning = reasoning

        # If both have content, reasoning is "thinking" and content is the answer
        return content

    # ── structured output ──────────────────────────────────────────

    def sort_files(
        self,
        file_descriptions: str,
        directory_path: str,
        vision_candidates: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Ask the SLM to categorise files; return parsed {filename: category}.

        *file_descriptions* is a human-readable summary of filenames and
        (for text files) a content sample. Automatically trimmed if too long.

        If *vision_candidates* is provided and mmproj is available,
        does a two-pass approach: text pass for all files, then a
        vision pass for image/video files whose results override.
        """
        # Trim file descriptions to avoid exceeding context window.
        # Default (8192 ctx): save ~3000 tokens for system/user prompt + ~2000 for output.
        # At ~4 chars/token, that's ~12000 chars for file descriptions.
        # YOLO mode (131072 ctx): allow ~60000 chars (plenty of room).
        if self.cfg.yolo:
            MAX_DESC_CHARS = 60000
        else:
            MAX_DESC_CHARS = 12000
        if len(file_descriptions) > MAX_DESC_CHARS:
            logger.info(
                "Trimming file descriptions from %d to %d chars",
                len(file_descriptions), MAX_DESC_CHARS,
            )
            file_descriptions = file_descriptions[:MAX_DESC_CHARS] + \
                "\n[... and more files, truncated]"

        system_prompt = (
            "You are a file organizer. Given a list of files with their names "
            "and content samples, organize them into meaningful category folders. "
            "Return ONLY valid JSON in this exact format: "
            '{"sorts": {"filename.ext": "CategoryName", ...}} '
            "Categories should be short, human-readable folder names without "
            "special characters or spaces (use CamelCase or hyphens). "
            "CRITICAL: You MUST copy filenames EXACTLY as shown — do NOT "
            "change, shorten, or paraphrase them. Every file in the input "
            "must appear in the output with its original name. "
            "Do NOT include any explanatory text, only the JSON object. "
            "NEVER use 'Images', 'Videos', 'Audio', 'Photos', or similar "
            "generic media-type names as categories — that just repeats the "
            "file type and is NOT helpful. Instead, infer what the media "
            "depicts: use categories like 'Vacation-Photos', 'Screenshots', "
            "'Work-Diagrams', 'Memes', 'Game-Captures', etc. If you cannot "
            "determine the content, group by likely context from filename "
            "patterns (e.g. IMG_ = camera photos, Screenshot_ = screenshots, "
            "Screen_Recording_ = screen recordings)."
        )

        # Apply sort strategy steering
        strategy = self.cfg.sort_strategy
        if strategy == "content":
            system_prompt += (
                " IMPORTANT STRATEGY: Group files by their CONTENT, TOPIC, or THEME. "
                "Do NOT use dates, timestamps, or time-based categories. "
                "For example, vacation photos should go in 'Vacation-Photos' "
                "not 'June-15' or '2025-06'. Group similar files together "
                "regardless of when they were created. Prefer semantic categories "
                "like 'Work-Documents', 'Game-Screenshots', 'Music', "
                "'Vacation-Photos', 'Code-Projects', etc."
            )
        elif strategy == "date":
            system_prompt += (
                " IMPORTANT STRATEGY: Group files by TIME PERIOD. "
                "Use categories based on creation dates such as months, "
                "seasons, or years. For example: '2025-June', '2025-Summer', "
                "'2024-December'. If you can infer a date from the filename or "
                "content, use that to group chronologically."
            )

        user_prompt = (
            f"Directory: {directory_path}\n\n"
            f"Files:\n{file_descriptions}\n\n"
            "Organize these files into categories and return the JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.chat_completion(messages, max_tokens=4096)
        logger.debug("SLM raw response:\n%s", raw)
        sorts = parse_sort_json(raw)

        # Post-process: reject generic media-type categories
        _GENERIC_CATEGORIES = {"images", "videos", "audio", "photos", "pictures",
                               "image", "video", "photo", "picture", "media"}
        for fname, cat in list(sorts.items()):
            if cat.lower().strip() in _GENERIC_CATEGORIES:
                # Try to infer a better category from the filename
                better = self._infer_category_from_filename(fname)
                if better:
                    logger.info(
                        "Replacing generic category %r with %r for %s",
                        cat, better, fname,
                    )
                    sorts[fname] = better

        # If vision candidates exist and mmproj is available, do a vision pass
        if vision_candidates and self._needs_vision(vision_candidates, directory_path):
            try:
                vision_sorts = self.sort_files_with_vision(
                    directory_path, vision_candidates
                )
                if vision_sorts:
                    # Vision results override text results for these files
                    sorts.update(vision_sorts)
                    logger.info(
                        "Vision pass updated %d file categories",
                        len(vision_sorts),
                    )
            except Exception as exc:
                logger.warning("Vision pass failed (non-fatal): %s", exc)

        return sorts

    def sort_files_with_retry(
        self,
        file_descriptions: str,
        directory_path: str,
        max_retries: int = 2,
        vision_candidates: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Wrap sort_files() with retry logic for JSON parse failures.

        If JSON parse fails, re-prompts the SLM asking it to fix the JSON.
        Up to *max_retries* attempts.
        """
        if self.cfg.yolo:
            MAX_DESC_CHARS = 60000
        else:
            MAX_DESC_CHARS = 12000
        if len(file_descriptions) > MAX_DESC_CHARS:
            file_descriptions = file_descriptions[:MAX_DESC_CHARS] + \
                "\n[... and more files, truncated]"

        last_error: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    return self.sort_files(
                        file_descriptions, directory_path, vision_candidates
                    )
                else:
                    # Re-prompt with the broken JSON and ask for fix
                    fix_prompt = (
                        f"The previous JSON was malformed. Please return valid JSON only.\n\n"
                        f"The broken JSON was:\n{last_raw}\n\n"
                        f"Directory: {directory_path}\n\n"
                        f"Original file descriptions:\n{file_descriptions}\n\n"
                        "Organize these files into categories and return ONLY valid JSON: "
                        '{"sorts": {"filename.ext": "CategoryName"}}'
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a file organizer. Return ONLY valid JSON: "
                                '{"sorts": {"filename.ext": "CategoryName"}}'
                            ),
                        },
                        {"role": "user", "content": fix_prompt},
                    ]
                    raw = self.chat_completion(messages, max_tokens=4096)
                    logger.debug("SLM retry %d response:\n%s", attempt, raw)
                    sorts = parse_sort_json(raw)

                    # Apply vision pass if needed
                    if vision_candidates and self._needs_vision(
                        vision_candidates, directory_path
                    ):
                        try:
                            vision_sorts = self.sort_files_with_vision(
                                directory_path, vision_candidates
                            )
                            if vision_sorts:
                                sorts.update(vision_sorts)
                        except Exception as exc:
                            logger.warning("Vision pass failed (non-fatal): %s", exc)

                    return sorts

            except ValueError as exc:
                last_error = str(exc)
                last_raw = getattr(exc, "raw_response", None)
                # Try to capture the raw response for the retry prompt
                logger.warning(
                    "JSON parse failed on attempt %d: %s", attempt, last_error
                )
                if attempt < max_retries:
                    console.print(
                        f"[yellow]⚠ JSON parse failed, retrying ({attempt + 1}/{max_retries})...[/]"
                    )

        raise ValueError(
            f"SLM returned invalid JSON after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _needs_vision(
        self,
        vision_candidates: List[str],
        directory_path: str,
    ) -> bool:
        """Check if mmproj_path is set and there are image/video files to analyze."""
        if not self.cfg.mmproj_path or not Path(self.cfg.mmproj_path).exists():
            return False
        if not vision_candidates:
            return False
        # Verify at least one candidate actually exists
        dir_path = Path(directory_path)
        for fname in vision_candidates:
            fpath = dir_path / fname
            if fpath.exists():
                from media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, GIF_EXTENSION
                ext = fpath.suffix.lower()
                if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS or ext in GIF_EXTENSION:
                    return True
        return False

    def _infer_category_from_filename(self, fname: str) -> Optional[str]:
        """When the SLM uses a generic category like 'Images', try to infer
        a better one from filename patterns.

        Returns None if no inference can be made (will keep the generic category).
        """
        import re as _re
        low = fname.lower()

        # Common camera photo patterns: IMG_YYYYMMDD, DSC_, DSCF_, etc.
        if _re.match(r"img_\d{8}", low) or low.startswith("dsc") or low.startswith("dscf"):
            return "Camera-Photos"

        # Screenshots
        if "screenshot" in low or "scrnshot" in low or low.startswith("screenshot"):
            return "Screenshots"

        # Screen recordings
        if "screen_recording" in low or "screenrecording" in low or "screen record" in low:
            return "Screen-Recordings"

        # Signal/WhatsApp/Telegram downloads
        if low.startswith("signal-") or "telegram" in low or "whatsapp" in low:
            return "Messenger-Media"

        # Downloads from browsers (often numeric)
        if _re.match(r"\d{5,}", low):
            return "Downloads"

        return None

    def sort_files_with_vision(
        self,
        directory_path: str,
        vision_candidates: List[str],
    ) -> Dict[str, str]:
        """Send image files as base64 in the OpenAI vision format.

        The message format uses content as a list with text and image_url types.
        Vision results provide more nuanced categories for images/videos.
        """
        dir_path = Path(directory_path)
        system_prompt = (
            "You are a file organizer. Given filenames and images, organize them "
            "into meaningful categories. Return ONLY valid JSON: "
            '{"sorts": {"filename.ext": "CategoryName"}} '
            "Categories should be descriptive but concise folder names (e.g. "
            "'Summer-Vacation', 'Work-Documents', 'Game-Screenshots'). "
            "Use CamelCase or hyphens. Every file in the input must appear in the output. "
            "NEVER use generic categories like 'Images', 'Photos', 'Videos', 'Audio' — "
            "describe what the content DEPICTS, not its file type."
        )

        # Apply sort strategy steering (same as text pass)
        strategy = self.cfg.sort_strategy
        if strategy == "content":
            system_prompt += (
                " IMPORTANT: Group by CONTENT and THEME, not by date. "
                "Use categories like 'Vacation-Photos', 'Memes', 'Screenshots' "
                "rather than 'June-15' or '2025-06'."
            )
        elif strategy == "date":
            system_prompt += (
                " IMPORTANT: Group by TIME PERIOD based on what you see. "
                "Use categories like '2025-Summer', '2024-Winter'."
            )

        # Build the content list for the user message
        content_parts: List[Dict[str, Any]] = []
        text_lines = [f"Directory: {directory_path}\n"]
        text_lines.append("Analyze these images/videos and suggest categories:\n")

        # We'll batch candidates — limit to avoid token overflow
        # Each image ~512x512 JPEG ≈ 50-80KB base64 ≈ ~2000 tokens
        # With 8192 ctx we can fit ~10 images; with yolo (131072) we can do ~50
        MAX_VISION_FILES = 50 if self.cfg.yolo else 20
        candidates_to_process = vision_candidates[:MAX_VISION_FILES]

        if len(vision_candidates) > MAX_VISION_FILES:
            logger.info(
                "Limiting vision pass to %d of %d candidates",
                MAX_VISION_FILES, len(vision_candidates),
            )

        for fname in candidates_to_process:
            fpath = dir_path / fname
            if not fpath.exists():
                continue

            ext = fpath.suffix.lower()
            from media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, GIF_EXTENSION

            if ext in IMAGE_EXTENSIONS:
                # Direct image file — resize and encode
                try:
                    from media import resize_image_for_vision
                    jpeg_bytes = resize_image_for_vision(fpath, max_size=512)
                    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                    text_lines.append(f"\nFile: {fname}")
                    content_parts.append({
                        "type": "text",
                        "text": f"\nFile: {fname}",
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    })
                except Exception as exc:
                    logger.warning("Failed to encode image %s: %s", fpath, exc)
                    text_lines.append(f"\nFile: {fname} [image encoding failed]")

            elif ext in GIF_EXTENSION:
                # GIF — extract a frame and treat as image
                try:
                    from media import extract_gif_frames, resize_image_for_vision
                    frames = extract_gif_frames(fpath, num_frames=1)
                    if frames:
                        jpeg_bytes = resize_image_for_vision(frames[0], max_size=512)
                        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                        content_parts.append({
                            "type": "text",
                            "text": f"\nFile: {fname} (GIF, first frame shown)",
                        })
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        })
                except Exception as exc:
                    logger.warning("Failed to process GIF %s: %s", fpath, exc)
                    text_lines.append(f"\nFile: {fname} [GIF processing failed]")

            elif ext in VIDEO_EXTENSIONS:
                # Video — extract a frame and treat as image
                try:
                    from media import extract_video_frames, resize_image_for_vision
                    frames = extract_video_frames(fpath, num_frames=1)
                    if frames:
                        jpeg_bytes = resize_image_for_vision(frames[0], max_size=512)
                        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                        content_parts.append({
                            "type": "text",
                            "text": f"\nFile: {fname} (video, key frame shown)",
                        })
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        })
                except Exception as exc:
                    logger.warning("Failed to process video %s: %s", fpath, exc)
                    text_lines.append(f"\nFile: {fname} [video processing failed]")

        # Add remaining text that wasn't part of image content
        if text_lines:
            remaining_text = "\n".join(text_lines)
            # Prepend text content
            content_parts.insert(0, {
                "type": "text",
                "text": remaining_text + "\nOrganize these files and return the JSON.",
            })
        else:
            content_parts.append({
                "type": "text",
                "text": "Organize these files and return the JSON.",
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        raw = self.chat_completion(messages, max_tokens=2048)
        logger.debug("Vision SLM raw response:\n%s", raw)
        return parse_sort_json(raw)


# ── JSON parsing ──────────────────────────────────────────────────────


def parse_sort_json(raw: str) -> Dict[str, str]:
    """Extract the sorts mapping from an SLM response.

    The SLM may wrap its JSON in markdown fences or add preamble text;
    we try to be tolerant. Also handles reasoning_content that may
    contain the JSON mixed with "thinking" text.

    Self-healing steps:
      1. Try normal JSON parsing (after extracting JSON from fences/braces)
      2. If that fails, apply regex-based fixes:
         - Remove trailing commas before } or ]
         - Fix unescaped quotes inside string values
         - Try robust {"sorts": {...}} pattern matching
      3. If regex fixes still fail, raise ValueError (caller can retry)
    """
    # Step 1: Extract candidate JSON string
    candidate = _extract_json_candidate(raw)

    # Step 2: Try direct parse
    first_error: Optional[str] = None
    try:
        parsed = json.loads(candidate)
        return _validate_sorts_dict(parsed)
    except json.JSONDecodeError as exc:
        first_error = str(exc)
        logger.info("Direct JSON parse failed: %s — attempting self-healing", exc)

    # Step 3: Regex-based self-healing
    healed = _heal_json(candidate)
    if healed != candidate:
        try:
            parsed = json.loads(healed)
            logger.info("Self-healed JSON parsed successfully")
            return _validate_sorts_dict(parsed)
        except json.JSONDecodeError as healed_error:
            logger.warning("Self-healed JSON still invalid: %s", healed_error)

    # Step 4: Try robust {"sorts": {...}} pattern matching
    robust_match = re.search(
        r'\{\s*"sorts"\s*:\s*\{[^}]*\}\s*\}',
        raw,
        re.DOTALL,
    )
    if robust_match:
        try:
            parsed = json.loads(robust_match.group(0))
            logger.info("Robust regex extracted valid JSON")
            return _validate_sorts_dict(parsed)
        except json.JSONDecodeError:
            pass

    # Step 5: Try healing the robust match
    if robust_match:
        healed_robust = _heal_json(robust_match.group(0))
        try:
            parsed = json.loads(healed_robust)
            logger.info("Healed robust regex match parsed successfully")
            return _validate_sorts_dict(parsed)
        except json.JSONDecodeError:
            pass

    logger.error("All JSON parsing attempts failed. Raw:\n%s", raw)
    exc = ValueError(
        f"SLM returned invalid JSON. Parse error: {first_error}"
    )
    exc.raw_response = raw  # type: ignore[attr-defined]
    raise exc


def _extract_json_candidate(raw: str) -> str:
    """Extract the most likely JSON string from the SLM response."""
    # Attempt 1: find a JSON block inside ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Attempt 2: find the outermost { ... } in the text
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return raw


def _heal_json(text: str) -> str:
    """Apply regex-based fixes to potentially broken JSON.

    Fixes applied:
      - Remove trailing commas before } or ]
      - Fix unescaped quotes inside string values
    """
    # Fix 1: Remove trailing commas before } or ]
    # E.g. {"sorts": {"a": "b",}} -> {"sorts": {"a": "b"}}
    healed = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix 2: Fix unescaped quotes inside string values
    # This is tricky — we look for patterns like:
    #   "key": "value with 'unescaped' quotes"
    # and try to escape the inner quotes.
    # Strategy: for each string value, replace unescaped " inside it with \"
    healed = _fix_unescaped_quotes(healed)

    return healed


def _fix_unescaped_quotes(text: str) -> str:
    """Try to fix unescaped double quotes inside JSON string values.

    Uses a state machine approach: track whether we're inside a string,
    and if we encounter an unescaped quote that seems to be inside a
    value (followed by non-JSON-structural chars), escape it.
    """
    result = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(text):
        ch = text[i]

        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue

        if ch == '\\':
            result.append(ch)
            escape_next = True
            i += 1
            continue

        if ch == '"':
            if in_string:
                # We might be ending the string, or this might be an
                # unescaped quote inside the string.
                # Look ahead: if after this quote there's a colon, comma,
                # closing brace/bracket, or end-of-text, it's likely the
                # end of the string. Otherwise, it's probably an inner quote
                # that needs escaping.
                rest = text[i + 1:].lstrip()
                if not rest:
                    # End of text — this is the string end
                    in_string = False
                    result.append(ch)
                elif rest[0] in (':', ',', '}', ']', '\n', '\r'):
                    # Structural character after — likely end of string
                    in_string = False
                    result.append(ch)
                else:
                    # Likely an unescaped quote inside a value — escape it
                    result.append('\\"')
                    # Stay in_string
            else:
                # Starting a string
                in_string = True
                result.append(ch)
        else:
            result.append(ch)

        i += 1

    return ''.join(result)


def _validate_sorts_dict(parsed: Any) -> Dict[str, str]:
    """Validate that the parsed JSON has the expected structure."""
    if not isinstance(parsed, dict):
        raise ValueError(f"SLM response is not a dict: {parsed}")

    if "sorts" not in parsed or not isinstance(parsed["sorts"], dict):
        raise ValueError(
            f"SLM response JSON missing 'sorts' key or wrong type: {parsed}"
        )

    return {str(k): str(v) for k, v in parsed["sorts"].items()}
