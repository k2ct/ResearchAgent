"""
MinerU API Client — optional backend for complex PDF parsing.

Per official MinerU API docs:

**Precise mode (local file upload):**
    1. POST /api/v4/file-urls/batch  → batch_id + file_urls[]
    2. PUT  {file_urls[0]}             → upload raw bytes (no Content-Type, no Auth)
    3. GET  /api/v4/extract-results/batch/{batch_id} → poll state
    4. GET  {full_zip_url}             → download zip
    5. extract_markdown_from_zip()     → prefer full.md

**Precise mode (URL task):**
    1. POST /api/v4/extract/task       → task_id
    2. GET  /api/v4/extract/task/{task_id} → poll state
    3. GET  {full_zip_url}             → download zip

**Agent mode:**
    1. POST /api/v1/agent/parse/file   → task_id + file_url (no Auth header)
    2. PUT  {file_url}                 → upload file
    3. GET  /api/v1/agent/parse/{task_id} → poll state
    4. GET  {markdown_url}             → download markdown text

API keys are NEVER printed.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def is_mineru_enabled() -> bool:
    backend = os.getenv("DOCUMENT_INGESTION_BACKEND", "local").strip().lower()
    api_key = os.getenv("MINERU_API_KEY", "").strip()
    return backend == "mineru" and len(api_key) > 0


def get_mineru_config() -> Dict[str, Any]:
    return {
        "api_key": os.getenv("MINERU_API_KEY", "").strip(),
        "base_url": os.getenv("MINERU_API_BASE_URL", "").strip().rstrip("/"),
        "api_mode": os.getenv("MINERU_API_MODE", "precise").strip().lower(),
        "model_version": os.getenv("MINERU_MODEL_VERSION", "vlm").strip(),
        "parse_endpoint": os.getenv("MINERU_PARSE_ENDPOINT", "").strip(),
        "file_urls_endpoint": os.getenv("MINERU_FILE_URLS_ENDPOINT", "").strip(),
        "batch_results_endpoint": os.getenv("MINERU_BATCH_RESULTS_ENDPOINT", "").strip(),
        "status_endpoint": os.getenv("MINERU_TASK_STATUS_ENDPOINT", "").strip(),
        "result_endpoint": os.getenv("MINERU_RESULT_ENDPOINT", "").strip(),
        "poll_interval_seconds": int(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "3")),
        "max_wait_seconds": int(os.getenv("MINERU_MAX_WAIT_SECONDS", "300")),
        "disable_proxy": os.getenv("MINERU_DISABLE_PROXY", "false").strip().lower() == "true",
        "http_proxy": os.getenv("MINERU_HTTP_PROXY", "").strip(),
        "https_proxy": os.getenv("MINERU_HTTPS_PROXY", "").strip(),
        "backend": "mineru",
    }


def _mask_key(key: str) -> str:
    if not key:
        return "(empty)"
    if len(key) <= 8:
        return key[:2] + "***"
    return key[:4] + "****" + key[-4:]


# ═══════════════════════════════════════════════════════════════════════════
# Result builder
# ═══════════════════════════════════════════════════════════════════════════

def _make_result(ok: bool, markdown_text: str = "", json_data: Optional[Dict] = None,
                 error: str = "", mineru_used: bool = False, mineru_error: str = "",
                 raw_output_path: Optional[str] = None) -> Dict[str, Any]:
    return {"ok": ok, "markdown_text": markdown_text, "json_data": json_data or {},
            "error": error, "backend": "mineru", "mineru_used": mineru_used,
            "mineru_error": mineru_error, "raw_output_path": raw_output_path}


# ═══════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_session(config: Dict[str, Any]) -> Any:
    import requests
    session = requests.Session()
    if config.get("disable_proxy"):
        session.trust_env = False
        return session
    proxies = {}
    if config.get("http_proxy"):
        proxies["http"] = config["http_proxy"]
    if config.get("https_proxy"):
        proxies["https"] = config["https_proxy"]
    if proxies:
        session.proxies.update(proxies)
    return session


def _auth_headers(config: Dict[str, Any]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {config['api_key']}"}


def _resolve_url(config: Dict[str, Any], endpoint_key: str, default_path: str) -> str:
    base = config["base_url"]
    endpoint = config.get(endpoint_key, "")
    if endpoint:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"{base}/{endpoint.lstrip('/')}"
    return f"{base}/{default_path.lstrip('/')}"


def _safe_data_id(file_stem: str) -> str:
    """Generate a safe data_id from a filename stem."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', file_stem)[:64]
    return safe or "doc"


# ═══════════════════════════════════════════════════════════════════════════
# Zip extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_markdown_from_zip(source: Any, *, output_dir: Optional[Path] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Extract markdown from a zip file (path or bytes).
    Priority order: full.md > main.md > shortest-path .md > any .md.
    Skips __MACOSX.  Collects JSON and image file lists.
    Returns (markdown_text, meta).
    """
    meta: Dict[str, Any] = {"md_files": [], "json_files": [], "image_files": [], "other_files": []}
    md_contents: List[Tuple[str, str]] = []

    def _process(zf: zipfile.ZipFile) -> None:
        for name in zf.namelist():
            if "__MACOSX" in name or name.endswith("/"):
                continue
            base_lower = name.split("/")[-1].lower()
            if base_lower.endswith(".md"):
                meta["md_files"].append(name)
                try:
                    text = zf.read(name).decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = zf.read(name).decode("utf-8-sig")
                    except Exception:
                        continue
                md_contents.append((name, text))
            elif base_lower.endswith((".json", ".jsonl")):
                meta["json_files"].append(name)
            elif base_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                meta["image_files"].append(name)
            else:
                meta["other_files"].append(name)

    saved_path: Optional[str] = None

    if isinstance(source, (str, Path)):
        path = Path(source)
        saved_path = str(path)
        try:
            with zipfile.ZipFile(path, "r") as zf:
                _process(zf)
        except (zipfile.BadZipFile, FileNotFoundError, OSError):
            return "", meta
    elif isinstance(source, bytes):
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            path = output_dir / f"mineru_result_{ts}.zip"
            path.write_bytes(source)
            saved_path = str(path)
        try:
            with zipfile.ZipFile(io.BytesIO(source)) as zf:
                _process(zf)
        except (zipfile.BadZipFile, Exception):
            return "", meta
    else:
        return "", meta

    meta["raw_output_path"] = saved_path
    if not md_contents:
        return "", meta

    # Priority: full.md > main.md > shortest path
    priority_order = {"full.md": 0, "main.md": 1}
    md_contents.sort(key=lambda x: (priority_order.get(Path(x[0]).name.lower(), 99), len(x[0]), x[0]))

    if len(md_contents) == 1:
        return md_contents[0][1], meta

    # Return the highest-priority single file
    return md_contents[0][1], meta


def _extract_md_from_bytes(data: bytes) -> str:
    text, _ = extract_markdown_from_zip(data)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Response helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_api_error(resp_data: dict) -> Optional[str]:
    """If the response looks like a MinerU error, return the error message."""
    code = resp_data.get("code", 0)
    if isinstance(code, int) and code != 0:
        msg = resp_data.get("msg", resp_data.get("message", "unknown"))
        trace_id = resp_data.get("trace_id", resp_data.get("traceId", ""))
        extra = f" trace_id={trace_id}" if trace_id else ""
        return f"API error code={code}: {msg}{extra}"
    return None


def _download_zip(url: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Download zip from url. Returns content bytes or None."""
    import requests
    session = _build_session(config)
    headers = _auth_headers(config)
    try:
        resp = session.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        if resp.content[:4] == b"PK\x03\x04":
            return resp.content
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Polling — state-based (per official docs)
# ═══════════════════════════════════════════════════════════════════════════

PROCESSING_STATES = {"waiting-file", "waiting_file", "pending", "running", "converting", "processing"}
TERMINAL_OK_STATES = {"done", "completed", "success", "succeed", "ready"}
TERMINAL_FAIL_STATES = {"failed", "error", "fail"}


def _poll_batch_results(batch_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Poll /api/v4/extract-results/batch/{batch_id}.
    Returns {"ok": bool, "data": dict, "error": str, "full_zip_url": str}.
    """
    import requests
    session = _build_session(config)
    headers = _auth_headers(config)
    results_url_tpl = _resolve_url(config, "batch_results_endpoint", "api/v4/extract-results/batch/{batch_id}")
    results_url = results_url_tpl.replace("{batch_id}", str(batch_id))
    poll_interval = config["poll_interval_seconds"]
    max_wait = config["max_wait_seconds"]
    deadline = time.time() + max_wait
    last_state = ""

    while time.time() < deadline:
        try:
            resp = session.get(results_url, headers=headers, timeout=30)
        except requests.RequestException as e:
            time.sleep(poll_interval)
            continue

        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue

        try:
            data = resp.json()
        except ValueError:
            time.sleep(poll_interval)
            continue

        err = _extract_api_error(data)
        if err:
            return {"ok": False, "data": data, "error": err, "full_zip_url": ""}

        inner = data.get("data", data)

        # extract_result is a LIST per official docs
        extract_result = inner.get("extract_result", inner.get("extractResult", []))
        if not isinstance(extract_result, list):
            extract_result = [extract_result] if extract_result else []

        # Check each result item
        all_done = True
        any_failed = False
        fail_msg = ""
        full_zip_url = ""

        for item in extract_result:
            if not isinstance(item, dict):
                continue
            state = (item.get("state", "") or "").lower()
            last_state = state
            if state in TERMINAL_OK_STATES:
                full_zip_url = item.get("full_zip_url", item.get("fullZipUrl", "")) or full_zip_url
            elif state in TERMINAL_FAIL_STATES:
                any_failed = True
                fail_msg = item.get("err_msg", item.get("errMsg", "batch item failed"))
            elif state in PROCESSING_STATES:
                all_done = False
            else:
                # Unknown state — check top-level state too
                all_done = False

        # Also check top-level state
        top_state = (inner.get("state", "") or "").lower()
        if top_state:
            last_state = top_state
            if top_state in TERMINAL_OK_STATES:
                all_done = True
                full_zip_url = inner.get("full_zip_url", inner.get("fullZipUrl", "")) or full_zip_url
            elif top_state in TERMINAL_FAIL_STATES:
                any_failed = True
                fail_msg = inner.get("err_msg", inner.get("errMsg", "failed"))
                all_done = True

        if any_failed:
            return {"ok": False, "data": data, "error": f"Batch failed: {fail_msg}", "full_zip_url": ""}
        if all_done:
            return {"ok": True, "data": data, "error": "", "full_zip_url": full_zip_url}

        time.sleep(poll_interval)

    return {"ok": False, "data": {}, "error": f"Batch {batch_id} timed out after {max_wait}s (last state: {last_state})", "full_zip_url": ""}


def _poll_task_status(task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Poll /api/v4/extract/task/{task_id}.
    Returns {"ok": bool, "data": dict, "error": str, "full_zip_url": str, "markdown_url": str}.
    """
    import requests
    session = _build_session(config)
    headers = _auth_headers(config)
    if config["status_endpoint"]:
        status_url = config["status_endpoint"].replace("{task_id}", task_id)
    else:
        parse_url = _resolve_url(config, "parse_endpoint", "api/v4/extract/task")
        status_url = f"{parse_url}/{task_id}"
    poll_interval = config["poll_interval_seconds"]
    max_wait = config["max_wait_seconds"]
    deadline = time.time() + max_wait
    last_state = ""

    while time.time() < deadline:
        try:
            resp = session.get(status_url, headers=headers, timeout=30)
        except requests.RequestException as e:
            time.sleep(poll_interval)
            continue
        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue
        try:
            data = resp.json()
        except ValueError:
            time.sleep(poll_interval)
            continue

        err = _extract_api_error(data)
        if err:
            return {"ok": False, "data": data, "error": err, "full_zip_url": "", "markdown_url": ""}

        inner = data.get("data", data)
        state = (inner.get("state", "") or "").lower()
        last_state = state

        if state in TERMINAL_OK_STATES:
            return {
                "ok": True, "data": data, "error": "",
                "full_zip_url": inner.get("full_zip_url", inner.get("fullZipUrl", "")),
                "markdown_url": inner.get("markdown_url", inner.get("markdownUrl", "")),
            }
        if state in TERMINAL_FAIL_STATES:
            err_msg = inner.get("err_msg", inner.get("errMsg", "task failed"))
            return {"ok": False, "data": data, "error": f"Task failed: {err_msg}", "full_zip_url": "", "markdown_url": ""}
        time.sleep(poll_interval)

    return {"ok": False, "data": {}, "error": f"Task {task_id} timed out after {max_wait}s (last state: {last_state})", "full_zip_url": "", "markdown_url": ""}


# ═══════════════════════════════════════════════════════════════════════════
# Precise mode: LOCAL FILE upload
# ═══════════════════════════════════════════════════════════════════════════

def _call_mineru_v4_local_file(file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a local file via MinerU v4 precise batch upload.

    Per official docs:
    1. POST /api/v4/file-urls/batch → batch_id + file_urls[]
    2. PUT {file_urls[0]} (binary, no Content-Type, no Authorization)
    3. GET /api/v4/extract-results/batch/{batch_id} → poll extract_result[]
    4. GET full_zip_url → download zip → extract full.md
    """
    import requests
    session = _build_session(config)
    headers = _auth_headers(config)
    model_version = config["model_version"]
    file_name = file_path.name
    file_stem = file_path.stem
    file_size = file_path.stat().st_size

    # ── Step 1: Request upload URL ──────────────────────────────────
    batch_url = _resolve_url(config, "file_urls_endpoint", "api/v4/file-urls/batch")
    batch_payload = {
        "files": [{"name": file_name, "data_id": _safe_data_id(file_stem)}],
        "model_version": model_version,
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }

    try:
        resp = session.post(batch_url, headers=headers, json=batch_payload, timeout=30)
    except requests.RequestException as e:
        return _make_result(ok=False, error=f"Batch request failed: {e}", mineru_used=True, mineru_error="batch_request_failed")

    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        return _make_result(ok=False, error=f"Batch HTTP {resp.status_code}: {detail}", mineru_used=True, mineru_error=f"batch_http_{resp.status_code}")

    try:
        batch_data = resp.json()
    except ValueError:
        return _make_result(ok=False, error="Batch: non-JSON response", mineru_used=True, mineru_error="batch_invalid_json")

    err = _extract_api_error(batch_data)
    if err:
        return _make_result(ok=False, error=err, json_data=batch_data, mineru_used=True, mineru_error="batch_api_error")

    inner = batch_data.get("data", batch_data)
    batch_id = inner.get("batch_id", inner.get("batchId", ""))
    # file_urls is a list of strings per official docs
    file_urls = inner.get("file_urls", inner.get("fileUrls", []))
    if isinstance(file_urls, list) and len(file_urls) > 0:
        upload_url = file_urls[0]
    else:
        upload_url = inner.get("upload_url", inner.get("uploadUrl", ""))

    if not batch_id:
        return _make_result(ok=False, error=f"No batch_id in batch response. data keys: {list(inner.keys())[:10]}", json_data=batch_data, mineru_used=True, mineru_error="no_batch_id")
    if not upload_url:
        return _make_result(ok=False, error=f"No file_urls in batch response. data keys: {list(inner.keys())[:10]}", json_data=batch_data, mineru_used=True, mineru_error="no_file_urls")

    # ── Step 2: Upload file via PUT (no Content-Type, no Auth) ──────
    try:
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()
    except OSError as e:
        return _make_result(ok=False, error=f"Cannot read file: {e}", mineru_used=True, mineru_error="file_read_error")

    try:
        put_resp = session.put(upload_url, data=file_bytes, timeout=120)
    except requests.RequestException as e:
        return _make_result(ok=False, error=f"PUT upload failed: {e}", mineru_used=True, mineru_error="upload_failed")

    if put_resp.status_code not in (200, 201, 204):
        detail = put_resp.text[:300] if put_resp.text else ""
        return _make_result(ok=False, error=f"PUT upload HTTP {put_resp.status_code}: {detail}", mineru_used=True, mineru_error=f"upload_http_{put_resp.status_code}")

    # ── Step 3: Poll batch results ───────────────────────────────────
    poll = _poll_batch_results(str(batch_id), config)
    if not poll["ok"]:
        return _make_result(ok=False, error=poll.get("error", "Batch polling failed"), json_data=poll.get("data", batch_data), mineru_used=True, mineru_error="batch_polling_failed")

    full_zip_url = poll.get("full_zip_url", "")
    if not full_zip_url:
        # Search extract_result list for full_zip_url
        inner_final = poll.get("data", {}).get("data", poll.get("data", {}))
        extract_result = inner_final.get("extract_result", inner_final.get("extractResult", []))
        if isinstance(extract_result, list):
            for item in extract_result:
                if isinstance(item, dict):
                    full_zip_url = item.get("full_zip_url", item.get("fullZipUrl", "")) or full_zip_url
        if not full_zip_url:
            full_zip_url = inner_final.get("full_zip_url", inner_final.get("fullZipUrl", ""))

    if not full_zip_url:
        return _make_result(ok=False, error="Batch done but no full_zip_url in result", json_data=poll.get("data", batch_data), mineru_used=True, mineru_error="no_zip_url")

    # ── Step 4 & 5: Download zip & extract markdown ──────────────────
    zip_bytes = _download_zip(full_zip_url, config)
    if not zip_bytes:
        return _make_result(ok=False, error="Failed to download result zip", json_data=poll.get("data", batch_data), mineru_used=True, mineru_error="zip_download_failed")

    output_dir = Path("data/ingested/_mineru_raw")
    md, meta = extract_markdown_from_zip(zip_bytes, output_dir=output_dir)
    if md:
        return _make_result(ok=True, markdown_text=md, json_data={**poll.get("data", {}), **meta}, mineru_used=True, raw_output_path=meta.get("raw_output_path"))
    # Show what was found
    found = meta.get("md_files", []) + meta.get("json_files", []) + meta.get("other_files", [])
    found_str = ", ".join(found[:20]) if found else "no files at all"
    return _make_result(ok=False, error=f"Zip downloaded but no .md found inside. Zip contents: [{found_str}]", json_data={**poll.get("data", {}), **meta}, mineru_used=True, mineru_error="no_md_in_zip")


# ═══════════════════════════════════════════════════════════════════════════
# Precise mode: URL task (for remote files only, not local uploads)
# ═══════════════════════════════════════════════════════════════════════════

def _call_mineru_v4_url_task(file_url: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit a URL-based parse task.  Only for files already hosted online.

    POST /api/v4/extract/task  {url, model_version, enable_formula, enable_table, language}
    → task_id → poll → full_zip_url → download zip → extract
    """
    import requests
    session = _build_session(config)
    parse_url = _resolve_url(config, "parse_endpoint", "api/v4/extract/task")
    headers = _auth_headers(config)
    model_version = config["model_version"]

    payload = {
        "url": file_url,
        "model_version": model_version,
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }

    try:
        resp = session.post(parse_url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        return _make_result(ok=False, error=f"URL task request failed: {e}", mineru_used=True, mineru_error="api_request_failed")

    if resp.status_code != 200:
        detail = resp.text[:500] if resp.text else ""
        return _make_result(ok=False, error=f"URL task HTTP {resp.status_code}: {detail}", mineru_used=True, mineru_error=f"http_{resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        return _make_result(ok=False, error="URL task: non-JSON response", mineru_used=True, mineru_error="invalid_json")

    err = _extract_api_error(data)
    if err:
        return _make_result(ok=False, error=err, json_data=data, mineru_used=True, mineru_error="url_task_api_error")

    inner = data.get("data", data)
    task_id = inner.get("task_id", inner.get("taskId", ""))
    if not task_id:
        return _make_result(ok=False, error=f"No task_id. Keys: {list(inner.keys())[:10]}", json_data=data, mineru_used=True, mineru_error="no_task_id")

    poll = _poll_task_status(str(task_id), config)
    if not poll["ok"]:
        return _make_result(ok=False, error=poll.get("error", "Task polling failed"), json_data=poll.get("data", data), mineru_used=True, mineru_error="task_polling_failed")

    full_zip_url = poll.get("full_zip_url", "")
    if not full_zip_url:
        inner_final = poll.get("data", {}).get("data", poll.get("data", {}))
        full_zip_url = inner_final.get("full_zip_url", inner_final.get("fullZipUrl", ""))

    if full_zip_url:
        zip_bytes = _download_zip(full_zip_url, config)
        if zip_bytes:
            output_dir = Path("data/ingested/_mineru_raw")
            md, meta = extract_markdown_from_zip(zip_bytes, output_dir=output_dir)
            if md:
                return _make_result(ok=True, markdown_text=md, json_data={**poll.get("data", {}), **meta}, mineru_used=True, raw_output_path=meta.get("raw_output_path"))
            return _make_result(ok=False, error="No .md in task result zip", json_data=poll.get("data", data), mineru_used=True, mineru_error="no_md_in_zip")
        return _make_result(ok=False, error="Failed to download task result zip", json_data=poll.get("data", data), mineru_used=True, mineru_error="zip_download_failed")

    return _make_result(ok=False, error="Task done but no full_zip_url", json_data=poll.get("data", data), mineru_used=True, mineru_error="no_zip_url")


# ═══════════════════════════════════════════════════════════════════════════
# Precise mode dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def _call_mineru_precise(file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Precise mode: always use local file upload flow (batch)."""
    return _call_mineru_v4_local_file(file_path, config)


# ═══════════════════════════════════════════════════════════════════════════
# Agent mode (per official docs: no Authorization header)
# ═══════════════════════════════════════════════════════════════════════════

def _call_mineru_agent(file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent lightweight parsing API.

    1. POST /api/v1/agent/parse/file  {file_name, language, enable_table, is_ocr, enable_formula}
       (No Authorization header per official docs)
    2. PUT {file_url} (upload file bytes)
    3. GET /api/v1/agent/parse/{task_id} → poll state
    4. GET {markdown_url} → download markdown text

    Limits: 10MB, 20 pages.
    """
    import requests
    session = _build_session(config)
    agent_url = _resolve_url(config, "parse_endpoint", "api/v1/agent/parse/file")
    file_name = file_path.name

    # ── Step 1: Request parse ───────────────────────────────────────
    # Agent API does NOT use Authorization header
    payload = {
        "file_name": file_name,
        "language": "ch",
        "enable_table": True,
        "is_ocr": False,
        "enable_formula": True,
    }

    try:
        resp = session.post(agent_url, json=payload, timeout=30)
    except requests.RequestException as e:
        return _make_result(ok=False, error=f"Agent request failed: {e}", mineru_used=True, mineru_error="agent_request_failed")

    if resp.status_code != 200:
        detail = resp.text[:500] if resp.text else ""
        return _make_result(ok=False, error=f"Agent HTTP {resp.status_code}: {detail}", mineru_used=True, mineru_error=f"agent_http_{resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        return _make_result(ok=False, error="Agent: non-JSON response", mineru_used=True, mineru_error="agent_invalid_json")

    err = _extract_api_error(data)
    if err:
        return _make_result(ok=False, error=err, json_data=data, mineru_used=True, mineru_error="agent_api_error")

    inner = data.get("data", data)
    task_id = inner.get("task_id", inner.get("taskId", ""))
    file_url = inner.get("file_url", inner.get("fileUrl", ""))

    if not task_id or not file_url:
        return _make_result(ok=False, error=f"Agent: no task_id or file_url. Keys: {list(inner.keys())[:10]}", json_data=data, mineru_used=True, mineru_error="agent_no_task_id")

    # ── Step 2: PUT upload ──────────────────────────────────────────
    try:
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()
    except OSError as e:
        return _make_result(ok=False, error=f"Cannot read file: {e}", mineru_used=True, mineru_error="file_read_error")

    try:
        put_resp = session.put(file_url, data=file_bytes, timeout=120)
    except requests.RequestException as e:
        return _make_result(ok=False, error=f"Agent PUT upload failed: {e}", mineru_used=True, mineru_error="agent_upload_failed")

    if put_resp.status_code not in (200, 201, 204):
        detail = put_resp.text[:300] if put_resp.text else ""
        return _make_result(ok=False, error=f"Agent PUT HTTP {put_resp.status_code}: {detail}", mineru_used=True, mineru_error=f"agent_upload_http_{put_resp.status_code}")

    # ── Step 3: Poll ────────────────────────────────────────────────
    poll_url = f"{agent_url.rstrip('/')}/{task_id}"
    poll_interval = config["poll_interval_seconds"]
    max_wait = config["max_wait_seconds"]
    deadline = time.time() + max_wait
    last_state = ""

    while time.time() < deadline:
        try:
            resp = session.get(poll_url, timeout=30)
        except requests.RequestException:
            time.sleep(poll_interval)
            continue
        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue
        try:
            poll_data = resp.json()
        except ValueError:
            time.sleep(poll_interval)
            continue

        err = _extract_api_error(poll_data)
        if err:
            return _make_result(ok=False, error=err, json_data=poll_data, mineru_used=True, mineru_error="agent_poll_api_error")

        inner_poll = poll_data.get("data", poll_data)
        state = (inner_poll.get("state", "") or "").lower()
        last_state = state

        if state in TERMINAL_OK_STATES:
            markdown_url = inner_poll.get("markdown_url", inner_poll.get("markdownUrl", ""))
            if markdown_url:
                try:
                    md_resp = session.get(markdown_url, timeout=60)
                    if md_resp.status_code == 200:
                        return _make_result(ok=True, markdown_text=md_resp.text, json_data=poll_data, mineru_used=True)
                except requests.RequestException:
                    pass
            return _make_result(ok=False, error="Agent done but no markdown_url", json_data=poll_data, mineru_used=True, mineru_error="agent_no_markdown_url")
        if state in TERMINAL_FAIL_STATES:
            err_msg = inner_poll.get("err_msg", inner_poll.get("errMsg", "agent task failed"))
            return _make_result(ok=False, error=f"Agent failed: {err_msg}", json_data=poll_data, mineru_used=True, mineru_error="agent_failed")

        time.sleep(poll_interval)

    return _make_result(ok=False, error=f"Agent task {task_id} timed out after {max_wait}s (last state: {last_state})", json_data={}, mineru_used=True, mineru_error="agent_timeout")


# ═══════════════════════════════════════════════════════════════════════════
# local_fastapi (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def _call_mineru_local_fastapi(file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    session = _build_session(config)
    parse_url = _resolve_url(config, "parse_endpoint", "file_parse")
    try:
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()
    except OSError as e:
        return _make_result(ok=False, error=f"Cannot read file: {e}", mineru_used=True, mineru_error="file_read_error")
    try:
        resp = session.post(parse_url, files={"file": (file_path.name, file_bytes)}, data={"return_md": "true"}, timeout=120)
    except requests.RequestException as e:
        return _make_result(ok=False, error=f"Local FastAPI request failed: {e}", mineru_used=True, mineru_error="api_request_failed")
    if resp.status_code != 200:
        detail = resp.text[:500] if resp.text else ""
        return _make_result(ok=False, error=f"Local FastAPI HTTP {resp.status_code}: {detail}", mineru_used=True, mineru_error=f"http_{resp.status_code}")
    if resp.content[:4] == b"PK\x03\x04":
        md = _extract_md_from_bytes(resp.content)
        if md:
            return _make_result(ok=True, markdown_text=md, mineru_used=True)
        return _make_result(ok=False, error="Zip but no .md inside", mineru_used=True, mineru_error="no_md_in_zip")
    try:
        data = resp.json()
    except ValueError:
        text = resp.text.strip()
        if text:
            return _make_result(ok=True, markdown_text=text, mineru_used=True)
        return _make_result(ok=False, error="Empty response", mineru_used=True, mineru_error="empty_response")
    # Check for markdown in JSON
    for key in ("markdown", "md", "content", "text"):
        if isinstance(data.get(key), str) and data[key].strip():
            return _make_result(ok=True, markdown_text=data[key], json_data=data, mineru_used=True)
    inner = data.get("data", data)
    task_id = inner.get("task_id", inner.get("taskId", ""))
    if task_id:
        poll = _poll_task_status(str(task_id), config)
        if poll["ok"] and poll.get("full_zip_url"):
            zip_bytes = _download_zip(poll["full_zip_url"], config)
            if zip_bytes:
                md, meta = extract_markdown_from_zip(zip_bytes)
                if md:
                    return _make_result(ok=True, markdown_text=md, json_data={**poll.get("data", {}), **meta}, mineru_used=True, raw_output_path=meta.get("raw_output_path"))
        return _make_result(ok=False, error=poll.get("error", "Polling failed"), json_data=data, mineru_used=True, mineru_error="local_polling_failed")
    return _make_result(ok=False, error="Local FastAPI: unexpected response", json_data=data, mineru_used=True, mineru_error="unexpected_local_response")


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def _call_mineru_api(file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    mode = config["api_mode"]
    if mode == "precise":
        return _call_mineru_precise(file_path, config)
    elif mode == "agent":
        return _call_mineru_agent(file_path, config)
    elif mode == "local_fastapi":
        return _call_mineru_local_fastapi(file_path, config)
    else:
        return _make_result(ok=False, error=f"Unknown mode: '{mode}'. Supported: precise, agent, local_fastapi", mineru_used=False, mineru_error=f"unknown_mode_{mode}")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def parse_document_with_mineru(file_path: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return _make_result(ok=True, markdown_text=f"# (Dry Run) MinerU parse of: {file_path.name}\n\nFile: {file_path}\nSuffix: {file_path.suffix}\n\nThis is a placeholder result from MinerU dry_run mode.\n", json_data={"dry_run": True, "file_name": file_path.name, "file_suffix": file_path.suffix}, mineru_used=True)
    config = get_mineru_config()
    if not config["api_key"] and config["api_mode"] != "agent":
        return _make_result(ok=False, error="MINERU_API_KEY is not set", mineru_used=False, mineru_error="no_api_key")
    return _call_mineru_api(file_path, config)


def fallback_to_local_if_failed(file_path: Path, local_func: Callable[[Path], Dict[str, Any]], *, dry_run: bool = False) -> Dict[str, Any]:
    mineru_result = parse_document_with_mineru(file_path, dry_run=dry_run)
    if mineru_result["ok"]:
        return mineru_result
    try:
        local_result = local_func(file_path)
    except Exception as exc:
        return _make_result(ok=False, error=f"MinerU failed and local fallback also failed: {exc}", mineru_used=False, mineru_error=mineru_result.get("mineru_error", "mineru_failed"))
    local_result["backend"] = "local"
    local_result["mineru_used"] = False
    local_result["mineru_error"] = mineru_result.get("mineru_error", "")
    return local_result
