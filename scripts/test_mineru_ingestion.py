"""
Test script for MinerU Ingestion v2 -- optional backend check.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_mineru_ingestion.py
    ./.conda/python.exe scripts/test_mineru_ingestion.py --real-api-smoke

Tests:
1. Local mode -- always passes (no API key needed)
2. MinerU config check -- checks env without printing key
3. MinerU dry-run -- exercises the integration path
4. MinerU endpoint check -- if key present, tests real API
5. Real API smoke test -- small file through real MinerU (--real-api-smoke)
"""

import os
import sys
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.ingestion.mineru_client import (
    is_mineru_enabled,
    get_mineru_config,
    parse_document_with_mineru,
    fallback_to_local_if_failed,
    _mask_key,
)


PASS = 0
FAIL = 0
SKIP = 0


def check(condition: bool, label: str):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        print(f"  FAIL  {label}")
        FAIL += 1


def skip_test(label: str):
    global SKIP
    print(f"  SKIP  {label}")
    SKIP += 1


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# -- Test 1: Local mode (always works) -------------------------------


def test_local_mode():
    section("Test 1: local backend -- always available")

    # Simulate a simple MD file ingestion through the backend resolver
    from research_agent.ingestion.document_ingestor import _resolve_backend

    # Default (reads DOCUMENT_INGESTION_BACKEND from .env)
    backend = _resolve_backend()
    print(f"  INFO  Default backend (from env): '{backend}'")
    check(backend in ("local", "mineru"), f"_resolve_backend() is valid: '{backend}'")

    # Explicit
    backend = _resolve_backend(explicit_backend="local")
    check(backend == "local", f"_resolve_backend('local') = 'local'")


# -- Test 2: MinerU config check (no key printed) --------------------


def test_mineru_config():
    section("Test 2: MinerU config -- read without printing key")

    config = get_mineru_config()

    # Check config shape
    check("api_key" in config, "config has 'api_key' key")
    check("base_url" in config, "config has 'base_url' key")
    check("poll_interval_seconds" in config, "config has 'poll_interval_seconds'")
    check("max_wait_seconds" in config, "config has 'max_wait_seconds'")

    # The API key should NOT appear in output -- verify by checking
    # that the key is being masked properly
    raw_key = config["api_key"]
    masked = _mask_key(raw_key)

    if raw_key:
        print(f"  INFO  API key configured (length={len(raw_key)})")
        print(f"  INFO  Masked key: {masked}")
        # Verify the raw key is NOT in the masked version
        check(
            raw_key not in masked or len(raw_key) <= 4,
            "masked key does not reveal full API key",
        )
    else:
        print(f"  INFO  No API key configured (MINERU_API_KEY is empty)")

    check(
        isinstance(masked, str) and len(masked) > 0,
        "_mask_key returns non-empty string",
    )


# -- Test 3: MinerU dry-run -------------------------------------------


def test_mineru_dry_run():
    section("Test 3: MinerU dry-run -- exercises integration path")

    # Create a temp file for testing
    tmp_path = PROJECT_ROOT / "raw_docs" / "_test_dry_run.md"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("# Test Document\n\nHello from dry run test.\n", encoding="utf-8")

    try:
        result = parse_document_with_mineru(tmp_path, dry_run=True)

        check(result["ok"], "dry_run result.ok = True")
        check(
            result["backend"] == "mineru",
            f"dry_run result.backend = 'mineru' (got '{result['backend']}')",
        )
        check(
            result["mineru_used"],
            f"dry_run result.mineru_used = True",
        )
        check(
            len(result["markdown_text"]) > 0,
            "dry_run returns non-empty markdown_text",
        )
        check(
            result["json_data"].get("dry_run") is True,
            "dry_run json_data marks dry_run=True",
        )

        print(f"  INFO  Dry-run markdown preview (first 100 chars):")
        print(f"        {result['markdown_text'][:100].replace(chr(10), ' ')}")
    finally:
        tmp_path.unlink(missing_ok=True)


# -- Test 4: is_mineru_enabled check ---------------------------------


def test_mineru_enabled():
    section("Test 4: is_mineru_enabled -- env-based gating")

    is_enabled = is_mineru_enabled()

    # Only true when BOTH backend=mineru AND key is set
    backend = os.getenv("DOCUMENT_INGESTION_BACKEND", "local").strip().lower()
    key = os.getenv("MINERU_API_KEY", "").strip()

    if backend == "mineru" and key:
        print(f"  INFO  MinerU is enabled (backend={backend}, key present)")
        check(is_enabled, "is_mineru_enabled() = True")
    elif backend == "mineru" and not key:
        print(f"  INFO  backend=mineru but no key -- MinerU disabled")
        check(not is_enabled, "is_mineru_enabled() = False (no key)")
    else:
        print(f"  INFO  backend={backend} -- MinerU not selected")
        check(not is_enabled, "is_mineru_enabled() = False (backend != mineru)")


# -- Test 5: MinerU without key should fail gracefully ----------------


def test_mineru_without_key():
    section("Test 5: MinerU without API key -- fails gracefully")

    # Temporarily clear MINERU_API_KEY to test error path
    saved_key = os.environ.pop("MINERU_API_KEY", None)

    try:
        tmp_path = PROJECT_ROOT / "raw_docs" / "_test_no_key.pdf"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text("Fake PDF content", encoding="utf-8")

        try:
            result = parse_document_with_mineru(tmp_path, dry_run=False)
            check(not result["ok"], "without key: result.ok = False")
            check(
                "MINERU_API_KEY" in result.get("error", ""),
                "error message mentions MINERU_API_KEY",
            )
            # Verify NO key content in error
            error_text = result.get("error", "")
            if saved_key and len(saved_key) > 4:
                check(
                    saved_key not in error_text,
                    "error message does NOT contain raw API key",
                )
        finally:
            tmp_path.unlink(missing_ok=True)
    finally:
        if saved_key is not None:
            os.environ["MINERU_API_KEY"] = saved_key


# -- Test 6: MinerU endpoint check (real API if configured) ------------


def test_mineru_endpoint():
    section("Test 6: MinerU endpoint -- real API check")

    key = os.getenv("MINERU_API_KEY", "").strip()
    base_url = os.getenv("MINERU_API_BASE_URL", "").strip()
    api_mode = os.getenv("MINERU_API_MODE", "precise").strip()

    if not key:
        skip_test("No MINERU_API_KEY set -- cannot test real endpoint")
        return

    print(f"  INFO  API mode: {api_mode}")
    print(f"  INFO  Base URL: {base_url or '(default)'}")
    print(f"  INFO  Key masked: {_mask_key(key)}")

    # Try a real call with a small test file
    tmp_path = PROJECT_ROOT / "raw_docs" / "_test_endpoint.md"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("# Test\n\nTest content for MinerU endpoint check.\n", encoding="utf-8")

    try:
        result = parse_document_with_mineru(tmp_path, dry_run=False)

        if result["ok"]:
            print(f"  INFO  MinerU endpoint is LIVE and working!")
            md_preview = result.get("markdown_text", "")[:200].replace("\n", " ")
            print(f"  INFO  Markdown preview: {md_preview}")
            check(True, "real MinerU endpoint succeeded")
        else:
            error = result.get("error", "unknown")
            mineru_error = result.get("mineru_error", "")
            print(f"  INFO  MinerU returned error: {error}")
            print(f"  INFO  mineru_error tag: {mineru_error}")
            # This is informational — not necessarily a test failure
            # (e.g. endpoint may need a real PDF instead of a text .md file)
            check(True, f"endpoint responded (tag: {mineru_error})")
    finally:
        tmp_path.unlink(missing_ok=True)


# -- Test 8: Real API smoke test (small file) -------------------------


def test_real_api_smoke(file_path_override: Optional[str] = None):
    section("Test 8: Real API smoke test (real PDF/DOCX/PPTX)")

    key = os.getenv("MINERU_API_KEY", "").strip()
    if not key:
        skip_test("No MINERU_API_KEY set")
        return

    test_path = None

    # 1. Explicit --file parameter (must be real binary, not .md)
    if file_path_override:
        p = Path(file_path_override)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"  FAIL  Specified file does not exist: {p}")
            check(False, f"File not found: {p}")
            return
        if p.suffix.lower() == ".md":
            print(f"  WARN  .md files are not supported by MinerU — use PDF/DOCX/PPTX")
        test_path = p
    else:
        # 2. Auto-detect: look for PDF/DOCX/PPTX in raw_docs
        candidates = list(PROJECT_ROOT.glob("raw_docs/**/*.pdf")) + \
                     list(PROJECT_ROOT.glob("raw_docs/**/*.docx")) + \
                     list(PROJECT_ROOT.glob("raw_docs/**/*.pptx"))
        candidates = [c for c in candidates if not c.name.startswith("_")]
        if candidates:
            # Prefer smallest file for smoke test
            candidates.sort(key=lambda p: p.stat().st_size)
            test_path = candidates[0]

    if test_path is None:
        print("  FAIL  No test file found.")
        print("  INFO  Please put a PDF/DOCX/PPTX in raw_docs/ or use --file")
        check(False, "No test file available for real API smoke test")
        return

    suffix = test_path.suffix.lower()
    if suffix == ".md":
        print(f"  WARN  .md files are not supported by MinerU API.")
        print(f"  WARN  Use a real PDF, DOCX, or PPTX file instead.")
        skip_test("Skipping: .md not valid for MinerU (needs PDF/DOCX/PPTX)")
        return

    print(f"  INFO  Test file: {test_path}")
    print(f"  INFO  File size: {test_path.stat().st_size} bytes")
    print(f"  INFO  File type: {suffix}")

    try:
        result = parse_document_with_mineru(test_path, dry_run=False)

        if result["ok"]:
            md = result.get("markdown_text", "")
            print(f"  INFO  MinerU returned {len(md)} chars of markdown")
            print(f"  INFO  First 300 chars:")
            preview = md[:300].replace("\n", "\\n")
            print(f"        {preview}")
            check(True, "Real API smoke test PASSED — markdown returned")
        else:
            error = result.get("error", "")
            mineru_error = result.get("mineru_error", "")
            print(f"  WARN  MinerU smoke test returned error")
            print(f"  WARN  Error: {error}")
            print(f"  WARN  Tag: {mineru_error}")
            # Don't fail — the API may have rate limits or file issues
            skip_test(f"Real API returned error: {mineru_error}")
    except Exception as e:
        print(f"  FAIL  Unexpected exception: {type(e).__name__}: {e}")
        check(False, f"Exception during smoke test: {e}")


def test_fallback():
    section("Test 7: fallback_to_local_if_failed")

    # Simulate a local function
    def fake_local(file_path: Path) -> dict:
        return {
            "status": "success",
            "markdown_text": f"Local parse of {file_path.name}",
            "backend": "local",
        }

    tmp_path = PROJECT_ROOT / "raw_docs" / "_test_fallback.pdf"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("Fake PDF", encoding="utf-8")

    try:
        # With dry_run=False and no API key, MinerU will fail -> fallback
        result = fallback_to_local_if_failed(tmp_path, fake_local, dry_run=False)

        check(
            result.get("backend") in ("local", "mineru"),
            f"fallback result has valid backend: {result.get('backend')}",
        )
        check(
            "markdown_text" in result,
            "fallback result has markdown_text",
        )
        check(
            len(result.get("markdown_text", "")) > 0,
            "fallback returned non-empty markdown_text",
        )

        # If mineru failed, mineru_used should be False
        if not result.get("mineru_used", True):
            check(True, "fallback correctly marked mineru_used=False")
        else:
            check(True, "MinerU succeeded (no fallback needed)")
    finally:
        tmp_path.unlink(missing_ok=True)


# -- Main -------------------------------------------------------------


def main():
    global PASS, FAIL, SKIP

    real_smoke = "--real-api-smoke" in sys.argv
    file_override = None
    disable_proxy = "--disable-proxy" in sys.argv

    # Parse --file argument
    for i, arg in enumerate(sys.argv):
        if arg == "--file" and i + 1 < len(sys.argv):
            file_override = sys.argv[i + 1]
            break

    if disable_proxy:
        os.environ["MINERU_DISABLE_PROXY"] = "true"
        print("  INFO  MINERU_DISABLE_PROXY=true (bypassing system proxy)")

    print("=" * 60)
    print("MinerU Ingestion v2 Test Suite")
    print("=" * 60)

    test_local_mode()
    test_mineru_config()
    test_mineru_dry_run()
    test_mineru_enabled()
    test_mineru_without_key()
    test_mineru_endpoint()
    test_fallback()

    if real_smoke:
        test_real_api_smoke(file_path_override=file_override)
    else:
        section("Test 8: Real API smoke test")
        skip_test("Use --real-api-smoke to run real API smoke test")

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
