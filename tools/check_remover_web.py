"""Exercise the running tester using repository fixtures (no detector requests).

    .venv/bin/python tools/check_remover_web.py

Requires the optional Playwright package and an installed Chromium browser.
Only recordings created by this check are deleted.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

from remover import ROOT

BASE = os.environ.get("REMOVER_WEB_URL", "http://127.0.0.1:7860").rstrip("/")
SOURCE = ROOT / "audio/quality_verified_english/before.wav"
HEADERS = {"Content-Type": "application/octet-stream", "X-Requested-With": "recording-tester"}


def request(path, method="GET", body=None, headers=None):
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=40) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.read()


def wait_job(job):
    # A reconstruction already in the single-worker queue may take minutes.
    for _ in range(600):
        code, body = request("/api/jobs/" + job)
        assert code == 200
        row = json.loads(body)
        if row["state"] in ("done", "error"):
            return row
        time.sleep(.5)
    raise AssertionError("Processing timeout")


def wait_browser(page, expression):
    # Poll through DevTools rather than evaluating a string inside the page;
    # the application's CSP deliberately disallows eval.
    for _ in range(200):
        if page.evaluate("() => (" + expression + ")"):
            return
        page.wait_for_timeout(100)
    raise AssertionError("Browser audio did not become ready")


def main():
    code, health_body = request('/api/health')
    assert code == 200
    default_method = json.loads(health_body)['default_method']
    expected, sample_rate = sf.read(ROOT / "audio/english_blend25/en_006_marked.wav")
    assert request("/api/jobs", "POST", b"hello")[0] == 403
    assert request("/api/jobs", "POST", b"", HEADERS)[0] == 400
    assert request("/api/jobs?strength=1.5", "POST", b"hello", HEADERS)[0] == 422
    assert request("/api/jobs", "POST", b"x", {**HEADERS, "Content-Length": str(100*1024*1024+1)})[0] == 413
    assert request("/api/jobs?method=invalid", "POST", b"hello", HEADERS)[0] == 422
    code, body = request("/api/jobs?filename=reference.wav&method=editor", "POST", SOURCE.read_bytes(), HEADERS)
    assert code == 202
    job = json.loads(body)["id"]
    row = wait_job(job)
    assert row["state"] == "done", row
    code, body = request("/api/jobs/" + job + "/audio/download")
    assert code == 200
    actual, actual_rate = sf.read(io.BytesIO(body))
    assert actual_rate == sample_rate and np.array_equal(actual, expected)
    assert request("/api/jobs/" + job + "/audio/processed", headers={"Range": "bytes=0-63"})[0] == 206
    assert request("/api/jobs/not-a-valid-id")[0] == 404
    code, body = request("/api/jobs?filename=bad.wav", "POST", b"not an audio file", HEADERS)
    assert code == 202
    bad = json.loads(body)["id"]
    assert wait_job(bad)["state"] == "error"
    assert request("/api/jobs/" + bad, "DELETE", headers=HEADERS)[0] == 200
    assert request("/api/jobs/" + job, "DELETE", headers=HEADERS)[0] == 200
    assert request("/api/jobs/" + job)[0] == 404
    print("API: exact PCM, range requests, invalid/empty/oversize upload, edit limits, request protection and delete pass.", flush=True)

    source_audio, source_rate = sf.read(SOURCE)
    for method in ('vocoder', 'reconstruct_fullflow'):
        code, body = request('/api/jobs?filename=reference.wav&method='+method, 'POST', SOURCE.read_bytes(), HEADERS)
        assert code == 202
        reconstruction_job = json.loads(body)['id']
        try:
            row = wait_job(reconstruction_job)
            assert row['state'] == 'done' and row['method'] == method, row
            code, body = request('/api/jobs/'+reconstruction_job+'/audio/download')
            assert code == 200
            actual, rate = sf.read(io.BytesIO(body))
            assert rate == source_rate and actual.shape == source_audio.shape
            assert np.isfinite(actual).all() and np.max(np.abs(actual)) < 1
            assert sf.info(io.BytesIO(body)).subtype == 'FLOAT'
        finally:
            assert request('/api/jobs/'+reconstruction_job, 'DELETE', headers=HEADERS)[0] == 200
    print('API: vocoder and boundary-safe full-flow uploads preserve rate, channels, length and FLOAT downloads.', flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        desktop = browser.new_context(viewport={"width": 1280, "height": 1050}, accept_downloads=True)
        page = desktop.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(BASE, wait_until="networkidle")
        assert page.locator("#method").input_value() == default_method
        assert page.locator("#editor-settings").is_hidden()
        page.screenshot(path=str(ROOT / "results/remover_web_desktop_initial.png"), full_page=True)
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        phone = mobile.new_page()
        phone.goto(BASE, wait_until="networkidle")
        phone.screenshot(path=str(ROOT / "results/remover_web_mobile_initial.png"), full_page=True)
        assert phone.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#method").select_option("editor")
        assert page.locator("#editor-settings").is_visible()
        page.locator("#file-input").set_input_files(str(SOURCE))
        page.locator("#process").click()
        page.locator("#result").wait_for(state="visible", timeout=300000)
        browser_job = page.evaluate("localStorage.getItem('recording-tester-last-job')")
        wait_browser(page, 'document.getElementById("player").readyState >= 2')
        page.evaluate('document.getElementById("player").currentTime=2.25')
        page.locator("#play-processed").click()
        wait_browser(page, 'document.getElementById("player").readyState >= 2 && Math.abs(document.getElementById("player").currentTime-2.25)<.1')
        with page.expect_download() as downloading:
            page.locator("#download").click()
        got, got_rate = sf.read(downloading.value.path())
        assert got_rate == sample_rate and np.array_equal(got, expected)
        page.screenshot(path=str(ROOT / "results/remover_web_desktop_result.png"), full_page=True)
        page.reload(wait_until="networkidle")
        page.locator("#result").wait_for(state="visible", timeout=20000)
        assert page.locator("#result-name").inner_text() == SOURCE.name
        phone.evaluate('(id)=>localStorage.setItem("recording-tester-last-job",id)', browser_job)
        phone.reload(wait_until="networkidle")
        phone.locator("#result").wait_for(state="visible", timeout=20000)
        assert phone.evaluate("document.documentElement.scrollWidth <= innerWidth")
        phone.screenshot(path=str(ROOT / "results/remover_web_mobile_result.png"), full_page=True)
        phone.locator("#delete").click()
        phone.locator("#result").wait_for(state="hidden")
        assert request("/api/jobs/" + browser_job)[0] == 404
        page.reload(wait_until="networkidle")
        page.locator("#error").wait_for(state="visible", timeout=20000)
        assert not errors, errors
        browser.close()
    print("Browser: desktop/mobile upload, native audio, A/B seek, exact download, resume and delete pass.", flush=True)
    freeze = json.loads((ROOT / "results/remover_quality_v3.freeze.json").read_text())
    for path, expected_hash in freeze["inference_code_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected_hash
    report = {"url": BASE, "models": ["V3", "smoothed-vocoder", "full-flow"], "frozen_inference_unchanged": True,
              "API_and_UI_checks_passed": True, "download_PCM_exact_to_verified_model": True,
              "default_method": default_method, "new_reconstruction_methods_preserve_format": True,
              "desktop_and_mobile": True, "browser_errors": errors,
              "checks": ["WAV upload and processing", "Exact output PCM versus verified V3 example",
                         "Vocoder and full-flow rate/channel/length preservation and FLOAT downloads",
                         "Audio range requests for seeking", "A/B switching retains playback position",
                         "Full-quality download through the UI", "Job survives browser refresh",
                         "Mobile layout without horizontal overflow", "Explicit delete removes output",
                         "Invalid and empty audio errors", "Upload size and edit limits",
                         "Cross-origin form protection"]}
    (ROOT / "results/remover_web_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
