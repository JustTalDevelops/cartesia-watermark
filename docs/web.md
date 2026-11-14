# Local recording tester

The upload app reconstructs speech, offers A/B playback, and downloads FLOAT WAV. Reference reconstruction is the default. Full-flow, stronger denoising, an experimental smoothed-vocoder method, and the V3 Small edits path are explicit choices. See [reconstruction setup](reconstruction.md) and [V3 limitations](editor.md).

## Start the app

Run from the repository root after installing the inference/web dependencies and reconstruction runtime:

```bash
.venv/bin/python tools/serve_remover.py --host 127.0.0.1 --port 7860
```

Open http://127.0.0.1:7860. Startup requires `ffmpeg`, `ffprobe`, the pinned V3 checkpoint, and the reconstruction Python environment. Reconstruction model assets must be installed separately.

For private remote access, bind to your machine's Tailscale address with `--host`. The app has no account login; run it on a trusted interface. The repository includes an [example user service](../web/cartesia-remover-web.service). Replace `/path/to/cartesia-watermark` with your checkout's absolute path before installing it:

```bash
mkdir -p ~/.config/systemd/user
cp web/cartesia-remover-web.service ~/.config/systemd/user/
# Edit ~/.config/systemd/user/cartesia-remover-web.service with your checkout path.
systemctl --user daemon-reload
systemctl --user enable --now cartesia-remover-web.service
```

The example binds to localhost. To use a tailnet address, edit `ExecStart` in your installed copy. Machine-specific deployment addresses and recordings stay out of the repository.

## Files and limits

- Uploads: up to 100 MB and 15 minutes; mono/stereo at 8–96 kHz. FFmpeg decodes WAV, MP3, M4A, FLAC, and other supported containers.
- Processing: one job at a time, with at most three pending/running jobs. Reconstruction is CPU intensive, especially for stereo.
- Retention: completed uploads expire after 24 hours, with at most 20 retained jobs. Use Delete to remove a completed recording immediately.
- Storage: `.cache/recording_tester/<job-id>/`. The latest result survives a page refresh while retained; `?job=<id>` opens it directly.
- Playback: original and processed previews use equal PCM encoding. The download retains FLOAT WAV precision.

The server never calls Cartesia. A displayed detection result comes only from a saved response matching the exact downloadable WAV hash; new uploads are unchecked. Local listening trials and their evidence expire with the recording.

The reconstruction model loads on the first reconstruction job and stays in an isolated worker for subsequent uploads. This saves repeated Python imports, checkpoint verification, and model loading. It retains model memory while the server runs. A failed or timed-out worker is discarded and recreated for the next job. Restart the server after updating reconstruction code or assets.

The optional Experimental reconstruction method uses the [separately installed smoothed-vocoder candidate](reconstruction.md#smoothed-vocoder-candidate) in its own lazy worker. Full-flow reconstruction is another explicit choice. The reference reconstruction mapping and saved recordings retain their original settings; choosing a new method does not rewrite existing results. Every app method remains local and unchecked unless a saved, exact-file detector response is available.

## Checks and interface notes

The optional browser check uses Playwright/Chromium and locally generated English fixtures. Set `REMOVER_WEB_URL` to target a separate test server; it defaults to localhost. It uploads and deletes only its own test jobs:

```bash
uv pip install --python .venv/bin/python playwright==1.65.1
.venv/bin/python -m playwright install chromium
REMOVER_WEB_URL=http://127.0.0.1:7860 .venv/bin/python tools/check_remover_web.py
```

See Playwright's [installation guide](https://playwright.dev/python/docs/intro) for supported systems and browser dependencies. If its bundled Node runtime cannot start on your image, `PLAYWRIGHT_NODEJS_PATH=/path/to/node` selects an existing compatible Node executable. The check covers the frozen V3 output, both new reconstruction routes, desktop/mobile playback, downloads, refresh, and deletion.

The [design](web/design.md), [product scope](web/product.md), and [upload flow](web/surface.md) explain the existing interface decisions.
