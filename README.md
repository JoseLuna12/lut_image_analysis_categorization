# LUT ML Model Workspace

This workspace builds and curates an image + LUT dataset and a lightweight LUT selection flow.

- Images: Unsplash research dataset and DIV2K samples are used as the base imagery.
- LUTs: Collected via web scraping; stored under `ml_luts/` as `.cube` files.
- Goal: Generate datasets, preview LUT applications on images, and export curated LUT picks.

## Project Layout
- `libs/lut_image.py` – LUT parsing and application utilities.
- `clients/api/` – FastAPI service exposing:
  - `GET /luts` list available LUT labels.
  - `POST /preview` accept an upload or server image path, apply LUTs, return `{ image, variants: [{ label, path, data_url }] }`.
  - `POST /export` save `exports/selection.json` with curated selections (developer-approved LUTs).
  - Run with your Python env: `cd clients/api && pip install -r requirements.txt && python main.py`.
- `server/` – Alternate FastAPI service using `libs/lut_api_final.py`.
  - Run: `cd server && pip install -r requirements.txt && python main.py`.
- `clients/` – React + Vite UI (bun) to upload or point to an image, request previews, pick labels, and trigger export.
  - Run: `cd clients && bun install && bun dev` (set `VITE_API_BASE` if not `http://localhost:8000`).
- `ml_luts/` – LUT files.
- `_raw_*`, `processed_images/`, `raw_unfiltered/` – dataset staging/output directories.
- `lut_cli.py` – Simple CLI for applying LUTs to images and folders.

## Using the Client
1) Start the API (see above).
2) In the UI, upload an image or provide a server-side path (e.g., `processed_images/example.png`).
3) Click “Generate previews” to see LUT-applied variants; click cards to pick labels or open a larger preview.
4) “Export selection” writes `clients/api/exports/selection.json` on the server with the chosen LUTs.

## Notes
- Exports are intended as curated LUT picks by the developer.
- The API mock-applies LUTs using the shared utilities and returns inline PNG previews.
- Adjust `LUT_DIR` or paths in `clients/api/main.py` if you relocate LUTs.
- The CLI supports `data_url`, `save`, and `batch` modes with `--max-images`, `--max-luts`, `--seed`, and `--max-size`.
