# Flood Risk Prediction from Satellite Images — Implementation Plan

This repository contains a starter implementation and plan for a research-grade project: Flood Risk Prediction from Satellite Images using Deep Learning.

## Purpose
Use Sentinel satellite images plus auxiliary data to detect and classify flood risk (Low / Medium / High) automatically using a CNN-based model. Deliverables include code, a Jupyter notebook, dataset links, and visualization guidance.

## What's included
- `requirements.txt` — Python dependencies
- `notebooks/starter.ipynb` — Starter notebook for data exploration, preprocessing, training and visualization
- `src/train.py` — Training/evaluation CLI and skeleton
- `data/` (not included) — place to store downloaded satellite imagery and auxiliary CSVs

## Dataset sources (recommended)
1. Sentinel-1 / Sentinel-2 (Copernicus Open Access Hub)
   - https://scihub.copernicus.eu/ (register for free)
2. Kaggle flood datasets
   - Flood detection (Sentinel-1 SAR) — search Kaggle for "flood detection sentinel-1"
   - Bangladesh Flood Dataset — https://www.kaggle.com
3. NASA EarthData — https://earthdata.nasa.gov/

Other useful data:
- SRTM / DEM elevation data: https://earthexplorer.usgs.gov/
- Precipitation / rainfall APIs: NOAA, OpenWeatherMap

## Environment & dependencies
Create a Python virtual environment (venv/conda). Install dependencies from `requirements.txt`.

Recommended Python version: 3.9 — 3.11.

## Folder structure
```
flods/
├─ data/                 # raw and processed imagery (not in repo)
├─ notebooks/            # exploratory notebooks
│  └─ starter.ipynb
├─ src/
│  └─ train.py           # training/eval CLI and utilities
├─ requirements.txt
└─ README.md
```

## Quick start (local)
1. Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place satellite imagery into `data/` (organize by region/date), or follow the notebook to download samples.

3. Run training skeleton (example):

```powershell
python src\train.py --data-dir data/ --epochs 3 --batch-size 8
```

Note: The provided `src/train.py` is a skeleton; update dataset loaders and model-building code before large runs.

## Notebook
`notebooks/starter.ipynb` includes:
- Imports and environment checks
- Sample data loader (placeholder)
- Preprocessing examples (resize, normalize, simple augmentation)
- A minimal model training example (using Keras/TensorFlow or PyTorch)
- Visualization: display images, overlay predictions, produce a heatmap

## Next steps / Enhancements
- Implement robust data pipeline for Sentinel (use `rasterio`, `xarray`, or `sentinelsat`)
- Add multispectral and SAR specific preprocessing
- Implement a transfer-learning pipeline using ResNet50/EfficientNet
- Add geospatial overlay with `folium` or export to GeoJSON for QGIS

---

If you want, I can now:
- Fill the notebook with concrete code for Sentinel-2 sample download and preprocessing, OR
- Implement a complete training pipeline with a small sample dataset and runnable tests.

Reply with which you'd like next: `notebook-code` or `full-train`.

## Sample download notes

If you want to fetch a small Sentinel-2 sample for experimenting locally, here are safe options:

- Use the Copernicus Open Access Hub with `sentinelsat` (requires registration).
- Use the AWS Public Datasets (if the tile you need is hosted there) with the AWS CLI for fast downloads:

```powershell
aws s3 cp "s3://<bucket>/<tile>.zip" . --no-sign-request
unzip <tile>.zip
move <extracted>.SAFE data\
```

- Or use the helper `src/download_sample.py` which prints instructions and includes a small `boto3`-based helper to download a single S3 object (public buckets only). Example:

```powershell
python src\download_sample.py
# or, from Python:
from src.download_sample import download_s3_object
download_s3_object('s3://<bucket>/<path-to-file>.zip', out_dir='data')
```

Be careful with disk usage and only download small sample scenes (<1-2 GB) for local experiments.
