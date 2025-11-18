# Flood Risk Prediction from Satellite Images

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)

A starter implementation and plan to detect and classify flood risk from satellite imagery using deep learning. This repository contains a runnable demo pipeline (synthetic/demo mode), helper scripts and a notebook with download/preprocessing templates.

Live repo: https://github.com/Anki2003tha/flood-risk

## Highlights
- Demo training pipeline using a small Keras model (`src/train.py` with `--demo` flag)
- Notebook with examples for downloading and exploring Sentinel data (`notebooks/starter.ipynb`)
- Placeholder dataset utilities and model builders in `src/` so you can plug real data in later

## Quick links
- Code: `src/` (training, data helpers)
- Notebook: `notebooks/starter.ipynb`
- Requirements: `requirements.txt`

## Quick start
1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the demo training (no external data required):

```powershell
python src\train.py --demo --epochs 1 --batch-size 4 --model-out model_demo.h5
```

This runs a short training job on a synthetic dataset and saves `model_demo.h5`.

3. To run on real data, implement/point `--data-dir` to your processed imagery and update the dataset loader in `src/data.py`.

## What you'll find in this repo
- `src/train.py` — CLI training script (supports `--demo` for a runnable quick test)
- `src/data.py` — dataset factory and synthetic demo dataset generator
- `src/model.py` — model builder functions (small CNN + transfer-learning hooks)
- `src/download_sample.py` — helper and examples to download small Sentinel samples
- `notebooks/starter.ipynb` — exploratory notebook with download & preprocessing templates
- `requirements.txt` — Python packages used for the demo and recommended extras

## Recommended workflow
1. Use the notebook to explore a small sample scene (AWS public datasets or Copernicus Hub).
2. Convert tiles to small tiles (e.g., 256x256) and store in `data/` with a label CSV describing flood / not-flood.
3. Implement a robust tf.data / PyTorch Dataset pipeline in `src/data.py` and set `--data-dir` when training.
4. Use transfer learning (ResNet/EfficientNet) for production training.

## Contributing
Contributions are welcome. Open an issue or a pull request describing your change. For large datasets or model files, use Git LFS or host artifacts externally (S3, Zenodo, Hugging Face Datasets).

## License
This project is released under the MIT License. See `LICENSE` for details.

## Contact
If you want help integrating real Sentinel-2/SAR data or adding CI, open an issue or reach out via GitHub Discussions on the repository.

---

If you'd like, I can also:
- Wire Git LFS for `*.h5` and `data/` patterns,
- Add a GitHub Actions workflow that runs the `tests/test_demo.py` on push,
- Or create a short one-page demo `docs/` with screenshots and a quick web viewer.

Reply with which of those you'd like next.
