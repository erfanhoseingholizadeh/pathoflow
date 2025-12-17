# 🧬 PathoFlow: High-Performance WSI Inference Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge\&logo=docker)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=for-the-badge\&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📖 Overview

**PathoFlow** is a **production-grade inference pipeline** for **Computational Pathology**, designed to process **Gigapixel Whole Slide Images (WSI)**—often exceeding **100,000 × 100,000 pixels**—*without exhausting system RAM*.

Unlike conventional image-processing scripts that fail on medical images, PathoFlow uses a **streaming generator architecture**. It detects tissue regions, tiles them on-the-fly, performs deep learning inference using **ResNet-18**, and reconstructs a **diagnostic probability heatmap**.

---

## ✨ Key Features

* 🐋 **Dockerized & Reproducible**
  Eliminates *dependency hell* (e.g., `libgl1`, `openslide` C-libraries) by packaging the entire runtime into a portable Linux container.

* ⚡ **Memory Efficient**
  Streams patches using Python generators (`yield`), enabling **10GB+ WSI processing** with **constant memory usage (<1GB RAM)**.

* 🔍 **Smart Tissue Detection**
  Automatically filters glass/whitespace using **Otsu Thresholding** and **Morphological Operations**.

* 🏎️ **Batch Inference**
  Maximizes GPU/CPU utilization via dynamic batching.

* 🛡️ **Type Safe**
  Built with **Pydantic** and **Typer** for robust validation and a clean CLI.

---

## 🚀 Getting Started

### Prerequisites

* **Docker Desktop** (Windows WSL2 / macOS / Linux)
* ❌ No local Python installation required

---

### 1️⃣ Build the Engine

Compile the Docker container. This installs OS dependencies, PyTorch, and the PathoFlow engine.

```bash
docker build -t pathoflow:latest .
```

---

### 2️⃣ Run Inference

PathoFlow runs inside a sealed container. To process your files, use **Docker volume mapping** (`-v`).

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  pathoflow:latest \
  /data/YOUR_SLIDE.svs \
  --output /data/heatmap.png \
  --verbose
```

#### Command Breakdown

* `-v $(pwd)/data:/data` → Mounts local `./data` into the container
* `/data/YOUR_SLIDE.svs` → Input slide (container-visible path)
* `--output` → Output heatmap location
* `--verbose` → Detailed progress logs

---

## 🐞 Debugging & Development

To validate tissue detection and tiling **without running full inference**, use the built-in debug tool.

```bash
# Runs debug_wsi.py inside the container
docker run --rm \
  -v $(pwd)/data:/data \
  --entrypoint python \
  pathoflow:latest debug_wsi.py
```

This extracts **5 sample patches** to `data/patches/` for visual sanity checks.

---

## 🏗️ Project Architecture

The codebase follows a **Ports and Adapters** style architecture, ensuring clean separation between core logic, CLI, and runtime.

```text
.
├── README.md              # Documentation & usage guide
├── LICENSE                # MIT license
├── Dockerfile             # Container blueprint
├── .dockerignore          # Docker exclusions
├── pyproject.toml         # Dependencies & metadata
├── debug_wsi.py           # [DEV] Visual sanity checker
├── src
│   └── pathoflow
│       ├── cli.py         # Typer CLI entry point
│       ├── core
│       │   ├── mask.py    # Tissue detection (CV2/Otsu)
│       │   ├── tiler.py   # Streaming grid generator
│       │   └── wsi.py     # OpenSlide wrapper
│       ├── engine
│       │   ├── cnn.py     # ResNet model wrapper
│       │   ├── heatmap.py # Heatmap stitching
│       │   └── interface.py # Model abstraction
│       └── utils
│           └── batching.py # Lazy batch generator
└── tests
    ├── conftest.py        # Pytest fixtures (mock OpenSlide)
    └── test_core.py       # Unit tests
```

---

## ⚠️ Model Status & Disclaimer

**Current State**
The repository includes a **ResNet-18 backbone with an untrained classification head**.

The generated heatmaps demonstrate **pipeline correctness** (tiling, batching, stitching), *not clinical accuracy*.

To use a trained model, load weights via:

```python
model.load_state_dict(...)
```

> **Disclaimer**
> This software is provided *"as is"* for **research and educational purposes only**. It is **not a medical device** and must not be used for clinical diagnosis or patient care. The authors assume no liability for decisions made using this software.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
