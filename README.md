# 🧬 PathoFlow: High-Performance WSI Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?\&logo=docker\&logoColor=white)](Dockerfile)

---

## 📌 Overview

**PathoFlow** is a **clinical-grade deep learning inference engine** for analyzing **gigapixel Whole Slide Images (WSI)** in digital pathology. It is designed to efficiently process massive histopathology slides, intelligently tile them into manageable patches, and apply a custom-trained **Convolutional Neural Network (CNN)** to detect **tumor regions**.

Unlike research-only scripts, PathoFlow is **engineered for production**:

* Memory-efficient WSI tiling
* Artifact and background filtration
* Modular design separating **model weights ("brain")** from inference logic

---

## 🧠 The AI Model

The core model is a **ResNet18** architecture fine-tuned for **histopathologic lymph node classification**.

* **Dataset:** PatchCamelyon (PCam), derived from the **Camelyon16 Challenge**
* **Tissue Type:** Sentinel lymph node sections
* **Objective:** Binary classification of **metastatic tumor tissue** vs **normal tissue** (glass, fat, healthy lymphoid)

---

## ✨ Key Features

### 🖼️ Whole Slide Image Support

* Native support for `.svs`, `.tif`, and `.tiff` formats
* Powered by **OpenSlide** for efficient slide access

### 🧩 Smart & Efficient Tiling

* Automatic detection and removal of background glass regions
* Up to **5× faster inference** by skipping non-informative areas

### 🧠 Deep Learning Engine

* **ResNet18 backbone** optimized for metastasis detection
* Batch-wise inference for memory stability

### 🐳 Portable & Reproducible

* Fully containerized with **Docker**
* Consistent behavior across machines and operating systems

### 📊 Clinical-Ready Reporting

* Visual **heatmaps** highlighting tumor probability
* Structured **text reports** suitable for clinical review

---

## 🔧 Prerequisites

### Hardware

* **RAM:** ≥ 8 GB (16 GB+ recommended for large WSIs)
* **CPU:** Multi-core processor (Intel i5/i7 or AMD Ryzen 5+)
* **Storage:** SSD strongly recommended for fast slide I/O

### Software

* **Docker Desktop** (recommended for isolation)
* **Python 3.11+** (for local execution)
* **Git** (to clone the repository)

---

## 📂 Project Structure

```text
pathoflow/
├── data/                          # Local data (slides & models)
│   └── pathoflow_resnet18_pro.pth # Model weights
├── experiments/                   # Research & training (not in Docker)
│   └── training_engine.py
├── outputs/                       # Generated results
├── src/                           # Production source code
│   └── pathoflow/
│       ├── __init__.py
│       ├── cli.py                 # Main entry point
│       ├── core/                  # WSI processing
│       │   ├── mask.py
│       │   ├── tiler.py
│       │   └── wsi.py
│       ├── engine/                # AI inference
│       │   ├── cnn.py
│       │   └── heatmap.py
│       └── utils/                 # Utilities
│           ├── batching.py
│           └── logger.py
├── .dockerignore
├── Dockerfile
├── LICENSE
├── NOTICES.md                     # Third-party attributions
├── pyproject.toml                 # Python dependencies
├── README.md
└── technical_report.md
```

---

## 📥 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/erfanhoseingholizadeh/pathoflow.git
cd pathoflow
```

---

### 2️⃣ Option A: Docker (Recommended)

Docker guarantees reproducibility by isolating all dependencies.

```bash
docker build -t pathoflow:latest .
```

---

### 2️⃣ Option B: Local Installation

```bash
# Install system dependency (Linux / WSL)
sudo apt-get update && sudo apt-get install -y openslide-tools

# Install Python package
pip install --upgrade pip
pip install -e .
```

---

## 🚀 Usage

Ensure your `.svs` slide and `.pth` model weights are placed in the `data/` directory.

### ▶️ Run with Docker (Universal)

> 💡 The `--smart` flag is highly recommended to skip empty glass regions.

```bash
docker run -it --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/data:/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  pathoflow:latest /data/Your_Slide.svs --smart --verbose
```

---

### ▶️ Run Locally (Python)

```bash
python -m pathoflow.cli "data/Your_Slide.svs" \
  --model-path "data/pathoflow_resnet18_pro.pth" \
  --smart --verbose
```

---

## 📊 Output & Reports

All results are saved to the `outputs/` directory.

### Generated Files

* **Heatmap (`.png`)**
  Color-coded overlay where **red = high tumor probability** and **blue = normal tissue**.

* **Report (`.txt`)**
  Structured diagnostic summary:

```text
--- PATHOLOGY AI REPORT ---
Slide:        CMU-1.svs
Date:         2025-12-23 14:00:00
Duration:     45.2 seconds
Threshold:    0.500

--- DIAGNOSIS ---
Total Tissues Scanned: 1540
Tumor Regions Found:   320
Tumor Burden:          20.78%
```

---

## ⚖️ License & Acknowledgments

### 🧾 License

This project is licensed under the **MIT License** — free to use and modify.

### 📚 Third-Party Components

* **Dataset:** PatchCamelyon (PCam) / Camelyon16 (CC0)
* **Libraries:** OpenSlide (LGPL), PyTorch (BSD)

See `NOTICES.md` for full legal and attribution details.
