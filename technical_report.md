# Technical Report: PathoFlow v2.0 Architecture & Implementation

**Project:** PathoFlow High-Performance WSI Inference Engine
**Version:** 2.0 (Stable Release)

---

## 1. Executive Summary

PathoFlow v2.0 is a specialized computer vision pipeline designed for the automated detection of metastatic tissue in Sentinel Lymph Node whole-slide images (WSIs). The system addresses the computational challenges inherent in processing gigapixel clinical slides (exceeding $100,000 \times 100,000$ pixels) through a containerized, streaming architecture.

The core solution leverages a **ResNet18** Convolutional Neural Network (CNN) fine-tuned on histopathologic benchmarks. To achieve production-viable latency, the engine incorporates a **Smart Tiling** algorithm that filters non-diagnostic regions prior to inference, reducing computational load by approximately 80%. The application is fully containerized via Docker to ensure cross-platform reproducibility and scalability.

---

## 2. System Architecture

The system follows a modular **Producer-Consumer** design pattern to handle memory-intensive I/O operations without causing resource exhaustion.

### 2.1. The Orchestrator (`cli.py`)
The Command Line Interface (CLI) functions as the central controller, implementing **Dependency Injection** to manage component lifecycles.
* **Role:** It instantiates the WSI loader, mask generator, and neural network classifier at runtime.
* **Process Flow:** The CLI manages a synchronous loop that fetches coordinate batches from the `tiler` (Producer) and feeds them to the `CNN` (Consumer), aggregating results in real-time. This ensures a constant memory footprint (O(1)) regardless of input slide dimensions.

### 2.2. Containerization (Docker)
To guarantee environmental consistency, the application is deployed as a stateless Docker container.
* **Base Image:** `python:3.11-slim` was selected for a minimal attack surface and reduced image size.
* **Dependencies:** The container builds the `openslide-tools` C-libraries at the operating system level, resolving cross-platform compatibility issues inherent to Windows and Linux environments.
* **Volume Management:** Data ingestion is handled via Docker Volumes, decoupling the inference logic from the storage layer.

---

## 3. Algorithmic Core: Smart Tiling & Preprocessing

Clinical WSIs typically contain 60-80% non-tissue background (glass). Processing these regions wastes computational resources and introduces false positives. PathoFlow implements a two-stage computer vision pipeline to address this.

### 3.1. Background Filtration (Otsu’s Method)
Before Deep Learning inference occurs, the system generates a boolean tissue mask:
1.  **Downsampling:** A low-resolution thumbnail is generated to facilitate rapid global analysis.
2.  **HSV Transformation:** The RGB thumbnail is converted to the HSV (Hue, Saturation, Value) color space.
3.  **Saturation Thresholding:** As H&E stained tissue exhibits high saturation compared to the white/gray background, the Saturation channel is isolated.
4.  **Otsu’s Binarization:** The algorithm automatically calculates a global threshold that minimizes intra-class variance, creating a binary mask that separates tissue (foreground) from glass (background).
5.  **Coordinate Mapping:** Patches are only extracted if their spatial coordinates map to the foreground mask.

### 3.2. Patch Extraction & Normalization
The `tiler` module extracts square regions from the raw WSI.
* **Input Resolution:** Patches are extracted and immediately resized to **$224 \times 224$ pixels**.
* **Rationale:** This resolution matches the standard input dimensions required by the ResNet architecture, ensuring spatial consistency with the pre-trained feature extractors.

---

## 4. Model Architecture: ResNet18

### 4.1. Network Selection
The **ResNet18** (Residual Network) architecture was selected as the backbone for the classification engine.
* **Architecture Logic:** Histopathologic features (nuclei texture, cellular density) are low-level visual patterns. ResNet18 provides sufficient depth to capture these features without the parameter overhead of deeper networks (e.g., ResNet50/101), resulting in faster inference times and reduced risk of overfitting on patch-based datasets.
* **Transfer Learning:** The network is initialized with **ImageNet** weights to leverage pre-learned low-level feature detectors (edges, gradients), accelerating convergence during training.

### 4.2. Training Protocol
The model was trained using the **PatchCamelyon (PCam)** benchmark dataset under the following configuration:
* **Objective:** Binary Classification (Tumor vs. Normal).
* **Loss Function:** `BCEWithLogitsLoss`. This function integrates a Sigmoid layer and Binary Cross Entropy Loss into a single class for improved numerical stability.
* **Optimizer:** `Adam` (Adaptive Moment Estimation).
* **Medical-Grade Augmentation:** To ensure generalization across different scanners and staining protocols, the training pipeline utilized:
    * `transforms.Resize((224, 224))`: Standardization of input dimensions.
    * `RandomHorizontalFlip` / `RandomVerticalFlip`: Accounting for the rotation-invariant nature of tissue sections.
    * `ColorJitter`: simulating variability in H&E stain intensity.

---

## 5. Performance Engineering

### 5.1. Batch Processing
To maximize hardware utilization, inference is performed in batches.
* The system aggregates $N$ patches (Default: 32) into a single tensor `(32, 3, 224, 224)`.
* This enables parallelized matrix operations on the CPU/GPU, significantly increasing throughput compared to serial processing.
---

## 6. Conclusion

The PathoFlow v2.0 architecture successfully transitions deep learning research into a robust software artifact. By combining heuristic computer vision techniques (Otsu filtration) with deep neural networks, the system achieves a significant reduction in computational cost while maintaining diagnostic precision. The modular, Docker-based design ensures that the tool is deployable in diverse clinical and research environments with zero configuration overhead.
