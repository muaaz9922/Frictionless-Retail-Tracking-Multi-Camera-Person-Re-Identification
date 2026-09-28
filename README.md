# Frictionless Retail Tracking  
### Multi-Camera Person Re-Identification System

> An end-to-end Computer Vision pipeline that assigns a unique global identity to a person and tracks them across multiple cameras — the core technology behind cashier-less stores like Amazon Go.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Tracking a person in a single camera is relatively easy.  
Tracking the **same person** as they move across 6–10 cameras with different viewpoints, lighting, resolutions, and occlusions is one of the hardest problems in Computer Vision.

This project implements a complete **Multi-Camera Multi-Target Tracking + Re-Identification** pipeline:

1. **Person Detection** (YOLO)
2. **Single-Camera Multi-Object Tracking** (ByteTrack / DeepSORT)
3. **Appearance-based Re-Identification** (deep metric learning)
4. **Cross-Camera Association** & Global ID management
5. Interactive deployment with Gradio

---

## Key Features

- Full phase-by-phase development in Kaggle notebook style
- Strong ReID model (ResNet-50 + Bottleneck + Cross-Entropy + Triplet Loss)
- Online Global ID Manager for multi-camera association
- Rich interactive visualizations at every stage
- Gradio web demo for cross-camera person search
- Clean, modular, and reproducible code

---

## Pipeline Architecture

Multiple Camera Streams

↓

[Detection]          → YOLOv8 / YOLOv11 (person class)

↓

[Single-Cam Tracking] → ByteTrack / DeepSORT (local tracklets)

↓

[Re-Identification]  → Deep embedding model (clothing + appearance)

↓

[Cross-Camera Matching] → Cosine similarity + Hungarian / greedy association

↓

[Global ID Manager]  → Consistent unique ID across the entire camera network

↓

[Deployment]         → Gradio / Streamlit dashboard
text



---

## Dataset

**Market-1501** (primary)

- 1,501 identities
- 6 cameras
- 32,668 bounding boxes
- Standard train / query / gallery split

The project is designed so that other multi-camera ReID datasets (DukeMTMC-reID, MSMT17, Wildtrack, etc.) can be easily plugged in.

---

## Project Structure (Phases)

| Phase | Description                              | Status |
|-------|------------------------------------------|--------|
| 01    | Data Acquisition & Exploration           | Done   |
| 02    | Crop Quality Analysis + Data Pipeline    | Done   |
| 03    | (Skipped – Detection focused)            | -      |
| 04    | ReID Model Training (CE + Triplet)       | Done   |
| 05    | Multi-Camera Association + Global IDs    | Done   |
| 06    | Gradio Deployment Demo                   | Done   |

---

## Installation

```bash
git clone https://github.com/yourusername/frictionless-retail-tracking.git
cd frictionless-retail-tracking

Main dependencies:

torch, torchvision
ultralytics (for YOLO)
gradio
supervision, lap
scikit-learn, plotly, seaborn
```


## Future Improvements

 - Full video multi-camera tracking (not just static images)
 
 - Stronger backbones (OSNet, Vision Transformers, CLIP-based)
 
 - Camera topology + temporal constraints
 
 - Real-time Streamlit / Gradio dashboard with multiple live streams
 Clothing-change robustness
 
 - Integration with product interaction detection (true Amazon Go style)


## Acknowledgments

Market-1501 dataset by Zheng et al.

Torchreid / FastReID community

ByteTrack & DeepSORT authors

Ultralytics YOLO


License
This project is released under the MIT License.

Built for learning, research, and real-world smart retail applications.
