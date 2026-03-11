
# Stereo Matching with Superpixel Graphs and Graph Neural Networks

This repository implements a stereo matching pipeline that operates on superpixels instead of pixels. It builds a region adjacency graph (RAG) from the left image, extracts deep features, and refines disparities using Graph Attention Networks (GATs). Three variants:

- `stereodepth1.py` – baseline using Fibonacci candidates and binary GraphSAGE.
- `stereog.py` – enhanced architecture with a ResNet‑50 backbone, ASPP, cost MLP, GAT refinement, and pixel‑level U‑Net.
- `strg.py` – experimental version adding unsupervised losses (photometric, LR consistency, plane prior).

## 🧠 Method Overview

1. **Superpixel segmentation** (SLIC) on the left image.
2. **Graph construction** – nodes = superpixels, edges = 4‑neighbour adjacency.
3. **Feature extraction** – ResNet‑50 (truncated) + ASPP produces dense feature maps; features are averaged per superpixel.
4. **Disparity candidate generation** – Fibonacci numbers + local dense band.
5. **Cost volume** – per node and candidate: cosine similarity, correlation, and patch‑based AD‑Census are fed into a small MLP to obtain a cost logit.
6. **Initial disparity** – soft‑argmin over the cost volume.
7. **Graph refinement** – two GAT layers refine disparities and predict confidence.
8. **Pixel‑level refinement** – a lightweight U‑Net upsamples to full resolution using the image and confidence map.

## 🗂️ Repository Structure
stereo-graph-matching/
├── src/
│   ├── __init__.py
│   ├── stereodepth1.py          # baseline version
│   ├── stereog.py                # enhanced architecture
│   └── strg.py                   # work‑in‑progress with extra losses
├── data/                          # (optional) small sample images for demo
│   └── sample_left.png
│   └── sample_right.png
├── checkpoints/                   # (empty) where trained models will be saved
├── logs/                           # (empty) training logs
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE                        

___________________________
Inference on a single pair
from src.stereog import StereoGraphNet

model = StereoGraphNet(dmax=128, device='cuda')
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

left = cv2.imread('left.png')[..., ::-1]   # RGB
right = cv2.imread('right.png')[..., ::-1]

result = model.forward_from_paths('left.png', 'right.png', 'disparity.png')
disp = result['pixel_disp_refined']        # numpy array (H, W)
