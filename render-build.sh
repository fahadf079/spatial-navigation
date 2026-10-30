#!/usr/bin/env bash
# Build step for Render (see render.yaml).
# 1. A CPU-only PyTorch: Render has no GPU, and the default build is several GB larger.
# 2. The requirements. ultralytics also installs the full OpenCV, which needs display libraries Render
#    does not have, so it is swapped for the headless build.
# 3. The YOLO weights are downloaded now so the first request does not have to wait for them.
set -euo pipefail
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-server.txt
pip uninstall -y opencv-python opencv-contrib-python || true
pip install --force-reinstall --no-deps opencv-python-headless
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('yolov8n.pt ready')"
