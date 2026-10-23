"""Detect objects in test.jpg and store them with the camera position and session.

Run live_run.py instead for continuous detection. This script is the single-image version.
The saved position is where the camera was, not where the object is.
"""
import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import connect, read_position

model = YOLO('yolov8n.pt')

img_path = 'test.jpg'
img = cv2.imread(img_path)
if img is None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)

results = model(img)[0]

conn = connect()
pos = read_position()
if pos is None:
    print('No current_position.json found. Run odometry_test.py or live_run.py first for real positions.')
    x = y = z = 0.0
    session_id = 0
else:
    x, y, z, session_id = pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0), pos['session_id']

count = 0
for box in results.boxes:
    label = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    conn.execute(
        'INSERT INTO detections (label, confidence, image_source, timestamp, x, y, z, session_id, pos_source) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (label, conf, img_path, datetime.datetime.now().isoformat(), x, y, z, session_id, 'camera'))
    count += 1
    print(f'{label}: {conf:.2f} at camera position ({x:.2f}, {y:.2f}, {z:.2f}), session {session_id}')

conn.commit()
conn.close()
print(f'Saved {count} detection(s) to spatial_memory.db')
