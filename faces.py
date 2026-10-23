"""Face detection and recognition (OpenCV YuNet + SFace). Optional.

The two small model files are downloaded once into the `.spatial_navigation/models` folder in
your user folder and checked against their published SHA-256 hashes. Without them, everything
else still works; people can be remembered by name but not recognised by face.

If the automatic download fails (no internet, blocked site), download these two files in a
browser and put them in that folder with exactly these names:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
from https://github.com/opencv/opencv_zoo/tree/main/models (face_detection_yunet and
face_recognition_sface).
"""
import hashlib
import os
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.expanduser('~'), '.spatial_navigation', 'models')
BASE = 'opencv/opencv_zoo/main/models'
MODELS = {
    'detector': ('face_detection_yunet_2023mar.onnx',
                 '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
                 f'{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx'),
    'recognizer': ('face_recognition_sface_2021dec.onnx',
                   '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
                   f'{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx'),
}
DETECT_SCORE = 0.8          # how sure the detector must be that it sees a face
MIN_FACE_PIXELS = 40        # ignore faces smaller than this (too far away to recognise)


class FaceUnavailable(Exception):
    """Face recognition cannot be used (models missing or could not be loaded)."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(path_in_repo, destination):
    urls = [f'https://media.githubusercontent.com/media/{path_in_repo}',
            f'https://github.com/{path_in_repo.replace("/main/", "/raw/main/", 1)}']
    last = None
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=60) as response, open(destination, 'wb') as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except Exception as exc:                 # try the next address
            last = exc
    raise FaceUnavailable(f'could not download {os.path.basename(destination)} ({last})')


def ensure_models(model_dir=None, download=_download):
    """Return the two model paths, downloading and verifying them if needed."""
    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for key, (filename, sha, repo_path) in MODELS.items():
        path = os.path.join(model_dir, filename)
        if not (os.path.exists(path) and _sha256(path) == sha):
            print(f'Face recognition: downloading {filename} (first time only)...')
            part = path + '.part'
            download(repo_path, part)
            if _sha256(part) != sha:
                os.remove(part)
                raise FaceUnavailable(f'{filename} did not match its published hash, so it was not used')
            os.replace(part, path)
        paths[key] = path
    return paths


class FaceEngine:
    """Finds faces in a picture and turns each one into a 128-number description."""

    def __init__(self, model_dir=None, download=_download):
        paths = ensure_models(model_dir, download)
        try:
            self.detector = cv2.FaceDetectorYN.create(paths['detector'], '', (320, 320), DETECT_SCORE, 0.3, 50)
            self.recognizer = cv2.FaceRecognizerSF.create(paths['recognizer'], '')
        except (cv2.error, AttributeError) as exc:
            raise FaceUnavailable(f'this OpenCV cannot load the face models ({exc})')

    def detect(self, frame):
        """Faces in the frame, each as (x, y, w, h, raw detector row), biggest first."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(frame)
        faces = []
        if rows is not None:
            for row in rows:
                x, y, fw, fh = (float(v) for v in row[:4])
                if min(fw, fh) >= MIN_FACE_PIXELS:
                    faces.append((x, y, fw, fh, row))
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def embed(self, frame, face):
        """The 128-number description of one detected face (see detect())."""
        aligned = self.recognizer.alignCrop(frame, face[4])
        return np.asarray(self.recognizer.feature(aligned), np.float32).ravel()


def faces_in_boxes(faces, boxes):
    """Pair each person box with the biggest face whose centre lies inside it.

    faces: output of FaceEngine.detect(). boxes: [(x1, y1, x2, y2)].
    Returns a list the same length as boxes: a face, or None.
    """
    out = []
    for x1, y1, x2, y2 in boxes:
        inside = [f for f in faces if x1 <= f[0] + f[2] / 2 <= x2 and y1 <= f[1] + f[3] / 2 <= y2]
        out.append(max(inside, key=lambda f: f[2] * f[3]) if inside else None)
    return out
