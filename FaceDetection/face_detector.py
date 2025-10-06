# face_detector.py — YOLO + MTCNN Hybrid Face Detector
from __future__ import annotations
import json
import logging
import os
import os.path as osp
import argparse
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from facedet_utils import FaceDataProcessor
from facedet_mtcnn import MTCNN


# Define a logger for debugging
def _setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("facedetector")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


LOGGER = _setup_logger()


# Utility functions to use in both detectors
def ensure_dir(path: str) -> None:
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def clip_box(x1, y1, x2, y2, w_img, h_img) -> Tuple[int, int, int, int]:
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(int(x2), int(w_img))
    y2 = min(int(y2), int(h_img))
    return x1, y1, x2, y2


def uniform_save_image(img: np.ndarray, save_dir: Optional[str], filename: str) -> Optional[str]:
    if save_dir is None:
        return None
    ensure_dir(save_dir)
    out_path = osp.join(save_dir, filename)
    ok = cv2.imwrite(out_path, img)
    if ok:
        LOGGER.info(f"Salvato: {out_path}")
        return out_path
    LOGGER.warning(f"Errore nel salvataggio: {out_path}")
    return None


def choose_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# FaceDetector (MTCNN)
class FaceDetector:
    def __init__(self, cfg_path="mtcnn.json", weight_path="mtcnn.pth", device=None, log_level=logging.INFO):
        LOGGER.setLevel(log_level)
        self.device = choose_device(device)
        self.model = self._build_mtcnn(cfg_path, weight_path)
        self.model.eval().to(self.device)
        self.data_processor = FaceDataProcessor(self.device)
        LOGGER.info(f"FaceDetector on: {self.device}")

    def _build_mtcnn(self, cfg_path, weight_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        model = MTCNN(**cfg)
        weights = torch.load(weight_path, map_location=self.device)
        model.load_state_dict(weights["state_dict"] if "state_dict" in weights else weights)
        return model

    def _ensure_image(self, img):
        if isinstance(img, str):
            assert osp.isfile(img), f"File not found: {img}"
            bgr = cv2.imread(img)
            if bgr is None:
                raise ValueError(f"CAnnot read: {img}")
            return bgr
        return img

    def detect(self, img, conf_thr=0.9, show=False):
        assert 0 <= conf_thr < 1
        bgr = self._ensure_image(img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        data = self.data_processor(rgb).to(self.device)
        with torch.no_grad():
            results = self.model(data)
        faces, landmarks = results[0][0], results[1][0]
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.cpu().numpy()
        keep = faces[:, -1] > conf_thr
        faces, landmarks = faces[keep], landmarks[keep]
        if show:
            vis = self.draw(bgr, faces, landmarks)
            cv2.imshow("faces", vis)
            cv2.waitKey(0)
        return faces, landmarks

    def draw(self, img, faces, landmarks=None):
        bgr = self._ensure_image(img).copy()
        for i in range(faces.shape[0]):
            x1, y1, x2, y2, prob = faces[i, :5]
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(bgr, f"prob:{prob:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if landmarks is not None:
                for j in range(5):
                    lx, ly = landmarks[i, j, 0], landmarks[i, j, 1]
                    cv2.circle(bgr, (int(lx), int(ly)), 2, (0, 255, 0), 2)
        return bgr

    def crop_faces(self, img, faces, img_scale=160, save_dir=None, save_prefix=None):
        bgr = self._ensure_image(img)
        h, w = bgr.shape[:2]
        crops = []
        for i in range(faces.shape[0]):
            x1, y1, x2, y2 = faces[i, :4].astype(int)
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
            crop = bgr[y1:y2, x1:x2]
            if img_scale is not None and crop.size > 0:
                crop = cv2.resize(crop, (img_scale, img_scale))
            crops.append(crop)
            if save_dir and save_prefix and crop.size > 0:
                fname = f"{save_prefix}_face_{i:02d}.jpg"
                uniform_save_image(crop, save_dir, fname)
        return crops

# HybridDetector (YOLO + MTCNN)
class HybridDetector:
    def __init__(self, yolo_model_size="m", mtcnn_cfg="mtcnn.json", mtcnn_weights="mtcnn.pth", device=None, log_level=logging.INFO):
        LOGGER.setLevel(log_level)
        self.device = choose_device(device)
        yolo_model_path = f"yolov8{yolo_model_size}-seg.pt"
        self.person_detector = YOLO(yolo_model_path)
        self.face_detector = FaceDetector(cfg_path=mtcnn_cfg, weight_path=mtcnn_weights, device=str(self.device))
        LOGGER.info(f"HybridDetector using YOLO='{yolo_model_path}' and device={self.device}")

    def _ensure_image(self, img):
        if isinstance(img, str):
            assert osp.isfile(img), f"File not found: {img}"
            bgr = cv2.imread(img)
            if bgr is None:
                raise ValueError(f"CAnnot read: {img}")
            return bgr
        return img

    def detect(self, img, conf_thr_person=0.8, conf_thr_face=0.9):
        bgr = self._ensure_image(img)
        h_img, w_img = bgr.shape[:2]
        persons, faces, landmarks = [], [], []

        yolo_res = self.person_detector(bgr, verbose=False)[0]
        use_masks = hasattr(yolo_res, "masks") and yolo_res.masks is not None

        for i, box in enumerate(yolo_res.boxes):
            if int(box.cls) != 0 or float(box.conf) < conf_thr_person:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w_img, h_img)
            persons.append([x1, y1, x2, y2, float(box.conf)])

            if use_masks:
                mask = yolo_res.masks.data[i].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (w_img, h_img))
                mask = (mask > 0).astype(np.uint8)
                masked = bgr.copy()
                masked[mask == 0] = 0
                person_crop = masked[y1:y2, x1:x2]
            else:
                person_crop = bgr[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            det_faces, det_landmarks = self.face_detector.detect(rgb_crop, conf_thr=conf_thr_face)

            for f, lm in zip(det_faces, det_landmarks):
                fx1, fy1, fx2, fy2, fconf = f
                gx1, gy1, gx2, gy2 = clip_box(fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1, w_img, h_img)
                faces.append([gx1, gy1, gx2, gy2, float(fconf)])
                landmarks.append(lm + np.array([x1, y1]))

        if len(persons) == 0:  # fallback
            det_faces, det_landmarks = self.face_detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), conf_thr=conf_thr_face)
            for f, lm in zip(det_faces, det_landmarks):
                x1, y1, x2, y2, fconf = f
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w_img, h_img)
                faces.append([x1, y1, x2, y2, float(fconf)])
                landmarks.append(lm)

        return np.array(persons), np.array(faces), np.array(landmarks)

    def draw(self, img, persons, faces, landmarks=None):
        bgr = self._ensure_image(img).copy()
        if persons is not None and len(persons) > 0:
            for (x1, y1, x2, y2, conf) in persons:
                cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(bgr, f"person:{conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if faces is not None and len(faces) > 0:
            for i, (x1, y1, x2, y2, conf) in enumerate(faces):
                cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(bgr, f"face:{conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if landmarks is not None and len(landmarks) > i:
                    for (lx, ly) in landmarks[i]:
                        cv2.circle(bgr, (int(lx), int(ly)), 2, (255, 0, 0), -1)
        return bgr

    def crop_faces(self, img, faces, img_scale=224, save_dir=None, save_prefix=None):
        bgr = self._ensure_image(img)
        h, w = bgr.shape[:2]
        crops = []
        for i, (x1, y1, x2, y2, _) in enumerate(faces):
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
            crop = bgr[y1:y2, x1:x2]
            if img_scale is not None and crop.size > 0:
                crop = cv2.resize(crop, (img_scale, img_scale))
            crops.append(crop)
            if save_dir and save_prefix and crop.size > 0:
                fname = f"{save_prefix}_face_{i:02d}.jpg"
                uniform_save_image(crop, save_dir, fname)
        return crops



### CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Face Detector (YOLO + MTCNN)")
    parser.add_argument("--img", type=str, required=True, help="Input img path")
    parser.add_argument("--save_dir", type=str, default="results", help="Output folder")
    parser.add_argument("--device", type=str, default='cuda', help="Device ('cuda' o 'cpu')")
    parser.add_argument("--yolo-size", type=str, default="m", help="Size YOLOv8 (n/s/m/l/x)")
    args = parser.parse_args()

    det = HybridDetector(yolo_model_size=args.yolo_size, device=args.device)
    persons, faces, landmarks = det.detect(args.img)
    vis = det.draw(args.img, persons, faces, landmarks)
    ensure_dir(args.save_dir)
    save_path = osp.join(args.save_dir, f"{args.img}_result.jpg")
    cv2.imwrite(save_path, vis)
    LOGGER.info(f"Saved in: {save_path}")
    det.crop_faces(args.img, faces, save_dir=args.save_dir, save_prefix=args.img)

    
