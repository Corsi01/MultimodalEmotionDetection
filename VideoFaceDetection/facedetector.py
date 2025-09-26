import json
import os.path as osp
import cv2
import cv2
import numpy as np
import torch
import os.path as osp

from utils.facedet_utils import FaceDataProcessor
from utils.facedet_mtcnn import MTCNN

class FaceDetector(object):

    def __init__(self,
                 cfg_path='mtcnn.json',
                 weight_path='mtcnn.pth',
                 device='cpu'):
        self.device = torch.device(device)
        self.model = self.build_mtcnn(cfg_path, weight_path)
        self.model.eval()
        self.model.to(self.device)
        self.data_processor = FaceDataProcessor(device)

    def build_mtcnn(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = MTCNN(**cfg)
        weights = torch.load(weight_path, map_location=self.device)
        if 'state_dict' in weights:
            model.load_state_dict(weights['state_dict'])
        else:
            model.load_state_dict(weights)
        return model

    def detect(self, img, conf_thr=0.5, show=False):
        assert conf_thr >= 0 and conf_thr < 1
        if isinstance(img, str):
            filename = img
            assert osp.isfile(filename)
            img = cv2.imread(filename)
            # Convert to RGB if needed (FaceDataProcessor expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = self.data_processor(img)
        data = data.to(self.device)
        with torch.no_grad():
            results = self.model(data)
        faces = results[0][0]
        landmarks = results[1][0]
        # Se sono tensori torch, porta su CPU
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu()
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.cpu()
        keep_idx = faces[:, -1] > conf_thr
        faces = faces[keep_idx]
        landmarks = landmarks[keep_idx]
        if show:
            # draw_face expects BGR for OpenCV display
            img_show = self.draw_face(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR), faces, landmarks)
            cv2.imshow('face', img_show)
            cv2.waitKey()
        return faces, landmarks

    def draw_face(self, img, faces, landmarks=None):
        num_faces = faces.shape[0]
        for i in range(num_faces):
            img = cv2.rectangle(img, (int(faces[i, 0]), int(faces[i, 1])),
                                (int(faces[i, 2]), int(faces[i, 3])),
                                (0, 255, 0), 2)
            if landmarks is not None:
                for j in range(5):
                    img = cv2.circle(
                        img,
                        (int(landmarks[i, j, 0]), int(landmarks[i, j, 1])), 2,
                        (0, 255, 0), 2)
            img = cv2.putText(img, 'prob:{:.2f}'.format(faces[i, -1]),
                              (int(faces[i, 0]), int(faces[i, 1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img

    def crop_face(self,
                  img,
                  faces,
                  save_dir=None,
                  save_prefix=None,
                  img_scale=160):
        num_faces = faces.shape[0]
        h, w = img.shape[:2]
        face_list = []
        for i in range(num_faces):
            x1, y1, x2, y2 = faces[i, :4].astype(np.int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face_img = img[y1:y2, x1:x2]
            if img_scale is not None:
                face_img = cv2.resize(face_img, (img_scale, img_scale))
            if save_dir is not None and save_prefix is not None:
                save_path = osp.join(save_dir,
                                     save_prefix + '_{:02d}.jpg'.format(i))
                cv2.imwrite(save_path, face_img)
            face_list.append(face_img)
        return face_list
