import json
import os.path as osp
import cv2
import cv2
import numpy as np
import torch
import os.path as osp

from ultralytics import YOLO
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


class HybridDetector:
    def __init__(self, 
                 yolo_model_size="m",   
                 mtcnn_cfg="mtcnn.json", 
                 mtcnn_weights="mtcnn.pth", 
                 person_confidence = 0.9,
                 face_confidence = 0.9,
                 device="cpu"):
        """
        Inizializza YOLOv8 and MTCNN FaceDetector.
        """
        
        yolo_model_path = f'yolov8{yolo_model_size}-seg.pt'
        self.device = device
        self.person_detector = YOLO(yolo_model_path)
        self.face_detector = FaceDetector(cfg_path=mtcnn_cfg, 
                                          weight_path=mtcnn_weights, 
                                          device=device)

    def detect(self, img, conf_thr_person=0.8, conf_thr_face=0.9):
        """
        Step 1: YOLO search people
        Step 2: MTCNN search faces in person
        Step 3: fallback → if YOLO can't find person/people use directly MTCC
        
        Use segmentation and then create a rectangular boxe and image padding with black pixels 
        (avoid faces are detected more than once, useful in cases where there are people near)
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        persons, faces, landmarks = [], [], []

        # Detect w/ YOLO
        results = self.person_detector(img, verbose=False)[0]
        use_masks = hasattr(results, "masks") and results.masks is not None

        for i, box in enumerate(results.boxes):
            if int(box.cls) != 0 or float(box.conf) < conf_thr_person:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            persons.append([int(x1), int(y1), int(x2), int(y2), conf])

            # use masks (verify mask existance allows to use other model like classic 'f'yolov8{model_size}.pt'' 
            # that use directly rectangular bboxes)
            if use_masks:
                mask = results.masks.data[i].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask = (mask > 0).astype(np.uint8)

                # apply
                masked_img = img.copy()
                masked_img[mask == 0] = 0

                # crop bbox with black background
                person_crop = masked_img[int(y1):int(y2), int(x1):int(x2)]
            else:
                # fallback: only bbox in original imagae
                person_crop = img[int(y1):int(y2), int(x1):int(x2)]

            if person_crop.size == 0:
                continue

            # Convert to RGB per MTCNN
            if person_crop.shape[2] == 3:
                person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            det_faces, det_landmarks = self.face_detector.detect(person_crop, conf_thr=conf_thr_face)

            if det_faces is not None and len(det_faces) > 0:
                for f, lm in zip(det_faces, det_landmarks):
                    fx1, fy1, fx2, fy2, fconf = f

                    # Return to reference system of the original img
                    fx1, fy1, fx2, fy2 = int(fx1 + x1), int(fy1 + y1), int(fx2 + x1), int(fy2 + y1)

                    #fx1 = max(fx1, int(x1))
                    #fy1 = max(fy1, int(y1))
                    #fx2 = min(fx2, int(x2))
                    #fy2 = min(fy2, int(y2))

                    faces.append([fx1, fy1, fx2, fy2, float(fconf)])
                    landmarks.append(lm + np.array([x1, y1]))

        # fallback (MTCNN on full image)
        if len(persons) == 0:
            det_faces, det_landmarks = self.face_detector.detect(img, conf_thr=conf_thr_face)
            if det_faces is not None and len(det_faces) > 0:
                for f, lm in zip(det_faces, det_landmarks):
                    fx1, fy1, fx2, fy2, fconf = f
                    faces.append([int(fx1), int(fy1), int(fx2), int(fy2), float(fconf)])
                    landmarks.append(lm)

        persons = np.array(persons) if len(persons) > 0 else np.array([])
        faces = np.array(faces) if len(faces) > 0 else np.array([])
        landmarks = np.array(landmarks) if len(landmarks) > 0 else np.array([])

        return persons, faces, landmarks

    def draw(self, img, persons, faces, landmarks=None):
        
        img_show = img.copy()
        if isinstance(img_show, str):
            img_show = cv2.imread(img_show)
            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        # people
        if persons is not None and len(persons) > 0:
            for (x1, y1, x2, y2, conf) in persons:
                cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_show, f'person:{conf:.2f}', (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # faces
        if faces is not None and len(faces) > 0:
            for i, (x1, y1, x2, y2, conf) in enumerate(faces):
                cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_show, f'face:{conf:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                if landmarks is not None and len(landmarks) > i:
                    for (lx, ly) in landmarks[i]:
                        cv2.circle(img_show, (int(lx), int(ly)), 2, (255,0,0), -1)

        return img_show

    #def crop_faces(self, img, faces, img_scale=224):

     #   crops = []
      #  h, w = img.shape[:2]
       # for (x1, y1, x2, y2, _) in faces:
        #    x1, y1 = max(0, int(x1)), max(0, int(y1))
         #   x2, y2 = min(w, int(x2)), min(h, int(y2))
          #  face_img = img[y1:y2, x1:x2]
           # if img_scale is not None and face_img.size > 0:
            #    face_img = cv2.resize(face_img, (img_scale, img_scale))
            #crops.append(face_img)
        #return crops

    def crop_faces(self, img, faces, img_scale=224):
        crops = []
        h_img, w_img = img.shape[:2]

        for (x1, y1, x2, y2, _) in faces:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # original bb size
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

        # compute coordinate to get a square crop
            d = abs(w - h)
            if w < h:
                x1 = max(0, x1 - d // 2)
                x2 = min(w_img, x2 + d - d // 2)
            elif h < w:
                y1 = max(0, y1 - d // 2)
                y2 = min(h_img, y2 + d - d // 2)

            face_img = img[y1:y2, x1:x2]

        # resize if requested
            if img_scale is not None and face_img.size > 0:
                face_img = cv2.resize(face_img, (img_scale, img_scale))

            crops.append(face_img)

        return crops