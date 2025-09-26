import numpy as np
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, ToTensor

class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self, img_scale, keep_ratio=False):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        if self.keep_ratio:
            h, w = img.shape[:2]
            scale = min(self.img_scale[0] / w, self.img_scale[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, self.img_scale, interpolation=cv2.INTER_AREA)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, keep_ratio={})').format(
            self.img_scale, self.keep_ratio)
        return repr_str


class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


class OneImageCollate(object):
    """To collate a image.

    Args:
        deveice (int): the device to put the image
    """

    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def __call__(self, img):
        img = img.unsqueeze(dim=0).to(self.device)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"
    
    
    
def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


class BaseDataProcessor(object):

    def __init__(self, device='cpu'):
        self.pipeline = self.build_data_pipline(device)

    def __call__(self, img):
        """process an image.

        Args:
            img (np.array<uint8>): the input image, in BGR
        """
        return self.pipeline(img)

    def build_data_pipline(self, gpu):
        raise NotImplementedError


class FaceDataProcessor(BaseDataProcessor):
    """image preprocess pipeline for place feature extractor."""

    def build_data_pipline(self, device):
        pipeline = Compose([
            Normalize(
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            ToTensor(),
            OneImageCollate(device)
        ])
        return pipeline



def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idx = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while idx.size > 0:
        i = idx[-1]
        pick[counter] = i
        counter += 1
        idx = idx[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        idx = idx[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    long_side = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - long_side * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - long_side * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + long_side.repeat(2, 1).permute(1, 0)

    return bboxA


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode='area')
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        out = cv2.resize(
            img[box[1]:box[3], box[0]:box[2]], (image_size, image_size),
            interpolation=cv2.INTER_AREA).copy()
    else:
        out = img.crop(box).copy().resize((image_size, image_size),
                                          Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    if isinstance(img, np.ndarray):
        return img.shape[1::-1]
    else:
        return img.size
