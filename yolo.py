"""
yolo.py

Simple wrapper code for loading a pre-trained YOLO-v5 Detector, with utilities for making predictions.
"""
import sys
sys.path.append('./yolov5')
from models.experimental import attempt_load, non_max_suppression, xyxy2xywh

import torch
import torch.nn as nn


class YOLODetector(nn.Module):
    def __init__(self, load_path, device, n_classes=None):
        super(YOLODetector, self).__init__()
        self.yolo = attempt_load(load_path, map_location=device)
        if n_classes is None:
            self.n_classes = len(self.yolo.names)
        else:
            self.n_classes = n_classes

        self.device = device
        self.to(device)

    def get_class_one_hot(self, img):
        pred, _ = self.yolo(img, augment=False)

        # Detections with shape: bxnx6 (x1, y1, x2, y2, conf, cls)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.25, agnostic=True)

        out = [None] * img.shape[0]
        for i, det in enumerate(pred):
            if det is None:
                continue
            for obj in det:
                cls = int(obj[5])
                obj_feat = torch.nn.functional.one_hot(torch.tensor(cls),
                                                       num_classes=self.n_classes).float().to(self.device)
                out[i] = obj_feat
                break
        return out

    def get_object_features(self, img, goal_cls=None):
        pred, _ = self.yolo(img, augment=False)

        # Detections with shape: bxnx6 (x1, y1, x2, y2, conf, cls)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.25, agnostic=True)

        out = [None] * img.shape[0]
        for i, det in enumerate(pred):
            if det is None:
                continue
            for obj in det:
                cls = int(obj[5])
                if goal_cls is None or cls == goal_cls:
                    xywh = (xyxy2xywh(obj[:4].view(1, 4) / img.shape[-1]))

                    # Location in Image + One-Hot Encoding of Object Class
                    obj_feat = torch.cat([
                        xywh[0, :2],
                        torch.nn.functional.one_hot(torch.tensor(cls),
                                                    num_classes=self.n_classes).float().to(self.device)
                    ])

                    out[i] = obj_feat
                    break
        return out
