"""
util.py

Preprocessing Utility Functions.
"""
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """ Pad image to square shape and resize. """

    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]                                               # Current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale Ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:                                                     # Only scale down (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]   # WH Padding
    if auto:                                                            # Minimum Rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)                         # WH Padding
    elif scaleFill:                                                     # Stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2                                                             # Divide Padding into 2 Sides
    dh /= 2

    if shape[::-1] != new_unpad:                                        # Resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img
