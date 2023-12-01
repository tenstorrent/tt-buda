from pathlib import Path

import cv2
import numpy as np
import torch
import yolov5
from PIL import Image
from yolov5.models.common import Detections
from yolov5.utils.dataloaders import exif_transpose, letterbox
from yolov5.utils.general import Profile, non_max_suppression, scale_boxes


def data_preprocessing(ims: Image.Image, size: tuple) -> tuple:
    """Data preprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : Image.Image
        Input image
    size : tuple
        Desired image size

    Returns
    -------
    tuple
        List of images, number of samples, filenames, image size, inference size, preprocessed images
    """

    n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames

    for i, im in enumerate(ims):
        f = f"image{i}"  # filename
        im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
        files.append(Path(f).with_suffix(".jpg").name)
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = max(size) / max(s)  # gain
        shape1.append([int(y * g) for y in s])
        ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    shape1 = [size[0] for _ in np.array(shape1).max(0)]  # inf shape
    x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
    x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x) / 255  # uint8 to fp16/32

    return ims, n, files, shape0, shape1, x


def data_postprocessing(
    ims: list,
    x_shape: torch.Size,
    pred: list,
    model: yolov5.models.common.AutoShape,
    n: int,
    shape0: list,
    shape1: list,
    files: list,
) -> Detections:
    """Data postprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : list
        List of input images
    x_shape : torch.Size
        Shape of each image
    pred : list
        List of model predictions
    model : yolov5.models.common.AutoShape
        Model
    n : int
        Number of input samples
    shape0 : list
        Image shape
    shape1 : list
        Inference shape
    files : list
        Filenames

    Returns
    -------
    Detections
        Detection object
    """

    # Create dummy dt tuple (not used but required for Detections)
    dt = (Profile(), Profile(), Profile())

    # Perform NMS
    y = non_max_suppression(
        prediction=pred,
        conf_thres=model.conf,
        iou_thres=model.iou,
        classes=None,
        agnostic=model.agnostic,
        multi_label=model.multi_label,
        labels=(),
        max_det=model.max_det,
    )

    # Scale bounding boxes
    for i in range(n):
        scale_boxes(shape1, y[i][:, :4], shape0[i])

    # Return Detections object
    return Detections(ims, y, files, times=dt, names=model.names, shape=x_shape)
