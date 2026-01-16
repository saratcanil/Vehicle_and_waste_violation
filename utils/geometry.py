"""Geometric utilities for bounding boxes and NMS"""

import cv2
import numpy as np

def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def merge_detections(dets, score_thresh=0.05, nms_thresh=0.45):
    """Merge detections using Non-Maximum Suppression"""
    if not dets:
        return []
    boxes = np.array([d['bbox'] for d in dets], dtype=float)
    scores = np.array([d['confidence'] for d in dets], dtype=float)
    rects = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores.tolist(), score_thresh, nms_thresh)
    if len(indices) == 0:
        return []
    try:
        indices = indices.flatten()
    except Exception:
        indices = list(indices)
    return [dets[i] for i in indices]

def scale_and_detect(model, image, scale=1.0, conf=0.15, imgsz=None):
    """Run YOLO on a scaled copy of image"""
    if scale != 1.0:
        new_w = max(2, int(image.shape[1] * scale))
        new_h = max(2, int(image.shape[0] * scale))
        img_for_det = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_for_det = image

    kwargs = {'verbose': False}
    if imgsz:
        kwargs['imgsz'] = imgsz

    results = model(img_for_det, conf=conf, **kwargs)
    found = []
    
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                bxy = box.xyxy[0].tolist()
                bconf = float(box.conf[0].item())
            except Exception:
                continue
            
            x1, y1, x2, y2 = map(int, bxy)
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            found.append({'bbox': [x1, y1, x2, y2], 'confidence': bconf})
    return found

def apply_nms(detections, iou_threshold=0.5):
    """Apply NMS to detections"""
    if not detections:
        return []
    boxes = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]] 
                      for d in detections], dtype=np.float32)
    scores = [d.get('confidence', 0.5) for d in detections]
    rects = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, 0.3, iou_threshold)
    
    if len(indices) > 0:
        try:
            indices = indices.flatten()
        except Exception:
            indices = list(indices)
        return [detections[i] for i in indices]
    return []