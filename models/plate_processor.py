"""Number plate detection and OCR processing"""

import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from utils.geometry import scale_and_detect, merge_detections

class NumberPlateProcessor:
    """Multi-scale plate detection with robust OCR"""
    
    def __init__(self, plate_model_path, confidence_threshold=0.2, scales=None):
        self.plate_model = YOLO(plate_model_path)
        self.confidence_threshold = confidence_threshold
        self.scales = scales or [1.0, 1.6, 2.0]
        self.strict_plate_regex = re.compile(r"^KL[0-9]{2}[A-Z]{1,2}[0-9]{4}$")
        
        self.ocr = PaddleOCR(
            lang='en',
            show_log=False,
            det_db_thresh=0.2,
            det_db_box_thresh=0.3,
            rec_image_shape='3, 48, 320',
            use_angle_cls=True,
            drop_score=0.2
        )
    
    def detect_plates(self, roi_img):
        """Run plate detector at multiple scales"""
        all_found = []
        for s in self.scales:
            found = scale_and_detect(
                self.plate_model, roi_img, 
                scale=s, conf=max(0.12, self.confidence_threshold)
            )
            all_found.extend(found)
        
        merged = merge_detections(all_found, score_thresh=0.05, nms_thresh=0.45)
        
        # Filter by size
        filtered = []
        H, W = roi_img.shape[:2]
        for f in merged:
            x1, y1, x2, y2 = f['bbox']
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            
            if w >= 14 and h >= 8 and w < W and h < H:
                if f['confidence'] >= (self.confidence_threshold * 0.45):
                    x1 = max(0, min(W-1, x1))
                    y1 = max(0, min(H-1, y1))
                    x2 = max(0, min(W, x2))
                    y2 = max(0, min(H, y2))
                    filtered.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': f['confidence']
                    })
        return filtered
    
    def preprocess_for_ocr(self, plate_img):
        """Upscale and enhance plate image for OCR"""
        if plate_img is None or plate_img.size == 0:
            return []
        
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Upscale if too small
        h, w = gray.shape[:2]
        target_h, target_w = 80, 240
        scale_h = target_h / h if h < target_h else 1.0
        scale_w = target_w / w if w < target_w else 1.0
        scale = max(scale_h, scale_w, 1.0)
        scale = min(scale, 4.0)
        
        if scale > 1.0:
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), 
                            interpolation=cv2.INTER_CUBIC)
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Unsharp mask
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        unsharp = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)
        
        versions = [
            ("gray", gray),
            ("clahe", enhanced),
            ("unsharp", unsharp)
        ]
        
        # Binarization
        try:
            _, otsu = cv2.threshold(unsharp, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            versions.append(("otsu", otsu))
            versions.append(("otsu_inv", cv2.bitwise_not(otsu)))
        except Exception:
            pass
        
        try:
            adapt = cv2.adaptiveThreshold(
                unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 3
            )
            versions.append(("adaptive", adapt))
        except Exception:
            pass
        
        # Morphological cleaning
        kernel = np.ones((2, 2), np.uint8)
        cleaned = []
        for name, img in versions:
            try:
                cl = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                cl = cv2.morphologyEx(cl, cv2.MORPH_CLOSE, kernel)
                cleaned.append((name, cl))
            except Exception:
                cleaned.append((name, img))
        
        return cleaned
    
    def extract_text_with_paddleocr(self, plate_img_versions):
        """Extract text using PaddleOCR on multiple versions"""
        candidates = []
        
        for version_name, plate_img in plate_img_versions:
            if plate_img is None or plate_img.size == 0:
                continue
            
            if len(plate_img.shape) == 2:
                plate_bgr = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
            else:
                plate_bgr = plate_img
            
            try:
                res = self.ocr.ocr(plate_bgr, cls=False)
            except Exception:
                continue
            
            if not res or not isinstance(res, list) or len(res) == 0:
                continue
            
            page = res[0]
            if not isinstance(page, list):
                continue
            
            extracted = []
            for line in page:
                try:
                    if isinstance(line, list) and len(line) == 2:
                        txt = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                        score = float(line[1][1]) if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
                    elif isinstance(line, tuple) and len(line) == 2:
                        txt = line[1][0] if len(line[1]) > 0 else ""
                        score = float(line[1][1]) if len(line[1]) > 1 else 0.0
                    else:
                        continue
                    
                    text = str(txt).strip()
                    if text:
                        extracted.append((text, score))
                except Exception:
                    continue
            
            if extracted:
                merged_text = "".join([t for t, _ in extracted])
                mean_conf = float(np.mean([s for _, s in extracted]))
                cleaned = self.clean_text(merged_text)
                if cleaned:
                    candidates.append((cleaned, mean_conf, version_name))
        
        if not candidates:
            return ""
        
        # Prefer valid Kerala plates
        valid = [(t, c, v) for t, c, v in candidates 
                if self.strict_plate_regex.match(t)]
        if valid:
            return max(valid, key=lambda x: x[1])[0]
        
        return max(candidates, key=lambda x: (len(x[0]), x[1]))[0]
    
    def clean_text(self, text):
        """Clean and normalize OCR text"""
        if not text:
            return ""
        t = text.upper()
        t = t.replace(" ", "").replace(":", "").replace(".", "")
        t = t.replace("0L", "KL").replace("OL", "KL").replace("XL", "KL")
        t = t.replace("I", "1").replace("O", "0")
        t = re.sub(r'[^A-Z0-9]', '', t)
        return t
    
    def process_plates(self, roi_img):
        """Detect and read plates in image"""
        detected = self.detect_plates(roi_img)
        results = []
        
        for p in detected:
            x1, y1, x2, y2 = p['bbox']
            pad = 8
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(roi_img.shape[1], x2 + pad)
            y2p = min(roi_img.shape[0], y2 + pad)
            
            crop = roi_img[y1p:y2p, x1p:x2p].copy()
            if crop.size == 0:
                continue
            
            versions = self.preprocess_for_ocr(crop)
            text = self.extract_text_with_paddleocr(versions)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'text': text if text else "UNREADABLE",
                'confidence': p['confidence']
            })
        
        return results