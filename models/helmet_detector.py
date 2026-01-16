"""Helmet detection for two-wheelers"""

import cv2
from ultralytics import YOLO



class HelmetDetector:
    """Detects helmet usage on bikes and scooters"""
    
    def __init__(self, helmet_model_path):
        self.model = YOLO(helmet_model_path)




    def detect_helmet(
        self, 
        image, 
        bbox, 
        conf_threshold=0.8,      # ✔ main adjustable parameter
        expand_ratio=(0.2, 0.4)  # ✔ optional
    ):
        """
        Detect helmet & triple riding in expanded region around vehicle bbox.
        """

        # ---- Safety: auto-fix wrong expand_ratio types ----
        if isinstance(expand_ratio, float) or isinstance(expand_ratio, int):
            # If user accidentally passes float, convert safely
            expand_ratio = (expand_ratio, expand_ratio)

        h, w = image.shape[:2]
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        expand_w = int(bbox_w * expand_ratio[0])
        expand_h = int(bbox_h * expand_ratio[1])

        x1 = max(0, bbox[0] - expand_w)
        y1 = max(0, bbox[1] - expand_h)
        x2 = min(w, bbox[2] + expand_w)
        y2 = min(h, bbox[3] + expand_h)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Try passing conf to model (YOLO accepts this)
        try:
            results = self.model(crop, conf=conf_threshold, verbose=False)
        except TypeError:
            results = self.model(crop, verbose=False)

        if not results or not results[0].boxes:
            return None

        with_helmet = 0
        without_helmet = 0

        # -------- Confidence filtering for each box --------
        for box in results[0].boxes:

            # confidence
            conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
            if conf < conf_threshold:
                continue

            # class name
            class_name = self.model.names[int(box.cls[0])]

            if class_name == "with_helmet":
                with_helmet += 1
            elif class_name == "without_helmet":
                without_helmet += 1

        total = with_helmet + without_helmet

        # triple riding
        if total >= 3:
            return "triple riding (without helmet)" if without_helmet > 0 else "triple riding (with helmet)"

        # normal cases
        if without_helmet > 0:
            return "without helmet"
        if with_helmet > 0:
            return "with helmet"

        return None




    def detect_helmet1(self, image, bbox, expand_ratio=(0.2, 0.4)):
        """
        Detect helmet & triple riding in expanded region around vehicle bbox.

        Returns:
            str:
                - "with helmet"
                - "without helmet"
                - "triple riding (with helmet)"
                - "triple riding (without helmet)"
                - None
        """
        h, w = image.shape[:2]
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        expand_w = int(bbox_w * expand_ratio[0])
        expand_h = int(bbox_h * expand_ratio[1])

        x1_exp = max(0, bbox[0] - expand_w)
        y1_exp = max(0, bbox[1] - expand_h)
        x2_exp = min(w, bbox[2] + expand_w)
        y2_exp = min(h, bbox[1] + bbox_h + expand_h)

        crop = image[y1_exp:y2_exp, x1_exp:x2_exp]

        if crop.size == 0:
            return None

        results = self.model(crop, verbose=False)

        if not results or not results[0].boxes:
            return None

        # Count riders
        with_helmet_count = 0
        without_helmet_count = 0

        for box in results[0].boxes:
            class_name = self.model.names[int(box.cls[0])]

            if class_name == "with_helmet":
                with_helmet_count += 1
            elif class_name == "without_helmet":
                without_helmet_count += 1

        total_riders = with_helmet_count + without_helmet_count

        # -----------------------------------------------------
        # ✔️ Triple riding logic
        # -----------------------------------------------------
        if total_riders >= 3:
            if without_helmet_count > 0:
                return "triple riding (without helmet)"
            else:
                return "triple riding (with helmet)"

        # -----------------------------------------------------
        # ✔️ Normal 1-2 rider logic
        # -----------------------------------------------------
        if without_helmet_count > 0:
            return "without helmet"

        if with_helmet_count > 0:
            return "with helmet"

        return None
   

