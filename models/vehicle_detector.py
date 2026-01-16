"""Vehicle detection logic"""

from ultralytics import YOLO

class VehicleDetector:
    """Handles vehicle detection using hierarchical YOLO models"""
    
    def __init__(self, model_stage1_path, model_stage2_path, living_being_path):
        self.model_stage1 = YOLO(model_stage1_path)
        self.model_stage2 = YOLO(model_stage2_path)
        self.model_lb = YOLO(living_being_path)
        
        self.vehicle_categories = [
            'car', 'scooter', 'bike', 'cycle', 'auto', 
            'truck', 'mini_truck', 'bus', 'van', 'emergency_vehicle'
        ]
        self.living_being_categories = ['person', 'cat', 'dog']
    
    def detect(self, image):
        """Detect vehicles and living beings in image"""
        all_detections = []
        
        # Stage 1: Top-level detection
        results_stage1 = self.model_stage1(image, verbose=False)
        
        if results_stage1 and results_stage1[0].boxes:
            for box in results_stage1[0].boxes:
                super_category = self.model_stage1.names[int(box.cls[0])]
                conf = box.conf[0].item()
                bbox = box.xyxy[0].cpu().numpy()
                
                if super_category == 'living_being':
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped = image[y1:y2, x1:x2]
                    
                    # Sub-classification for living beings
                    class_ids_lb = [
                        k for k, v in self.model_lb.names.items() 
                        if v in self.living_being_categories
                    ]
                    results_lb = self.model_lb(cropped, verbose=False, classes=class_ids_lb)
                    
                    if results_lb and results_lb[0].boxes:
                        for lb_box in results_lb[0].boxes:
                            lb_category = self.model_lb.names[int(lb_box.cls[0])]
                            lb_conf = lb_box.conf[0].item()
                            if lb_conf >= 0.6:
                                all_detections.append({
                                    'label': lb_category,
                                    'bbox': bbox,
                                    'confidence': lb_conf
                                })
                
                elif super_category == 'vehicle':
                    # Sub-classification for vehicles
                    class_ids_stage2 = [
                        k for k, v in self.model_stage2.names.items() 
                        if v in self.vehicle_categories
                    ]
                    results_stage2 = self.model_stage2(image, verbose=False, classes=class_ids_stage2)
                    
                    if results_stage2 and results_stage2[0].boxes:
                        for sub_box in results_stage2[0].boxes:
                            sub_category = self.model_stage2.names[int(sub_box.cls[0])]
                            sub_conf = sub_box.conf[0].item()
                            all_detections.append({
                                'label': sub_category,
                                'bbox': sub_box.xyxy[0].cpu().numpy(),
                                'confidence': sub_conf,
                                'plate': None,
                                'helmet': None
                            })
        
        return all_detections