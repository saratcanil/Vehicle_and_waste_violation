"""Configuration management for vehicle detection system"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelPaths:
    """Model file paths configuration"""
    top_level: str = '/home/user/Vehicle_model/vehicle_detection/yolo-models/top-level/best.pt'
    vehicle_class: str = '/home/user/Vehicle_model/vehicle_detection/yolo-models/vehicle-class/best.pt'
    living_class: str = '/home/user/Vehicle_model/vehicle_detection/yolo-models/living_class/yolo11s.pt'
    helmet: str = '/home/user/Vehicle_model/vehicle_detection/yolo-models/helmet/best.pt'
    plate: str = '/home/user/Vehicle_model/vehicle_detection/yolo-models/plate/best.pt'
    car_brand: str = '/home/user/Vehicle_model/vehicle_detection/yolo-models/car_brand/best.pt'  # NEW

@dataclass
class DetectionConfig:
    """Detection thresholds and parameters"""
    plate_confidence: float = 0.2
    plate_scales: List[float] = field(default_factory=lambda: [1.0, 1.6, 2.0])
    nms_iou_threshold: float = 0.4
    tracker_iou_threshold: float = 0.3
    tracker_max_lost: int = 12
    plate_confirm_threshold: int = 2
    brand_confidence_threshold: float = 0.5
    brand_confirm_threshold: int = 4  # Need 4 consistent detections

@dataclass
class Config:
    """Main configuration class"""
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    
    def validate_paths(self):
        """Validate that all model paths exist"""
        paths = [
            self.model_paths.top_level,
            self.model_paths.vehicle_class,
            self.model_paths.living_class,
            self.model_paths.helmet,
            self.model_paths.plate
        ]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        return True