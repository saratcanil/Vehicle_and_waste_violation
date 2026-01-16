"""Car brand detection for vehicles"""

from ultralytics import YOLO

class CarBrandDetector:
    """Detects car brands from cropped vehicle images"""
    
    def __init__(self, brand_model_path, confidence_threshold=0.5):
        """
        Initialize car brand detector
        
        Args:
            brand_model_path: Path to YOLO brand detection model
            confidence_threshold: Minimum confidence for brand detection
        """
        self.model = YOLO(brand_model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect_brand(self, vehicle_crop):
        """
        Detect car brand from cropped vehicle image
        
        Args:
            vehicle_crop: Cropped image of the vehicle
            
        Returns:
            tuple: (brand_name, confidence) or (None, 0.0) if no detection
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0
        
        try:
            results = self.model(vehicle_crop, verbose=False)
            
            if results and results[0].boxes:
                # Get the detection with highest confidence
                best_conf = 0.0
                best_brand = None
                
                for box in results[0].boxes:
                    conf = float(box.conf[0].item())
                    if conf > best_conf and conf >= self.confidence_threshold:
                        best_conf = conf
                        brand_id = int(box.cls[0])
                        best_brand = self.model.names[brand_id]
                
                if best_brand:
                    return best_brand, best_conf
        
        except Exception as e:
            print(f"Error in brand detection: {e}")
        
        return None, 0.0
    
    def get_brand_names(self):
        """Get list of all detectable brand names"""
        return list(self.model.names.values())