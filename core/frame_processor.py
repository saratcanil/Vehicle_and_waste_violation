"""Main frame processing logic"""

import cv2
import numpy as np
from utils.geometry import apply_nms

class FrameProcessor:
    """Coordinates all detection models for frame processing"""
    
    def __init__(self, vehicle_detector, plate_processor, helmet_detector, 
                 brand_detector, tracker):
        self.vehicle_detector = vehicle_detector
        self.plate_processor = plate_processor
        self.helmet_detector = helmet_detector
        self.brand_detector = brand_detector  # NEW
        self.tracker = tracker
    
    def process_frame(self, frame):
        """Process a single frame through all detection stages"""
        roi_img = frame.copy()
        
        # Stage 1 & 2: Vehicle and living being detection
        all_detections = self.vehicle_detector.detect(roi_img)
        final_detections = apply_nms(all_detections, iou_threshold=0.4)
        
        # Update tracker
        if final_detections:
            detection_track_ids = self.tracker.update(final_detections)
        else:
            self.tracker.frame_index += 1
            detection_track_ids = []
        
        # Stage 3: Plate detection and tracking
        plate_results = self.plate_processor.process_plates(roi_img)
        self._assign_plates_to_vehicles(final_detections, detection_track_ids, plate_results)
        
        # Stage 4: Brand detection (NEW)
        self._detect_car_brands(roi_img, final_detections, detection_track_ids)
        
        # Stage 5: Helmet detection
        self._detect_helmets(roi_img, final_detections)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, final_detections, detection_track_ids)
        
        return annotated_frame
    
    def _assign_plates_to_vehicles(self, detections, track_ids, plate_results):
        """Assign detected plates to vehicles"""
        used_plates = set()
        
        for i, vehicle in enumerate(detections):
            vehicle_label = vehicle['label'].split(':')[0]
            
            # Only process vehicles that can have plates
            if vehicle_label not in ['car', 'truck', 'bus', 'van', 'emergency_vehicle', 
                                     'auto', 'mini_truck', 'bike', 'scooter']:
                continue
            
            vehicle_bbox = vehicle['bbox'].astype(int)
            best_plate = None
            best_overlap = 0
            best_plate_idx = -1
            
            for plate_idx, plate_result in enumerate(plate_results):
                if plate_idx in used_plates:
                    continue
                
                plate_bbox = np.array(plate_result['bbox'])
                x1_p, y1_p, x2_p, y2_p = plate_bbox
                x1_v, y1_v, x2_v, y2_v = vehicle_bbox
                
                tolerance = 20
                is_inside = (x1_p >= (x1_v - tolerance) and 
                           y1_p >= (y1_v - tolerance) and
                           x2_p <= (x2_v + tolerance) and 
                           y2_p <= (y2_v + tolerance))
                
                if is_inside:
                    plate_area = (x2_p - x1_p) * (y2_p - y1_p)
                    if plate_area > best_overlap:
                        best_overlap = plate_area
                        best_plate = plate_result
                        best_plate_idx = plate_idx
            
            if best_plate and best_plate['text'] != "UNREADABLE":
                plate_text = best_plate['text']
                
                if self.plate_processor.strict_plate_regex.match(plate_text):
                    if i < len(track_ids):
                        track_id = track_ids[i]
                        self.tracker.propose_plate(track_id, plate_text, confirm_threshold=2)
                        used_plates.add(best_plate_idx)
                        detections[i]['plate'] = plate_text
    
    # NEW: Car brand detection method
    def _detect_car_brands(self, image, detections, track_ids):
        """
        Detect car brands for vehicles
        
        Only detects brands for cars, and uses tracking to confirm
        consistent detections over multiple frames
        """
        for i, detection in enumerate(detections):
            vehicle_label = detection['label'].split(':')[0]
            
            # Only detect brands for cars (not bikes, trucks, etc.)
            if vehicle_label not in ['car']:
                continue
            
            # Get vehicle crop
            bbox = detection['bbox'].astype(int)
            x1, y1, x2, y2 = bbox
            
            # Add small padding
            h, w = image.shape[:2]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            vehicle_crop = image[y1:y2, x1:x2].copy()
            
            if vehicle_crop.size == 0:
                continue
            
            # Detect brand
            brand_name, confidence = self.brand_detector.detect_brand(vehicle_crop)
            
            if brand_name and confidence >= 0.5:
                # Get track ID
                track_id = track_ids[i] if i < len(track_ids) else None
                
                if track_id is not None:
                    # Propose brand to tracker (needs 4 consistent detections)
                    self.tracker.propose_brand(track_id, brand_name, confidence, 
                                             confirm_threshold=4)
                    
                    # Store current detection (even if not confirmed yet)
                    detections[i]['brand'] = brand_name
                    detections[i]['brand_confidence'] = confidence
    
    def _detect_helmets(self, image, detections):
        """Detect helmets on two-wheelers"""
        rider_indices = [i for i, d in enumerate(detections) 
                        if d['label'].split(':')[0] in ['bike', 'scooter']]
        
        for rider_idx in rider_indices:
            rider_detection = detections[rider_idx]
            rider_bbox = rider_detection['bbox'].astype(int)
            
            conf_threshold = 0.4
            helmet_status = self.helmet_detector.detect_helmet(image, rider_bbox, conf_threshold)
            if helmet_status:
                detections[rider_idx]['helmet'] = helmet_status
    
    def _annotate_frame(self, frame, detections, track_ids):
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox'].astype(int)
            base_label = detection['label']
            plate_text = detection.get('plate')
            helmet_info = detection.get('helmet')
            brand_info = detection.get('brand')  # NEW
            
            # Get track ID
            track_id = track_ids[i] if i < len(track_ids) else None
            
            # Get tracked plate if available
            if track_id is not None:
                plate_text_tracked = self.tracker.get_plate(track_id)
                plate_text = plate_text_tracked or plate_text
                
                # Get tracked (confirmed) brand if available (NEW)
                brand_confirmed = self.tracker.get_brand(track_id)
                if brand_confirmed:
                    brand_info = brand_confirmed
                else:
                    # Don't show unconfirmed brand
                    brand_info = None
            
            # Build label parts
            parts = [base_label]
            if brand_info:  # NEW: Add brand to label
                parts.append(f"Brand: {brand_info}")
            if plate_text:
                parts.append(str(plate_text))
            if helmet_info:
                parts.append(helmet_info)
            
            # Choose color based on detection type
            color = (255, 255, 0)  # Default: yellow
            if helmet_info in ['without helmet', 'triple riding (without helmet)', 'triple riding (with helmet)']:
                color = (0, 0, 255)  # Red
            elif helmet_info == 'with helmet':
                color = (0, 255, 0)  # Green
            elif base_label in ['car', 'truck', 'bus', 'van']:
                color = (255, 0, 0)  # Blue
            elif base_label in ['bike', 'scooter']:
                color = (0, 255, 255)  # Cyan
            
            # Draw bounding box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw labels with background
            x, y = bbox[0], max(15, bbox[1] - 10)
            for j, line in enumerate(parts):
                display_text = line if len(line) <= 40 else (line[:37] + "...")
                text_y = y + j * 22
                
                # Add text background for better visibility
                (text_w, text_h), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(annotated, (x-2, text_y-text_h-2), 
                            (x+text_w+2, text_y+2), (0, 0, 0), -1)
                cv2.putText(annotated, display_text, (x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated