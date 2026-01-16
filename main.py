"""Main entry point for vehicle detection system"""

import os
import cv2
from tqdm import tqdm

from config import Config
from models import VehicleDetector, NumberPlateProcessor, HelmetDetector, CarBrandDetector  # NEW
from tracking import VehicleTracker
from core import FrameProcessor
from utils.video_utils import get_valid_path

def load_models(config):
    """Load all detection models"""
    print("Loading detection models...")
    try:
        vehicle_detector = VehicleDetector(
            config.model_paths.top_level,
            config.model_paths.vehicle_class,
            config.model_paths.living_class
        )
        
        plate_processor = NumberPlateProcessor(
            config.model_paths.plate,
            confidence_threshold=config.detection.plate_confidence,
            scales=config.detection.plate_scales
        )
        
        helmet_detector = HelmetDetector(config.model_paths.helmet)
        
        # NEW: Load car brand detector
        brand_detector = CarBrandDetector(
            config.model_paths.car_brand,
            confidence_threshold=config.detection.brand_confidence_threshold
        )
        
        tracker = VehicleTracker(
            iou_threshold=config.detection.tracker_iou_threshold,
            max_lost=config.detection.tracker_max_lost
        )
        
        print("All models loaded successfully!")
        return vehicle_detector, plate_processor, helmet_detector, brand_detector, tracker
    
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def process_video(config):
    """Process video file with vehicle detection"""
    # Load models
    vehicle_detector, plate_processor, helmet_detector, brand_detector, tracker = load_models(config)
    
    # Create frame processor (NEW: added brand_detector)
    frame_processor = FrameProcessor(
        vehicle_detector, 
        plate_processor, 
        helmet_detector,
        brand_detector,  # NEW
        tracker
    )
    
    # Get input video path
    video_path = get_valid_path(
        "Enter the path to the .mp4 video file (or type 'exit' to quit): "
    )
    if video_path is None:
        print("Exiting program.")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Setup output video
    output_path = f"annotated_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    with tqdm(total=max(total_frames, 1), desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = frame_processor.process_frame(frame)
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing completed! Processed {frame_count} frames.")
    print(f"Annotated video saved as: {output_path}")

def main():
    """Main entry point"""
    print("=" * 60)
    print("Vehicle Detection System with Brand Recognition")  # NEW
    print("=" * 60)
    
    # Load configuration
    config = Config()
    
    # Validate model paths
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease update model paths in config/config.py")
        return
    
    # Process video
    process_video(config)

if __name__ == "__main__":
    main()