"""Vehicle tracking with plate and brand confirmation logic"""

import itertools
import numpy as np
from utils.geometry import iou

class VehicleTracker:
    """Lightweight IOU tracker with plate and brand confirmation logic"""
    
    def __init__(self, iou_threshold=0.3, max_lost=12):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self._id_iter = itertools.count(1)
        self.frame_index = 0

    def _create_track(self, bbox):
        """Create a new track"""
        tid = next(self._id_iter)
        self.tracks[tid] = {
            'id': tid,
            'bbox': np.array(bbox, dtype=int),
            
            # Plate tracking
            'plate_text': None,
            'plate_confirmed_at': None,
            'plate_candidate': None,
            'plate_candidate_count': 0,
            
            # Brand tracking (NEW)
            'brand_name': None,
            'brand_confirmed_at': None,
            'brand_candidate': None,
            'brand_candidate_count': 0,
            'brand_history': [],  # Store recent brand detections
            
            'lost': 0
        }
        return tid

    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_index += 1
        det_boxes = [np.array(d['bbox'], dtype=int) for d in detections]
        assigned_tracks = [-1] * len(det_boxes)
        
        if len(self.tracks) == 0:
            for i, b in enumerate(det_boxes):
                assigned_tracks[i] = self._create_track(b)
            return assigned_tracks

        track_ids = list(self.tracks.keys())
        if len(track_ids) == 0:
            for i, b in enumerate(det_boxes):
                assigned_tracks[i] = self._create_track(b)
            return assigned_tracks

        # Build IOU matrix
        iou_matrix = np.zeros((len(track_ids), len(det_boxes)), dtype=float)
        for ti, tid in enumerate(track_ids):
            tbox = self.tracks[tid]['bbox']
            for di, dbox in enumerate(det_boxes):
                iou_matrix[ti, di] = iou(tbox, dbox)

        # Greedy matching
        used_t = set()
        used_d = set()
        matches = []
        pairs = sorted(
            [(ti, di, iou_matrix[ti, di]) 
             for ti in range(iou_matrix.shape[0]) 
             for di in range(iou_matrix.shape[1])],
            key=lambda x: x[2], reverse=True
        )
        
        for ti, di, score in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di, score))

        # Update matched tracks
        for ti, di, score in matches:
            tid = track_ids[ti]
            assigned_tracks[di] = tid
            self.tracks[tid]['bbox'] = det_boxes[di]
            self.tracks[tid]['lost'] = 0

        # Create new tracks for unmatched detections
        for di in range(len(det_boxes)):
            if assigned_tracks[di] == -1:
                assigned_tracks[di] = self._create_track(det_boxes[di])

        # Handle lost tracks
        matched_tids = set(assigned_tracks)
        to_delete = []
        for tid in list(self.tracks.keys()):
            if tid not in matched_tids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]
            
        return assigned_tracks

    def set_plate(self, track_id, plate_text):
        """Set confirmed plate text for a track"""
        if track_id not in self.tracks:
            return
        self.tracks[track_id]['plate_text'] = plate_text
        self.tracks[track_id]['plate_confirmed_at'] = self.frame_index
        self.tracks[track_id]['lost'] = 0

    def propose_plate(self, track_id, plate_text, confirm_threshold=2):
        """Propose a plate candidate and confirm if seen enough times"""
        if track_id not in self.tracks or not plate_text:
            return
        
        track = self.tracks[track_id]
        if track['plate_text'] is not None:
            return
        
        if track['plate_candidate'] == plate_text:
            track['plate_candidate_count'] += 1
        else:
            track['plate_candidate'] = plate_text
            track['plate_candidate_count'] = 1
        
        if track['plate_candidate_count'] >= confirm_threshold:
            track['plate_text'] = track['plate_candidate']
            track['plate_confirmed_at'] = self.frame_index
            track['plate_candidate'] = None
            track['plate_candidate_count'] = 0
            track['lost'] = 0

    def get_plate(self, track_id):
        """Get confirmed plate text for a track"""
        if track_id not in self.tracks:
            return None
        return self.tracks[track_id].get('plate_text')

    # NEW: Brand tracking methods
    def propose_brand(self, track_id, brand_name, confidence, confirm_threshold=4):
        """
        Propose a brand candidate and confirm if seen consistently
        
        Args:
            track_id: ID of the track
            brand_name: Detected brand name
            confidence: Detection confidence
            confirm_threshold: Number of consistent detections needed
        """
        if track_id not in self.tracks or not brand_name:
            return
        
        track = self.tracks[track_id]
        
        # If already confirmed, don't update
        if track['brand_name'] is not None:
            return
        
        # Add to history (keep last 10 detections)
        track['brand_history'].append((brand_name, confidence))
        if len(track['brand_history']) > 10:
            track['brand_history'].pop(0)
        
        # Check if current candidate matches
        if track['brand_candidate'] == brand_name:
            track['brand_candidate_count'] += 1
        else:
            # New candidate, reset count
            track['brand_candidate'] = brand_name
            track['brand_candidate_count'] = 1
        
        # Confirm if threshold reached
        if track['brand_candidate_count'] >= confirm_threshold:
            # Additional check: ensure confidence is good
            recent_confidences = [c for b, c in track['brand_history'][-confirm_threshold:] 
                                 if b == brand_name]
            
            if len(recent_confidences) >= confirm_threshold:
                avg_conf = sum(recent_confidences) / len(recent_confidences)
                
                if avg_conf >= 0.5:  # Average confidence above 50%
                    track['brand_name'] = track['brand_candidate']
                    track['brand_confirmed_at'] = self.frame_index
                    track['brand_candidate'] = None
                    track['brand_candidate_count'] = 0
                    track['lost'] = 0

    def get_brand(self, track_id):
        """Get confirmed brand name for a track"""
        if track_id not in self.tracks:
            return None
        return self.tracks[track_id].get('brand_name')

    def debug_tracks(self):
        """Get debug information about all tracks"""
        return {
            tid: {
                'bbox': t['bbox'].tolist(),
                'plate': t['plate_text'],
                'brand': t['brand_name'],  # NEW
                'lost': t['lost']
            }
            for tid, t in self.tracks.items()
        }