import os
import cv2
import face_recognition
import numpy as np
import logging
import json
import gc
import time
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

@dataclass
class AnalysisConfig:
    """Configuration parameters for face analysis"""
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    min_out_of_view_duration: float = 1.0
    frame_processing_interval: int = 15
    face_detection_model: str = "hog"
    face_detection_scale: float = 0.5
    confidence_threshold: float = 0.7
    use_gpu: bool = False
    log_level: str = "INFO"
    max_faces_per_frame: int = 5

@dataclass
class FaceEvent:
    """Data structure for face detection events"""
    event_type: str
    timestamp: float
    face_id: Optional[str] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None

class OutOfViewDetector:
    """Detects and tracks periods when no faces are visible in video"""

    def __init__(self, min_duration: float = 1.0):
        self.min_duration = min_duration
        self.events = []
        self.current_start_time = None
        self.is_out_of_view = False
        self.face_events = []

    def update_status(self, frame_time: float, faces_detected: bool, face_count: int = 0):
        """Update detection status based on current frame"""
        if not faces_detected and not self.is_out_of_view:
            # Start tracking out-of-view period
            self.is_out_of_view = True
            self.current_start_time = frame_time
            self.face_events.append(FaceEvent(
                event_type="out_of_view_start",
                timestamp=frame_time
            ))

        elif faces_detected and self.is_out_of_view:
            # End out-of-view period
            duration = frame_time - self.current_start_time

            if duration >= self.min_duration:
                event_data = {
                    'start': round(self.current_start_time, 2),
                    'end': round(frame_time, 2),
                    'duration': round(duration, 2),
                    'type': 'out_of_view',
                    'severity': 'high' if duration > 5.0 else 'medium' if duration > 2.0 else 'low',
                    'frame_range': {
                        'start_frame': int(self.current_start_time * 30),
                        'end_frame': int(frame_time * 30)
                    }
                }
                self.events.append(event_data)
                self.face_events.append(FaceEvent(
                    event_type="out_of_view_end",
                    timestamp=frame_time,
                    duration=duration
                ))

            self.is_out_of_view = False
            self.current_start_time = None

    def finalize(self, video_duration: float):
        """Handle any ongoing out-of-view period at video end"""
        if self.is_out_of_view and self.current_start_time is not None:
            duration = video_duration - self.current_start_time

            if duration >= self.min_duration:
                event_data = {
                    'start': round(self.current_start_time, 2),
                    'end': round(video_duration, 2),
                    'duration': round(duration, 2),
                    'type': 'out_of_view_final',
                    'severity': 'high' if duration > 5.0 else 'medium' if duration > 2.0 else 'low',
                    'frame_range': {
                        'start_frame': int(self.current_start_time * 30),
                        'end_frame': int(video_duration * 30)
                    }
                }
                self.events.append(event_data)

class FaceAnalyzer:
    """face detection and analysis system for video files"""

    def __init__(self, config: AnalysisConfig = None):
        """Initialize analyzer with configuration"""
        self.config = config or AnalysisConfig()
        self.logger = self._setup_logging()

        # Detection data storage
        self.face_encodings: List[np.ndarray] = []
        self.face_metadata: List[Dict] = []
        self.processing_stats = {}

        # Components
        self.out_of_view_detector = OutOfViewDetector(self.config.min_out_of_view_duration)

        # Performance tracking
        self.frame_processing_times = []
        self.total_frames_processed = 0

    def _setup_logging(self) -> logging.Logger:
        """Configure logging system"""
        logger = logging.getLogger(f"{__name__}.FaceAnalyzer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _validate_input(self, video_path: str) -> bool:
        """Validate video file exists and is accessible"""
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return False

        if not os.path.isfile(video_path):
            self.logger.error(f"Path is not a file: {video_path}")
            return False

        # Check file size (warn if > 1GB)
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        if file_size > 1000:
            self.logger.warning(f"Large file detected: {file_size:.1f}MB")

        return True

    def _get_video_info(self, cap: cv2.VideoCapture) -> Dict:
        """Extract video metadata"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_sec': round(duration, 2),
            'estimated_processing_frames': frame_count // self.config.frame_processing_interval
        }

    def _process_frame(self, frame: np.ndarray, frame_time: float, frame_idx: int) -> Tuple[bool, List]:
        """Detect faces in a single frame"""
        try:
            start_time = time.time()

            # Resize for performance
            small_frame = cv2.resize(frame, (0, 0),
                                   fx=self.config.face_detection_scale,
                                   fy=self.config.face_detection_scale)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(
                rgb_frame, model=self.config.face_detection_model
            )

            faces_detected = len(face_locations) > 0
            processing_time = time.time() - start_time
            self.frame_processing_times.append(processing_time)

            # Limit faces per frame
            if len(face_locations) > self.config.max_faces_per_frame:
                face_locations = face_locations[:self.config.max_faces_per_frame]

            return faces_detected, face_locations

        except Exception as e:
            self.logger.error(f"Error processing frame {frame_idx}: {e}")
            return False, []

    def _extract_face_encodings(self, rgb_frame: np.ndarray, face_locations: List,
                               frame_idx: int, frame_time: float):
        """Extract face encodings for clustering"""
        try:
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for encoding, location in zip(encodings, face_locations):
                self.face_encodings.append(encoding)
                self.face_metadata.append({
                    'frame': frame_idx,
                    'timestamp': round(frame_time, 3),
                    'location': location,
                    'confidence': self.config.confidence_threshold
                })

        except Exception as e:
            self.logger.error(f"Error extracting encodings at frame {frame_idx}: {e}")

    def analyze_video(self, video_path: str) -> Dict:
        """Main analysis method - processes entire video"""
        if not self._validate_input(video_path):
            raise ValueError(f"Invalid video file: {video_path}")

        self.logger.info(f"Starting analysis: {os.path.basename(video_path)}")
        analysis_start = time.time()
        self._reset_state()

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            video_info = self._get_video_info(cap)
            frame_idx = 0
            frames_with_faces = 0

            # Process video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_idx / video_info['fps'] if video_info['fps'] > 0 else 0

                # Process every nth frame
                if frame_idx % self.config.frame_processing_interval == 0:
                    faces_detected, face_locations = self._process_frame(frame, frame_time, frame_idx)
                    self.out_of_view_detector.update_status(frame_time, faces_detected, len(face_locations))

                    if faces_detected:
                        frames_with_faces += 1
                        small_frame = cv2.resize(frame, (0, 0),
                                               fx=self.config.face_detection_scale,
                                               fy=self.config.face_detection_scale)
                        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        self._extract_face_encodings(rgb_frame, face_locations, frame_idx, frame_time)

                frame_idx += 1

            self.total_frames_processed = frame_idx
            self.out_of_view_detector.finalize(video_info['duration_sec'])

            # Store events before processing
            captured_events = [dict(event) for event in self.out_of_view_detector.events]

            # Cluster faces and generate results
            clustered_faces = self._cluster_faces()
            total_time = time.time() - analysis_start
            self._calculate_stats(total_time, video_info, frames_with_faces)

            results = self._generate_results(video_path, video_info, clustered_faces, total_time, captured_events)

            self.logger.info(f"Analysis complete: {results['face_analysis']['unique_faces_detected']} unique faces, "
                           f"{len(captured_events)} out-of-view events")

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

        finally:
            if cap is not None:
                cap.release()
            self._cleanup()

    def _reset_state(self):
        """Reset analyzer state for new analysis"""
        self.face_encodings.clear()
        self.face_metadata.clear()
        self.frame_processing_times.clear()
        self.total_frames_processed = 0
        self.processing_stats.clear()

        # Reset detector
        self.out_of_view_detector.events.clear()
        self.out_of_view_detector.face_events.clear()
        self.out_of_view_detector.is_out_of_view = False
        self.out_of_view_detector.current_start_time = None

    def _cluster_faces(self) -> Dict:
        """Cluster face encodings to identify unique individuals"""
        if not self.face_encodings:
            return {}

        try:
            dbscan = DBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples,
                metric='euclidean',
                n_jobs=-1
            )

            cluster_labels = dbscan.fit_predict(np.array(self.face_encodings))
            unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            # Group faces by cluster
            clustered_faces = {}
            for i, (label, metadata) in enumerate(zip(cluster_labels, self.face_metadata)):
                cluster_name = f"face_{label + 1}" if label != -1 else "noise"

                if cluster_name not in clustered_faces:
                    clustered_faces[cluster_name] = {
                        'face_id': cluster_name,
                        'appearances': [],
                        'cluster_label': int(label) if label != -1 else -1
                    }

                clustered_faces[cluster_name]['appearances'].append(metadata)

            # Calculate statistics for each face
            for face_id, data in clustered_faces.items():
                appearances = data['appearances']
                if appearances:
                    timestamps = [app['timestamp'] for app in appearances]
                    data.update({
                        'first_seen': round(min(timestamps), 2),
                        'last_seen': round(max(timestamps), 2),
                        'total_detections': len(appearances),
                        'avg_confidence': round(np.mean([app.get('confidence', 0.6) for app in appearances]), 3),
                        'duration_in_video': round(max(timestamps) - min(timestamps), 2)
                    })

            return clustered_faces

        except Exception as e:
            self.logger.error(f"Face clustering failed: {e}")
            return {}

    def _calculate_stats(self, total_time: float, video_info: Dict, frames_with_faces: int):
        """Calculate processing performance statistics"""
        processed_frames = len(self.frame_processing_times)

        self.processing_stats = {
            'total_processing_time_sec': round(total_time, 2),
            'avg_frame_processing_time': round(np.mean(self.frame_processing_times), 4) if self.frame_processing_times else 0,
            'frames_processed': processed_frames,
            'frames_with_faces': frames_with_faces,
            'face_detection_rate': round(frames_with_faces / processed_frames * 100, 1) if processed_frames > 0 else 0,
            'processing_fps': round(processed_frames / total_time, 1) if total_time > 0 else 0,
            'video_fps': video_info['fps'],
            'processing_efficiency': round(processed_frames / video_info['frame_count'] * 100, 1)
        }

    def _generate_results(self, video_path: str, video_info: Dict, clustered_faces: Dict, 
                         processing_time: float, captured_events: List[Dict]) -> Dict:
        """Generate final analysis results"""
        unique_faces = {k: v for k, v in clustered_faces.items() if k != "noise"}

        # Remove detailed appearances data
        for face_id, data in unique_faces.items():
            if 'appearances' in data:
                del data['appearances']

        noise_faces = clustered_faces.get('noise', {})

        # Face analysis summary
        face_analysis = {
            'unique_faces_detected': len(unique_faces),
            'total_face_detections': len(self.face_encodings),
            'noise_detections': len(noise_faces.get('appearances', [])),
            'face_detection_confidence': self.config.confidence_threshold,
            'unique_faces': unique_faces,
            'noise_faces': noise_faces if noise_faces.get('appearances') else {}
        }

        # Out-of-view analysis
        total_duration = sum(event['duration'] for event in captured_events)
        out_of_view_analysis = {
            'total_events': len(captured_events),
            'total_duration_sec': round(total_duration, 2),
            'events': captured_events,
            'percentage_out_of_view': 0
        }

        if video_info['duration_sec'] > 0:
            out_of_view_analysis['percentage_out_of_view'] = round(
                (total_duration / video_info['duration_sec']) * 100, 1
            )

        # Generate integrity assessment
        integrity_analysis = self._assess_integrity(face_analysis, out_of_view_analysis, video_info)

        return {
            'video_metadata': {
                'path': os.path.basename(video_path),
                'full_path': video_path,
                **video_info,
                'analysis_timestamp': time.time()
            },
            'face_analysis': face_analysis,
            'out_of_view_analysis': out_of_view_analysis,
            'integrity_analysis': integrity_analysis,
            'processing_statistics': self.processing_stats,
            'analysis_parameters': {
                'dbscan_eps': self.config.dbscan_eps,
                'dbscan_min_samples': self.config.dbscan_min_samples,
                'min_out_of_view_duration': self.config.min_out_of_view_duration,
                'frame_processing_interval': self.config.frame_processing_interval,
                'face_detection_model': self.config.face_detection_model
            }
        }

    def _assess_integrity(self, face_analysis: Dict, out_of_view_analysis: Dict, video_info: Dict) -> Dict:
        """Assess video integrity based on face detection patterns"""
        integrity_score = 1.0
        risk_factors = {}
        suspicious_patterns = []

        # Multiple faces detected
        if face_analysis['unique_faces_detected'] > 1:
            suspicious_patterns.append('multiple_unique_faces_detected')
            risk_score = min(0.4, face_analysis['unique_faces_detected'] * 0.1)
            risk_factors['multiple_faces'] = {
                'unique_faces_count': face_analysis['unique_faces_detected'],
                'risk_contribution': risk_score
            }
            integrity_score -= risk_score

        # High out-of-view percentage
        if out_of_view_analysis['percentage_out_of_view'] > 20:
            suspicious_patterns.append('high_out_of_view_percentage')
            risk_score = min(0.3, out_of_view_analysis['percentage_out_of_view'] / 100)
            risk_factors['out_of_view'] = {
                'percentage': out_of_view_analysis['percentage_out_of_view'],
                'risk_contribution': risk_score
            }
            integrity_score -= risk_score

        # No faces detected
        if face_analysis['unique_faces_detected'] == 0:
            suspicious_patterns.append('no_faces_detected')
            risk_factors['no_faces'] = {'risk_contribution': 0.8}
            integrity_score -= 0.8

        # High noise ratio
        total_detections = face_analysis['total_face_detections']
        noise_detections = face_analysis['noise_detections']
        if total_detections > 0:
            noise_ratio = noise_detections / total_detections
            if noise_ratio > 0.3:
                suspicious_patterns.append('high_noise_in_detections')
                risk_score = min(0.2, noise_ratio)
                risk_factors['high_noise'] = {
                    'noise_ratio': round(noise_ratio, 2),
                    'risk_contribution': risk_score
                }
                integrity_score -= risk_score

        integrity_score = max(0.0, integrity_score)

        # Determine risk level
        if integrity_score >= 0.8:
            risk_level, recommendation = 'LOW', 'ACCEPT'
        elif integrity_score >= 0.6:
            risk_level, recommendation = 'MEDIUM', 'REVIEW'
        elif integrity_score >= 0.4:
            risk_level, recommendation = 'HIGH', 'REVIEW'
        else:
            risk_level, recommendation = 'CRITICAL', 'REJECT'

        return {
            'integrity_score': round(integrity_score, 3),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'suspicious_patterns': suspicious_patterns,
            'risk_factors': risk_factors,
            'primary_face_detected': face_analysis['unique_faces_detected'] == 1,
            'consistent_presence': out_of_view_analysis['percentage_out_of_view'] < 10
        }

    def cleanup(self):
        """Clean up resources"""
        try:
            self.face_encodings.clear()
            self.face_metadata.clear()
            self.frame_processing_times.clear()
            self.processing_stats.clear()
            gc.collect()
        except Exception as e:
            self.logger.warning(f"Cleanup issues: {e}")

    def _cleanup(self):
        """Internal cleanup"""
        gc.collect()


def analyze_video(video_path: str, config: AnalysisConfig = None) -> Dict:
    """Convenience function for video face analysis"""
    analyzer = FaceAnalyzer(config)
    try:
        return analyzer.analyze_video(video_path)
    finally:
        analyzer.cleanup()


