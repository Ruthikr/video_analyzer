import os
import cv2
import json
import datetime
import time
import gc
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
from gaze_tracking import GazeTracking

warnings.filterwarnings("ignore")

@dataclass
class GazeConfig:
    """Configuration parameters for gaze analysis"""
    min_event_duration: float = 2.0
    horizontal_thresholds: Tuple[float, float] = (0.35, 0.65)
    vertical_thresholds: Tuple[float, float] = (0.43, 0.57)
    frame_processing_interval: int = 3
    confidence_threshold: float = 0.7
    log_level: str = "INFO"
    max_tracking_failures: int = 100
    early_stop_failure_rate: float = 0.95
    max_consecutive_failures: int = 30

@dataclass
class GazeEvent:
    """Data structure for gaze tracking events"""
    event_type: str
    direction: str
    start_timestamp: float
    end_timestamp: float
    duration: float
    start_frame: int
    end_frame: int
    horizontal_ratio: Optional[float] = None
    vertical_ratio: Optional[float] = None
    confidence: float = 0.7

class GazeAnalyzer:
    """Advanced gaze tracking system for video analysis"""

    def __init__(self, config: GazeConfig = None):
        """Initialize analyzer with configuration"""
        self.config = config or GazeConfig()
        self.logger = self._setup_logging()

        # Initialize gaze tracking
        try:
            self.gaze_tracker = GazeTracking()
        except Exception as e:
            self.logger.error(f"Failed to initialize gaze tracking: {e}")
            raise

        # Tracking state
        self.events: List[GazeEvent] = []
        self.current_gaze = None
        self.current_event_start = None
        self.current_event_start_frame = None

        # Statistics and monitoring
        self.processing_stats = {}
        self.frame_processing_times = []
        self.tracking_failures = 0
        self.consecutive_failures = 0
        self.total_frames_processed = 0

    def _setup_logging(self) -> logging.Logger:
        """Configure logging system"""
        logger = logging.getLogger(f"{__name__}.GazeAnalyzer")
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

        # Check file size (warn if > 2GB for gaze tracking)
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        if file_size > 2000:
            self.logger.warning(f"Large file detected: {file_size:.1f}MB - processing may be slow")

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

    def _get_gaze_direction(self) -> Optional[str]:
        """Detect gaze direction with configurable thresholds"""
        try:
            if not self.gaze_tracker.pupils_located:
                return None

            h_ratio = self.gaze_tracker.horizontal_ratio()
            v_ratio = self.gaze_tracker.vertical_ratio()

            if h_ratio is None or v_ratio is None:
                return None

            # Use configurable thresholds
            right_threshold, left_threshold = self.config.horizontal_thresholds
            up_threshold, down_threshold = self.config.vertical_thresholds

            # Determine horizontal direction
            if h_ratio < right_threshold:
                horizontal_dir = "RIGHT"
            elif h_ratio > left_threshold:
                horizontal_dir = "LEFT"
            else:
                horizontal_dir = "CENTER"

            # Determine vertical direction
            if v_ratio < up_threshold:
                vertical_dir = "UP"
            elif v_ratio > down_threshold:
                vertical_dir = "DOWN"
            else:
                vertical_dir = "CENTER"

            # Combine directions
            if horizontal_dir == "CENTER" and vertical_dir == "CENTER":
                return "CENTER"
            elif horizontal_dir == "CENTER":
                return vertical_dir
            elif vertical_dir == "CENTER":
                return horizontal_dir
            else:
                return f"{horizontal_dir}_{vertical_dir}"

        except Exception as e:
            self.logger.warning(f"Error in gaze direction detection: {e}")
            return None

    def _process_frame(self, frame: np.ndarray, frame_time: float, frame_idx: int) -> Optional[str]:
        """Process single frame for gaze detection"""
        try:
            start_time = time.time()
            self.gaze_tracker.refresh(frame)
            direction = self._get_gaze_direction()

            processing_time = time.time() - start_time
            self.frame_processing_times.append(processing_time)

            # Track failures
            if direction is None:
                self.consecutive_failures += 1
                self.tracking_failures += 1

                # Log excessive failures periodically
                if self.consecutive_failures == self.config.max_tracking_failures:
                    self.logger.warning(f"Tracking failure threshold reached: {self.consecutive_failures} consecutive failures")
                elif self.consecutive_failures > self.config.max_tracking_failures and self.consecutive_failures % 300 == 0:
                    self.logger.warning(f"Extended tracking failures: {self.consecutive_failures} consecutive failures")
            else:
                # Reset consecutive failures on successful detection
                if self.consecutive_failures > self.config.max_tracking_failures:
                    self.logger.info("Gaze tracking recovered")
                self.consecutive_failures = 0

            return direction

        except Exception as e:
            self.logger.error(f"Error processing frame {frame_idx}: {e}")
            self.tracking_failures += 1
            self.consecutive_failures += 1
            return None

    def _validate_video_for_tracking(self, video_path: str) -> Dict:
        """Pre-validate video suitability for gaze tracking"""
        validation_results = {
            'is_suitable': True,
            'warnings': [],
            'recommendations': []
        }

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                validation_results['is_suitable'] = False
                validation_results['warnings'].append("Cannot open video file")
                return validation_results

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = min(10, frame_count // 10)
            faces_detected_count = 0
            frames_checked = 0

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            for i in range(sample_frames):
                frame_pos = int((i / sample_frames) * frame_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    frames_checked += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        faces_detected_count += 1

            cap.release()

            # Analyze results
            face_detection_rate = faces_detected_count / frames_checked if frames_checked > 0 else 0

            if face_detection_rate < 0.3:
                validation_results['warnings'].append(f"Low face detection rate: {face_detection_rate:.1%}")
                validation_results['recommendations'].append("Video may not be suitable for reliable gaze tracking")

            if face_detection_rate < 0.1:
                validation_results['is_suitable'] = False
                validation_results['warnings'].append("Very low face visibility - gaze tracking likely to fail")

            validation_results['face_detection_rate'] = round(face_detection_rate, 2)
            validation_results['frames_sampled'] = frames_checked

        except Exception as e:
            self.logger.warning(f"Video validation failed: {e}")
            validation_results['warnings'].append(f"Validation error: {e}")

        return validation_results

    def _should_continue_processing(self, frame_idx: int, video_info: Dict) -> bool:
        """Determine if processing should continue based on failure rate"""
        # Stop if past 25% of video with >95% failure rate
        if frame_idx > video_info['frame_count'] * 0.25:
            if self.tracking_failures > 0:
                failure_rate = self.tracking_failures / frame_idx
                if failure_rate > self.config.early_stop_failure_rate:
                    self.logger.warning(f"Stopping due to high failure rate: {failure_rate:.1%}")
                    return False
        return True

    def analyze_video(self, video_path: str) -> Dict:
        """Main video gaze analysis method"""
        if not self._validate_input(video_path):
            raise ValueError(f"Invalid video file: {video_path}")

        # Pre-validate video
        validation = self._validate_video_for_tracking(video_path)
        self.logger.info(f"Video validation - Face detection rate: {validation.get('face_detection_rate', 0):.1%}")

        for warning in validation.get('warnings', []):
            self.logger.warning(warning)

        if not validation['is_suitable']:
            self.logger.error("Video not suitable for gaze tracking")
            return self._generate_failed_results(video_path, "Video unsuitable for gaze tracking")

        self.logger.info(f"Starting gaze analysis: {os.path.basename(video_path)}")
        analysis_start = time.time()
        self._reset_state()

        cap = None
        early_stop = False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            video_info = self._get_video_info(cap)
            consecutive_gaze_failures = 0
            max_allowed_failures = self.config.max_consecutive_failures

            frame_count = 0
            frames_with_gaze = 0
            last_progress_log = 0

            # Process video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / video_info['fps'] if video_info['fps'] > 0 else frame_count * 0.033

                # Check if processing should continue
                if frame_count > 300 and not self._should_continue_processing(frame_count, video_info):
                    early_stop = True
                    break

                # Process frame
                if frame_count % self.config.frame_processing_interval == 0:
                    gaze_direction = self._process_frame(frame, frame_time, frame_count)

                    if gaze_direction:
                        frames_with_gaze += 1
                        consecutive_gaze_failures = 0

                        # Handle direction changes
                        if gaze_direction != self.current_gaze:
                            self._handle_gaze_change(gaze_direction, frame_time, frame_count)
                    else:
                        # Handle consecutive failures
                        consecutive_gaze_failures += 1

                        if (consecutive_gaze_failures >= max_allowed_failures and 
                            self.current_gaze is not None):
                            # End current event due to tracking loss
                            failure_start_time = frame_time - (consecutive_gaze_failures * self.config.frame_processing_interval / video_info['fps'])
                            failure_start_frame = frame_count - (consecutive_gaze_failures * self.config.frame_processing_interval)
                            self._handle_gaze_change(None, failure_start_time, failure_start_frame)
                            consecutive_gaze_failures = 0

                frame_count += 1
                self.total_frames_processed = frame_count

                # Progress logging
                progress_interval = max(1800, video_info['frame_count'] // 20)
                if frame_count - last_progress_log >= progress_interval:
                    progress = (frame_count / video_info['frame_count']) * 100
                    self.logger.info(f"Progress: {frame_count}/{video_info['frame_count']} frames ({progress:.1f}%)")
                    last_progress_log = frame_count

            # Finalize processing
            if early_stop:
                self.logger.warning("Processing stopped early due to excessive tracking failures")

            final_duration = frame_count / video_info['fps'] if video_info['fps'] > 0 else frame_count * 0.033
            self._finalize_last_event(final_duration, frame_count)

            total_time = time.time() - analysis_start
            self._calculate_stats(total_time, video_info, frames_with_gaze)

            results = self._generate_results(video_path, video_info, total_time)

            if early_stop:
                results['processing_statistics']['early_stop'] = True
                results['processing_statistics']['frames_processed_percentage'] = round((frame_count / video_info['frame_count']) * 100, 1)

            if len(self.events) > 0:
                self.logger.info(f"Analysis complete: {len(self.events)} gaze events, "
                               f"{len(self._get_unique_directions())} unique directions")
            else:
                self.logger.warning(f"No gaze events detected. Failure rate: {self.processing_stats.get('tracking_failure_rate', 0):.1f}%")

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

        finally:
            if cap is not None:
                cap.release()
            self._cleanup()

    def _handle_gaze_change(self, new_direction: Optional[str], current_time: float, frame_idx: int):
        """Handle gaze direction changes"""
        try:
            # End previous event if it meets duration threshold
            if self.current_gaze is not None and self.current_event_start is not None:
                event_duration = current_time - self.current_event_start

                if event_duration >= self.config.min_event_duration:
                    h_ratio = self.gaze_tracker.horizontal_ratio()
                    v_ratio = self.gaze_tracker.vertical_ratio()

                    event = GazeEvent(
                        event_type="GAZE_DIRECTION",
                        direction=self.current_gaze,
                        start_timestamp=round(self.current_event_start, 3),
                        end_timestamp=round(current_time, 3),
                        duration=round(event_duration, 3),
                        start_frame=self.current_event_start_frame,
                        end_frame=frame_idx,
                        horizontal_ratio=h_ratio,
                        vertical_ratio=v_ratio,
                        confidence=self.config.confidence_threshold
                    )

                    self.events.append(event)

            # Handle new direction or tracking loss
            if new_direction is None:
                # Reset tracking state
                self.current_gaze = None
                self.current_event_start = None
                self.current_event_start_frame = None
            else:
                # Start new event
                self.current_gaze = new_direction
                self.current_event_start = current_time
                self.current_event_start_frame = frame_idx

        except Exception as e:
            self.logger.error(f"Error handling gaze change: {e}")

    def _finalize_last_event(self, video_duration: float, total_frames: int):
        """Finalize the last gaze event"""
        try:
            if self.current_gaze is not None and self.current_event_start is not None:
                event_duration = video_duration - self.current_event_start

                if event_duration >= self.config.min_event_duration:
                    h_ratio = self.gaze_tracker.horizontal_ratio() or 0.5
                    v_ratio = self.gaze_tracker.vertical_ratio() or 0.5

                    event = GazeEvent(
                        event_type="GAZE_DIRECTION",
                        direction=self.current_gaze,
                        start_timestamp=round(self.current_event_start, 3),
                        end_timestamp=round(video_duration, 3),
                        duration=round(event_duration, 3),
                        start_frame=self.current_event_start_frame,
                        end_frame=total_frames,
                        horizontal_ratio=h_ratio,
                        vertical_ratio=v_ratio,
                        confidence=self.config.confidence_threshold
                    )

                    self.events.append(event)

        except Exception as e:
            self.logger.error(f"Error finalizing last event: {e}")

    def _reset_state(self):
        """Reset analyzer state for new analysis"""
        self.events.clear()
        self.current_gaze = None
        self.current_event_start = None
        self.current_event_start_frame = None
        self.frame_processing_times.clear()
        self.tracking_failures = 0
        self.consecutive_failures = 0
        self.total_frames_processed = 0
        self.processing_stats.clear()

    def _get_unique_directions(self) -> List[str]:
        """Get list of unique gaze directions detected"""
        return list(set(event.direction for event in self.events if event.event_type == "GAZE_DIRECTION"))

    def _calculate_gaze_stats(self) -> Dict:
        """Calculate gaze direction statistics"""
        gaze_events = [e for e in self.events if e.event_type == "GAZE_DIRECTION"]
        gaze_stats = {}

        for event in gaze_events:
            direction = event.direction
            if direction not in gaze_stats:
                gaze_stats[direction] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "max_duration": 0,
                    "min_duration": float('inf')
                }

            stats = gaze_stats[direction]
            stats["count"] += 1
            stats["total_duration"] += event.duration
            stats["max_duration"] = max(stats["max_duration"], event.duration)
            stats["min_duration"] = min(stats["min_duration"], event.duration)

        # Calculate averages
        for direction, stats in gaze_stats.items():
            if stats["count"] > 0:
                stats["avg_duration"] = round(stats["total_duration"] / stats["count"], 2)
                stats["total_duration"] = round(stats["total_duration"], 2)
                stats["max_duration"] = round(stats["max_duration"], 2)
                stats["min_duration"] = round(stats["min_duration"], 2)

        return gaze_stats

    def _calculate_stats(self, total_time: float, video_info: Dict, frames_with_gaze: int):
        """Calculate processing performance statistics"""
        processed_frames = len(self.frame_processing_times)

        self.processing_stats = {
            'total_processing_time_sec': round(total_time, 2),
            'avg_frame_processing_time': round(np.mean(self.frame_processing_times), 4) if self.frame_processing_times else 0,
            'frames_processed': processed_frames,
            'frames_with_gaze_detected': frames_with_gaze,
            'gaze_detection_rate': round(frames_with_gaze / processed_frames * 100, 1) if processed_frames > 0 else 0,
            'processing_fps': round(processed_frames / total_time, 1) if total_time > 0 else 0,
            'video_fps': video_info['fps'],
            'tracking_failure_rate': round((self.tracking_failures / processed_frames) * 100, 1) if processed_frames > 0 else 0,
            'total_tracking_failures': self.tracking_failures
        }

    def _assess_integrity(self, gaze_stats: Dict, video_info: Dict) -> Dict:
        """Assess gaze tracking integrity"""
        integrity_score = 1.0
        risk_factors = {}
        suspicious_patterns = []

        total_events = len(self.events)
        video_duration = video_info['duration_sec']

        # No gaze detected
        if total_events == 0:
            suspicious_patterns.append('no_gaze_events_detected')
            risk_factors['no_gaze'] = {'risk_contribution': 0.9}
            integrity_score -= 0.9

        # High tracking failure rate
        failure_rate = self.processing_stats.get('tracking_failure_rate', 0)
        if failure_rate > 50:
            suspicious_patterns.append('high_tracking_failure_rate')
            risk_score = min(0.6, failure_rate / 100)
            risk_factors['tracking_failures'] = {
                'failure_rate': failure_rate,
                'risk_contribution': risk_score
            }
            integrity_score -= risk_score

        # Limited gaze diversity
        unique_directions = len(gaze_stats)
        if unique_directions <= 1 and total_events > 0:
            suspicious_patterns.append('limited_gaze_movement')
            risk_factors['limited_movement'] = {
                'unique_directions': unique_directions,
                'risk_contribution': 0.4
            }
            integrity_score -= 0.4

        # Excessive gaze changes
        if video_duration > 0:
            change_rate = total_events / (video_duration / 60)
            if change_rate > 30:
                suspicious_patterns.append('excessive_gaze_changes')
                risk_score = min(0.3, change_rate / 100)
                risk_factors['excessive_changes'] = {
                    'changes_per_minute': round(change_rate, 1),
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
            'gaze_diversity_score': min(1.0, unique_directions / 5),
            'tracking_reliability': max(0.0, 1.0 - (failure_rate / 100))
        }

    def _generate_results(self, video_path: str, video_info: Dict, processing_time: float) -> Dict:
        """Generate comprehensive analysis results"""
        gaze_stats = self._calculate_gaze_stats()

        # Convert events to serializable format
        events_data = []
        for event in self.events:
            events_data.append({
                'event_type': event.event_type,
                'direction': event.direction,
                'start_timestamp': event.start_timestamp,
                'end_timestamp': event.end_timestamp,
                'duration': event.duration,
                'start_frame': event.start_frame,
                'end_frame': event.end_frame,
                'details': {
                    'horizontal_ratio': event.horizontal_ratio,
                    'vertical_ratio': event.vertical_ratio,
                    'confidence': event.confidence
                }
            })

        # Gaze analysis summary
        gaze_analysis = {
            'total_gaze_events': len(self.events),
            'unique_directions_detected': len(gaze_stats),
            'supported_directions': ["LEFT", "RIGHT", "UP", "DOWN", "CENTER", "LEFT_UP", "LEFT_DOWN", "RIGHT_UP", "RIGHT_DOWN"],
            'detected_directions': list(gaze_stats.keys()),
            'gaze_statistics': gaze_stats,
            'events': events_data
        }

        integrity_analysis = self._assess_integrity(gaze_stats, video_info)

        return {
            'video_metadata': {
                'path': os.path.basename(video_path),
                'full_path': video_path,
                **video_info,
                'analysis_timestamp': time.time()
            },
            'gaze_analysis': gaze_analysis,
            'integrity_analysis': integrity_analysis,
            'processing_statistics': self.processing_stats,
            'analysis_parameters': {
                'min_event_duration': self.config.min_event_duration,
                'horizontal_thresholds': self.config.horizontal_thresholds,
                'vertical_thresholds': self.config.vertical_thresholds,
                'confidence_threshold': self.config.confidence_threshold,
                'max_tracking_failures': self.config.max_tracking_failures,
                'max_consecutive_failures': self.config.max_consecutive_failures
            }
        }

    def _generate_failed_results(self, video_path: str, reason: str) -> Dict:
        """Generate results structure for failed analysis"""
        return {
            'video_metadata': {
                'path': os.path.basename(video_path),
                'full_path': video_path,
                'analysis_timestamp': time.time(),
                'error': reason
            },
            'gaze_analysis': {
                'total_gaze_events': 0,
                'unique_directions_detected': 0,
                'supported_directions': ["LEFT", "RIGHT", "UP", "DOWN", "CENTER", "LEFT_UP", "LEFT_DOWN", "RIGHT_UP", "RIGHT_DOWN"],
                'detected_directions': [],
                'gaze_statistics': {},
                'events': []
            },
            'integrity_analysis': {
                'integrity_score': 0.0,
                'risk_level': 'CRITICAL',
                'recommendation': 'REJECT',
                'suspicious_patterns': ['video_unsuitable_for_gaze_tracking'],
                'risk_factors': {'unsuitable_video': {'risk_contribution': 1.0}},
                'gaze_diversity_score': 0.0,
                'tracking_reliability': 0.0
            },
            'processing_statistics': {
                'total_processing_time_sec': 0,
                'tracking_failure_rate': 100.0,
                'processing_failed': True,
                'failure_reason': reason
            },
            'analysis_parameters': {
                'min_event_duration': self.config.min_event_duration,
                'horizontal_thresholds': self.config.horizontal_thresholds,
                'vertical_thresholds': self.config.vertical_thresholds
            }
        }

    def cleanup(self):
        """Clean up resources"""
        try:
            self.events.clear()
            self.frame_processing_times.clear()
            self.processing_stats.clear()
            
            # Reset gaze tracker state
            self.current_gaze = None
            self.current_event_start = None
            self.current_event_start_frame = None
            
            gc.collect()
        except Exception as e:
            self.logger.warning(f"Cleanup issues: {e}")

    def _cleanup(self):
        """Internal cleanup"""
        gc.collect()


def analyze_video_gaze(video_path: str, config: GazeConfig = None) -> Dict:
    """Convenience function for video gaze analysis"""
    analyzer = GazeAnalyzer(config)
    try:
        return analyzer.analyze_video(video_path)
    finally:
        analyzer.cleanup()


