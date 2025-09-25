
import os
import cv2
import json
import time
import gc
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
from gaze_tracking.gaze_tracking import GazeTracking

warnings.filterwarnings("ignore")

@dataclass
class GazeTrackerConfig:
    """Configuration for gaze tracking parameters"""

    # Core thresholds
    min_event_duration: float = 1.0
    horizontal_thresholds: Tuple[float, float] = (0.35, 0.65)
    vertical_thresholds: Tuple[float, float] = (0.43, 0.57)
    frame_skip: int = 12
    confidence_threshold: float = 0.7

    # Processing control
    log_level: str = "INFO"
    max_tracking_failures: int = 100
    early_stop_threshold: float = 0.95
    max_consecutive_failures: int = 30
    tracking_loss_timeout: float = 0.3

    # Optional features
    face_detection_enabled: bool = False
    calibration_duration: float = 2.0
    smoothing_factor: float = 0.25
    hysteresis_margin: float = 0.05
    dwell_time_ms: int = 200

@dataclass
class GazeEvent:
    """Represents a gaze tracking event"""

    event_type: str
    direction: str
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    horizontal_ratio: Optional[float] = None
    vertical_ratio: Optional[float] = None
    confidence: float = 0.7

class VideoGazeTracker:
    """Professional gaze tracking system for video analysis"""

    def __init__(self, config: GazeTrackerConfig = None):
        self.config = config or GazeTrackerConfig()
        self.logger = self._init_logger()

        try:
            self.tracker = GazeTracking()
        except Exception as e:
            self.logger.error(f"Failed to initialize gaze tracker: {e}")
            raise

        # State variables
        self.events: List[GazeEvent] = []
        self.current_direction = None
        self.current_event_start = None
        self.current_event_frame = None

        # Tracking state
        self.last_valid_time: Optional[float] = None
        self.last_valid_frame: Optional[int] = None
        self.last_valid_direction: Optional[str] = None

        # Face detector (optional)
        try:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.face_detector = None

        # Calibration state
        self.calibration_start: Optional[float] = None
        self.is_calibrated: bool = False
        self.h_calibration_samples: List[float] = []
        self.v_calibration_samples: List[float] = []
        self.h_center: float = 0.5
        self.v_center: float = 0.5
        self.h_smoothed: Optional[float] = None
        self.v_smoothed: Optional[float] = None
        self.pending_direction: Optional[str] = None
        self.pending_start_time: Optional[float] = None
        self.tracking_active: bool = False

        # Statistics
        self.processing_stats = {}
        self.frame_times = []
        self.failure_count = 0
        self.consecutive_failures = 0
        self.total_frames = 0

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.VideoGazeTracker")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _validate_video(self, video_path: str) -> bool:
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return False

        if not os.path.isfile(video_path):
            self.logger.error(f"Path is not a file: {video_path}")
            return False

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 2000:
            self.logger.warning(f"Large file detected: {file_size_mb:.1f}MB")

        return True

    def _get_video_metadata(self, cap: cv2.VideoCapture) -> Dict:
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
            'processing_frames': frame_count // self.config.frame_skip
        }

    def _detect_face(self, frame: np.ndarray) -> bool:
        if not self.config.face_detection_enabled or self.face_detector is None:
            return True

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 3)
            return len(faces) > 0
        except:
            return True

    def _start_calibration(self, timestamp: float):
        if self.calibration_start is None:
            self.calibration_start = timestamp

    def _update_calibration(self, h_ratio: float, v_ratio: float, timestamp: float):
        if self.is_calibrated:
            return

        self._start_calibration(timestamp)

        if timestamp - self.calibration_start <= self.config.calibration_duration:
            self.h_calibration_samples.append(h_ratio)
            self.v_calibration_samples.append(v_ratio)
        else:
            if len(self.h_calibration_samples) >= 10:
                self.h_center = float(np.median(self.h_calibration_samples))
                self.v_center = float(np.median(self.v_calibration_samples))
                self.is_calibrated = True

    def _apply_smoothing(self, prev: Optional[float], current: float, alpha: float) -> float:
        return current if prev is None else (alpha * current + (1 - alpha) * prev)

    def _get_stable_direction(self, h_ratio: float, v_ratio: float, timestamp: float) -> Optional[str]:
        # Apply exponential smoothing
        self.h_smoothed = self._apply_smoothing(self.h_smoothed, h_ratio, self.config.smoothing_factor)
        self.v_smoothed = self._apply_smoothing(self.v_smoothed, v_ratio, self.config.smoothing_factor)

        h = self.h_smoothed
        v = self.v_smoothed

        # Update calibration
        self._update_calibration(h, v, timestamp)

        # Define thresholds with calibration offset
        right_threshold = self.config.horizontal_thresholds[0]
        left_threshold = self.config.horizontal_thresholds[1]
        up_threshold = self.config.vertical_thresholds[0]
        down_threshold = self.config.vertical_thresholds[1]

        margin = self.config.hysteresis_margin

        # Adjust thresholds based on calibrated center
        right_th = (self.h_center + (right_threshold - 0.5)) - margin
        left_th = (self.h_center + (left_threshold - 0.5)) + margin
        up_th = (self.v_center + (up_threshold - 0.5)) - margin
        down_th = (self.v_center + (down_threshold - 0.5)) + margin

        # Determine horizontal direction
        if h < right_th:
            horizontal = "RIGHT"
        elif h > left_th:
            horizontal = "LEFT"
        else:
            horizontal = "CENTER"

        # Determine vertical direction
        if v < up_th:
            vertical = "UP"
        elif v > down_th:
            vertical = "DOWN"
        else:
            vertical = "CENTER"

        # Combine directions
        if horizontal == "CENTER" and vertical == "CENTER":
            candidate = "CENTER"
        elif horizontal == "CENTER":
            candidate = vertical
        elif vertical == "CENTER":
            candidate = horizontal
        else:
            candidate = f"{horizontal}_{vertical}"

        # Apply dwell time to prevent rapid changes
        dwell_duration = self.config.dwell_time_ms / 1000.0

        if self.pending_direction != candidate:
            self.pending_direction = candidate
            self.pending_start_time = timestamp
            return None
        else:
            if (timestamp - (self.pending_start_time or timestamp)) >= dwell_duration:
                return self.pending_direction
            return None

    def _process_frame(self, frame: np.ndarray, timestamp: float, frame_idx: int) -> Optional[str]:
        try:
            start_time = time.time()
            self.tracker.refresh(frame)

            # Check tracking quality
            pupils_detected = bool(getattr(self.tracker, "pupils_located", False))
            h_ratio = self.tracker.horizontal_ratio() if pupils_detected else None
            v_ratio = self.tracker.vertical_ratio() if pupils_detected else None
            ratios_valid = (h_ratio is not None) and (v_ratio is not None)
            face_present = self._detect_face(frame)

            tracking_ok = pupils_detected and ratios_valid and face_present
            direction = None

            if tracking_ok:
                direction = self._get_stable_direction(h_ratio, v_ratio, timestamp)

            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)

            # Update tracking state
            self.tracking_active = tracking_ok

            if not tracking_ok:
                self.consecutive_failures += 1
                self.failure_count += 1

                if self.consecutive_failures == self.config.max_tracking_failures:
                    self.logger.warning(f"Tracking failures reached: {self.consecutive_failures}")
            else:
                if direction is not None:
                    self.last_valid_time = timestamp
                    self.last_valid_frame = frame_idx
                    self.last_valid_direction = direction

                if self.consecutive_failures > self.config.max_tracking_failures:
                    self.logger.info("Gaze tracking recovered")
                self.consecutive_failures = 0

            return direction

        except Exception as e:
            self.logger.error(f"Frame processing error at {frame_idx}: {e}")
            self.failure_count += 1
            self.consecutive_failures += 1
            self.tracking_active = False
            return None

    def _validate_video_suitability(self, video_path: str) -> Dict:
        validation = {
            'is_suitable': True,
            'warnings': [],
            'recommendations': []
        }

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                validation['is_suitable'] = False
                validation['warnings'].append("Cannot open video file")
                return validation

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_count = min(10, frame_count // 10)
            faces_detected = 0
            frames_checked = 0

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            for i in range(max(1, sample_count)):
                frame_pos = int((i / max(1, sample_count)) * max(1, frame_count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if ret:
                    frames_checked += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        faces_detected += 1

            cap.release()

            face_rate = faces_detected / frames_checked if frames_checked > 0 else 0

            if face_rate < 0.3:
                validation['warnings'].append(f"Low face detection rate: {face_rate:.1%}")
                validation['recommendations'].append("Video may not be suitable for gaze tracking")

            if face_rate < 0.1:
                validation['is_suitable'] = False
                validation['warnings'].append("Very low face visibility")

            validation['face_detection_rate'] = round(face_rate, 2)
            validation['frames_sampled'] = frames_checked

        except Exception as e:
            self.logger.warning(f"Video validation failed: {e}")
            validation['warnings'].append(f"Validation error: {e}")

        return validation

    def _should_continue_processing(self, frame_idx: int, video_info: Dict) -> bool:
        if frame_idx > video_info['frame_count'] * 0.25:
            if self.failure_count > 0:
                failure_rate = self.failure_count / frame_idx
                if failure_rate > self.config.early_stop_threshold:
                    self.logger.warning(f"Stopping due to high failure rate: {failure_rate:.1%}")
                    return False
        return True

    def analyze_video(self, video_path: str) -> Dict:
        if not self._validate_video(video_path):
            raise ValueError(f"Invalid video file: {video_path}")

        # Validate video suitability
        validation = self._validate_video_suitability(video_path)
        self.logger.info(f"Face detection rate: {validation.get('face_detection_rate', 0):.1%}")

        for warning in validation.get('warnings', []):
            self.logger.warning(warning)

        if not validation['is_suitable']:
            self.logger.error("Video not suitable for gaze tracking")
            return self._create_failed_result(video_path, "Video unsuitable for gaze tracking")

        self.logger.info(f"Starting analysis: {os.path.basename(video_path)}")
        start_time = time.time()
        self._reset_state()

        cap = None
        early_stop = False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            video_info = self._get_video_metadata(cap)
            consecutive_tracking_failures = 0
            frame_count = 0
            frames_with_gaze = 0
            last_log_frame = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / video_info['fps'] if video_info['fps'] > 0 else frame_count * 0.033

                # Check if should continue processing
                if frame_count > 300 and not self._should_continue_processing(frame_count, video_info):
                    early_stop = True
                    break

                # Process frame at intervals
                if frame_count % self.config.frame_skip == 0:
                    direction = self._process_frame(frame, timestamp, frame_count)

                    if self.tracking_active:
                        consecutive_tracking_failures = 0
                        if direction:
                            frames_with_gaze += 1

                        # Handle direction changes
                        if direction != self.current_direction:
                            self._handle_direction_change(direction, timestamp, frame_count)
                    else:
                        consecutive_tracking_failures += 1

                        # Calculate tracking loss duration
                        fps_safe = max(video_info['fps'], 1e-6)
                        loss_duration = (consecutive_tracking_failures * self.config.frame_skip) / fps_safe

                        # End current event after timeout
                        if self.current_direction is not None and loss_duration >= self.config.tracking_loss_timeout:
                            self._end_current_event()
                            consecutive_tracking_failures = 0

                frame_count += 1
                self.total_frames = frame_count

                # Progress logging
                progress_interval = max(1800, video_info['frame_count'] // 20)
                if frame_count - last_log_frame >= progress_interval:
                    progress = (frame_count / video_info['frame_count']) * 100
                    self.logger.info(f"Progress: {frame_count}/{video_info['frame_count']} frames ({progress:.1f}%)")
                    last_log_frame = frame_count

            # Finalize processing
            if early_stop:
                self.logger.warning("Processing stopped early due to tracking failures")

            final_duration = frame_count / video_info['fps'] if video_info['fps'] > 0 else frame_count * 0.033
            self._finalize_analysis(final_duration, frame_count)

            total_time = time.time() - start_time
            self._calculate_statistics(total_time, video_info, frames_with_gaze)

            results = self._generate_results(video_path, video_info, total_time)

            if early_stop:
                results['processing_statistics']['early_stop'] = True
                results['processing_statistics']['completion_percentage'] = round((frame_count / video_info['frame_count']) * 100, 1)

            if len(self.events) > 0:
                self.logger.info(f"Analysis complete: {len(self.events)} events, {len(self._get_unique_directions())} directions")
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

    def _end_current_event(self):
        try:
            if self.current_direction is None or self.current_event_start is None:
                self.current_direction = None
                self.current_event_start = None
                self.current_event_frame = None
                return

            end_time = self.last_valid_time if self.last_valid_time is not None else self.current_event_start
            end_frame = self.last_valid_frame if self.last_valid_frame is not None else self.current_event_frame

            event_duration = end_time - self.current_event_start

            if event_duration >= self.config.min_event_duration:
                h_ratio = self.tracker.horizontal_ratio() or 0.5
                v_ratio = self.tracker.vertical_ratio() or 0.5

                event = GazeEvent(
                    event_type="GAZE_DIRECTION",
                    direction=self.current_direction,
                    start_time=round(self.current_event_start, 3),
                    end_time=round(end_time, 3),
                    duration=round(event_duration, 3),
                    start_frame=self.current_event_frame,
                    end_frame=end_frame,
                    horizontal_ratio=h_ratio,
                    vertical_ratio=v_ratio,
                    confidence=self.config.confidence_threshold
                )

                self.events.append(event)

            # Reset state
            self.current_direction = None
            self.current_event_start = None
            self.current_event_frame = None

        except Exception as e:
            self.logger.error(f"Error ending current event: {e}")

    def _handle_direction_change(self, new_direction: Optional[str], timestamp: float, frame_idx: int):
        try:
            # End previous event if it meets duration requirement
            if self.current_direction is not None and self.current_event_start is not None:
                event_duration = timestamp - self.current_event_start

                if event_duration >= self.config.min_event_duration:
                    h_ratio = self.tracker.horizontal_ratio()
                    v_ratio = self.tracker.vertical_ratio()

                    event = GazeEvent(
                        event_type="GAZE_DIRECTION",
                        direction=self.current_direction,
                        start_time=round(self.current_event_start, 3),
                        end_time=round(timestamp, 3),
                        duration=round(event_duration, 3),
                        start_frame=self.current_event_frame,
                        end_frame=frame_idx,
                        horizontal_ratio=h_ratio,
                        vertical_ratio=v_ratio,
                        confidence=self.config.confidence_threshold
                    )

                    self.events.append(event)

            # Start new event or reset state
            if new_direction is None:
                self.current_direction = None
                self.current_event_start = None
                self.current_event_frame = None
            else:
                self.current_direction = new_direction
                self.current_event_start = timestamp
                self.current_event_frame = frame_idx

        except Exception as e:
            self.logger.error(f"Error handling direction change: {e}")

    def _finalize_analysis(self, video_duration: float, total_frames: int):
        try:
            if self.current_direction is not None and self.current_event_start is not None:
                end_time = self.last_valid_time if self.last_valid_time is not None else video_duration
                end_frame = self.last_valid_frame if self.last_valid_frame is not None else total_frames

                event_duration = end_time - self.current_event_start

                if event_duration >= self.config.min_event_duration:
                    h_ratio = self.tracker.horizontal_ratio() or 0.5
                    v_ratio = self.tracker.vertical_ratio() or 0.5

                    event = GazeEvent(
                        event_type="GAZE_DIRECTION",
                        direction=self.current_direction,
                        start_time=round(self.current_event_start, 3),
                        end_time=round(end_time, 3),
                        duration=round(event_duration, 3),
                        start_frame=self.current_event_frame,
                        end_frame=end_frame,
                        horizontal_ratio=h_ratio,
                        vertical_ratio=v_ratio,
                        confidence=self.config.confidence_threshold
                    )

                    self.events.append(event)

        except Exception as e:
            self.logger.error(f"Error finalizing analysis: {e}")

    def _reset_state(self):
        self.events.clear()
        self.current_direction = None
        self.current_event_start = None
        self.current_event_frame = None
        self.frame_times.clear()
        self.failure_count = 0
        self.consecutive_failures = 0
        self.total_frames = 0
        self.processing_stats.clear()

        # Reset tracking state
        self.last_valid_time = None
        self.last_valid_frame = None
        self.last_valid_direction = None

        # Reset calibration
        self.calibration_start = None
        self.is_calibrated = False
        self.h_calibration_samples.clear()
        self.v_calibration_samples.clear()
        self.h_center = 0.5
        self.v_center = 0.5
        self.h_smoothed = None
        self.v_smoothed = None
        self.pending_direction = None
        self.pending_start_time = None
        self.tracking_active = False

    def _get_unique_directions(self) -> List[str]:
        return list(set(event.direction for event in self.events if event.event_type == "GAZE_DIRECTION"))

    def _calculate_direction_stats(self) -> Dict:
        gaze_events = [e for e in self.events if e.event_type == "GAZE_DIRECTION"]
        direction_stats = {}

        for event in gaze_events:
            direction = event.direction
            if direction not in direction_stats:
                direction_stats[direction] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "max_duration": 0,
                    "min_duration": float('inf')
                }

            stats = direction_stats[direction]
            stats["count"] += 1
            stats["total_duration"] += event.duration
            stats["max_duration"] = max(stats["max_duration"], event.duration)
            stats["min_duration"] = min(stats["min_duration"], event.duration)

        # Calculate averages
        for direction, stats in direction_stats.items():
            if stats["count"] > 0:
                stats["avg_duration"] = round(stats["total_duration"] / stats["count"], 2)
                stats["total_duration"] = round(stats["total_duration"], 2)
                stats["max_duration"] = round(stats["max_duration"], 2)
                stats["min_duration"] = round(stats["min_duration"], 2)

        return direction_stats

    def _calculate_statistics(self, total_time: float, video_info: Dict, frames_with_gaze: int):
        processed_frames = len(self.frame_times)

        self.processing_stats = {
            'total_processing_time': round(total_time, 2),
            'avg_frame_time': round(np.mean(self.frame_times), 4) if self.frame_times else 0,
            'frames_processed': processed_frames,
            'frames_with_gaze': frames_with_gaze,
            'gaze_detection_rate': round(frames_with_gaze / processed_frames * 100, 1) if processed_frames > 0 else 0,
            'processing_fps': round(processed_frames / total_time, 1) if total_time > 0 else 0,
            'video_fps': video_info['fps'],
            'tracking_failure_rate': round((self.failure_count / processed_frames) * 100, 1) if processed_frames > 0 else 0,
            'total_failures': self.failure_count
        }

    def _assess_tracking_quality(self, direction_stats: Dict, video_info: Dict) -> Dict:
        quality_score = 1.0
        issues = {}
        warnings = []

        total_events = len(self.events)
        video_duration = video_info['duration_sec']

        # No events detected
        if total_events == 0:
            warnings.append('no_gaze_events')
            issues['no_events'] = {'impact': 0.9}
            quality_score -= 0.9

        # High failure rate
        failure_rate = self.processing_stats.get('tracking_failure_rate', 0)
        if failure_rate > 50:
            warnings.append('high_failure_rate')
            risk_score = min(0.6, failure_rate / 100)
            issues['tracking_failures'] = {
                'failure_rate': failure_rate,
                'impact': risk_score
            }
            quality_score -= risk_score

        # Limited movement
        unique_directions = len(direction_stats)
        if unique_directions <= 1 and total_events > 0:
            warnings.append('limited_movement')
            issues['limited_movement'] = {
                'unique_directions': unique_directions,
                'impact': 0.4
            }
            quality_score -= 0.4

        # Excessive changes
        if video_duration > 0:
            change_rate = total_events / (video_duration / 60)
            if change_rate > 30:
                warnings.append('excessive_changes')
                risk_score = min(0.3, change_rate / 100)
                issues['excessive_changes'] = {
                    'changes_per_minute': round(change_rate, 1),
                    'impact': risk_score
                }
                quality_score -= risk_score

        quality_score = max(0.0, quality_score)

        # Determine quality level
        if quality_score >= 0.8:
            level, recommendation = 'HIGH', 'ACCEPT'
        elif quality_score >= 0.6:
            level, recommendation = 'MEDIUM', 'REVIEW'
        elif quality_score >= 0.4:
            level, recommendation = 'LOW', 'REVIEW'
        else:
            level, recommendation = 'POOR', 'REJECT'

        return {
            'quality_score': round(quality_score, 3),
            'quality_level': level,
            'recommendation': recommendation,
            'warnings': warnings,
            'issues': issues,
            'movement_diversity': min(1.0, unique_directions / 5),
            'tracking_reliability': max(0.0, 1.0 - (failure_rate / 100))
        }

    def _generate_results(self, video_path: str, video_info: Dict, processing_time: float) -> Dict:
        direction_stats = self._calculate_direction_stats()

        # Convert events to serializable format
        events_data = []
        for event in self.events:
            events_data.append({
                'event_type': event.event_type,
                'direction': event.direction,
                'start_time': event.start_time,
                'end_time': event.end_time,
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
            'total_events': len(self.events),
            'unique_directions': len(direction_stats),
            'supported_directions': ["LEFT", "RIGHT", "UP", "DOWN", "CENTER", "LEFT_UP", "LEFT_DOWN", "RIGHT_UP", "RIGHT_DOWN"],
            'detected_directions': list(direction_stats.keys()),
            'direction_statistics': direction_stats,
            'events': events_data
        }

        quality_analysis = self._assess_tracking_quality(direction_stats, video_info)

        return {
            'video_metadata': {
                'filename': os.path.basename(video_path),
                'full_path': video_path,
                **video_info,
                'analysis_timestamp': time.time()
            },
            'gaze_analysis': gaze_analysis,
            'quality_analysis': quality_analysis,
            'processing_statistics': self.processing_stats,
            'configuration': {
                'min_event_duration': self.config.min_event_duration,
                'horizontal_thresholds': self.config.horizontal_thresholds,
                'vertical_thresholds': self.config.vertical_thresholds,
                'confidence_threshold': self.config.confidence_threshold,
                'max_tracking_failures': self.config.max_tracking_failures,
                'max_consecutive_failures': self.config.max_consecutive_failures,
                'tracking_loss_timeout': self.config.tracking_loss_timeout,
                'face_detection_enabled': self.config.face_detection_enabled,
                'calibration_duration': self.config.calibration_duration,
                'smoothing_factor': self.config.smoothing_factor,
                'hysteresis_margin': self.config.hysteresis_margin,
                'dwell_time_ms': self.config.dwell_time_ms
            }
        }

    def _create_failed_result(self, video_path: str, reason: str) -> Dict:
        return {
            'video_metadata': {
                'filename': os.path.basename(video_path),
                'full_path': video_path,
                'analysis_timestamp': time.time(),
                'error': reason
            },
            'gaze_analysis': {
                'total_events': 0,
                'unique_directions': 0,
                'supported_directions': ["LEFT", "RIGHT", "UP", "DOWN", "CENTER", "LEFT_UP", "LEFT_DOWN", "RIGHT_UP", "RIGHT_DOWN"],
                'detected_directions': [],
                'direction_statistics': {},
                'events': []
            },
            'quality_analysis': {
                'quality_score': 0.0,
                'quality_level': 'POOR',
                'recommendation': 'REJECT',
                'warnings': ['video_unsuitable'],
                'issues': {'unsuitable_video': {'impact': 1.0}},
                'movement_diversity': 0.0,
                'tracking_reliability': 0.0
            },
            'processing_statistics': {
                'total_processing_time': 0,
                'tracking_failure_rate': 100.0,
                'processing_failed': True,
                'failure_reason': reason
            },
            'configuration': {
                'min_event_duration': self.config.min_event_duration,
                'horizontal_thresholds': self.config.horizontal_thresholds,
                'vertical_thresholds': self.config.vertical_thresholds
            }
        }

    def cleanup(self):
        try:
            self.events.clear()
            self.frame_times.clear()
            self.processing_stats.clear()

            # Reset state
            self.current_direction = None
            self.current_event_start = None
            self.current_event_frame = None

            # Reset tracking state
            self.last_valid_time = None
            self.last_valid_frame = None
            self.last_valid_direction = None
            self.calibration_start = None
            self.is_calibrated = False
            self.h_calibration_samples.clear()
            self.v_calibration_samples.clear()
            self.h_center = 0.5
            self.v_center = 0.5
            self.h_smoothed = None
            self.v_smoothed = None
            self.pending_direction = None
            self.pending_start_time = None
            self.tracking_active = False

            gc.collect()

        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

    def _cleanup(self):
        gc.collect()


# Convenience function for easy usage
def analyze_gaze_tracking(video_path: str, config: GazeTrackerConfig = None) -> Dict:
    """Analyze gaze tracking in a video file"""
    tracker = VideoGazeTracker(config)
    try:
        return tracker.analyze_video(video_path)
    finally:
        tracker.cleanup()


# Example usage for FastAPI integration
class GazeTrackingAPI:
    """FastAPI-ready gaze tracking service"""

    def __init__(self, config: GazeTrackerConfig = None):
        self.config = config or GazeTrackerConfig()

    async def analyze_video_file(self, video_path: str) -> Dict:
        """Async video analysis for FastAPI endpoints"""
        return analyze_gaze_tracking(video_path, self.config)

    def get_default_config(self) -> Dict:
        """Get default configuration as dictionary"""
        return {
            'min_event_duration': self.config.min_event_duration,
            'horizontal_thresholds': self.config.horizontal_thresholds,
            'vertical_thresholds': self.config.vertical_thresholds,
            'frame_skip': self.config.frame_skip,
            'confidence_threshold': self.config.confidence_threshold,
            'face_detection_enabled': self.config.face_detection_enabled,
            'calibration_duration': self.config.calibration_duration,
            'smoothing_factor': self.config.smoothing_factor
        }


if __name__ == "__main__":
    # Example usage
    config = GazeTrackerConfig(
        tracking_loss_timeout=0.3,
        face_detection_enabled=True,
        frame_skip=12,
        calibration_duration=2.0,
        smoothing_factor=0.25,
        hysteresis_margin=0.05,
        dwell_time_ms=200
    )

    result = analyze_gaze_tracking("./videos/sample_video.mp4", config)
    print(json.dumps(result, indent=2))
