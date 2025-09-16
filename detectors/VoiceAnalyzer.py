import os
import json
import logging
import gc
import numpy as np
import librosa
import soundfile as sf
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import warnings
import time
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import torchaudio
from collections import defaultdict

warnings.filterwarnings("ignore")

@dataclass
class SpeakerSegment:
    """Data structure for speaker segments"""
    speaker_id: str
    start_time: str
    end_time: str
    duration_sec: float
    confidence_score: float
    overlap_detected: bool = False
    overlap_with: List[str] = None

@dataclass
class VoiceEvent:
    """Data structure for voice change events"""
    from_speaker: str
    to_speaker: str
    transition_time: str
    end_time: str
    duration_sec: float
    type: str
    gap_duration: float = 0.0
    overlap_duration: float = 0.0

class VoiceAnalyzer:
    """Advanced voice analysis system for detecting multiple speakers in audio/video"""

    def __init__(self,
                 hf_token: str,
                 use_cuda: bool = True,
                 min_speaker_duration: float = 1.0,
                 min_voice_change_gap: float = 0.5,
                 confidence_threshold: float = 0.8,
                 audio_sample_rate: int = 16000,
                 overlap_threshold: float = 0.1,
                 max_allowed_overlap: float = 0.05,
                 log_level: str = "INFO"):
        """
        Initialize voice analyzer
        
        Args:
            hf_token: HuggingFace access token for pyannote models
            use_cuda: Whether to use GPU acceleration
            min_speaker_duration: Minimum duration to consider a speaker segment
            min_voice_change_gap: Minimum gap between speaker changes
            confidence_threshold: Minimum confidence for reliable detections
            audio_sample_rate: Target sample rate for audio processing
            overlap_threshold: Threshold for detecting overlaps
            max_allowed_overlap: Maximum allowed overlap before correction
            log_level: Logging level
        """
        if not hf_token:
            raise ValueError("HuggingFace token required! Get token from: https://hf.co/settings/tokens")

        self.hf_token = hf_token
        self.min_speaker_duration = min_speaker_duration
        self.min_voice_change_gap = min_voice_change_gap
        self.confidence_threshold = confidence_threshold
        self.audio_sample_rate = audio_sample_rate
        self.overlap_threshold = overlap_threshold
        self.max_allowed_overlap = max_allowed_overlap

        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Device setup
        self.device = self._setup_device(use_cuda)
        
        # Initialize pipeline
        self._initialize_pipeline()

        # Tracking variables
        self.detected_speakers: List[SpeakerSegment] = []
        self.voice_events: List[VoiceEvent] = []
        self.processing_stats = {}
        self.overlap_corrections = []

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Configure logging system"""
        logger = logging.getLogger(__name__ + ".VoiceAnalyzer")
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_device(self, use_cuda: bool) -> str:
        """Setup compute device with GPU detection"""
        if use_cuda and torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.randn(10, 10).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                return "cuda"
            except Exception as e:
                self.logger.warning(f"CUDA requested but failed: {e}")
                self.logger.info("Falling back to CPU")
                return "cpu"
        else:
            self.logger.info("Using CPU")
            return "cpu"

    def _initialize_pipeline(self):
        """Initialize pyannote.audio pipeline"""
        try:
            self.logger.info("Loading speaker diarization pipeline...")
            
            # Load the model
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
                cache_dir=None
            )

            # Move to appropriate device
            if self.device == "cuda":
                try:
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                    self.logger.info("Pipeline moved to GPU")
                except Exception as e:
                    self.logger.warning(f"Failed to move pipeline to GPU: {e}")
                    self.device = "cpu"

            # Apply optimizations
            self._optimize_pipeline()
            self.logger.info(f"Pipeline initialized on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            self.logger.error("Check HuggingFace token and model access permissions")
            raise

    def _optimize_pipeline(self):
        """Optimize pipeline parameters for better performance"""
        try:
            # Access internal components safely
            if hasattr(self.pipeline, '_segmentation') and self.pipeline._segmentation is not None:
                seg_model = self.pipeline._segmentation
                
                # Update VAD thresholds
                if hasattr(seg_model, 'onset'):
                    seg_model.onset = 0.4
                elif hasattr(seg_model, '_onset'):
                    seg_model._onset = 0.4
                    
                if hasattr(seg_model, 'offset'):
                    seg_model.offset = 0.6
                elif hasattr(seg_model, '_offset'):
                    seg_model._offset = 0.6

            # Clustering optimization
            if hasattr(self.pipeline, '_clustering') and self.pipeline._clustering is not None:
                cluster_model = self.pipeline._clustering
                if hasattr(cluster_model, 'threshold'):
                    cluster_model.threshold = 0.8
                elif hasattr(cluster_model, '_threshold'):
                    cluster_model._threshold = 0.8

        except Exception as e:
            self.logger.warning(f"Parameter optimization failed: {e}")

    def _preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio for optimal diarization accuracy"""
        try:
            start_time = time.time()
            
            # Load audio
            audio, original_sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / original_sr

            if duration < 1.0:
                self.logger.warning(f"Audio file is very short ({duration:.2f} seconds)")
            if duration > 3600:
                self.logger.warning(f"Audio file is very long ({duration/60:.1f} minutes)")

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            # Resample to target sample rate
            if original_sr != self.audio_sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=original_sr,
                    target_sr=self.audio_sample_rate,
                    res_type='kaiser_best'
                )

            # Apply processing pipeline
            audio = self._reduce_noise(audio)
            audio = self._normalize_audio(audio)
            audio = self._filter_audio(audio)

            # Validate quality
            self._validate_audio_quality(audio)

            # Save preprocessed audio
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio_path, audio, self.audio_sample_rate)

            # Calculate statistics
            processing_time = time.time() - start_time
            rms_level = np.sqrt(np.mean(audio**2))
            snr_estimate = self._estimate_snr(audio)

            self.processing_stats['preprocessing'] = {
                'duration_sec': round(duration, 2),
                'original_sr': original_sr,
                'target_sr': self.audio_sample_rate,
                'rms_level': round(rms_level, 4),
                'snr_estimate_db': round(snr_estimate, 1),
                'processing_time_sec': round(processing_time, 2)
            }

            return temp_audio_path

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return audio_path

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction using spectral subtraction"""
        try:
            stft = librosa.stft(audio, n_fft=2048, hop_length=512, window='hann')
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise from quiet segments
            noise_frames = min(int(0.5 * self.audio_sample_rate / 512), magnitude.shape[1] // 4)
            noise_spectrum_start = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

            frame_energy = np.mean(magnitude, axis=0)
            quiet_threshold = np.percentile(frame_energy, 20)
            quiet_frames = magnitude[:, frame_energy < quiet_threshold]

            if quiet_frames.shape[1] > 0:
                noise_spectrum_quiet = np.mean(quiet_frames, axis=1, keepdims=True)
                noise_spectrum = (noise_spectrum_start + noise_spectrum_quiet) / 2
            else:
                noise_spectrum = noise_spectrum_start

            # Apply spectral subtraction
            alpha = 2.0
            beta = 0.01
            clean_magnitude = magnitude - alpha * noise_spectrum
            clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)

            # Smooth spectrum
            clean_magnitude = self._smooth_spectrum(clean_magnitude)

            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft, hop_length=512)

            return clean_audio

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio

    def _smooth_spectrum(self, spectrum: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Smooth spectrum using median filtering"""
        try:
            from scipy import ndimage
            return ndimage.median_filter(spectrum, size=(kernel_size, 1))
        except Exception:
            return spectrum

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Intelligent audio normalization preserving dynamics"""
        try:
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))

            target_rms = 0.1
            target_peak = 0.8

            # Choose normalization method based on audio characteristics
            if peak > 0.95:  # Clipped audio
                scale_factor = target_rms / (rms + 1e-8)
            elif rms < 0.01:  # Very quiet audio
                scale_factor = target_peak / (peak + 1e-8)
            else:
                # Balanced normalization
                rms_scale = target_rms / (rms + 1e-8)
                peak_scale = target_peak / (peak + 1e-8)
                scale_factor = min(rms_scale, peak_scale)

            # Apply gain limiting
            scale_factor = min(scale_factor, 10.0)
            normalized_audio = audio * scale_factor

            # Soft limiting
            normalized_audio = np.tanh(normalized_audio * 0.9) / 0.9

            return normalized_audio

        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return librosa.util.normalize(audio)

    def _filter_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio filtering for speech clarity"""
        try:
            from scipy import signal

            nyquist = self.audio_sample_rate / 2

            # High-pass filter
            low_freq_energy = np.mean(np.abs(librosa.stft(audio, n_fft=2048)[:50, :]))
            cutoff_freq = 100 if low_freq_energy > 0.1 else 80
            normalized_cutoff = cutoff_freq / nyquist

            b, a = signal.butter(4, normalized_cutoff, btype='high')
            filtered_audio = signal.filtfilt(b, a, audio)

            # Band-pass filter for speech frequencies (300-3400 Hz)
            speech_low = 300 / nyquist
            speech_high = 3400 / nyquist

            if speech_high < 1.0:
                b_bp, a_bp = signal.butter(2, [speech_low, speech_high], btype='band')
                bp_filtered = signal.filtfilt(b_bp, a_bp, audio)
                filtered_audio = 0.7 * filtered_audio + 0.3 * bp_filtered

            return filtered_audio

        except Exception as e:
            self.logger.warning(f"Audio filtering failed: {e}")
            return audio

    def _validate_audio_quality(self, audio: np.ndarray) -> bool:
        """Validate audio quality for diarization"""
        try:
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.001:
                self.logger.warning("Audio appears to be mostly silent")
                return False

            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            if clipping_ratio > 0.01:
                self.logger.warning(f"Audio has {clipping_ratio*100:.1f}% clipping")

            # Check dynamic range
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (rms + 1e-8))
            if dynamic_range < 10:
                self.logger.warning(f"Low dynamic range: {dynamic_range:.1f} dB")

            return True

        except Exception as e:
            self.logger.warning(f"Audio quality validation failed: {e}")
            return True

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        try:
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)

            frame_energy = np.mean(magnitude, axis=0)
            speech_threshold = np.percentile(frame_energy, 70)
            speech_power = np.mean(frame_energy[frame_energy > speech_threshold])
            noise_power = np.mean(frame_energy[frame_energy <= speech_threshold])

            snr_db = 10 * np.log10((speech_power + 1e-8) / (noise_power + 1e-8))
            return max(snr_db, -10)

        except Exception:
            return 0.0

    def _to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp string"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        microseconds = td.microseconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}.{microseconds:06d}"

    def _parse_time(self, ts_str: str) -> timedelta:
        """Parse timestamp string to timedelta object"""
        try:
            parts = ts_str.strip().split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            sec_parts = parts[2].split(".")
            seconds = int(sec_parts[0])
            microseconds = int(sec_parts[1][:6].ljust(6, '0')) if len(sec_parts) > 1 else 0
            return timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
        except Exception as e:
            self.logger.error(f"Failed to parse timestamp {ts_str}: {e}")
            return timedelta(0)

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp to seconds"""
        td = self._parse_time(timestamp)
        return td.total_seconds()

    def _post_process_diarization(self, diarization: Annotation) -> Annotation:
        """Post-process diarization with overlap removal"""
        try:
            # Remove short segments
            filtered_diarization = Annotation()
            short_segments_removed = 0

            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if segment.duration >= self.min_speaker_duration:
                    filtered_diarization[segment] = speaker
                else:
                    short_segments_removed += 1

            if short_segments_removed > 0:
                self.logger.info(f"Removed {short_segments_removed} short segments")

            # Remove overlapping segments
            clean_diarization = self._remove_overlaps(filtered_diarization)

            # Smooth speaker transitions
            smoothed_diarization = self._smooth_transitions(clean_diarization)

            original_segments = len(list(diarization.itertracks()))
            final_segments = len(list(smoothed_diarization.itertracks()))
            self.logger.info(f"Post-processing: {original_segments} → {final_segments} segments")

            return smoothed_diarization

        except Exception as e:
            self.logger.warning(f"Post-processing failed: {e}")
            return diarization

    def _remove_overlaps(self, diarization: Annotation) -> Annotation:
        """Remove overlapping segments with conflict resolution"""
        try:
            # Get all segments with timestamps
            all_segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                all_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.duration,
                    'speaker': speaker,
                    'segment': segment
                })

            # Sort by start time
            all_segments.sort(key=lambda x: x['start'])

            # Detect and resolve overlaps
            resolved_segments = []
            overlaps_found = 0

            for i, current_seg in enumerate(all_segments):
                # Check for overlaps with subsequent segments
                conflicts = []
                for j, other_seg in enumerate(all_segments[i+1:], i+1):
                    overlap_start = max(current_seg['start'], other_seg['start'])
                    overlap_end = min(current_seg['end'], other_seg['end'])
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > self.max_allowed_overlap:
                        conflicts.append((j, other_seg, overlap_duration))

                if conflicts:
                    overlaps_found += len(conflicts)
                    current_seg = self._resolve_conflict(current_seg, conflicts, all_segments)

                resolved_segments.append(current_seg)

            # Create clean diarization
            clean_diarization = Annotation()
            for seg_data in resolved_segments:
                if seg_data is not None:
                    new_segment = Segment(seg_data['start'], seg_data['end'])
                    clean_diarization[new_segment] = seg_data['speaker']

            if overlaps_found > 0:
                self.logger.info(f"Resolved {overlaps_found} overlapping segments")

            return clean_diarization

        except Exception as e:
            self.logger.warning(f"Overlap removal failed: {e}")
            return diarization

    def _resolve_conflict(self, current_seg: Dict, conflicts: List, all_segments: List) -> Dict:
        """Resolve conflicts between overlapping segments"""
        try:
            for conflict_idx, conflict_seg, overlap_duration in conflicts:
                # For small overlaps, adjust boundaries
                if overlap_duration < 0.5:
                    overlap_start = max(current_seg['start'], conflict_seg['start'])
                    overlap_end = min(current_seg['end'], conflict_seg['end'])
                    split_point = (overlap_start + overlap_end) / 2

                    if current_seg['start'] < conflict_seg['start']:
                        current_seg['end'] = min(current_seg['end'], split_point)
                    else:
                        current_seg['start'] = max(current_seg['start'], split_point)

                    if conflict_seg['start'] < current_seg['start']:
                        all_segments[conflict_idx]['end'] = min(conflict_seg['end'], split_point)
                    else:
                        all_segments[conflict_idx]['start'] = max(conflict_seg['start'], split_point)

                # For larger overlaps, keep the longer segment
                else:
                    if current_seg['duration'] >= conflict_seg['duration']:
                        if conflict_seg['start'] < current_seg['end']:
                            all_segments[conflict_idx]['start'] = current_seg['end'] + 0.01
                    else:
                        if current_seg['end'] > conflict_seg['start']:
                            current_seg['end'] = conflict_seg['start'] - 0.01

                # Record the correction
                self.overlap_corrections.append({
                    'type': 'boundary_adjustment',
                    'original_overlap': overlap_duration,
                    'speakers': [current_seg['speaker'], conflict_seg['speaker']]
                })

            # Ensure segment is still valid
            current_seg['duration'] = current_seg['end'] - current_seg['start']
            if current_seg['duration'] < self.min_speaker_duration:
                return None

            return current_seg

        except Exception as e:
            self.logger.warning(f"Conflict resolution failed: {e}")
            return current_seg

    def _smooth_transitions(self, diarization: Annotation, max_gap: float = 0.5) -> Annotation:
        """Smooth speaker transitions by merging nearby segments"""
        try:
            segments_by_speaker = defaultdict(list)

            # Group by speaker
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments_by_speaker[speaker].append(segment)

            smoothed_diarization = Annotation()

            for speaker, segments in segments_by_speaker.items():
                # Sort by start time
                segments.sort(key=lambda s: s.start)

                # Merge nearby segments
                merged_segments = []
                for segment in segments:
                    if not merged_segments:
                        merged_segments.append(segment)
                    else:
                        last_segment = merged_segments[-1]
                        gap = segment.start - last_segment.end

                        if gap <= max_gap:
                            # Merge segments
                            merged_segment = Segment(last_segment.start, segment.end)
                            merged_segments[-1] = merged_segment
                        else:
                            merged_segments.append(segment)

                # Add to final diarization
                for segment in merged_segments:
                    smoothed_diarization[segment] = speaker

            return smoothed_diarization

        except Exception as e:
            self.logger.warning(f"Speaker transition smoothing failed: {e}")
            return diarization

    def _detect_voice_events(self, diarization: Annotation) -> List[VoiceEvent]:
        """Detect voice change events with detailed timing"""
        events = []
        segments = list(diarization.itertracks(yield_label=True))

        if len(segments) < 1:
            return events

        segments.sort(key=lambda x: x[0].start)

        prev_segment = None
        prev_speaker = None

        for segment, _, current_speaker in segments:
            if prev_speaker is None:
                # First speaker detected
                event = VoiceEvent(
                    from_speaker='silence',
                    to_speaker=current_speaker,
                    transition_time=self._to_timestamp(segment.start),
                    end_time=self._to_timestamp(segment.end),
                    duration_sec=round(segment.duration, 3),
                    type='new_speaker',
                    gap_duration=0.0
                )
                events.append(event)

            elif prev_speaker != current_speaker:
                # Calculate gap or overlap
                gap_duration = segment.start - prev_segment.end
                overlap_duration = max(0, prev_segment.end - segment.start)

                # Determine event type
                if overlap_duration > 0.05:
                    event_type = 'overlap'
                elif gap_duration > 1.0:
                    event_type = 'gap'
                else:
                    event_type = 'speaker_change'

                event = VoiceEvent(
                    from_speaker=prev_speaker,
                    to_speaker=current_speaker,
                    transition_time=self._to_timestamp(segment.start),
                    end_time=self._to_timestamp(segment.end),
                    duration_sec=round(segment.duration, 3),
                    type=event_type,
                    gap_duration=round(max(0, gap_duration), 3),
                    overlap_duration=round(overlap_duration, 3)
                )
                events.append(event)

            prev_segment = segment
            prev_speaker = current_speaker

        return events

    def _calculate_speaker_stats(self, diarization: Annotation) -> Dict:
        """Calculate comprehensive speaker statistics"""
        speaker_stats = {}
        total_speech_time = 0.0
        all_segments = list(diarization.itertracks(yield_label=True))

        if not all_segments:
            return speaker_stats

        # Calculate per-speaker statistics
        for segment, _, speaker in all_segments:
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'avg_segment_duration': 0.0,
                    'min_segment_duration': float('inf'),
                    'max_segment_duration': 0.0,
                    'first_appearance': segment.start,
                    'last_appearance': segment.end,
                    'speech_percentage': 0.0,
                    'segments': []
                }

            stats = speaker_stats[speaker]
            stats['total_duration'] += segment.duration
            stats['segment_count'] += 1
            stats['min_segment_duration'] = min(stats['min_segment_duration'], segment.duration)
            stats['max_segment_duration'] = max(stats['max_segment_duration'], segment.duration)
            stats['first_appearance'] = min(stats['first_appearance'], segment.start)
            stats['last_appearance'] = max(stats['last_appearance'], segment.end)
            stats['segments'].append({
                'start': segment.start,
                'end': segment.end,
                'duration': segment.duration
            })

            total_speech_time += segment.duration

        # Calculate derived statistics
        for speaker, stats in speaker_stats.items():
            if stats['segment_count'] > 0:
                stats['avg_segment_duration'] = stats['total_duration'] / stats['segment_count']
                stats['speech_percentage'] = (stats['total_duration'] / total_speech_time * 100) if total_speech_time > 0 else 0

                # Calculate speaking pattern metrics
                stats['speaking_consistency'] = self._calculate_consistency(stats['segments'])
                stats['silence_tolerance'] = self._calculate_silence_tolerance(stats['segments'])

                # Round values
                for key in ['total_duration', 'avg_segment_duration', 'min_segment_duration',
                           'max_segment_duration', 'speech_percentage', 'speaking_consistency', 'silence_tolerance']:
                    if key in stats:
                        stats[key] = round(stats[key], 3)

        return speaker_stats

    def _calculate_consistency(self, segments: List[Dict]) -> float:
        """Calculate speaking consistency based on segment durations"""
        if len(segments) < 2:
            return 1.0

        durations = [seg['duration'] for seg in segments]
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        cv = std_duration / (mean_duration + 1e-8)
        consistency = max(0, 1 - cv)
        return consistency

    def _calculate_silence_tolerance(self, segments: List[Dict]) -> float:
        """Calculate average gap between segments"""
        if len(segments) < 2:
            return 0.0

        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i]['start'] - segments[i-1]['end']
            if gap > 0:
                gaps.append(gap)

        return np.mean(gaps) if gaps else 0.0

    def _analyze_integrity(self, speaker_stats: Dict, voice_events: List[VoiceEvent], audio_duration: float) -> Dict:
        """Analyze audio integrity based on speaker patterns"""
        integrity_analysis = {
            'is_single_speaker': len(speaker_stats) == 1,
            'multiple_speakers_detected': len(speaker_stats) > 1,
            'suspicious_patterns': [],
            'risk_factors': {},
            'integrity_score': 1.0,
            'risk_level': 'LOW',
            'recommendation': 'ACCEPT',
            'overlap_analysis': {},
            'event_analysis': {}
        }

        risk_score = 0.0
        risk_factors = {}

        # Analyze overlaps
        overlap_events = [e for e in voice_events if e.type == 'overlap']
        total_overlap_duration = sum(e.overlap_duration for e in overlap_events)

        overlap_analysis = {
            'overlap_count': len(overlap_events),
            'total_overlap_duration': round(total_overlap_duration, 3),
            'overlap_percentage': round((total_overlap_duration / audio_duration * 100), 2) if audio_duration > 0 else 0,
            'avg_overlap_duration': round(total_overlap_duration / len(overlap_events), 3) if overlap_events else 0,
            'corrections_applied': len(self.overlap_corrections)
        }
        integrity_analysis['overlap_analysis'] = overlap_analysis

        # Event type analysis
        event_types = defaultdict(int)
        for event in voice_events:
            event_types[event.type] += 1

        event_analysis = {
            'total_events': len(voice_events),
            'event_types': dict(event_types),
            'avg_gap_duration': round(np.mean([e.gap_duration for e in voice_events if e.gap_duration > 0]), 3) if voice_events else 0,
            'changes_per_minute': round(len(voice_events) / (audio_duration / 60), 1) if audio_duration > 0 else 0
        }
        integrity_analysis['event_analysis'] = event_analysis

        # Risk factor analysis
        # Multiple speakers
        if len(speaker_stats) > 1:
            integrity_analysis['suspicious_patterns'].append('multiple_speakers_present')
            risk_factors['multiple_speakers'] = {
                'count': len(speaker_stats),
                'risk_contribution': 0.5
            }
            risk_score += 0.5

            # Speaker balance analysis
            speaker_durations = [stats['total_duration'] for stats in speaker_stats.values()]
            max_duration = max(speaker_durations)
            min_duration = min(speaker_durations)
            balance_ratio = max_duration / (min_duration + 0.001)

            if balance_ratio > 10:
                integrity_analysis['suspicious_patterns'].append('possible_coaching_pattern')
                risk_factors['coaching_pattern'] = {
                    'balance_ratio': round(balance_ratio, 2),
                    'risk_contribution': 0.3
                }
                risk_score += 0.3
            elif balance_ratio < 3:
                integrity_analysis['suspicious_patterns'].append('balanced_conversation')
                risk_factors['balanced_conversation'] = {
                    'balance_ratio': round(balance_ratio, 2),
                    'risk_contribution': 0.4
                }
                risk_score += 0.4

        # Overlap analysis
        if overlap_analysis['overlap_count'] > 0:
            integrity_analysis['suspicious_patterns'].append('speaker_overlaps_detected')
            overlap_risk = min(0.3, overlap_analysis['overlap_count'] * 0.05 + overlap_analysis['overlap_percentage'] * 0.01)
            risk_factors['overlaps'] = {
                'overlap_count': overlap_analysis['overlap_count'],
                'overlap_percentage': overlap_analysis['overlap_percentage'],
                'risk_contribution': overlap_risk
            }
            risk_score += overlap_risk

        # Voice change frequency
        if audio_duration > 0:
            change_rate = len(voice_events) / (audio_duration / 60)
            if change_rate > 8:
                integrity_analysis['suspicious_patterns'].append('high_speaker_change_frequency')
                risk_factors['high_change_frequency'] = {
                    'changes_per_minute': round(change_rate, 1),
                    'risk_contribution': 0.2
                }
                risk_score += 0.2

        # Background voices
        for speaker, stats in speaker_stats.items():
            if stats['speech_percentage'] < 5 and stats['segment_count'] > 3:
                integrity_analysis['suspicious_patterns'].append('background_voice_detected')
                risk_factors['background_voice'] = {
                    'speaker': speaker,
                    'speech_percentage': stats['speech_percentage'],
                    'segment_count': stats['segment_count'],
                    'risk_contribution': 0.4
                }
                risk_score += 0.4
                break

        # Calculate final scores
        risk_score = min(risk_score, 1.0)
        integrity_score = max(0.0, 1.0 - risk_score)

        # Determine risk level
        if integrity_score >= 0.8:
            risk_level, recommendation = 'LOW', 'ACCEPT'
        elif integrity_score >= 0.6:
            risk_level, recommendation = 'MEDIUM', 'REVIEW'
        elif integrity_score >= 0.4:
            risk_level, recommendation = 'HIGH', 'REVIEW'
        else:
            risk_level, recommendation = 'CRITICAL', 'REJECT'

        # Calculate dominant speaker percentage
        if speaker_stats:
            max_duration = max(stats['total_duration'] for stats in speaker_stats.values())
            dominant_percentage = (max_duration / audio_duration * 100) if audio_duration > 0 else 0
        else:
            dominant_percentage = 0

        integrity_analysis.update({
            'risk_factors': risk_factors,
            'integrity_score': round(integrity_score, 3),
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'dominant_speaker_percentage': round(dominant_percentage, 1),
            'voice_change_frequency': round(len(voice_events) / (audio_duration / 60), 1) if audio_duration > 0 else 0
        })

        return integrity_analysis

    def detect_multiple_voices(self, audio_path: str, expected_speakers: Optional[int] = None, preprocess_audio: bool = True) -> Dict:
        """Main method for detecting multiple voices in audio"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.logger.info(f"Starting voice detection: {audio_path}")
        analysis_start_time = time.time()

        processed_audio_path = audio_path
        temp_file_created = False

        try:
            # Reset tracking variables
            self.overlap_corrections = []

            # Preprocess audio
            if preprocess_audio:
                processed_audio_path = self._preprocess_audio(audio_path)
                temp_file_created = (processed_audio_path != audio_path)

            # Run speaker diarization
            self.logger.info("Running speaker diarization...")
            diarization_start_time = time.time()

            if expected_speakers is not None:
                diarization = self.pipeline(processed_audio_path, num_speakers=expected_speakers)
            else:
                diarization = self.pipeline(processed_audio_path)

            diarization_time = time.time() - diarization_start_time

            # Post-process with overlap removal
            post_process_start = time.time()
            diarization = self._post_process_diarization(diarization)
            post_process_time = time.time() - post_process_start

            # Extract speaker segments
            speaker_segments = []
            unique_speakers = set()

            for segment, _, speaker_label in diarization.itertracks(yield_label=True):
                # Check for remaining overlaps
                overlap_detected = False
                overlap_with = []

                for other_segment, _, other_speaker in diarization.itertracks(yield_label=True):
                    if other_speaker != speaker_label:
                        overlap_start = max(segment.start, other_segment.start)
                        overlap_end = min(segment.end, other_segment.end)
                        if overlap_end > overlap_start:
                            overlap_detected = True
                            overlap_with.append(other_speaker)

                segment_data = SpeakerSegment(
                    speaker_id=speaker_label,
                    start_time=self._to_timestamp(segment.start),
                    end_time=self._to_timestamp(segment.end),
                    duration_sec=round(segment.duration, 3),
                    confidence_score=0.95,
                    overlap_detected=overlap_detected,
                    overlap_with=overlap_with if overlap_with else None
                )

                speaker_segments.append(segment_data)
                unique_speakers.add(speaker_label)

            # Sort by start time
            speaker_segments.sort(key=lambda x: self._parse_time(x.start_time))

            # Calculate statistics
            speaker_stats = self._calculate_speaker_stats(diarization)
            voice_events = self._detect_voice_events(diarization)

            # Get audio duration
            try:
                audio_duration = librosa.get_duration(path=audio_path)
            except:
                audio_duration = max((seg.duration_sec for seg in speaker_segments), default=0.0)

            # Integrity analysis
            integrity_analysis = self._analyze_integrity(speaker_stats, voice_events, audio_duration)

            # Determine primary speaker
            primary_speaker = None
            if speaker_stats:
                primary_speaker = max(speaker_stats.keys(), key=lambda s: speaker_stats[s]['total_duration'])

            # Calculate total processing time
            total_processing_time = time.time() - analysis_start_time

            # Build results
            results = {
                "audio_metadata": {
                    "path": os.path.basename(audio_path),
                    "duration_sec": round(audio_duration, 2),
                    "sample_rate": self.audio_sample_rate,
                    "total_processing_time_sec": round(total_processing_time, 2),
                    "diarization_time_sec": round(diarization_time, 2),
                    "post_processing_time_sec": round(post_process_time, 2),
                    "preprocessing_applied": preprocess_audio,
                    "device_used": self.device,
                    **self.processing_stats
                },
                "multiple_voices": {
                    "count_unique_speakers": len(unique_speakers),
                    "primary_speaker": primary_speaker,
                    "speakers_detected": sorted(list(unique_speakers)),
                    "total_segments": len(speaker_segments),
                    "segments": [
                        {
                            "speaker_id": seg.speaker_id,
                            "start": seg.start_time,
                            "end": seg.end_time,
                            "duration_sec": seg.duration_sec,
                            "confidence": seg.confidence_score,
                            "overlap_detected": seg.overlap_detected,
                            "overlap_with": seg.overlap_with
                        }
                        for seg in speaker_segments
                    ]
                },
                "speaker_statistics": speaker_stats,
                "voice_change_events": {
                    "count_events": len(voice_events),
                    "events": [
                        {
                            "from_speaker": event.from_speaker,
                            "to_speaker": event.to_speaker,
                            "transition_time": event.transition_time,
                            "end_time": event.end_time,
                            "duration_sec": event.duration_sec,
                            "type": event.type,
                            "gap_duration": event.gap_duration,
                            "overlap_duration": event.overlap_duration
                        }
                        for event in voice_events
                    ]
                },
                "integrity_analysis": integrity_analysis,
                "processing_corrections": {
                    "overlap_corrections": len(self.overlap_corrections),
                    "correction_details": self.overlap_corrections
                }
            }

            # Log results
            self.logger.info(f"Voice Detection Results:")
            self.logger.info(f"Speakers detected: {len(unique_speakers)}")
            self.logger.info(f"Speech segments: {len(speaker_segments)}")
            self.logger.info(f"Voice events: {len(voice_events)}")
            self.logger.info(f"Integrity score: {integrity_analysis['integrity_score']}")
            self.logger.info(f"Processing time: {total_processing_time:.2f}s")

            return results

        except Exception as e:
            self.logger.error(f"Voice detection failed: {e}")
            raise

        finally:
            # Cleanup temporary files
            if temp_file_created and os.path.exists(processed_audio_path):
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        temp_audio_path = None

        try:
            temp_audio_path = tempfile.mktemp(suffix='.wav')

            # ffmpeg command for audio extraction
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.audio_sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-af', 'highpass=f=80',  # High-pass filter
                '-y',  # Overwrite output file
                '-hide_banner',  # Reduce output
                '-loglevel', 'error',  # Only show errors
                temp_audio_path
            ]

            self.logger.info(f"Extracting audio from video: {os.path.basename(video_path)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                if "Output file does not contain any stream" in result.stderr:
                    self.logger.warning("No audio stream found in video")
                    return None
                else:
                    raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            # Verify extracted audio
            if not os.path.exists(temp_audio_path):
                raise RuntimeError("Audio extraction failed - no output file created")

            try:
                audio_duration = librosa.get_duration(path=temp_audio_path)
                if audio_duration < 0.1:
                    self.logger.warning("Extracted audio is too short or empty")
                    return None
                self.logger.info(f"Audio extracted: {audio_duration:.2f} seconds")
            except Exception as e:
                self.logger.warning(f"Could not verify extracted audio: {e}")
                return None

            return temp_audio_path

        except FileNotFoundError:
            self.logger.error("ffmpeg not found. Please install ffmpeg")
            raise
        except Exception as e:
            # Clean up temp file on any exception
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            self.logger.error(f"Audio extraction failed: {e}")
            raise

    def analyze_video_voices(self, video_path: str, **kwargs) -> Dict:
        """Analyze voices in video file"""
        audio_path = None

        try:
            self.logger.info(f"Analyzing voices in video: {os.path.basename(video_path)}")
            audio_path = self.extract_audio_from_video(video_path)

            # If no audio stream, return default structure
            if audio_path is None:
                video_duration = 0
                try:
                    video_duration = librosa.get_duration(filename=video_path)
                except Exception:
                    pass

                return {
                    "audio_metadata": {
                        "path": os.path.basename(video_path),
                        "error": "No audio stream found in video",
                        "duration_sec": video_duration,
                        "extracted_from_video": True,
                        "video_path": os.path.basename(video_path)
                    },
                    "multiple_voices": {
                        "count_unique_speakers": 0,
                        "primary_speaker": None,
                        "speakers_detected": [],
                        "total_segments": 0,
                        "segments": []
                    },
                    "speaker_statistics": {},
                    "voice_change_events": {
                        "count_events": 0,
                        "events": []
                    },
                    "integrity_analysis": {
                        "is_single_speaker": True,
                        "multiple_speakers_detected": False,
                        "suspicious_patterns": ["no_audio_stream_detected"],
                        "risk_factors": {},
                        "integrity_score": 1.0,
                        "risk_level": "LOW",
                        "recommendation": "ACCEPT",
                        "overlap_analysis": {},
                        "event_analysis": {}
                    },
                    "processing_corrections": {
                        "overlap_corrections": 0,
                        "correction_details": []
                    }
                }

            # Analyze extracted audio
            results = self.detect_multiple_voices(audio_path, **kwargs)
            results["audio_metadata"]["extracted_from_video"] = True
            results["audio_metadata"]["video_path"] = os.path.basename(video_path)

            return results

        finally:
            # Cleanup extracted audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

    def cleanup(self):
        """Clean up resources"""
        try:
            # Preserve pipeline for reuse
            if not hasattr(self, 'pipeline'):
                try:
                    self._initialize_pipeline()
                except Exception as e:
                    self.logger.warning(f"Failed to reinitialize pipeline: {e}")

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

    def reinitialize(self):
        """Reinitialize components if needed"""
        if not hasattr(self, 'pipeline'):
            self._initialize_pipeline()

    def reset(self):
        """Reset detector state for new analysis"""
        self.detected_speakers.clear()
        self.voice_events.clear()
        self.processing_stats.clear()
        self.overlap_corrections.clear()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



