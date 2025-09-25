import os
import asyncio
import tempfile
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List
import json
import gc
from pathlib import Path
import shutil
from dataclasses import asdict
import traceback
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import video analysis modules
from VoiceAnalyzer import VoiceAnalyzer
from FaceAnalyzer import FaceAnalyzer, AnalysisConfig as FaceConfig
from GazeAnalyzer import VideoGazeTracker, GazeTrackerConfig


class VideoAnalysisResult(BaseModel):
    """Response model for video analysis results"""
    success: bool
    processing_time: float
    chunk_id: Optional[str] = None
    video_metadata: Dict
    voice_analysis: Optional[Dict] = None
    face_analysis: Optional[Dict] = None
    gaze_analysis: Optional[Dict] = None
    errors: Dict[str, str] = {}
    warnings: List[str] = []


class VideoAnalysisService:
    """Main service class for video analysis operations"""

    def __init__(self):
        self.app = FastAPI(
            title="Video Analysis API",
            description="Real-time video analysis service for voice, face, and gaze detection",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Load HuggingFace token for voice analysis
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        if not self.huggingface_token:
            print("Warning: No HuggingFace token found. Voice analysis will be disabled.")
        else:
            print(f"HuggingFace token loaded successfully: {self.huggingface_token[:10]}...")

        # Configure CORS for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize logging
        self.logger = self._configure_logging()

        # Create thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Setup API routes
        self._configure_routes()

        # Track active analyses with thread-safe access
        self.active_analyses = {}
        self.analysis_counter = 0
        self._analysis_lock = asyncio.Lock()

    def _configure_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("VideoAnalysisAPI")

    def _configure_routes(self):
        """Setup API endpoints"""

        @self.app.post("/analyze", response_model=VideoAnalysisResult)
        async def analyze_video_endpoint(
            file: UploadFile = File(...),
            enable_voice: str = Form("true"),
            enable_face: str = Form("true"),
            enable_gaze: str = Form("true"),
            chunk_id: Optional[str] = Form(None),
            background_tasks: BackgroundTasks = None
        ):
            """
            Analyze uploaded video file for voice, face, and gaze features
            """
            return await self.process_video_file(
                file, enable_voice, enable_face, enable_gaze, chunk_id, background_tasks
            )

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "active_analyses": len(self.active_analyses),
                "version": "1.0.0",
                "voice_analysis_available": bool(self.huggingface_token)
            }

        @self.app.get("/config")
        async def get_configuration():
            """Get current API configuration"""
            return {
                "voice_analysis": {
                    "enabled": bool(self.huggingface_token),
                    "token_configured": bool(self.huggingface_token)
                },
                "supported_formats": ["webm", "mp4", "mov", "avi"],
                "max_file_size": "50MB"
            }

    async def process_video_file(
        self,
        file: UploadFile,
        enable_voice: str,
        enable_face: str,
        enable_gaze: str,
        chunk_id: Optional[str],
        background_tasks: BackgroundTasks
    ) -> VideoAnalysisResult:
        """Process uploaded video file and run selected analyses"""

        start_time = time.time()
        temp_video_path = None

        # Generate unique execution ID 
        self.analysis_counter += 1
        execution_id = f"{chunk_id or 'auto'}_{self.analysis_counter}_{int(time.time()*1000)}"
        display_id = chunk_id or f"video_{self.analysis_counter}"

        # Track this analysis with unique execution ID
        async with self._analysis_lock:
            self.active_analyses[execution_id] = {
                "chunk_id": display_id,
                "start_time": start_time,
                "status": "processing"
            }

        try:
            # Parse boolean flags
            voice_enabled = enable_voice.lower() in ['true', '1', 'yes', 'on']
            face_enabled = enable_face.lower() in ['true', '1', 'yes', 'on']
            gaze_enabled = enable_gaze.lower() in ['true', '1', 'yes', 'on']

            self.logger.info(f"Processing {display_id}: voice={voice_enabled}, face={face_enabled}, gaze={gaze_enabled}")

            # Save uploaded file to temporary location
            temp_video_path = await self._save_uploaded_file(file, display_id)

            # Extract video metadata
            video_metadata = self._extract_video_metadata(temp_video_path)
            video_metadata["chunk_id"] = display_id

            # Run parallel analysis
            analysis_results = await self._run_parallel_analyses(
                temp_video_path, voice_enabled, face_enabled, gaze_enabled, display_id
            )

            processing_time = time.time() - start_time

            # Create response
            response = VideoAnalysisResult(
                success=True,
                processing_time=round(processing_time, 2),
                chunk_id=display_id,
                video_metadata=video_metadata,
                voice_analysis=analysis_results.get("voice"),
                face_analysis=analysis_results.get("face"),
                gaze_analysis=analysis_results.get("gaze"),
                errors=analysis_results.get("errors", {}),
                warnings=analysis_results.get("warnings", [])
            )

            # Update analysis status safely
            async with self._analysis_lock:
                if execution_id in self.active_analyses:
                    self.active_analyses[execution_id]["status"] = "completed"
                    self.active_analyses[execution_id]["processing_time"] = processing_time

            # Schedule cleanup tasks
            if background_tasks:
                background_tasks.add_task(self._cleanup_temporary_file, temp_video_path)
                background_tasks.add_task(self._cleanup_analysis_record, execution_id)

            self.logger.info(f"{display_id} completed in {processing_time:.2f}s")
            return response

        except Exception as e:
            self.logger.error(f"{display_id} failed: {str(e)}")
            self.logger.error(f"Error details: {traceback.format_exc()}")

            # Update analysis status safely
            async with self._analysis_lock:
                if execution_id in self.active_analyses:
                    self.active_analyses[execution_id]["status"] = "failed"
                    self.active_analyses[execution_id]["error"] = str(e)

            # Cleanup on error
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

            return VideoAnalysisResult(
                success=False,
                processing_time=time.time() - start_time,
                chunk_id=display_id,
                video_metadata={"chunk_id": display_id, "error": str(e)},
                errors={"system": f"Processing failed: {str(e)}"}
            )

    async def _save_uploaded_file(self, file: UploadFile, display_id: str) -> str:
        """Save uploaded file to temporary location"""
        try:
            content = await file.read()

            # Create temporary file with appropriate extension
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.webm',
                prefix=f"video_{display_id.replace('/', '_')}_",
                dir=tempfile.gettempdir()
            )

            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(content)

            self.logger.info(f"Saved {display_id} ({len(content)} bytes)")
            return temp_path

        except Exception as e:
            self.logger.error(f"File save failed for {display_id}: {e}")
            raise Exception(f"File processing error: {str(e)}")

    def _extract_video_metadata(self, video_path: str) -> Dict:
        """Extract basic metadata from video file"""
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Cannot read video file"}

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            return {
                "duration_sec": round(duration, 2),
                "fps": fps,
                "frame_count": frame_count,
                "file_size_mb": round(os.path.getsize(video_path) / (1024*1024), 2)
            }

        except Exception as e:
            return {
                "error": f"Metadata extraction failed: {str(e)}",
                "file_size_mb": round(os.path.getsize(video_path) / (1024*1024), 2) if os.path.exists(video_path) else 0
            }

    async def _run_parallel_analyses(
        self,
        video_path: str,
        enable_voice: bool,
        enable_face: bool,
        enable_gaze: bool,
        display_id: str
    ) -> Dict:
        """Execute multiple analyses in parallel"""

        results = {
            "voice": None,
            "face": None,
            "gaze": None,
            "errors": {},
            "warnings": []
        }

        # Prepare analysis tasks
        tasks = []
        if enable_voice and self.huggingface_token:
            tasks.append(("voice", self._run_voice_analysis(video_path, display_id)))
        elif enable_voice:
            results["errors"]["voice"] = "Voice analysis requested but HuggingFace token not available"

        if enable_face:
            tasks.append(("face", self._run_face_analysis(video_path, display_id)))

        if enable_gaze:
            tasks.append(("gaze", self._run_gaze_analysis(video_path, display_id)))

        # Execute tasks in parallel
        if tasks:
            completed_tasks = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

            for (analysis_type, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    error_msg = str(result)
                    results["errors"][analysis_type] = error_msg
                    self.logger.warning(f"{display_id} - {analysis_type}: {error_msg}")
                else:
                    results[analysis_type] = result
                    self.logger.info(f"{display_id} - {analysis_type} completed")

        return results

    async def _run_voice_analysis(self, video_path: str, display_id: str) -> Dict:
        """Execute voice analysis on video file"""

        def analyze():
            try:
                analyzer = VoiceAnalyzer(
                    hf_token=self.huggingface_token,
                    use_cuda=True,
                    min_speaker_duration=1.0,
                    min_voice_change_gap=1.0,
                    confidence_threshold=0.8,
                    log_level="ERROR"
                )

                self.logger.info(f"{display_id} - Starting voice analysis")
                result = analyzer.analyze_video_voices(video_path)

                # Cleanup
                del analyzer
                gc.collect()
                return result

            except Exception as e:
                raise Exception(f"Voice analysis failed: {str(e)}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, analyze)

    async def _run_face_analysis(self, video_path: str, display_id: str) -> Dict:
        """Execute face detection and analysis on video file"""

        def analyze():
            try:
                config = FaceConfig(
                    frame_processing_interval=15,  # Process every 15th frame for efficiency
                    face_detection_scale=0.5,
                    confidence_threshold=0.8,
                    log_level="ERROR"
                )

                analyzer = FaceAnalyzer(config)
                self.logger.info(f"{display_id} - Starting face analysis")
                result = analyzer.analyze_video(video_path)

                # Cleanup
                analyzer.cleanup()
                del analyzer
                gc.collect()
                return result

            except Exception as e:
                raise Exception(f"Face analysis failed: {str(e)}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, analyze)

    async def _run_gaze_analysis(self, video_path: str, display_id: str) -> Dict:
        """Execute gaze tracking analysis on video file"""

        def analyze():
            try:
                config = GazeTrackerConfig(
                    frame_skip=12,  # Skip frames for faster processing
                    min_event_duration=1.0,
                    confidence_threshold=0.8,
                    face_detection_enabled=False,
                    log_level="ERROR"
                )

                tracker = VideoGazeTracker(config)
                self.logger.info(f"{display_id} - Starting gaze analysis")
                result = tracker.analyze_video(video_path)

                # Cleanup
                tracker.cleanup()
                del tracker
                gc.collect()
                return result

            except Exception as e:
                raise Exception(f"Gaze analysis failed: {str(e)}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, analyze)

    async def _cleanup_temporary_file(self, file_path: str):
        """Clean up temporary files after processing"""
        try:
            await asyncio.sleep(5)  # Wait before cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
                self.logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")

    async def _cleanup_analysis_record(self, execution_id: str):
        """Remove analysis record from active tracking"""
        try:
            await asyncio.sleep(60)  # Keep record for 1 minute
            async with self._analysis_lock:
                if execution_id in self.active_analyses:
                    display_id = self.active_analyses[execution_id].get("chunk_id", "unknown")
                    del self.active_analyses[execution_id]
                    self.logger.debug(f"Cleaned up analysis record: {display_id}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup analysis record {execution_id}: {e}")

    def get_app(self):
        """Return FastAPI application instance"""
        return self.app


# Initialize the service
video_service = VideoAnalysisService()
app = video_service.get_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
