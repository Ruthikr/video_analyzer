import os
import asyncio
import tempfile
import logging
import gc
import psutil
import torch
import multiprocessing
import platform
import weakref
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Conditional uvloop import for better performance on Linux/macOS
try:
    if platform.system() != "Windows":
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import detection systems
from detectors.FaceAnalyzer import FaceAnalyzer, AnalysisConfig
from detectors.GazeAnalyzer import GazeAnalyzer, GazeConfig
from detectors.VoiceAnalyzer import VoiceAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DetectionAPI")

class ResourceMonitor:
    """Monitor and manage system resources"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.request_count = 0
        self.last_cleanup = time.time()
        self._lock = threading.Lock()

    def get_memory_usage(self):
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        
        return {
            'cpu_memory_mb': round(memory_info.rss / (1024**2), 2),
            'gpu_memory_mb': round(gpu_memory, 2),
            'cpu_percent': self.process.cpu_percent()
        }

    def should_force_cleanup(self):
        """Determine if forced cleanup is needed"""
        current_memory = self.get_memory_usage()
        memory_growth = current_memory['cpu_memory_mb'] - self.initial_memory['cpu_memory_mb']
        
        # Force cleanup if memory grows by more than 500MB or every 10 requests
        return (memory_growth > 500 or 
                self.request_count % 10 == 0 or 
                time.time() - self.last_cleanup > 300)  # 5 minutes

    def force_cleanup(self):
        """Force comprehensive cleanup"""
        with self._lock:
            logger.info("Performing forced cleanup...")
            
            # Python garbage collection
            collected = gc.collect()
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Update tracking
            self.last_cleanup = time.time()
            logger.info(f"Cleanup complete - collected {collected} objects")

class SystemResources:
    """Auto-detect and configure system resources"""

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = torch.cuda.is_available()
        self.gpu_name = None
        self.gpu_memory_gb = 0
        self.platform = platform.system()

        if self.gpu_available:
            try:
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
                self.gpu_available = False

        # Configure for sustained performance
        self.max_workers = min(3, max(1, self.cpu_count // 2))
        self.use_gpu = self.gpu_available and self.gpu_memory_gb > 4

        logger.info(f"System Resources - Platform: {self.platform}, CPU: {self.cpu_count} cores, "
                   f"Memory: {self.memory_gb:.1f}GB, GPU: {self.gpu_available}, "
                   f"Max Workers: {self.max_workers}")

# Initialize system resources and monitor
system_resources = SystemResources()
resource_monitor = ResourceMonitor()

class ModelPool:
    """Pool of pre-initialized models to avoid repeated initialization"""

    def __init__(self):
        self._voice_analyzers = []
        self._max_pool_size = 2
        self._lock = threading.Lock()
        self._hf_token = os.getenv("HF_TOKEN")

    def get_voice_analyzer(self):
        """Get a voice analyzer from pool or create new one"""
        with self._lock:
            if self._voice_analyzers and self._hf_token:
                return self._voice_analyzers.pop()
            elif self._hf_token:
                return VoiceAnalyzer(
                    hf_token=self._hf_token,
                    use_cuda=system_resources.use_gpu,
                    min_speaker_duration=1.0,
                    confidence_threshold=0.75,
                    log_level="WARNING"
                )
            return None

    def return_voice_analyzer(self, analyzer):
        """Return analyzer to pool"""
        with self._lock:
            if analyzer and len(self._voice_analyzers) < self._max_pool_size:
                # Reset analyzer state but keep it initialized
                analyzer.reset()
                self._voice_analyzers.append(analyzer)
            elif analyzer:
                # Clean up excess analyzer
                try:
                    analyzer.cleanup()
                except:
                    pass

# Initialize model pool
model_pool = ModelPool()

class ThreadPoolExecutor(ThreadPoolExecutor):
    """Enhanced thread pool with proper cleanup"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_tasks = set()

    def submit(self, fn, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        self._active_tasks.add(future)
        future.add_done_callback(self._active_tasks.discard)
        return future

# Create executor
executor = ThreadPoolExecutor(max_workers=system_resources.max_workers, thread_name_prefix="DetectionWorker")

class DetectionResponse(BaseModel):
    """Response model for combined detection results"""
    success: bool
    processing_time_sec: float
    face_detection: Optional[Dict[str, Any]] = None
    gaze_detection: Optional[Dict[str, Any]] = None
    voice_detection: Optional[Dict[str, Any]] = None
    errors: Dict[str, str] = {}
    system_info: Dict[str, Any] = {}

class DetectionService:
    """Service for running detection algorithms"""

    def __init__(self):
        self.request_counter = 0

    async def run_face_detection(self, video_path: str) -> Dict[str, Any]:
        """Run face detection with optimized settings"""
        try:
            logger.debug("Starting face detection")
            config = AnalysisConfig(
                frame_processing_interval=15,
                use_gpu=system_resources.use_gpu,
                face_detection_model="hog",
                face_detection_scale=0.5,
                log_level="WARNING"
            )

            def face_task():
                analyzer = None
                try:
                    analyzer = FaceAnalyzer(config)
                    result = analyzer.analyze_video(video_path)
                    return result
                finally:
                    if analyzer:
                        analyzer.cleanup()
                    del analyzer
                    gc.collect()

            result = await asyncio.get_event_loop().run_in_executor(executor, face_task)
            logger.debug("Face detection completed")
            return {"result": result, "error": None}

        except Exception as e:
            error_msg = f"Face detection failed: {str(e)}"
            logger.error(error_msg)
            return {"result": None, "error": error_msg}

    async def run_gaze_detection(self, video_path: str) -> Dict[str, Any]:
        """Run gaze detection with optimized settings"""
        try:
            logger.debug("Starting gaze detection")
            config = GazeConfig(
                frame_processing_interval=8,
                min_event_duration=2.0,
                max_tracking_failures=50,
                max_consecutive_failures=20,
                early_stop_failure_rate=0.9,
                log_level="WARNING"
            )

            def gaze_task():
                analyzer = None
                try:
                    analyzer = GazeAnalyzer(config)
                    result = analyzer.analyze_video(video_path)
                    return result
                finally:
                    if analyzer:
                        analyzer.cleanup()
                    del analyzer
                    gc.collect()

            result = await asyncio.get_event_loop().run_in_executor(executor, gaze_task)
            logger.debug("Gaze detection completed")
            return {"result": result, "error": None}

        except Exception as e:
            error_msg = f"Gaze detection failed: {str(e)}"
            logger.error(error_msg)
            return {"result": None, "error": error_msg}

    async def run_voice_detection(self, video_path: str) -> Dict[str, Any]:
        """Run voice detection with model pooling"""
        try:
            analyzer = model_pool.get_voice_analyzer()
            if not analyzer:
                return {"result": None, "error": "Voice detection not available - HF token not set"}

            logger.debug("Starting voice detection")

            def voice_task():
                try:
                    result = analyzer.analyze_video_voices(video_path, preprocess_audio=False)
                    return result
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            result = await asyncio.get_event_loop().run_in_executor(executor, voice_task)

            # Return analyzer to pool
            model_pool.return_voice_analyzer(analyzer)
            logger.debug("Voice detection completed")
            return {"result": result, "error": None}

        except Exception as e:
            # Make sure analyzer is returned even on error
            if 'analyzer' in locals() and analyzer:
                model_pool.return_voice_analyzer(analyzer)
            
            error_msg = f"Voice detection failed: {str(e)}"
            logger.error(error_msg)
            return {"result": None, "error": error_msg}

    async def process_video(self, video_path: str) -> DetectionResponse:
        """Process video with resource management"""
        self.request_counter += 1
        resource_monitor.request_count += 1
        start_time = asyncio.get_event_loop().time()

        # Check if we need forced cleanup before processing
        if resource_monitor.should_force_cleanup():
            resource_monitor.force_cleanup()

        # Run detections in parallel
        tasks = [
            self.run_face_detection(video_path),
            self.run_gaze_detection(video_path),
            self.run_voice_detection(video_path)
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            results = [{"result": None, "error": str(e)}] * 3

        processing_time = asyncio.get_event_loop().time() - start_time

        # Process results
        face_result, gaze_result, voice_result = results
        errors = {}

        # Handle face detection results
        face_detection = None
        if isinstance(face_result, dict) and face_result.get("result"):
            face_detection = face_result["result"]
        elif isinstance(face_result, dict) and face_result.get("error"):
            errors["face_detection"] = face_result["error"]
        elif isinstance(face_result, Exception):
            errors["face_detection"] = f"Face detection exception: {str(face_result)}"

        # Handle gaze detection results
        gaze_detection = None
        if isinstance(gaze_result, dict) and gaze_result.get("result"):
            gaze_detection = gaze_result["result"]
        elif isinstance(gaze_result, dict) and gaze_result.get("error"):
            errors["gaze_detection"] = gaze_result["error"]
        elif isinstance(gaze_result, Exception):
            errors["gaze_detection"] = f"Gaze detection exception: {str(gaze_result)}"

        # Handle voice detection results
        voice_detection = None
        if isinstance(voice_result, dict) and voice_result.get("result"):
            voice_detection = voice_result["result"]
        elif isinstance(voice_result, dict) and voice_result.get("error"):
            errors["voice_detection"] = voice_result["error"]
        elif isinstance(voice_result, Exception):
            errors["voice_detection"] = f"Voice detection exception: {str(voice_result)}"

        success = bool(face_detection or gaze_detection or voice_detection)

        # Get current resource usage
        memory_usage = resource_monitor.get_memory_usage()

        return DetectionResponse(
            success=success,
            processing_time_sec=round(processing_time, 2),
            face_detection=face_detection,
            gaze_detection=gaze_detection,
            voice_detection=voice_detection,
            errors=errors,
            system_info={
                "platform": system_resources.platform,
                "request_number": self.request_counter,
                "memory_usage": memory_usage,
                "cpu_cores": system_resources.cpu_count,
                "gpu_available": system_resources.gpu_available
            }
        )

# Initialize FastAPI app
app = FastAPI(
    title="Video Detection API",
    description="Video analysis with face, gaze and voice detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection service
detection_service = DetectionService()

async def cleanup_files(file_path: str):
    """Clean up temporary files and memory"""
    try:
        await asyncio.sleep(0.5)
        
        # Remove file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove parent directory if empty
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
            os.rmdir(parent_dir)

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception as e:
        logger.warning(f"Cleanup error for {file_path}: {e}")

@app.post("/analyze", response_model=DetectionResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...)
):
    """Video analysis endpoint"""
    temp_dir = None
    temp_path = None

    try:
        # Validate file
        if not video_file.filename:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No filename provided",
                    "processing_time_sec": 0,
                    "errors": {"upload": "No filename provided"}
                }
            )

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="vid_")
        temp_path = os.path.join(temp_dir, video_file.filename)

        # Save uploaded file
        logger.info(f"Processing video: {video_file.filename} (Request #{resource_monitor.request_count + 1})")

        # Stream file to disk to avoid loading entire file in memory
        with open(temp_path, "wb") as temp_file:
            while chunk := await video_file.read(8192):  # Read in 8KB chunks
                temp_file.write(chunk)

        # Get file size from disk
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.1f}MB")

        if file_size_mb > 500:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"File too large: {file_size_mb:.1f}MB (max 500MB)",
                    "processing_time_sec": 0,
                    "errors": {"upload": "File size exceeds limit"}
                }
            )

        # Schedule cleanup
        background_tasks.add_task(cleanup_files, temp_path)

        # Process video
        result = await detection_service.process_video(temp_path)

        logger.info(f"Request #{result.system_info['request_number']} completed - "
                   f"Success: {result.success}, Time: {result.processing_time_sec}s, "
                   f"Memory: {result.system_info['memory_usage']['cpu_memory_mb']}MB")

        return result

    except Exception as e:
        error_msg = f"API processing error: {str(e)}"
        logger.error(error_msg)

        # Still schedule cleanup
        if temp_path and os.path.exists(temp_path):
            background_tasks.add_task(cleanup_files, temp_path)

        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "processing_time_sec": 0,
                "errors": {"api": error_msg},
                "system_info": {
                    "memory_usage": resource_monitor.get_memory_usage(),
                    "platform": system_resources.platform
                }
            }
        )

@app.get("/health")
async def health_check():
    """Health check with resource monitoring"""
    memory_usage = resource_monitor.get_memory_usage()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "requests_processed": resource_monitor.request_count,
        "memory_usage": memory_usage,
        "system_resources": {
            "platform": system_resources.platform,
            "cpu_cores": system_resources.cpu_count,
            "memory_gb": round(system_resources.memory_gb, 1),
            "gpu_available": system_resources.gpu_available,
            "gpu_name": system_resources.gpu_name,
            "max_workers": system_resources.max_workers
        }
    }

@app.get("/memory")
async def memory_status():
    """Get detailed memory status"""
    return {
        "memory_usage": resource_monitor.get_memory_usage(),
        "initial_memory": resource_monitor.initial_memory,
        "requests_processed": resource_monitor.request_count,
        "last_cleanup": resource_monitor.last_cleanup,
        "should_cleanup": resource_monitor.should_force_cleanup()
    }

@app.post("/cleanup")
async def force_cleanup():
    """Manually trigger cleanup"""
    resource_monitor.force_cleanup()
    return {
        "message": "Cleanup completed",
        "memory_usage": resource_monitor.get_memory_usage()
    }

if __name__ == "__main__":
    import uvicorn
    
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "log_level": "info",
        "access_log": False,
    }

    if platform.system() != "Windows":
        try:
            import uvloop
            config["loop"] = "uvloop"
        except ImportError:
            pass

    uvicorn.run("main:app", **config)

