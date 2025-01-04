# gui/webcam_tracker.py

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox

class TrackingMode(Enum):
    """Enum for different tracking modes"""
    FACE = "face"
    EYES = "eyes"
    HEAD_POSE = "head_pose"
    FULL = "full"

@dataclass
class TrackingConfig:
    """Configuration for the webcam tracker"""
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    face_detection_scale: float = 1.1
    face_detection_neighbors: int = 5
    min_face_size: Tuple[int, int] = (30, 30)
    enable_landmarks: bool = True
    enable_head_pose: bool = True
    enable_eye_tracking: bool = True
    flip_image: bool = False
    debug_mode: bool = False

class WebcamTracker(QObject):
    # Define signals for various events
    frame_ready = pyqtSignal(QImage)
    face_detected = pyqtSignal(dict)
    tracking_error = pyqtSignal(str)
    landmarks_detected = pyqtSignal(dict)
    head_pose_updated = pyqtSignal(dict)
    fps_updated = pyqtSignal(float)

    def __init__(self, parent=None, config: Optional[TrackingConfig] = None):
        super().__init__(parent)
        self.config = config or TrackingConfig()
        self.parent = parent
        self.cap = None
        self.is_tracking = False
        self.frame_count = 0
        self.last_fps_time = cv2.getTickCount()
        
        # Initialize timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.calculate_fps)
        
        # Load required models
        self._load_models()
        
        # Initialize tracking state
        self.tracking_state = {
            'last_face_position': None,
            'face_landmarks': None,
            'head_rotation': None,
            'eye_states': None,
            'fps': 0.0
        }

    def _load_models(self):
        """Load all required detection and tracking models"""
        try:
            # Load face detection model
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Load eye detection model if enabled
            if self.config.enable_eye_tracking:
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_eye.xml'
                )
            
            # Load facial landmarks model if enabled
            if self.config.enable_landmarks:

                face_landmark_path = Path("models/shape_predictor_68_face_landmarks.dat")
                if face_landmark_path.exists():
                    import dlib
                    self.landmark_detector = dlib.shape_predictor(str(face_landmark_path))
                else:
                    logging.warning("Facial landmarks model not found. Landmarks disabled.")
                    self.config.enable_landmarks = False
            
        except Exception as e:
            logging.error(f"Failed to load detection models: {e}")
            raise RuntimeError("Failed to initialize tracking models")

    def start(self, mode: TrackingMode = TrackingMode.FULL):
        """Start webcam tracking with specified mode"""
        try:
            self.cap = cv2.VideoCapture(self.config.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError("Unable to access webcam")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Start tracking
            self.is_tracking = True
            self.tracking_mode = mode
            self.timer.start(int(1000 / self.config.fps))
            self.fps_timer.start(1000)  # Update FPS every second
            
            logging.info(f"Webcam tracking started in {mode.value} mode")
            
        except Exception as e:
            self.tracking_error.emit(str(e))
            logging.error(f"Failed to start webcam: {e}")
            raise

    def stop(self):
        """Stop webcam tracking and clean up resources"""
        try:
            self.is_tracking = False
            self.timer.stop()
            self.fps_timer.stop()
            if self.cap is not None:
                self.cap.release()
            logging.info("Webcam tracking stopped")
        except Exception as e:
            logging.error(f"Error stopping webcam: {e}")

    def update_frame(self):
        """Process and update the current frame"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam")

            # Flip image if configured
            if self.config.flip_image:
                frame = cv2.flip(frame, 1)

            # Process frame based on tracking mode
            processed_frame = self._process_frame(frame)
            
            # Convert to Qt format and emit
            qt_image = self._convert_frame_to_qt(processed_frame)
            self.frame_ready.emit(qt_image)
            
            # Update frame count for FPS calculation
            self.frame_count += 1
            
        except Exception as e:
            self.tracking_error.emit(str(e))
            logging.error(f"Error updating frame: {e}")
            self.stop()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame based on current tracking mode"""
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.face_detection_scale,
                minNeighbors=self.config.face_detection_neighbors,
                minSize=self.config.min_face_size
            )
            
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Process face region based on tracking mode
                face_roi = gray[y:y+h, x:x+w]
                
                if self.tracking_mode in [TrackingMode.EYES, TrackingMode.FULL]:
                    self._track_eyes(frame, face_roi, x, y, w, h)
                
                if self.tracking_mode in [TrackingMode.HEAD_POSE, TrackingMode.FULL]:
                    self._estimate_head_pose(frame, face_roi, x, y, w, h)
                
                if self.config.enable_landmarks:
                    self._detect_landmarks(frame, face_roi, x, y, w, h)
                
                # Emit face detection data
                self.face_detected.emit({
                    'position': (x, y, w, h),
                    'confidence': 1.0  # Could be calculated based on detection parameters?
                })
            
            # Add debug information if enabled
            if self.config.debug_mode:
                self._add_debug_info(frame)
            
            return frame
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    def _track_eyes(self, frame, face_roi, x, y, w, h):
        """Track eyes within the face region"""
        try:
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            for (ex, ey, ew, eh) in eyes:
                # Convert to absolute coordinates
                abs_ex = x + ex
                abs_ey = y + ey
                
                # Draw eye rectangles
                cv2.rectangle(frame, (abs_ex, abs_ey), 
                            (abs_ex + ew, abs_ey + eh), (255, 0, 0), 2)
                
                # Calculate eye center
                eye_center = (abs_ex + ew//2, abs_ey + eh//2)
                cv2.circle(frame, eye_center, 2, (0, 255, 0), -1)
        except Exception as e:
            logging.error(f"Error tracking eyes: {e}")

    def _estimate_head_pose(self, frame, face_roi, x, y, w, h):
        """Estimate head pose using facial landmarks or other methods"""
        try:
            # This is a simplified head pose estimation
            # You might want to use more sophisticated methods
            face_center = (x + w//2, y + h//2)
            cv2.circle(frame, face_center, 3, (0, 0, 255), -1)
            
            # Calculate relative position from center
            frame_center = (frame.shape[1]//2, frame.shape[0]//2)
            rel_x = (face_center[0] - frame_center[0]) / frame_center[0]
            rel_y = (face_center[1] - frame_center[1]) / frame_center[1]
            
            # Emit head pose data
            self.head_pose_updated.emit({
                'x_rotation': rel_y * 45,  # Approximate degree rotation
                'y_rotation': rel_x * 45,
                'z_rotation': 0.0
            })
            
        except Exception as e:
            logging.error(f"Error estimating head pose: {e}")

    def detect_landmarks(self, frame, face_roi, x, y, w, h):
        """
        Detect and draw facial landmarks, then track face movement
        
        Args:
            frame: Input video frame
            face_roi: Region of interest containing the face
            x, y: Top-left coordinates of the face region
            w, h: Width and height of the face region
        """
        try:
            if hasattr(self, 'landmark_detector'):
                # Convert ROI to grayscale if needed
                if len(face_roi.shape) == 3:
                    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                else:
                    face_roi_gray = face_roi
                    
                # Detect landmarks
                landmarks = self.landmark_detector.fit(face_roi_gray)
                
                if landmarks is None:
                    return
                    
                # Initialize tracking points if this is the first detection
                if not hasattr(self, 'prev_landmarks'):
                    self.prev_landmarks = landmarks
                    self.prev_gray = face_roi_gray.copy()
                    return
                    
                # Calculate optical flow for tracking
                if hasattr(self, 'prev_gray') and hasattr(self, 'prev_landmarks'):
                    # Convert landmarks to numpy array of points
                    prev_points = np.float32([point for point in self.prev_landmarks])
                    curr_points = np.float32([point for point in landmarks])
                    
                    # Calculate optical flow
                    flow, status = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray,
                        face_roi_gray,
                        prev_points,
                        None,
                        winSize=(15,15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    
                    # Filter good points
                    good_new = curr_points[status==1]
                    good_old = prev_points[status==1]
                    
                    # Draw landmarks and tracking lines
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        # Convert to absolute coordinates
                        abs_x_new = x + int(new[0])
                        abs_y_new = y + int(new[1])
                        abs_x_old = x + int(old[0])
                        abs_y_old = y + int(old[1])
                        
                        # Draw the tracking line
                        cv2.line(frame, (abs_x_old, abs_y_old), (abs_x_new, abs_y_new), 
                                (0, 255, 0), 1)
                        # Draw the current landmark
                        cv2.circle(frame, (abs_x_new, abs_y_new), 1, (0, 255, 255), -1)
                    
                    # Calculate movement metrics
                    if len(good_new) > 0 and len(good_old) > 0:
                        movement = {
                            'displacement': np.mean(np.sqrt(np.sum((good_new - good_old)**2, axis=1))),
                            'direction': np.arctan2(np.mean(good_new[:, 1] - good_old[:, 1]),
                                                np.mean(good_new[:, 0] - good_old[:, 0]))
                        }
                    else:
                        movement = {'displacement': 0, 'direction': 0}
                    
                    # Update previous frame data
                    self.prev_gray = face_roi_gray.copy()
                    self.prev_landmarks = landmarks
                    
                    # Emit tracking data
                    self.landmarks_detected.emit({
                        'landmarks': landmarks.tolist(),
                        'face_position': (x, y, w, h),
                        'movement': movement
                    })
                    
        except Exception as e:
            logging.error(f"Error in landmark detection and tracking: {e}")
            self.prev_landmarks = None
            self.prev_gray = None

    def _add_debug_info(self, frame):
        """Add debug information to the frame"""
        # Add FPS
        cv2.putText(frame, f"FPS: {self.tracking_state['fps']:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add tracking mode
        cv2.putText(frame, f"Mode: {self.tracking_mode.value}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _convert_frame_to_qt(self, frame: np.ndarray) -> QImage:
        """Convert OpenCV frame to QImage"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def calculate_fps(self):
        """Calculate and update FPS"""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_fps_time) / cv2.getTickFrequency()
        self.tracking_state['fps'] = self.frame_count / time_diff
        self.fps_updated.emit(self.tracking_state['fps'])
        
        # Reset counters
        self.frame_count = 0
        self.last_fps_time = current_time

    def update_config(self, new_config: TrackingConfig):
        """Update tracker configuration"""
        restart_required = (
            new_config.camera_id != self.config.camera_id or
            new_config.frame_width != self.config.frame_width or
            new_config.frame_height != self.config.frame_height or
            new_config.fps != self.config.fps
        )
        
        self.config = new_config
        
        if restart_required and self.is_tracking:
            self.stop()
            self.start(self.tracking_mode)

    def get_tracking_state(self) -> Dict:
        """Get current tracking state"""
        return self.tracking_state.copy()

    def set_tracking_mode(self, mode: TrackingMode):
        """Change tracking mode during runtime"""
        self.tracking_mode = mode
        logging.info(f"Tracking mode changed to {mode.value}")