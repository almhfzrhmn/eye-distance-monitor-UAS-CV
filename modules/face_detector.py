"""
OBJECT DETECTION (FACE & EYE DETECTION)
Modul untuk deteksi wajah dan ekstraksi landmark iris menggunakan MediaPipe
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from config import settings

class FaceDetector:
    """
    Kelas untuk mendeteksi wajah dan mengekstrak posisi iris
    Menggunakan MediaPipe Face Landmarker dengan landmark points
    """

    def __init__(self):
        """Inisialisasi MediaPipe Face Landmarker detector"""
        # Setup MediaPipe Face Landmarker (Materi 7: Deep Learning based Detection)
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=settings.MAX_NUM_FACES,
            min_face_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=settings.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def detect_face(self, rgb_frame):
        """
        Deteksi wajah pada frame RGB

        Args:
            rgb_frame: Frame dalam format RGB (MediaPipe requirement)

        Returns:
            Mock results object compatible with old API atau None
        """
        # Proses deteksi menggunakan neural network (Materi 7)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)

        # Convert to old API format for compatibility
        if result.face_landmarks:
            # Create mock landmark objects
            landmarks = []
            for lm in result.face_landmarks[0]:
                class Landmark:
                    def __init__(self, x, y, z):
                        self.x = x
                        self.y = y
                        self.z = z
                landmarks.append(Landmark(lm.x, lm.y, lm.z))

            class FaceLandmarks:
                def __init__(self, landmarks):
                    self.landmark = landmarks

            class MockResults:
                def __init__(self, face_landmarks):
                    self.multi_face_landmarks = face_landmarks

            mock_results = MockResults([FaceLandmarks(landmarks)])
            return mock_results
        else:
            return None
    
    def get_iris_positions(self, landmarks, frame_width, frame_height):
        """
        FEATURE EXTRACTION
        Ekstraksi koordinat pusat iris kiri dan kanan
        
        Args:
            landmarks: Array landmark dari MediaPipe (normalized 0-1)
            frame_width: Lebar frame dalam pixel
            frame_height: Tinggi frame dalam pixel
        
        Returns:
            Tuple: (iris_distance_pixel, left_center, right_center)
        """
        # Konversi normalized coordinates ke pixel coordinates
        # Hitung pusat iris kiri dari 4 landmark points (averaging)
        left_iris_center = np.mean([
            [landmarks[i].x * frame_width, landmarks[i].y * frame_height]
            for i in settings.LEFT_IRIS
        ], axis=0)
        
        # Hitung pusat iris kanan dari 4 landmark points
        right_iris_center = np.mean([
            [landmarks[i].x * frame_width, landmarks[i].y * frame_height]
            for i in settings.RIGHT_IRIS
        ], axis=0)
        
        # Hitung jarak Euclidean antar iris (Interpupillary Distance dalam pixel)
        # Rumus: d = sqrt((x2-x1)² + (y2-y1)²) - Distance Metrics
        iris_distance_pixel = np.linalg.norm(left_iris_center - right_iris_center)
        
        return iris_distance_pixel, left_iris_center, right_iris_center
    
    def close(self):
        """Release MediaPipe resources"""
        self.detector.close()