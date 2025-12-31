"""
Konfigurasi konstanta untuk aplikasi Monitor Jarak Mata
Memisahkan konfigurasi dari logic untuk maintainability
"""

# ========== KONSTANTA KALIBRASI  ==========
KNOWN_DISTANCE = 60.0  # Jarak kalibrasi dalam cm
KNOWN_WIDTH = 6.3      # Lebar rata-rata IPD manusia dalam cm
CALIBRATION_FRAMES = 30  # Jumlah frame untuk kalibrasi focal length

# ========== THRESHOLD & WARNING ==========
DISTANCE_THRESHOLD = 50.0  # Jarak minimum aman dalam cm

# ========== MEDIAPIPE CONFIGURATION  ==========
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True  # Aktifkan deteksi iris

# ========== LANDMARK INDICES  ==========
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ========== CAMERA SETTINGS ==========
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ========== VISUAL SETTINGS ==========
BLUR_KERNEL_SIZE = (35, 35)  # Gaussian kernel size
WARNING_COLOR = (0, 0, 255)  # BGR: Merah
SAFE_COLOR = (0, 255, 0)     # BGR: Hijau
IRIS_COLOR = (255, 0, 0)     # BGR: Biru
LINE_COLOR = (255, 255, 0)   # BGR: Cyan