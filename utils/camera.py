"""
Modul untuk handling webcam operations
Abstraksi untuk memudahkan testing dan maintenance
"""

import cv2
from config import settings

class Camera:
    """
    Kelas untuk mengelola operasi webcam
    Wrapping OpenCV VideoCapture untuk kemudahan penggunaan
    """
    
    def __init__(self, camera_index=None):
        """
        Inisialisasi kamera
        
        Args:
            camera_index: Index kamera (default dari settings)
        """
        if camera_index is None:
            camera_index = settings.CAMERA_INDEX
        
        # Inisialisasi VideoCapture dengan index kamera
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set resolusi kamera untuk performa optimal
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        
        # Validasi apakah kamera berhasil dibuka
        if not self.cap.isOpened():
            raise RuntimeError("Gagal membuka kamera!")
    
    def read_frame(self):
        """
        Membaca frame dari webcam
        
        Returns:
            Tuple: (success, frame)
            - success: Boolean, True jika berhasil
            - frame: Image array atau None
        """
        success, frame = self.cap.read()
        
        if success:
            # Flip horizontal untuk efek mirror (user experience)
            frame = cv2.flip(frame, 1)
        
        return success, frame
    
    def is_opened(self):
        """
        Cek apakah kamera masih terbuka
        
        Returns:
            Boolean
        """
        return self.cap.isOpened()
    
    def release(self):
        """Release camera resource"""
        self.cap.release()
    
    def get_frame_dimensions(self):
        """
        Mendapatkan dimensi frame
        
        Returns:
            Tuple: (width, height)
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height