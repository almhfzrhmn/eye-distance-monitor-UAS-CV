"""
Modul untuk menghitung estimasi jarak menggunakan depth estimation monocular
"""

from config import settings

class DistanceCalculator:
    """
    Kelas untuk menghitung jarak objek ke kamera menggunakan prinsip similar triangles
    Implementasi depth estimation dengan single camera (monocular)
    """
    
    def __init__(self):
        """Inisialisasi calculator dengan focal length None (akan dikalibrasi)"""
        self.focal_length = None
        self.calibration_count = 0
        self.is_calibrated = False
    
    def calibrate(self, pixel_width):
        """
        CAMERA CALIBRATION
        Kalibrasi focal length kamera menggunakan reference distance
        
        Args:
            pixel_width: Lebar objek dalam pixel (IPD dalam pixel)
        
        Returns:
            Boolean: True jika kalibrasi selesai
        """
        if self.is_calibrated:
            return True
        
        # Hitung focal length menggunakan formula pinhole camera model
        # f = (P × D) / W (Materi 10: Camera Model)
        temp_focal = (pixel_width * settings.KNOWN_DISTANCE) / settings.KNOWN_WIDTH
        
        # Smoothing dengan exponential moving average untuk stabilitas
        if self.focal_length is None:
            self.focal_length = temp_focal
        else:
            # Alpha = 0.9 untuk smoothing (mengurangi noise dari jittering)
            self.focal_length = 0.9 * self.focal_length + 0.1 * temp_focal
        
        self.calibration_count += 1
        
        # Kalibrasi selesai setelah N frames
        if self.calibration_count >= settings.CALIBRATION_FRAMES:
            self.is_calibrated = True
            return True
        
        return False
    
    def calculate_distance(self, pixel_width):
        """
        DEPTH ESTIMATION
        Menghitung jarak objek ke kamera menggunakan similar triangles principle
        
        Prinsip: Semakin jauh objek, semakin kecil proyeksinya di image plane
        
        Args:
            pixel_width: Lebar objek dalam pixel
        
        Returns:
            Estimasi jarak dalam cm, atau 0 jika belum dikalibrasi
        """
        if not self.is_calibrated or pixel_width == 0:
            return 0
        
        # Aplikasi formula depth estimation
        distance = (settings.KNOWN_WIDTH * self.focal_length) / pixel_width
        
        return distance
    
    def get_calibration_progress(self):
        """
        Mendapatkan progress kalibrasi
        
        Returns:
            Tuple: (current_count, total_frames, percentage)
        """
        percentage = (self.calibration_count / settings.CALIBRATION_FRAMES) * 100
        return self.calibration_count, settings.CALIBRATION_FRAMES, percentage