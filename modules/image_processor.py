import cv2
from config import settings

class ImageProcessor:
    
    @staticmethod
    def apply_gaussian_blur(frame):
        """
        
        Gaussian Blur menggunakan convolution dengan Gaussian kernel:
        G(x,y) = (1/2πσ²) * e^(-(x²+y²)/2σ²)
        
        Kernel yang lebih besar = blur yang lebih kuat
        
        Args:
            frame: Input image array (BGR)
        
        Returns:
            Blurred image array
        """
        
        blurred = cv2.GaussianBlur(frame, settings.BLUR_KERNEL_SIZE, 0)
        
        return blurred
    
    @staticmethod
    def add_warning_text(frame, text, position=(50, 100), color=None):
        """
        Menambahkan teks warning pada frame
        
        Args:
            frame: Input frame
            text: Teks yang akan ditampilkan
            position: Posisi teks (x, y)
            color: Warna teks dalam BGR
        """
        if color is None:
            color = settings.WARNING_COLOR
        
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_DUPLEX, 2, color, 4)
        
        return frame
    
    @staticmethod
    def add_distance_info(frame, distance, is_safe=True):
        """
        Menambahkan informasi jarak pada frame
        
        Args:
            frame: Input frame
            distance: Jarak dalam cm
            is_safe: Boolean, True jika jarak aman
        """
        color = settings.SAFE_COLOR if is_safe else settings.WARNING_COLOR
        text = f'Jarak: {distance:.1f} cm'
        
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame
    
    @staticmethod
    def draw_iris_visualization(frame, left_center, right_center):
        """
        
        Menggambar visualisasi posisi iris untuk debugging
        
        Args:
            frame: Input frame
            left_center: Koordinat pusat iris kiri (x, y)
            right_center: Koordinat pusat iris kanan (x, y)
        """
        # Gambar lingkaran pada pusat iris kiri
        cv2.circle(frame, tuple(left_center.astype(int)), 
                   5, settings.IRIS_COLOR, -1)
        
        # Gambar lingkaran pada pusat iris kanan
        cv2.circle(frame, tuple(right_center.astype(int)), 
                   5, settings.IRIS_COLOR, -1)
        
        # Gambar garis penghubung (IPD line)
        cv2.line(frame, tuple(left_center.astype(int)),
                 tuple(right_center.astype(int)), 
                 settings.LINE_COLOR, 2)
        
        return frame