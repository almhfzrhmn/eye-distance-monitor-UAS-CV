"""
EYE DISTANCE MONITOR

Materi yang diimplementasikan:
- Materi 7: Object Detection (Face & Eye Detection)
- Materi 10: Stereo Vision & 3D Reconstruction (Depth Estimation)
- Materi 2: Filtering (Gaussian Blur)

"""

import cv2
import sys

# Import modul-modul yang sudah dibuat
from modules import FaceDetector, DistanceCalculator, ImageProcessor
from utils import Camera
from config import settings

def print_instructions():
    """Menampilkan instruksi penggunaan program"""
    print("=" * 60)
    print("EYE DISTANCE MONITOR")
    print("=" * 60)
    print("\nINSTRUKSI:")
    print("1. Posisikan wajah pada jarak ~60 cm untuk kalibrasi")
    print("2. Tunggu hingga kalibrasi selesai (30 frame)")
    print("3. Jika jarak < 50 cm, layar akan blur (WARNING!)")
    print("4. Tekan 'q' untuk keluar dari program")
    print("=" * 60)
    print("\nMemulai program...\n")

def main():
    """Fungsi utama aplikasi"""
    
    # Tampilkan instruksi
    print_instructions()
    
    # ========== INISIALISASI MODUL-MODUL ==========
    try:
        # Inisialisasi kamera
        camera = Camera()
        
        # Inisialisasi detector wajah (MATERI 7)
        face_detector = FaceDetector()
        
        # Inisialisasi calculator jarak (MATERI 10)
        distance_calc = DistanceCalculator()
        
        # Image processor untuk filtering (MATERI 2)
        img_processor = ImageProcessor()
        
        print("Semua modul berhasil diinisialisasi!")
        print("Kamera aktif. Mulai deteksi...\n")
        
    except Exception as e:
        print(f"Error saat inisialisasi: {e}")
        sys.exit(1)
    
    try:
        while camera.is_opened():
            # Baca frame dari webcam
            success, frame = camera.read_frame()
            
            if not success:
                print("Gagal membaca frame dari kamera")
                break
            
            # Konversi BGR ke RGB untuk MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Dimensi frame
            h, w, _ = frame.shape
            
            # ========== FACE DETECTION ==========
            results = face_detector.detect_face(rgb_frame)

            if results and results.multi_face_landmarks:
                # Ambil landmark wajah pertama
                face_landmarks = results.multi_face_landmarks[0]
                
                # Ekstraksi posisi iris 
                iris_distance_px, left_center, right_center = \
                    face_detector.get_iris_positions(face_landmarks.landmark, w, h)
                
                if not distance_calc.is_calibrated:
                    # Kalibrasi focal length
                    distance_calc.calibrate(iris_distance_px)
                    
                    # Tampilkan progress kalibrasi
                    current, total, percentage = distance_calc.get_calibration_progress()
                    cv2.putText(frame, f'Kalibrasi: {current}/{total} ({percentage:.0f}%)',
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (255, 255, 0), 2)
                    
                    if current == total:
                        print(f"Kalibrasi selesai!")
                        print(f"   Focal Length: {distance_calc.focal_length:.2f} pixel\n")
                
                # ========== ESTIMASI JARAK ==========
                else:
                    distance = distance_calc.calculate_distance(iris_distance_px)
                    
                    # Cek apakah jarak aman
                    is_safe = distance >= settings.DISTANCE_THRESHOLD
                    
                    if not is_safe:
                        # Aplikasi Gaussian Blur sebagai warning
                        frame = img_processor.apply_gaussian_blur(frame)
                        frame = img_processor.add_warning_text(frame, 'TERLALU DEKAT!')
                    
                    # Tambahkan informasi jarak ke frame
                    frame = img_processor.add_distance_info(frame, distance, is_safe)
                    
                    # Visualisasi posisi iris
                    frame = img_processor.draw_iris_visualization(
                        frame, left_center, right_center
                    )
            
            else:
                # Feedback jika wajah tidak terdeteksi
                cv2.putText(frame, 'Wajah tidak terdeteksi', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Tampilkan frame hasil processing
            cv2.imshow('Monitor Jarak Mata - Informatika', frame)
            
            # Break loop jika 'q' ditekan
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("\nProgram dihentikan oleh user")
                break
    
    except KeyboardInterrupt:
        print("\nProgram dihentikan (Ctrl+C)")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        # ========== CLEANUP ==========
        print("\nMembersihkan resources...")
        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("Program selesai. Terima kasih!\n")

if __name__ == "__main__":
    main()