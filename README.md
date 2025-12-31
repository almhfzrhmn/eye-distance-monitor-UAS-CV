#  Monitor Jarak Mata ke Layar

Aplikasi Computer Vision untuk memonitor jarak mata pengguna ke layar laptop menggunakan OpenCV dan MediaPipe.

## 🎓 Materi yang Diimplementasikan
Berikut materi yang diimplementasikan pada sistem

### Materi 7: Object Detection (Face & Eye Detection)
- Deteksi wajah menggunakan MediaPipe Face Mesh
- Ekstraksi landmark iris (468 landmark points)
- Feature extraction untuk interpupillary distance

### Materi 10: Stereo Vision & 3D Reconstruction
- Depth estimation monocular (single camera)
- Camera calibration untuk focal length
- Implementasi similar triangles principle

### Materi 2: Filtering
- Gaussian Blur sebagai visual feedback
- Low-pass filtering untuk smoothing
- Conditional image processing

## 📦 Instalasi

### 1. Clone/Download Project
```bash
cd eye-distance-monitor
```

### 2. Buat Virtual Environment (Recommended)
```bash
python -m venv venv

# Aktivasi:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
Berikut library python yang diperlukan dalam project ini : 
- numpy
- mediapipe
- opencv

## 🚀 Cara Menjalankan
```bash
python main.py
```

## ⚙️ Konfigurasi

Edit file `config/settings.py` untuk mengubah:
- Jarak kalibrasi (default: 60 cm)
- Threshold jarak aman (default: 50 cm)
- Jumlah frame kalibrasi
- Ukuran kernel Gaussian Blur
- Warna dan visual settings

## 📊 Cara Kerja

1. **Kalibrasi**: Program akan melakukan kalibrasi focal length kamera selama 30 frame pertama
2. **Deteksi**: MediaPipe mendeteksi wajah dan mengekstrak posisi iris
3. **Estimasi**: Menghitung jarak menggunakan formula: `Distance = (Known_Width × Focal_Length) / Pixel_Width`
4. **Warning**: Jika jarak < 50 cm, aplikasikan Gaussian Blur + teks warning

## 🎯 Fitur

- ✅ Real-time face detection
- ✅ Accurate distance estimation
- ✅ Visual warning system
- ✅ Modular architecture
