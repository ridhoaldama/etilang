# 🚀 Pendeteksian Pelanggaran Helm dengan YOLO

Proyek ini bertujuan untuk mendeteksi pengendara motor yang tidak memakai helm menggunakan model YOLO dan OpenCV. Program akan membaca video, mendeteksi kendaraan, mengidentifikasi apakah pengendara memakai helm atau tidak, serta mengenali TNKB (nomor kendaraan). Data pelanggar akan disimpan secara otomatis dalam file CSV dan gambar mereka akan disimpan untuk referensi lebih lanjut.

## 📌 Fitur Utama
- **Deteksi Objek dengan YOLO**: Menggunakan model YOLO untuk mendeteksi motor, pengendara tanpa helm, dan TNKB.
- **Penyimpanan Data Pelanggar**: Data pelanggar disimpan dalam file CSV secara real-time.
- **Penyimpanan Gambar Pelanggar**: Gambar motor dan TNKB disimpan dalam folder khusus.
- **Konfigurasi Threshold**: Dapat mengatur batas confidence untuk berbagai kategori deteksi.
- **Visualisasi dengan OpenCV**: Bounding box ditampilkan pada frame video secara real-time.

## 🛠️ Instalasi dan Penggunaan
### 1️⃣ Persyaratan
Pastikan Anda memiliki Python versi **3.8 - 3.12** dan menginstal pustaka berikut:
```bash
pip install opencv-python numpy ultralytics
```

### 2️⃣ Clone Repository
```bash
git clone https://github.com/username/repository-name.git
cd etilang
```

### 3️⃣ Letakkan Model YOLO dan Video
- **Model YOLO (`best.pt`)**: Letakkan model YOLO yang telah dilatih di dalam folder proyek. Jika belum memiliki model, silakan latih menggunakan YOLOv8 atau gunakan model yang sudah ada.
- **File Video (`tes.mp4`)**: Gantilah dengan video yang ingin Anda analisis. Pastikan format dan resolusinya sesuai dengan kebutuhan.

### 4️⃣ Jalankan Program
```bash
python main.py
```

## 📂 Struktur Proyek
```
📂 repository-name/
├── 📄 main.py              # Program utama
├── 📄 requirements.txt     # Daftar dependensi
├── 📄 README.md            # Dokumentasi
├── 📂 image_pelanggar/     # Folder untuk menyimpan gambar pelanggar
├── 📄 pelanggar.csv        # Data pelanggar dalam format CSV
├── 📄 best.pt              # Model YOLO (edit sesuai model Anda)
└── 📄 tes.mp4              # Video input (ganti dengan video yang ingin dianalisis)
```

## 📊 Format CSV
File `pelanggar.csv` menyimpan data pelanggar dengan format:
| No | Jenis Kendaraan | Gambar Pelanggar | Jenis Pelanggaran | No TNKB |
|----|----------------|------------------|-------------------|---------|
| 1  | Motor         | motor_1.jpg      | Tidak Pakai Helm  | tnkb_1.jpg |
| 2  | Motor         | motor_2.jpg      | Tidak Pakai Helm  | unknown |

## 🔧 Konfigurasi
Anda dapat menyesuaikan threshold dan parameter gambar di dalam kode:
```python
MOTOR_CONF_THRESHOLD = 0.4
WITHOUT_HELMET_CONF_THRESHOLD = 0.3
TNKB_CONF_THRESHOLD_LOW = 0.2
TNKB_CONF_THRESHOLD_HIGH = 0.5
```

## 🤝 Kontribusi
Jika Anda ingin berkontribusi, silakan fork repository ini, buat branch baru, dan ajukan pull request. Kami sangat menghargai kontribusi Anda!

## 📜 Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

🚀 **Selamat Mendeteksi Pelanggar!** Jika ada pertanyaan, jangan ragu untuk membuka issue atau menghubungi saya. 😊

