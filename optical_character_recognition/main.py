# optical_character_recognition/ocr_utils.py
import cv2
import os
import uuid
from ultralytics import YOLO
import sys
import time
import numpy as np

def simple_loading(message="Loading", duration=1):
    """Simple loading animation untuk OCR"""
    animation_chars = ["/", "-", "\\", "|"]
    start_time = time.time()
    
    while time.time() - start_time < duration:
        for char in animation_chars:
            sys.stdout.write(f'\r{message} {char}')
            sys.stdout.flush()
            time.sleep(0.1)
    
    sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line

def load_ocr_model(model_path: str):
    """Load model OCR sekali saja."""
    simple_loading("Loading OCR model", 1)
    model = YOLO(model_path)
    print("✅ OCR Model loaded")
    return model

def preprocess_plate_image(img):
    """Preprocessing: grayscale, blur, CLAHE dengan grid output."""
    # Simpan gambar asli untuk grid
    original_image = img.copy()
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Bilateral filter untuk noise reduction
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. CLAHE untuk contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    # 4. Convert back to BGR untuk YOLO
    final_result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # BUAT GRID PREPROCESSING DAN SIMPAN
    grid_image = create_ocr_preprocessing_grid(original_image, gray, bilateral, enhanced, final_result)
    
    # Simpan grid ke folder preprocessing
    grid_dir = "optical_character_recognition/output/preprocess_grids"
    os.makedirs(grid_dir, exist_ok=True)
    
    grid_filename = f"preproc_grid_{uuid.uuid4().hex}.jpg"
    grid_path = os.path.join(grid_dir, grid_filename)
    cv2.imwrite(grid_path, grid_image)
    
    print(f"✅ Grid preprocessing disimpan: {os.path.basename(grid_path)}")
    
    return final_result

def create_ocr_preprocessing_grid(original, gray, bilateral, clahe, final):
    """Membuat grid gambar dengan semua tahap preprocessing OCR"""
    # Resize semua gambar ke ukuran yang sama (250x80 untuk konsistensi plat)
    size = (250, 80)
    
    original_resized = cv2.resize(original, size)
    gray_resized = cv2.resize(gray, size)
    bilateral_resized = cv2.resize(bilateral, size)
    clahe_resized = cv2.resize(clahe, size)
    final_resized = cv2.resize(final, size)
    
    # Convert grayscale images to BGR untuk konsistensi tampilan
    gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    bilateral_bgr = cv2.cvtColor(bilateral_resized, cv2.COLOR_GRAY2BGR)
    clahe_bgr = cv2.cvtColor(clahe_resized, cv2.COLOR_GRAY2BGR)
    
    # Tambahkan label/text pada setiap gambar
    def add_label(img, text):
        labeled_img = img.copy()
        # Tambahkan background untuk text
        cv2.rectangle(labeled_img, (0, 0), (250, 25), (0, 0, 0), -1)
        # Tambahkan text
        cv2.putText(labeled_img, text, (5, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return labeled_img
    
    # Label setiap gambar
    original_labeled = add_label(original_resized, "1. Original")
    gray_labeled = add_label(gray_bgr, "2. Grayscale")
    bilateral_labeled = add_label(bilateral_bgr, "3. Bilateral Filter")
    clahe_labeled = add_label(clahe_bgr, "4. CLAHE Enhanced")
    final_labeled = add_label(final_resized, "5. Final Result")
    
    # Buat grid 2x3 (2 baris, 3 kolom)
    # Baris pertama: Original, Grayscale, Bilateral Filter
    row1 = np.hstack([original_labeled, gray_labeled, bilateral_labeled])
    # Baris kedua: CLAHE, Final Result, Kosong
    row2 = np.hstack([clahe_labeled, final_labeled, np.zeros_like(original_labeled)])
    
    # Gabungkan baris
    grid = np.vstack([row1, row2])
    
    return grid

def run_ocr_on_plate(crop_path: str,
                     model_ocr,
                     preprocess_dir: str,
                     det_dir: str,
                     conf_threshold: float = 0.5):
    """
    OCR process dengan loading animation
    Returns: plate_string
    """
    print("\n🔤 OCR PROCESS")
    print("=" * 20)
    
    # Buat folder output
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    base_name = os.path.basename(crop_path)
    
    # 1. Load image
    simple_loading("Membaca gambar plat", 0.5)
    img = cv2.imread(crop_path)

    if img is None:
        print("❌ Gagal membaca gambar plat")
        return ""

    print("✅ Gambar plat terbaca")
    
    # 2. Preprocessing (sekarang otomatis menyimpan grid)
    simple_loading("Preprocessing gambar", 1)
    processed = preprocess_plate_image(img)  # Fungsi ini sekarang otomatis save grid

    # Save preprocessed image untuk OCR process
    preprocess_filename = f"proc_{uuid.uuid4().hex}_{base_name}"
    preprocess_path = os.path.join(preprocess_dir, preprocess_filename)
    cv2.imwrite(preprocess_path, processed)
    
    print("✅ Preprocessing selesai")
    
    # 3. OCR Detection
    simple_loading("Running OCR detection", 1.5)
    results = model_ocr(processed, conf=conf_threshold, verbose=False)

    # 4. Save detection result
    det_image = results[0].plot()
    det_filename = f"ocr_{uuid.uuid4().hex}_{base_name}"
    det_path = os.path.join(det_dir, det_filename)
    cv2.imwrite(det_path, det_image)
    
    print("✅ OCR detection selesai")
    
    # 5. Extract characters
    simple_loading("Extracting karakter", 1)
    chars = []
    names = model_ocr.names

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        char = names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        chars.append({
            "char": char,
            "x": x_center,
            "y": y_center
        })

    # Jika tidak ada karakter terdeteksi
    if not chars:
        print("❌ Tidak ada karakter terdeteksi")
        return ""

    # 6. Filter dan sort karakter
    simple_loading("Processing karakter", 0.5)
    
    # Kelompokkan berdasarkan baris (ambil baris atas saja)
    top_line_y = min(c["y"] for c in chars)
    Y_TOL = 25  # Toleransi variasi tinggi karakter

    # Filter karakter yang berada di baris yang sama
    filtered = [c for c in chars if abs(c["y"] - top_line_y) < Y_TOL]

    # Sort dari kiri ke kanan
    filtered = sorted(filtered, key=lambda c: c["x"])

    # Gabungkan menjadi string
    plate_string = "".join(c["char"] for c in filtered)
    
    print(f"✅ Plate terbaca: {plate_string}")
    return plate_string

# Alternatif version dengan threading loading (lebih smooth)
import threading

class OCRLoading:
    def __init__(self, message="OCR Processing"):
        self.message = message
        self.loading = True
        self.animation_chars = ["/", "-", "\\", "|"]
        self.current_char = 0
        self.thread = None
        
    def _animate(self):
        while self.loading:
            sys.stdout.write(f'\r{self.message} {self.animation_chars[self.current_char]}')
            sys.stdout.flush()
            self.current_char = (self.current_char + 1) % len(self.animation_chars)
            time.sleep(0.1)
    
    def start(self):
        self.loading = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, success_message=None):
        self.loading = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
        if success_message:
            print(f"✅ {success_message}")

def run_ocr_on_plate_smooth(crop_path: str,
                           model_ocr,
                           preprocess_dir: str,
                           det_dir: str,
                           conf_threshold: float = 0.5):
    """
    OCR process dengan smooth loading animation
    Returns: plate_string
    """
    print("\n🔤 OCR PROCESS")
    print("=" * 20)
    
    # Buat folder output
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    base_name = os.path.basename(crop_path)
    
    # 1. Load image
    loading = OCRLoading("Membaca gambar plat")
    loading.start()
    img = cv2.imread(crop_path)
    loading.stop("Gambar plat terbaca")

    if img is None:
        print("❌ Gagal membaca gambar plat")
        return ""

    # 2. Preprocessing (otomatis menyimpan grid)
    loading = OCRLoading("Preprocessing gambar")
    loading.start()
    processed = preprocess_plate_image(img)  # Otomatis save grid di dalam fungsi
    loading.stop("Preprocessing selesai")

    # Save preprocessed image untuk OCR process
    preprocess_filename = f"proc_{uuid.uuid4().hex}_{base_name}"
    preprocess_path = os.path.join(preprocess_dir, preprocess_filename)
    cv2.imwrite(preprocess_path, processed)
    
    # 3. OCR Detection
    loading = OCRLoading("Running OCR detection")
    loading.start()
    results = model_ocr(processed, conf=conf_threshold, verbose=False)
    loading.stop("OCR detection selesai")

    # 4. Save detection result
    det_image = results[0].plot()
    det_filename = f"ocr_{uuid.uuid4().hex}_{base_name}"
    det_path = os.path.join(det_dir, det_filename)
    cv2.imwrite(det_path, det_image)
    
    # 5. Extract characters
    loading = OCRLoading("Extracting karakter")
    loading.start()
    chars = []
    names = model_ocr.names

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        char = names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        chars.append({
            "char": char,
            "x": x_center,
            "y": y_center
        })

    # Jika tidak ada karakter terdeteksi
    if not chars:
        loading.stop("Tidak ada karakter terdeteksi")
        return ""

    # 6. Filter dan sort karakter
    top_line_y = min(c["y"] for c in chars)
    Y_TOL = 25

    filtered = [c for c in chars if abs(c["y"] - top_line_y) < Y_TOL]
    filtered = sorted(filtered, key=lambda c: c["x"])
    plate_string = "".join(c["char"] for c in filtered)
    
    loading.stop(f"Plate terbaca: {plate_string}")
    return plate_string