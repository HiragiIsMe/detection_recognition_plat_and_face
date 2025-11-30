# face_recog/main.py
from deepface import DeepFace
import cv2
import numpy as np
import uuid
import os
from utils.loading import LoadingAnimation

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess_face_manual(face_image):
    """PREPROCESSING MANUAL untuk memenuhi syarat PCV dengan grid semua proses"""
    loading = LoadingAnimation("Preprocessing wajah")
    loading.start()
    
    # Simpan gambar asli untuk grid
    original_image = face_image.copy()
    
    # 1. Convert to grayscale (manual)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Manual Histogram Equalization
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()
    cdf = hist_norm.cumsum()
    equalized = np.interp(gray.flatten(), range(256), 255 * cdf)
    equalized = equalized.reshape(gray.shape).astype(np.uint8)
    
    # 3. Gaussian Blur manual
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    blurred = cv2.filter2D(equalized, -1, kernel)
    
    # 4. Convert back to BGR
    result = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    # BUAT GRID DENGAN SEMUA PROSES
    grid_image = create_preprocessing_grid(original_image, gray, equalized, blurred, result)
    
    # SIMPAN GRID
    root_dir = get_project_root()
    grid_dir = os.path.join(root_dir, "face_recog", "preprocessing_grids")
    os.makedirs(grid_dir, exist_ok=True)
    
    # Extract UUID dari filename asli atau generate baru
    face_uuid = str(uuid.uuid4())
    
    # Save grid image
    grid_filename = f"preproc_grid_{face_uuid}.jpg"
    grid_path = os.path.join(grid_dir, grid_filename)
    cv2.imwrite(grid_path, grid_image)
    
    loading.stop("Preprocessing wajah selesai")
    return result

def create_preprocessing_grid(original, gray, equalized, blurred, final):
    """Membuat grid gambar dengan semua tahap preprocessing"""
    # Resize semua gambar ke ukuran yang sama (200x200 untuk konsistensi)
    size = (200, 200)
    
    original_resized = cv2.resize(original, size)
    gray_resized = cv2.resize(gray, size)
    equalized_resized = cv2.resize(equalized, size)
    blurred_resized = cv2.resize(blurred, size)
    final_resized = cv2.resize(final, size)
    
    # Convert grayscale images to BGR untuk konsistensi tampilan
    gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    equalized_bgr = cv2.cvtColor(equalized_resized, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.cvtColor(blurred_resized, cv2.COLOR_GRAY2BGR)
    
    # Tambahkan label/text pada setiap gambar
    def add_label(img, text):
        labeled_img = img.copy()
        # Tambahkan background untuk text
        cv2.rectangle(labeled_img, (0, 0), (200, 30), (0, 0, 0), -1)
        # Tambahkan text
        cv2.putText(labeled_img, text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return labeled_img
    
    # Label setiap gambar
    original_labeled = add_label(original_resized, "1. Original")
    gray_labeled = add_label(gray_bgr, "2. Grayscale")
    equalized_labeled = add_label(equalized_bgr, "3. Hist Equalized")
    blurred_labeled = add_label(blurred_bgr, "4. Gaussian Blur")
    final_labeled = add_label(final_resized, "5. Final Result")
    
    # Buat grid 2x3 (2 baris, 3 kolom)
    # Baris pertama: Original, Grayscale, Hist Equalized
    row1 = np.hstack([original_labeled, gray_labeled, equalized_labeled])
    # Baris kedua: Gaussian Blur, Final Result, Kosong (atau bisa diisi lainnya)
    row2 = np.hstack([blurred_labeled, final_labeled, np.zeros_like(original_labeled)])
    
    # Gabungkan baris
    grid = np.vstack([row1, row2])
    
    return grid

def generate_face_encoding(face_image_path):
    """Generate face encoding dengan preprocessing manual"""
    try:
        loading = LoadingAnimation("Membaca gambar wajah")
        loading.start()
        
        # Baca gambar wajah asli
        original_face = cv2.imread(face_image_path)
        if original_face is None:
            loading.stop("❌ Gagal membaca gambar wajah")
            return None
        
        loading.stop("✅ Gambar wajah terbaca")
        
        # PREPROCESSING MANUAL (sekarang otomatis menyimpan grid)
        preprocessed_face = preprocess_face_manual(original_face)
        
        # SIMPAN DI FACE_RECOG/IMG (tetap seperti semula)
        root_dir = get_project_root()
        preprocessed_dir = os.path.join(root_dir, "face_recog", "img")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # Extract UUID dari filename asli
        original_filename = os.path.basename(face_image_path)
        face_uuid = original_filename.replace("face_", "").replace(".jpg", "")
        
        # Generate filename preprocessing
        preprocessed_filename = f"preproc_face_{face_uuid}.jpg"
        preprocessed_path = os.path.join(preprocessed_dir, preprocessed_filename)
        
        cv2.imwrite(preprocessed_path, preprocessed_face)
        
        # Generate encoding
        loading = LoadingAnimation("Generating face encoding")
        loading.start()
        
        embedding_objs = DeepFace.represent(
            img_path=preprocessed_path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False
        )
        
        if embedding_objs:
            face_encoding = embedding_objs[0]["embedding"]
            loading.stop(f"✅ Face encoding berhasil ({len(face_encoding)} dimensi)")
            return face_encoding
        else:
            loading.stop("❌ Tidak ada encoding yang dihasilkan")
            return None
            
    except Exception as e:
        print(f"\r❌ Error: {e}")
        return None

def process_face_recognition(face_crop_path):
    """Pure face recognition process"""
    print("\n🎭 FACE RECOGNITION")
    print("=" * 30)
    
    face_encoding = generate_face_encoding(face_crop_path)
    
    if face_encoding is not None:
        return face_encoding
    else:
        return None