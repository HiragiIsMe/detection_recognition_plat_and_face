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
    """PREPROCESSING MANUAL untuk memenuhi syarat PCV"""
    loading = LoadingAnimation("Preprocessing wajah")
    loading.start()
    
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
    
    loading.stop("Preprocessing wajah selesai")
    return result

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
        
        # PREPROCESSING MANUAL
        preprocessed_face = preprocess_face_manual(original_face)
        
        # SIMPAN DI FACE_RECOG/IMG
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