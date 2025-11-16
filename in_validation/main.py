# in_validation/main.py
import cv2
import uuid
import os
import sys
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import setup untuk suppress warnings
from utils.setup import setup_environment
setup_environment()

# === IMPORT MODULES ===
from optical_character_recognition.main import load_ocr_model, run_ocr_on_plate
from face_recog.main import process_face_recognition
from utils.database import insert_entry, create_table_if_not_exists
from utils.loading import LoadingAnimation

# Folder output crop -> "img"
CROP_DIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(CROP_DIR, exist_ok=True)

def sensor_detect_vehicle():
    return True

def run_detection(frame):
    loading = LoadingAnimation("Deteksi objek (plat & wajah)")
    loading.start()
    
    model = YOLO('../model/detection.pt')
    results = model(frame)
    detections = results[0]

    crops = {"plate": [], "face": []}

    for box in detections.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            label = "plate"
        elif cls == 1:
            label = "face"
        else:
            continue

        crop_img = frame[y1:y2, x1:x2]
        crop_id = str(uuid.uuid4())
        crop_path = os.path.join(CROP_DIR, f"{label}_{crop_id}.jpg")
        cv2.imwrite(crop_path, crop_img)

        crops[label].append({
            "path": crop_path,
            "confidence": conf,
            "uuid": crop_id
        })

    loading.stop(f"✅ {len(crops['plate'])} plat & {len(crops['face'])} wajah terdeteksi")
    return crops

def main():
    print("🚗 IN VALIDATION SERVICE")
    print("=" * 40)

    # Inisialisasi database
    create_table_if_not_exists()

    # Load model OCR
    ocr_model = load_ocr_model("../model/ocr.pt")

    # Sensor mendeteksi kendaraan
    if sensor_detect_vehicle():
        print("📷 Mengambil gambar kendaraan...")
        frame = cv2.imread('../test.jpg')

        # Deteksi objek
        crops = run_detection(frame)

        # Variabel untuk menyimpan hasil
        plate_text = "UNKNOWN"
        plate_confidence = 0.0
        plate_crop_path = ""
        face_encoding = None
        face_crop_path = ""
        
        # PROSES OCR
        if len(crops["plate"]) > 0:
            plate_crop_path = crops["plate"][0]["path"]
            plate_confidence = crops["plate"][0]["confidence"]
            
            loading = LoadingAnimation("OCR plat nomor")
            loading.start()
            
            plate_text = run_ocr_on_plate(
                crop_path=plate_crop_path,
                model_ocr=ocr_model,
                preprocess_dir="../optical_character_recognition/output/preprocess",
                det_dir="../optical_character_recognition/output/detection"
            )
            
            loading.stop(f"✅ OCR: {plate_text}")

        # PROSES FACE RECOGNITION
        if len(crops["face"]) > 0:
            face_crop_path = crops["face"][0]["path"]
            face_encoding = process_face_recognition(face_crop_path)

        # SIMPAN KE DATABASE
        if face_encoding is not None and plate_text != "UNKNOWN":
            loading = LoadingAnimation("Menyimpan ke database")
            loading.start()
            
            db_entry_id = insert_entry(
                plate_text=plate_text,
                plate_conf=plate_confidence,
                face_vector=face_encoding,
                plate_path=plate_crop_path,
                face_path=face_crop_path
            )
            
            loading.stop(f"✅ Data tersimpan (ID: {db_entry_id[:8]}...)")
            
            print(f"\n🎉 PROSES MASUK BERHASIL!")
            print(f"   📋 Plat: {plate_text}")
            print(f"   👤 Wajah: Terdaftar")
            print(f"   💾 Database: Tersimpan")
            
        else:
            print("\n❌ Gagal memproses. Data tidak disimpan.")

        print("\n" + "=" * 40)
        print("IN VALIDATION SELESAI")

if __name__ == "__main__":
    main()