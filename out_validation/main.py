# out_validation/main.py
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
from optical_character_recognition.main import load_ocr_model, run_ocr_on_plate_smooth
from face_recog.main import process_face_recognition
from utils.database import get_active_entry_by_plate, mark_entry_exited
from utils.loading import LoadingAnimation
from sklearn.metrics.pairwise import cosine_similarity

CROP_DIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(CROP_DIR, exist_ok=True)

def sensor_detect_vehicle():
    return True

def open_servo_gate():
    print("🚪 Gate terbuka!")
    return True

def trigger_alarm():
    print("🚨 Alarm berbunyi!")
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
        })

    loading.stop(f"✅ {len(crops['plate'])} plat & {len(crops['face'])} wajah terdeteksi")
    return crops

def compare_face_encodings(encoding1, encoding2, threshold=0.6):
    loading = LoadingAnimation("Membandingkan wajah")
    loading.start()
    
    try:
        similarity = cosine_similarity([encoding1], [encoding2])[0][0]
        
        if similarity >= threshold:
            loading.stop(f"✅ Wajah cocok ({similarity:.2%})")
            return True
        else:
            loading.stop(f"❌ Wajah tidak cocok ({similarity:.2%})")
            return False
            
    except Exception as e:
        loading.stop(f"❌ Error: {e}")
        return False

def main():
    print("🚗 OUT VALIDATION SERVICE")
    print("=" * 40)

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
        face_encoding = None
        
        # PROSES OCR
        if len(crops["plate"]) > 0:
            plate_crop_path = crops["plate"][0]["path"]
            plate_confidence = crops["plate"][0]["confidence"]
            
            # Gunakan OCR dengan smooth loading
            plate_text = run_ocr_on_plate_smooth(
                crop_path=plate_crop_path,
                model_ocr=ocr_model,
                preprocess_dir="../optical_character_recognition/output/preprocess",
                det_dir="../optical_character_recognition/output/detection"
            )

        # PROSES FACE RECOGNITION
        if len(crops["face"]) > 0:
            face_crop_path = crops["face"][0]["path"]
            face_encoding = process_face_recognition(face_crop_path)

        # VALIDATION LOGIC
        print("\n🔍 VALIDASI KELUAR")
        print("-" * 20)
        
        if face_encoding is not None and plate_text != "UNKNOWN":
            # QUERY HANYA DATA YANG MASIH ACTIVE
            db_entry = get_active_entry_by_plate(plate_text)
            
            if db_entry is None:
                print("❌ Plat tidak terdaftar atau sudah keluar!")
                trigger_alarm()
            else:
                # COMPARE FACE ENCODINGS
                is_face_match = compare_face_encodings(face_encoding, db_entry['face_vector'])
                
                if is_face_match:
                    print("\n🎉 VALIDASI BERHASIL!")
                    print(f"   📋 Plat: {plate_text}")
                    print(f"   👤 Wajah: Cocok")
                    print(f"   🆔 Entry ID: {db_entry['id'][:8]}...")
                    print(f"   ⏰ Masuk: {db_entry['entry_time']}")
                    
                    # UPDATE STATUS KE EXITED
                    loading = LoadingAnimation("Update status keluar")
                    loading.start()
                    mark_entry_exited(db_entry['id'])
                    loading.stop("Status updated ke exited")
                    
                    # BUKA GATE
                    open_servo_gate()
                    
                else:
                    print("\n❌ Wajah tidak cocok dengan pemilik plat!")
                    trigger_alarm()
                    
        else:
            print("❌ Data tidak lengkap (wajah atau plat tidak terdeteksi)")
            trigger_alarm()

        print("\n" + "=" * 40)
        print("OUT VALIDATION SELESAI")

if __name__ == "__main__":
    main()