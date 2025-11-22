import cv2
import uuid
import os
import sys
import glob
import time
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
from utils.sensor import sensor_detect_vehicle_continuous  # existing sensor function
from utils.camera import capture_vehicle_image

# Folder output crop -> "img"
CROP_DIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(CROP_DIR, exist_ok=True)

# Folder input (images captured by sensor)
IMG_IN_DIR = os.path.join(os.path.dirname(__file__), "img-in")
os.makedirs(IMG_IN_DIR, exist_ok=True)

# Load YOLO model once
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'detection.pt')

# OCR model path
OCR_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'ocr.pt')


def run_detection(frame, yolo_model):
    """Deteksi plat & wajah pada satu frame. Kembalikan dict crops.
    crops = { "plate": [{path, confidence, uuid}, ...], "face": [...] }
    """
    loading = LoadingAnimation("Deteksi objek (plat & wajah)")
    loading.start()

    results = yolo_model(frame)
    detections = results[0]

    crops = {"plate": [], "face": []}

    for box in detections.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # safety crop bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

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


def process_image_file(img_path, ocr_model, yolo_model):
    """Proses satu file gambar (path). Mengembalikan True jika tersimpan di DB."""
    print(f"\n🖼️ Memproses file: {img_path}")

    frame = cv2.imread(img_path)
    if frame is None:
        print("❌ Gagal membaca gambar, menghapus file")
        try:
            os.remove(img_path)
        except Exception:
            pass
        return False

    # Deteksi
    crops = run_detection(frame, yolo_model)

    plate_text = "UNKNOWN"
    plate_confidence = 0.0
    plate_crop_path = ""
    face_encoding = None
    face_crop_path = ""

    # OCR
    if len(crops["plate"]) > 0:
        plate_crop_path = crops["plate"][0]["path"]
        plate_confidence = crops["plate"][0]["confidence"]

        loading = LoadingAnimation("OCR plat nomor")
        loading.start()

        plate_text = run_ocr_on_plate(
            crop_path=plate_crop_path,
            model_ocr=ocr_model,
            preprocess_dir=os.path.join(os.path.dirname(__file__), '..', 'optical_character_recognition', 'output', 'preprocess'),
            det_dir=os.path.join(os.path.dirname(__file__), '..', 'optical_character_recognition', 'output', 'detection')
        )

        loading.stop(f"✅ OCR: {plate_text}")

    # FACE
    if len(crops["face"]) > 0:
        face_crop_path = crops["face"][0]["path"]
        face_encoding = process_face_recognition(face_crop_path)

    # SAVE TO DB
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

        return True
    else:
        print("\n❌ Gagal memproses. Data tidak disimpan.")
        return False


def process_pending_images(ocr_model, yolo_model):
    """Baca semua file di IMG_IN_DIR dan proses satu-satu."""
    files = sorted(glob.glob(os.path.join(IMG_IN_DIR, "*.jpg")))
    if len(files) == 0:
        return 0

    processed = 0
    for img_path in files:
        try:
            ok = process_image_file(img_path, ocr_model, yolo_model)
        except Exception as e:
            print(f"❌ Error saat memproses {img_path}: {e}")
            ok = False

        # Hapus file input apapun hasilnya
        try:
            os.remove(img_path)
        except Exception:
            pass

        if ok:
            processed += 1

    return processed

def main():
    print("🚗 IN VALIDATION SERVICE (queue-based)")
    print("=" * 50)
    print("Sistem akan terus mendeteksi kendaraan -> capture -> simpan ke img-in -> worker memproses")
    print("Tekan Ctrl+C untuk berhenti")
    print("=" * 50)

    # Inisialisasi database
    create_table_if_not_exists()

    # Load models ONCE
    print("🔁 Memuat model YOLO dan OCR (sekali saja)...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_model = load_ocr_model(OCR_MODEL_PATH)

    vehicle_count = 0

    try:
        while True:

            # 2) Worker: proses semua file yang ada di folder img-in
            processed = process_pending_images(ocr_model, yolo_model)
            if processed > 0:
                print(f"✅ Selesai memproses {processed} file dari '{IMG_IN_DIR}'")

            # 3) Sleep singkat agar loop tidak 100% CPU
            time.sleep(0.25)

    except KeyboardInterrupt:
        print('\n\n🛑 Dihentikan oleh user (Ctrl+C)')

    print("\n" + "=" * 50)
    print("🔚 IN VALIDATION SERVICE STOPPED")
    print(f"Total event sensor terdeteksi: {vehicle_count}")
    print("Terima kasih telah menggunakan sistem ini!")


if __name__ == "__main__":
    main()