import cv2
import uuid
import os
import sys
import numpy as np
from ultralytics import YOLO
import time
import serial
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


TRIGGER_DIR = os.path.join(os.path.dirname(__file__), "triggers")
os.makedirs(TRIGGER_DIR, exist_ok=True)
# === PATH HANDLING ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.setup import setup_environment
setup_environment()

from optical_character_recognition.main import load_ocr_model, run_ocr_on_plate_smooth
from face_recog.main import process_face_recognition
from utils.database import get_active_entry_by_plate, mark_entry_exited

# === CONFIG ===
CROP_DIR = os.path.join(os.path.dirname(__file__), "img-live")
os.makedirs(CROP_DIR, exist_ok=True)

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

serial_conn = None


# ================================ #
#       SERIAL COMMUNICATION       #
# ================================ #

def setup_serial(port=SERIAL_PORT):
    global serial_conn
    try:
        serial_conn = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"✅ Serial ready di {port}")
        return True
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return False


def send_serial(cmd):
    try:
        serial_conn.write((cmd + "\n").encode())
    except:
        pass



# ================================ #
#         YOLO DETECTION           #
# ================================ #

model_det = YOLO('../model/detection.pt')

def detect_objects(frame):
    results = model_det(frame)[0]

    crops = {"plate": [], "face": []}

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = "plate" if cls == 0 else "face" if cls == 1 else None
        if not label:
            continue

        crop = frame[y1:y2, x1:x2]
        crop_id = str(uuid.uuid4())
        path = os.path.join(CROP_DIR, f"{label}_{crop_id}.jpg")
        cv2.imwrite(path, crop)

        crops[label].append({"path": path})

    return crops


def compare_encoding(a, b, threshold=0.5):
    sim = cosine_similarity([a], [b])[0][0]
    return sim >= threshold, sim


# ================================ #
#        MAIN VALIDATION           #
# ================================ #

def process_vehicle(frame, ocr_model):

    print("\n📸 Running detection...")

    crops = detect_objects(frame)

    # -------- OCR --------
    plate_text = "UNKNOWN"
    if crops["plate"]:
        plate_text = run_ocr_on_plate_smooth(
            crops["plate"][0]["path"],
            ocr_model,
            "../optical_character_recognition/output/preprocess",
            "../optical_character_recognition/output/detection"
        )

    # -------- FACE RECOG --------
    face_enc = None
    if crops["face"]:
        face_enc = process_face_recognition(crops["face"][0]["path"])

    if plate_text == "UNKNOWN" or face_enc is None:
        print("❌ Tidak ada wajah / plat")
        send_serial("buzz")
        return False

    # -------- DB CHECK --------
    db = get_active_entry_by_plate(plate_text)
    if not db:
        print("❌ Plat tidak terdaftar / sudah keluar")
        send_serial("buzz")
        return False

    # -------- FACE MATCH --------
    match, sim = compare_encoding(face_enc, db['face_vector'])
    if not match:
        print(f"❌ Wajah tidak cocok ({sim:.2f})")
        send_serial("buzz")
        return False

    # -------- SUCCESS --------
    print("✅ Validasi berhasil")
    mark_entry_exited(db['id'])

    send_serial("silent")
    send_serial("o")  # open gate
    time.sleep(2)
    return True


# ================================ #
#      MANUAL TRIGGER CHECK        #
# ================================ #
# (Bagian Baru Ditambahkan Di Sini)

def check_manual_trigger():
    """
    Cek apakah API Server mengirim request lewat file text.
    Mendukung: Buka Gate (trigger_open.txt) & Matikan Buzzer (trigger_mute.txt)
    """
    
    # --- A. LOGIKA BUKA GATE ---
    gate_files = [
        "trigger_open.txt", 
        "../out_validation/trigger_open.txt", 
        "out_validation/trigger_open.txt"
    ]
    
    for file_path in gate_files:
        if os.path.exists(file_path):
            print(f"\n⚡ [INTERRUPT] PERINTAH APP: BUKA GATE")
            
            # 1. Kirim perintah serial
            send_serial("silent") # Matikan buzzer dulu
            send_serial("o")      # Buka Gate
            
            # 2. Hapus file trigger
            try:
                os.remove(file_path)
                print("🧹 File trigger gate dihapus.")
            except Exception as e:
                print(f"Error hapus file: {e}")
            return True 

    # --- B. LOGIKA MATIKAN BUZZER ---
    buzzer_files = [
        "trigger_mute.txt", 
        "../out_validation/trigger_mute.txt", 
        "out_validation/trigger_mute.txt"
    ]
    
    for file_path in buzzer_files:
        if os.path.exists(file_path):
            print(f"\n🔕 [INTERRUPT] PERINTAH APP: MATIKAN BUZZER")
            
            # 1. Kirim perintah serial
            send_serial("silent") # Matikan buzzer saja
            
            # 2. Hapus file trigger
            try:
                os.remove(file_path)
                print("🧹 File trigger buzzer dihapus.")
            except Exception as e:
                print(f"Error hapus file: {e}")
            return True

    return False


# ================================ #
#          LIVE CAMERA             #
# ================================ #

def main():
    print("🚗 OUT VALIDATION LIVE SERVICE")
    print("=" * 60)

    # --- CAMERA ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka")
        return
    print("📷 Kamera aktif")

    # --- SERIAL ---
    if not setup_serial(SERIAL_PORT):
        return

    # --- OCR MODEL ---
    ocr_model = load_ocr_model("../model/ocr.pt")
    print("✅ OCR Model loaded")

    buffer = ""

    while True:
        # -------- CEK TRIGGER DARI APLIKASI (BARU) --------
        check_manual_trigger()

        # -------- LIVE VIDEO --------
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("LIVE CAMERA", frame)

        # -------- LISTEN SERIAL --------
        if serial_conn.in_waiting:
            data = serial_conn.read().decode(errors="ignore")

            if data in ["\n", "\r"]:
                line = buffer.strip()
                buffer = ""

                if line:
                    print(f"[SERIAL] {line}")

                    if line == "VEHICLE_DETECTED":

                        print("🚗 Sensor: VEHICLE DETECTED")
                        print("📸 Capturing fresh frame...")

                        # ambil 3 frame terbaru agar clear
                        fresh_frame = None
                        for _ in range(3):
                            ret, fresh_frame = cap.read()
                            time.sleep(0.05)

                        # jalankan proses validasi
                        process_vehicle(fresh_frame, ocr_model)

                        time.sleep(1)

            else:
                buffer += data

        # -------- EXIT KEY --------
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    serial_conn.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()