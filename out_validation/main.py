import cv2
import uuid
import os
import sys
import numpy as np
from ultralytics import YOLO
import time
import serial

# === PATH HANDLING ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.setup import setup_environment
setup_environment()

# === IMPORTS ===
from optical_character_recognition.main import load_ocr_model, run_ocr_on_plate_smooth
from face_recog.main import process_face_recognition
from utils.database import get_active_entry_by_plate, mark_entry_exited
from utils.loading import LoadingAnimation
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
CROP_DIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(CROP_DIR, exist_ok=True)

serial_conn = None


# ================================ #
#   SERIAL COMMUNICATION (CLEAN)   #
# ================================ #

def setup_serial(port='COM10'):
    """Setup serial untuk sensor, servo, dan buzzer dalam satu jalur"""
    global serial_conn
    try:
        if serial_conn and serial_conn.is_open:
            serial_conn.close()
            time.sleep(1)

        serial_conn = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1
        )
        time.sleep(2)
        print(f"✅ Serial ready di {port}")
        serial_conn.reset_input_buffer()
        return True
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return False


def send_serial(cmd):
    """Kirim perintah ke Arduino"""
    global serial_conn
    if serial_conn and serial_conn.is_open:
        try:
            serial_conn.write((cmd + "\n").encode())
            time.sleep(0.1)
            return True
        except:
            return False
    return False


# ================================ #
#        SENSOR DETECTION          #
# ================================ #

def wait_vehicle_detected():
    """
    Tunggu sensor mengirim 'VEHICLE_DETECTED' atau '1'
    """
    global serial_conn
    loading = LoadingAnimation("Menunggu kendaraan")
    loading.start()

    try:
        last_data_time = time.time()

        while True:
            if serial_conn.in_waiting > 0:
                raw = serial_conn.readline()

                try:
                    text = raw.decode().strip()
                except:
                    continue

                if "VEHICLE_DETECTED" in text or text == "1":
                    loading.stop("🚗 Kendaraan terdeteksi")
                    return True

                last_data_time = time.time()

            # Timeout 20 detik tidak ada data sama sekali
            if time.time() - last_data_time > 20:
                loading.stop("❌ Tidak ada data sensor")
                return False

            time.sleep(0.1)

    except Exception as e:
        loading.stop(f"❌ Sensor error: {e}")
        return False


# ================================ #
#        HARDWARE ACTIONS          #
# ================================ #

def open_gate():
    print("🔓 Membuka gate...")
    send_serial("o")
    time.sleep(3)


def alarm_on():
    print("🚨 ALARM ON")
    send_serial("buzz")


def alarm_off():
    print("🔇 ALARM OFF")
    send_serial("silent")


# ================================ #
#   CHECK MANUAL OPEN FROM API     #
# ================================ #

def wait_manual_decision():
    """
    Jika validasi gagal → tunggu keputusan admin dari aplikasi HP.
    Aplikasi akan memanggil /api/open-gate → API menulis trigger_open.txt
    """
    print("\n📲 Menunggu keputusan admin...")

    while True:
        if os.path.exists("trigger_open.txt"):
            print("⚡ Perintah manual diterima: OPEN GATE")

            alarm_off()
            open_gate()

            try:
                os.remove("trigger_open.txt")
            except:
                pass

            print("➡️ Lanjut ke kendaraan berikutnya")
            return True

        time.sleep(1)


# ================================ #
#        DETECTION FUNCTIONS        #
# ================================ #

def run_detection(frame):
    model = YOLO('../model/detection.pt')
    results = model(frame)
    detections = results[0]

    crops = {"plate": [], "face": []}

    for box in detections.boxes:
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


def compare_encoding(a, b, threshold=0.6):
    similarity = cosine_similarity([a], [b])[0][0]
    return similarity >= threshold, similarity


# ================================ #
#         MAIN PROCESSING           #
# ================================ #

def capture_frame():
    """
    Capture 1 frame dari kamera laptop.
    Kamera dibuka → ambil gambar → tutup.
    """
    cap = cv2.VideoCapture(0)   # 0 = kamera laptop
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka")
        return None

    # Ambil 1 frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Gagal capture frame")
        return None

    return frame

def process_vehicle(ocr_model):
    print("📷 Capture kendaraan...")
    frame = capture_frame()

    crops = run_detection(frame)
    plate_text = "UNKNOWN"
    face_enc = None

    # OCR
    if crops["plate"]:
        plate_text = run_ocr_on_plate_smooth(
            crops["plate"][0]["path"],
            ocr_model,
            "../optical_character_recognition/output/preprocess",
            "../optical_character_recognition/output/detection"
        )

    # FACE
    if crops["face"]:
        face_enc = process_face_recognition(crops["face"][0]["path"])

    # VALIDATION
    if plate_text == "UNKNOWN" or face_enc is None:
        print("❌ Tidak ada wajah / plat")
        alarm_on()
        return False

    db = get_active_entry_by_plate(plate_text)
    if not db:
        print("❌ Plat tidak terdaftar/ sudah keluar")
        alarm_on()
        return False

    match, sim = compare_encoding(face_enc, db['face_vector'])

    if not match:
        print(f"❌ Wajah tidak cocok ({sim:.2f})")
        alarm_on()
        return False

    # SUCCESS
    print("✅ Validasi berhasil")
    mark_entry_exited(db['id'])

    alarm_off()
    open_gate()
    return True


# ================================ #
#               MAIN               #
# ================================ #

def main():
    print("🚗 OUT VALIDATION SERVICE\n")

    if not setup_serial('COM10'):
        return

    ocr_model = load_ocr_model("../model/ocr.pt")

    while True:
        print("\n🔄 Menunggu kendaraan...")
        detected = wait_vehicle_detected()

        if not detected:
            continue

        success = process_vehicle(ocr_model)

        if not success:
            print("\n❌ VALIDASI GAGAL → Menunggu keputusan admin.")
            wait_manual_decision()

        time.sleep(3)


if __name__ == "__main__":
    main()
