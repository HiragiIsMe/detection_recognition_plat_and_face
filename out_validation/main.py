# out_validation/main.py
import cv2
import uuid
import os
import sys
import numpy as np
from ultralytics import YOLO
import time
import serial

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
from utils.sensor import sensor_detect_vehicle_continuous
from sklearn.metrics.pairwise import cosine_similarity

CROP_DIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(CROP_DIR, exist_ok=True)

# Serial connection untuk SEMUA: sensor, servo & buzzer
serial_conn = None

def setup_serial(port='COM3'):
    """Setup koneksi serial untuk SEMUA: sensor, servo & buzzer"""
    global serial_conn
    try:
        # Close existing connection if any
        if serial_conn and serial_conn.is_open:
            serial_conn.close()
            time.sleep(1)
        
        serial_conn = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1,
            write_timeout=1
        )
        time.sleep(2)  # Tunggu Arduino reset
        print(f"✅ Serial terkoneksi di {port} untuk SENSOR, SERVO & BUZZER")
        
        # Clear buffer
        serial_conn.reset_input_buffer()
        return True
    except Exception as e:
        print(f"❌ Gagal koneksi serial: {e}")
        return False

def send_serial_command(command):
    """Kirim command ke Arduino"""
    global serial_conn
    if serial_conn and serial_conn.is_open:
        try:
            serial_conn.write(f"{command}\n".encode())
            print(f"📡 Command dikirim: {command}")
            time.sleep(0.1)  # Beri waktu untuk Arduino memproses
            return True
        except Exception as e:
            print(f"❌ Gagal kirim command: {e}")
    return False

def sensor_detect_vehicle_continuous_out():
    """
    Fungsi sensor khusus untuk out_validation menggunakan koneksi serial yang sama
    TANPA menampilkan data sensor real-time di console
    """
    global serial_conn
    
    loading = LoadingAnimation("Mendeteksi kendaraan")
    loading.start()
    
    try:
        # Pastikan koneksi serial aktif
        if not serial_conn or not serial_conn.is_open:
            loading.stop("❌ Koneksi serial tidak aktif")
            return False
        
        last_data_time = time.time()
        data_count = 0
        
        while True:
            # Baca data serial
            if serial_conn.in_waiting > 0:
                try:
                    raw_data = serial_conn.readline()
                    
                    # Coba decode
                    try:
                        text = raw_data.decode('utf-8').strip()
                    except:
                        try:
                            text = raw_data.decode('latin-1').strip()
                        except:
                            continue
                    
                    # HITUNG DATA COUNTER (untuk debugging internal saja)
                    data_count += 1
                    last_data_time = time.time()
                    
                    # Cek jika kendaraan terdeteksi
                    if "VEHICLE_DETECTED" in text or "1" in text:
                        loading.stop("✅ Kendaraan terdeteksi!")
                        return True
                            
                except Exception as e:
                    # Skip error tanpa print ke console
                    continue
            
            # Optional: Timeout jika tidak ada data sama sekali dalam 30 detik
            if time.time() - last_data_time > 30 and data_count == 0:
                loading.stop("❌ Timeout: Tidak ada data dari sensor")
                return False
            
            time.sleep(0.1)
        
    except Exception as e:
        loading.stop(f"❌ Error deteksi kendaraan: {e}")
        return False

def open_servo_gate():
    """Buka servo gate 90°"""
    print("🚪 Membuka gate...")
    if send_serial_command("o"):
        time.sleep(3)  # Tunggu 3 detik gate terbuka
        # Tutup gate setelah delay
        close_servo_gate()
        return True
    return False

def close_servo_gate():
    """Tutup servo gate"""
    print("🚪 Menutup gate...")
    return send_serial_command("c")

def trigger_alarm():
    """Nyalakan alarm selama 5 detik"""
    print("🚨 Menyalakan alarm...")
    if send_serial_command("buzz"):
        time.sleep(5)  # Alarm menyala 5 detik
        stop_alarm()
        return True
    return False

def stop_alarm():
    """Matikan alarm"""
    print("🔇 Mematikan alarm...")
    return send_serial_command("silent")

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

def process_vehicle(ocr_model):
    """
    Proses validasi satu kendaraan keluar
    """
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
            return False
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
                
                # BUKA GATE (otomatis tutup setelah delay)
                open_servo_gate()
                return True
                
            else:
                print("\n❌ Wajah tidak cocok dengan pemilik plat!")
                trigger_alarm()
                return False
                
    else:
        print("❌ Data tidak lengkap (wajah atau plat tidak terdeteksi)")
        trigger_alarm()
        return False

def main():
    print("🚗 OUT VALIDATION SERVICE")
    print("=" * 50)
    print("Mode: CONTINUOUS DETECTION")
    print("Sistem akan terus mendeteksi kendaraan keluar...")
    print("Tekan Ctrl+C untuk berhenti")
    print("=" * 50)

    # Setup serial UNTUK SEMUA: sensor, servo & buzzer
    if not setup_serial(port='COM3'):
        print("❌ Gagal koneksi serial, sistem tidak dapat berjalan")
        return

    # Load model OCR sekali saja
    ocr_model = load_ocr_model("../model/ocr.pt")

    vehicle_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    try:
        while True:
            vehicle_count += 1
            
            print(f"\n🔄 PROSES KENDARAAN KE-{vehicle_count} (KELUAR)")
            print("-" * 40)
            
            try:
                # 1. DETEKSI KENDARAAN menggunakan koneksi serial yang sama
                vehicle_detected = sensor_detect_vehicle_continuous_out()
                
                if vehicle_detected:
                    # 2. PROSES validasi kendaraan keluar
                    success = process_vehicle(ocr_model)
                    
                    if success:
                        print(f"✅ Kendaraan #{vehicle_count} berhasil validasi KELUAR")
                        consecutive_errors = 0
                    else:
                        print(f"❌ Kendaraan #{vehicle_count} gagal validasi")
                        consecutive_errors += 1
                    
                    # 3. Delay sebelum kendaraan berikutnya
                    print("\n⏳ Menunggu 5 detik untuk kendaraan berikutnya...")
                    time.sleep(5)
                    
                else:
                    print("⏭️  Tidak ada kendaraan terdeteksi, lanjut monitoring...")
                    consecutive_errors += 1
                    
            except Exception as e:
                print(f"🚨 Error proses kendaraan: {e}")
                consecutive_errors += 1
                time.sleep(2)
            
            # Safety mechanism
            if consecutive_errors >= max_consecutive_errors:
                print(f"🔄 Terlalu banyak error ({consecutive_errors}), restarting serial...")
                setup_serial(port='COM3')
                consecutive_errors = 0
                time.sleep(2)
                    
    except KeyboardInterrupt:
        print('\n\n🛑 Dihentikan oleh user (Ctrl+C)')
    
    # Cleanup
    if serial_conn and serial_conn.is_open:
        try:
            serial_conn.close()
            print("🔌 Serial connection closed")
        except:
            pass
    
    print("\n" + "=" * 50)
    print("🔚 OUT VALIDATION SERVICE STOPPED")
    print(f"Total kendaraan divalidasi: {vehicle_count - 1}")
    print("Terima kasih telah menggunakan sistem ini!")

if __name__ == "__main__":
    main()