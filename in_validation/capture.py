import cv2
import serial
import time
import os
from datetime import datetime

# ========== CONFIG ==========

SERIAL_PORT = "COM9"
BAUD_RATE = 115200

IMG_IN_DIR = os.path.join(os.path.dirname(__file__), "img-in")
os.makedirs(IMG_IN_DIR, exist_ok=True)

# ============================

def open_serial():
    print(f"🔌 Opening serial port {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)  # wait ESP8266 reset
    print("✅ Serial connected")
    return ser

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def main():
    print("🚗 SENSOR + CAMERA LIVE SERVICE")
    print("=" * 60)
    
    # ---------- OPEN CAMERA ----------
    print("📷 Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera gagal dibuka")
        return

    print("✅ Camera OK")
    
    # ---------- OPEN SERIAL ----------
    ser = open_serial()

    event_id = 0
    buffer = ""

    print("🎥 Kamera hidup. Menunggu VEHICLE DETECTED...\n")

    while True:
        # ======== CAMERA LOOP ========
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame gagal dibaca")
            continue

        # Tampilkan live camera
        cv2.imshow("LIVE CAMERA", frame)
        
        # ======== SENSOR SERIAL LISTEN ========
        if ser.in_waiting:
            data = ser.read().decode(errors="ignore")

            if data == "\n" or data == "\r":
                line = buffer.strip()
                buffer = ""

                if line:
                    print(f"[SERIAL] {line}")

                    # ==== jika kendaraan terdeteksi ====
                    if line == "VEHICLE DETECTED":
                        event_id += 1
                        fname = f"vehicle_{timestamp()}.jpg"
                        fpath = os.path.join(IMG_IN_DIR, fname)

                        cv2.imwrite(fpath, frame)
                        print(f"📸 Captured → {fpath}")

            else:
                buffer += data

        # ======== EXIT WITH Q ========
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n🛑 EXIT")
    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
