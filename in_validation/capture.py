import time
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.sensor import sensor_detect_vehicle_continuous
from utils.camera import capture_vehicle_image

IMG_IN_DIR = os.path.join(os.path.dirname(__file__), "img-in")
os.makedirs(IMG_IN_DIR, exist_ok=True)

def main():
    print("🚗 SENSOR CAPTURE SERVICE")
    print("=" * 50)
    print("Sensor memantau kendaraan -> capture -> simpan ke img-in")
    print("Worker akan memproses di file lain.")
    print("Tekan Ctrl+C untuk berhenti")
    print("=" * 50)

    event_id = 0

    try:
        while True:
            detected = sensor_detect_vehicle_continuous(port="COM10")

            if detected:
                event_id += 1
                print(f"\n🔔 Kendaraan terdeteksi (event: {event_id})")
                print("📸 Mengambil gambar...")

                saved_path = capture_vehicle_image(output_dir=IMG_IN_DIR)

                if saved_path:
                    print(f"📥 Gambar tersimpan: {saved_path}")
                else:
                    print("⚠️ Gagal capture gambar")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 SENSOR CAPTURE dihentikan")

if __name__ == "__main__":
    main()
