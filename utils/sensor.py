# utils/sensor_simple.py
import serial
import time

# Global serial instance + last detect time
ser_instance = None
_last_detect_time = 0  


def init_sensor(port='COM3'):
    """Init serial hanya sekali di awal."""
    global ser_instance

    if ser_instance is None:
        try:
            ser_instance = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=0.05  # cepat dan non-blocking
            )
            time.sleep(2)  # tunggu arduino reset
            ser_instance.reset_input_buffer()
            print(f"✅ Sensor pada {port} tersambung")
        except Exception as e:
            print(f"❌ Gagal membuka sensor: {e}")
            ser_instance = None


def sensor_detect_vehicle_continuous(port='COM3', debounce_ms=1500):
    """Non-blocking sensor check. Dipanggil berkali-kali dalam loop utama."""
    global ser_instance, _last_detect_time

    # init serial jika belum
    if ser_instance is None:
        init_sensor(port)

    if ser_instance is None:
        return False  # sensor tidak ada

    # baca 1 line saja (non-blocking)
    try:
        if ser_instance.in_waiting > 0:
            raw = ser_instance.readline()
            try:
                text = raw.decode('utf-8').strip()
            except:
                text = raw.decode('latin-1').strip()

            if "VEHICLE_DETECTED" in text:
                now = time.time() * 1000
                if now - _last_detect_time > debounce_ms:
                    _last_detect_time = now
                    return True

    except Exception:
        pass

    return False
