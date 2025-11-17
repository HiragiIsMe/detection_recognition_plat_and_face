# utils/sensor_simple.py
import serial
import time
from utils.loading import LoadingAnimation

def sensor_detect_vehicle_continuous(port='COM3'):
    """
    Fungsi sensor yang terus mendeteksi dengan loading animation
    """
    print(f"🔍 Mencoba koneksi ke {port}...")
    
    try:
        # Koneksi serial sederhana
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1
        )
        
        # Tunggu Arduino reset
        time.sleep(2)
        
        # Clear buffer garbage data
        ser.reset_input_buffer()
        
        print(f"✅ Terkoneksi ke {port}")
        
        # LOADING CONTINUOUS - akan terus berjalan
        loading = LoadingAnimation("Mendeteksi kendaraan")
        loading.start()
        
        while True:  # Infinite loop
            # Baca data serial
            if ser.in_waiting > 0:
                try:
                    # Baca raw data
                    raw_data = ser.readline()
                    
                    # Coba decode, skip jika error
                    try:
                        text = raw_data.decode('utf-8').strip()
                    except:
                        try:
                            text = raw_data.decode('latin-1').strip()
                        except:
                            # Skip data yang tidak bisa di-decode
                            continue
                    
                    # Cek jika kendaraan terdeteksi
                    if "VEHICLE_DETECTED" in text:
                        loading.stop("✅ Kendaraan terdeteksi!")
                        ser.close()
                        return True
                            
                except Exception as e:
                    # Skip error decoding
                    continue
            
            time.sleep(0.1)
        
    except Exception as e:
        print(f"❌ Gagal koneksi ke {port}: {e}")
        # ⚠️ JANGAN return True, biarkan program berhenti
        return False  # ⚠️ PERBAIKAN: Return False jika gagal koneksi

# Fungsi untuk single detection
def sensor_detect_vehicle(port='COM3', timeout=30):
    """
    Fungsi sensor untuk single detection
    """
    return sensor_detect_vehicle_continuous(port=port)

# Fungsi tanpa parameter untuk compatibility
def sensor_detect_vehicle_simple():
    return sensor_detect_vehicle(port='COM3')