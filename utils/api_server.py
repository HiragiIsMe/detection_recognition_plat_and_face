# utils/api_server.py
from flask import Flask, jsonify, request  # 👈 TAMBAHKAN request DI SINI
import sys
import os
from datetime import datetime

# === SETUP PATH ===
# Karena file ini ada di dalam folder 'utils', kita perlu menambahkan
# folder 'root project' (satu level di atasnya) ke sistem path
# agar bisa melakukan import dengan gaya 'from utils.database ...'
current_dir = os.path.dirname(os.path.abspath(__file__)) # Path ke folder utils
project_root = os.path.dirname(current_dir)              # Path ke folder project utama
sys.path.append(project_root)

# === IMPORT ===
# Import dari database.py yang berada di folder yang sama (utils)
from utils.database import get_vehicle

app = Flask(__name__)

# === ENDPOINT ===
@app.route('/api/vehicle', methods=['GET'])
def get_history():
    try:
        # Ambil data
        data_kendaraan = get_vehicle()
        
        return jsonify({
            "status": "success",
            "total": len(data_kendaraan),
            "data": data_kendaraan
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "server_loc": "utils/api_server.py"
    })

@app.route('/api/open-gate', methods=['POST'])
def manual_gate_control():
    try:
        # 1. Terima request (sekadar untuk validasi token/auth jika perlu)
        data = request.json
        action = data.get('action') # misal: "OPEN"
        
        print(f"MANUAL OVERRIDE RECEIVED: Action {action}")

        # 2. LANGSUNG KE HARDWARE (Matikan Alarm & Buka Gate)
        # Contoh logika kirim ke Arduino via Serial
        if action == "OPEN":
            # Kirim kode ke Arduino, misal 'O' untuk Open, 'S' untuk Stop Alarm
            # ser.write(b'O') 
            print("🔌 [HARDWARE] Mengirim sinyal BUKA GATE ke Server...")
            print("🔌 [HARDWARE] Mengirim sinyal MATIKAN BUZZER ke Server...")
            
            # Opsional: Catat di file text biasa (bukan DB) biar ada jejak
            with open("manual_logs.txt", "a") as f:
                f.write(f"{datetime.now()} - Gate dibuka manual via App\n")

        return jsonify({
            "status": "success", 
            "message": "Perintah buka gate terkirim ke hardware"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 API Server berjalan dari: {current_dir}")
    # Host 0.0.0.0 agar bisa diakses dari luar (HP/Laptop lain)
    app.run(host='0.0.0.0', port=5000, debug=True)