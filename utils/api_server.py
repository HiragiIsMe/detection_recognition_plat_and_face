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

# ==========================================
# API 1: KHUSUS OPEN GATE
# ==========================================
@app.route('/api/open-gate', methods=['POST'])
def manual_open_gate():
    try:
        # Path menuju folder out_validation
        out_validation_dir = os.path.join(project_root, "out_validation")
        os.makedirs(out_validation_dir, exist_ok=True)

        # 1. Buat file trigger UNTUK GATE
        trigger_path = os.path.join(out_validation_dir, "trigger_open.txt")

        with open(trigger_path, "w") as f:
            f.write("OPEN")

        # 2. Catat Log
        with open("manual_logs.txt", "a") as f:
            f.write(f"{datetime.now()} - Gate dibuka manual via App\n")

            return jsonify({
                "status": "success",
                "message": "Gate akan dibuka (trigger dikirim)"
            }), 200
        return jsonify({
            "status": "success",
            "message": "Perintah BUKA GATE dikirim."
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==========================================
# API 2: KHUSUS MATIKAN BUZZER
# ==========================================
@app.route('/api/stop-buzzer', methods=['POST'])
def manual_stop_buzzer():
    try:
        # Path menuju folder out_validation
        out_validation_dir = os.path.join(project_root, "out_validation")
        os.makedirs(out_validation_dir, exist_ok=True)

        # 1. Buat file trigger UNTUK BUZZER (Nama file beda)
        trigger_path = os.path.join(out_validation_dir, "trigger_mute.txt")

        with open(trigger_path, "w") as f:
            f.write("MUTE")

        # 2. Catat Log
        with open("manual_logs.txt", "a") as f:
            f.write(f"{datetime.now()} - Buzzer dimatikan manual via App\n")

        return jsonify({
            "status": "success",
            "message": "Perintah MATIKAN BUZZER dikirim."
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print(f"🚀 API Server berjalan dari: {current_dir}")
    # Host 0.0.0.0 agar bisa diakses dari luar (HP/Laptop lain)
    app.run(host='0.0.0.0', port=5000, debug=True)