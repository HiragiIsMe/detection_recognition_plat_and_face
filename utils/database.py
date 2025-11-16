# utils/database.py
import mysql.connector
import json
import uuid
from datetime import datetime

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="gate_system"
    )

def generate_uuid():
    return str(uuid.uuid4())

def insert_entry(plate_text, plate_conf, face_vector, plate_path, face_path):
    conn = get_connection()
    cursor = conn.cursor()

    entry_id = generate_uuid()

    sql = """
    INSERT INTO entries (id, plate_text, plate_conf, face_vector, plate_image, face_image, entry_time, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
    """

    cursor.execute(sql, (
        entry_id,
        plate_text,
        plate_conf,
        json.dumps(face_vector),
        plate_path,
        face_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[DB] Entry inserted - ID: {entry_id}, Plate: {plate_text}")
    return entry_id

def mark_entry_exited(entry_id):
    """
    Update status menjadi 'exited' dan set exit_time
    """
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    UPDATE entries 
    SET status = 'exited', exit_time = %s 
    WHERE id = %s AND status = 'active'
    """

    cursor.execute(sql, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        entry_id
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[DB] Entry {entry_id} marked as exited")

def get_active_entry_by_plate(plate_text):
    """
    Query database berdasarkan plat nomor yang masih active
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    sql = """
    SELECT * FROM entries 
    WHERE plate_text = %s AND status = 'active' 
    ORDER BY entry_time DESC LIMIT 1
    """
    
    cursor.execute(sql, (plate_text,))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if result:
        print(f"[DB] Active data ditemukan untuk plat: {plate_text}")
        result['face_vector'] = json.loads(result['face_vector'])
        return result
    else:
        print(f"[DB] Tidak ada active data untuk plat: {plate_text}")
        return None

def create_table_if_not_exists():
    """
    Buat table dengan kolom status dan exit_time
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    sql = """
    CREATE TABLE IF NOT EXISTS entries (
        id VARCHAR(36) PRIMARY KEY,
        plate_text VARCHAR(50),
        plate_conf FLOAT,
        face_vector JSON,
        plate_image VARCHAR(500),
        face_image VARCHAR(500),
        entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        exit_time TIMESTAMP NULL,
        status ENUM('active', 'exited') DEFAULT 'active'
    )
    """
    
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Table 'entries' ready dengan status management")