# utils/camera.py
import cv2
import os
import uuid


def capture_vehicle_image(output_dir="img-in", camera_index=0, resize_to=None):
    """Capture satu frame dari webcam dan simpan ke folder.

    Args:
        output_dir (str): folder output
        camera_index (int): index device untuk cv2.VideoCapture
        resize_to (tuple|None): (width, height) jika ingin resize sebelum simpan

    Returns:
        str|None: path file yang disimpan atau None kalau gagal
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Kamera gagal dibuka (index={})".format(camera_index))
        return None

    # beri waktu kamera auto-adjust (opsional)
    time_wait_s = 0.2
    cap.read()  # discard first frame
    # Jika kamu butuh buffer lebih lama, tambahkan sleep

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Gagal capture frame dari kamera")
        return None

    if resize_to is not None:
        frame = cv2.resize(frame, resize_to)

    img_id = str(uuid.uuid4())
    file_path = os.path.join(output_dir, f"{img_id}.jpg")
    cv2.imwrite(file_path, frame)

    print(f"📸 Gambar tersimpan: {file_path}")
    return file_path