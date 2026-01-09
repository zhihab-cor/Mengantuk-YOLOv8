import cv2
import time
from ultralytics import YOLO

# 1. Load model hasil augmentasi kamu
model = YOLO('models/train2_aug/weights/best.pt')

# 2. Buka file video
video_path = 'scripts/kamera_dascam.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Mulai hitung waktu
    start_time = time.time()

    # Jalankan deteksi
    results = model.predict(frame, conf=0.4, verbose=False)
    
    # Selesai hitung waktu
    end_time = time.time()
    
    # Hitung FPS (1 dibagi durasi proses)
    fps = 1 / (end_time - start_time)

    # Gambar hasil deteksi (box, labels)
    annotated_frame = results[0].plot()

    # Tulis teks FPS di pojok kiri atas frame (Warna Putih, Background Biru)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Tampilkan video
    cv2.imshow("YOLOv8 Real-time Test - Mobil", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()