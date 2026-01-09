import cv2
import time
from ultralytics import YOLO

# 1. Load model kamu
model = YOLO("/Users/test/Documents/Data Mining/Mengantuk-YOLOv8/models/train2_aug/weights/best.pt")

# 2. Buka Kamera (source 0 adalah webcam bawaan)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Mulai hitung waktu awal proses
    start_time = time.time()

    # Jalankan prediksi (stream=True lebih efisien untuk kamera)
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # Hitung waktu selesai dan konversi ke FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Gambar hasil deteksi pada frame
    annotated_frame = results[0].plot()

    # Tampilkan teks FPS di pojok kiri atas (Warna hijau)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan jendela video
    cv2.imshow("Webcam Real-time Test - Monitoring Mengantuk", annotated_frame)

    # Berhenti jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()