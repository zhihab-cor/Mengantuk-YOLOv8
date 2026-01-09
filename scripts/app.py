import streamlit as st
import cv2
from ultralytics import YOLO
import time
import pygame
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Driver Monitoring System", layout="wide")

# Custom CSS untuk tampilan yang lebih "Automotive"
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    div[data-testid="stMetricValue"] { color: #00ff00; }
    </style>
    """, unsafe_allow_html=True)

# --- INISIALISASI ---
if 'start_sleep_time' not in st.session_state:
    st.session_state.start_sleep_time = None

# Inisialisasi Suara (Safe for Cloud)
try:
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False
    # Di cloud tidak ada audio device, jadi kita abaikan errornya
    # Agar app tidak crash
    pass

def play_alarm():
    if AUDIO_AVAILABLE:
        if not pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.load("scripts/biohazard-alarm-143105 (1).mp3") 
                pygame.mixer.music.play(-1) # Loop alarm sampai dimatikan
            except:
                pass
    else:
        # Fallback visual jika audio tidak support (misal di Cloud)
        pass 

def stop_alarm():
    if AUDIO_AVAILABLE:
        pygame.mixer.music.stop()

# Load Model
@st.cache_resource
def load_model():
    return YOLO("models/train2_aug/weights/best.pt")

model = load_model()

# --- UI LAYOUT ---
st.title("🚗 AI Driver, Detection Microsleep untuk pengendara")
st.markdown("---")

# Sidebar untuk Kontrol
st.sidebar.header("Control Panel")
start_monitor = st.sidebar.button("🎥 Mulai Deteksi", use_container_width=True)
stop_monitor = st.sidebar.button("🛑 Berhenti", use_container_width=True)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
alert_duration = st.sidebar.slider("Durasi Alarm (detik)", 1, 30, 3)

# Layout Utama: 2 Kolom
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Feed")
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("System Metrics")
    status_placeholder = st.empty()
    fps_placeholder = st.empty()
    timer_placeholder = st.empty()

# --- LOGIKA PROGRAM ---
if start_monitor:
    cap = cv2.VideoCapture(0)
    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera.")
            break

        # Hitung FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Deteksi YOLO
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # Logika Deteksi Mata
        eyes_closed = False
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            # GANTI 'mata_tertutup' sesuai label asli di modelmu (cek model.names)
            if "tutup" in label.lower() or "close" in label.lower():
                eyes_closed = True
                break

        # Logika Timer & Alarm
        if eyes_closed:
            if st.session_state.start_sleep_time is None:
                st.session_state.start_sleep_time = time.time()
            
            elapsed = time.time() - st.session_state.start_sleep_time
            
            status_placeholder.metric("Status", "WARNING", delta="- Bahaya", delta_color="inverse")
            timer_placeholder.metric("Mata Tertutup", f"{elapsed:.1f}s")
            
            if elapsed >= alert_duration:
                st.toast("🚨 BANGUN! ANDA MENGANTUK!", icon="🚨")
                play_alarm()
        else:
            st.session_state.start_sleep_time = None
            stop_alarm()
            status_placeholder.metric("Status", "ACTIVE", delta="Aman")
            timer_placeholder.metric("Mata Tertutup", "0.0s")

        # Tampilkan FPS
        fps_placeholder.metric("Frame Rate", f"{int(fps)} FPS")

        # Update Frame ke Web
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        if stop_monitor:
            stop_alarm()
            break
    
    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Klik tombol 'Mulai Deteksi' di sidebar untuk mengaktifkan kamera.")