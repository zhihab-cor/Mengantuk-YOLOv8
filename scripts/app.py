import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Driver Monitoring System", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00ff00; }
    </style>
    """, unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
# Variabel global untuk settings (bisa diubah admin, berlaku untuk semua user)
if 'global_conf_threshold' not in st.session_state:
    st.session_state.global_conf_threshold = 0.5
if 'global_alert_duration' not in st.session_state:
    st.session_state.global_alert_duration = 2.0

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan path ini sesuai dengan struktur folder di repo GitHub Anda
    return YOLO("models/train2_aug/weights/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- CLASS PROCESSOR ANTIGRAVITY ---
class VideoProcessor:
    def __init__(self):
        self.start_sleep_time = None
        self.conf_threshold = 0.5
        self.alert_duration = 3
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Deteksi YOLO
        results = model.predict(img, conf=self.conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # 2. Logika Deteksi Mata
        eyes_closed = False
        try:
            for box in results[0].boxes:
                # Ambil label dari ID kelas
                cls_id = int(box.cls[0])
                if cls_id < len(model.names):
                    label = model.names[cls_id]
                    # Sesuaikan kata kunci ini dengan label di dataset Anda!
                    if "tutup" in label.lower() or "close" in label.lower():
                        eyes_closed = True
                        break
        except Exception as e:
            pass # Hindari crash jika ada error parsing boxes

        # 3. Logika Timer & Alarm Visual
        current_time = time.time()
        
        if eyes_closed:
            if self.start_sleep_time is None:
                self.start_sleep_time = current_time
            
            elapsed = current_time - self.start_sleep_time
            
            # Tambahkan info text ke layar
            cv2.putText(annotated_frame, f"Mata Tertutup: {elapsed:.1f}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            if elapsed >= self.alert_duration:
                # ALARM VISUAL: Tulis text besar merah
                cv2.putText(annotated_frame, "BAHAYA! BANGUN!", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                # Gambar border merah tebal
                cv2.rectangle(annotated_frame, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 10)
                st.audio("scripts/biohazard-alarm-143105.mp3", format="audio/mpeg")
        else:
            self.start_sleep_time = None
            cv2.putText(annotated_frame, "Status: AMAN", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- FUNGSI LOGIN ---
def check_login(username, password):
    # Ganti dengan kredensial yang Anda inginkan
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def show_login_form():
    st.sidebar.markdown("### 🔐 Login Admin")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.show_login = False
                st.rerun()
            else:
                st.sidebar.error("Username atau password salah!")
    with col2:
        if st.button("Batal", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()


# --- SIDEBAR ---
# Button Login/Logout di header sidebar
if st.session_state.logged_in:
    st.sidebar.header("Control Panel")
    st.sidebar.success("✅ Logged in as Admin")
    st.session_state.global_conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.1, 1.0,
        st.session_state.global_conf_threshold,
    )
    st.session_state.global_alert_duration = st.sidebar.slider(
        "Durasi Alarm (detik)", 
        1.0, 5.0, 
        st.session_state.global_alert_duration,
    )
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
else:
    if st.sidebar.button("🔑 Login Admin", use_container_width=True, type="primary"):
        st.session_state.show_login = True
        st.rerun()

# Tampilkan form login jika diperlukan
if st.session_state.show_login and not st.session_state.logged_in:
    show_login_form()
    st.stop()

# --- UI LAYOUT ---
# Main Area
st.title("🚗 AI Driver Microsleep Detection (Cloud Optimized)")
st.caption("Menggunakan WebRTC untuk akses kamera browser")
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("### Live Camera Feed")
    # Konfigurasi WebRTC
    ctx = webrtc_streamer(
        key="driver-monitoring",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.gabriel.com:3478"]},
                {"urls": ["stun:stun.zerocharge.com:3478"]},
            ]
        }),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Hacky way to pass metrics to processor update (not perfectly Thread-safe but works for simple settings)
    if ctx.video_processor:
        ctx.video_processor.conf_threshold = st.session_state.global_conf_threshold
        ctx.video_processor.alert_duration = st.session_state.global_alert_duration

with col2:
    st.write("### Panduan")
    st.info("""
    1. Klik **START** untuk mengaktifkan kamera.
    2. Izinkan akses browser pop-up.
    3. Sistem akan mendeteksi mata tertutup.
    4. Jika mata tertutup > durasi, layar akan memberi peringatan **BAHAYA**.
    """)
    st.warning("Catatan: Suara alarm tidak didukung di versi Cloud ini (keterbatasan browser). Peringatan hanya visual.")