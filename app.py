import streamlit as st
import time
import av
import cv2
import numpy as np
import torch  # ← neu hinzugefügt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# Fix für den UnpicklingError (PyTorch 2.6+)
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# YOLOv8n laden (vortrainiert)
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")   # lädt automatisch herunter

yolo_model = load_yolo()

class VideoProcessor:
    def __init__(self):
        self.model = yolo_model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # YOLO Inference
        results = self.model(img, conf=0.5, verbose=False)
        annotated = results[0].plot()

        # Handy erkannt? (COCO class 67 = cell phone)
        phone_detected = any(int(box.cls[0]) == 67 for box in results[0].boxes) if results[0].boxes else False

        # Rote Warnung nur in der Arbeitsphase
        if phone_detected and st.session_state.get("timer_phase") == "work":
            cv2.putText(annotated, "HANDY ERKENNT! Leg es weg!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="FocusMate - YOLO", layout="wide")
st.title("🎯 FocusMate – Handy-Erkennung mit YOLOv8n")

st.markdown("**Nur mit YOLOv8n** (vortrainiert auf COCO). Erkennt jedes Handy als 'cell phone'.")

# Sidebar: Timer
st.sidebar.header("Pomodoro Timer")
col1, col2 = st.sidebar.columns(2)

if col1.button("▶️ Arbeitsphase starten (25 min)", type="primary"):
    st.session_state.timer_phase = "work"
    st.session_state.timer_start = time.time()
    st.session_state.timer_duration = 25 * 60
    st.session_state.timer_running = True

if col2.button("☕ Pause starten (5 min)"):
    st.session_state.timer_phase = "pause"
    st.session_state.timer_start = time.time()
    st.session_state.timer_duration = 5 * 60
    st.session_state.timer_running = True

if st.sidebar.button("⏹️ Timer stoppen"):
    st.session_state.timer_running = False

# Timer-Anzeige
timer_placeholder = st.empty()

if st.session_state.get("timer_running", False) and st.session_state.get("timer_start"):
    elapsed = time.time() - st.session_state.timer_start
    remaining = st.session_state.timer_duration - elapsed
    if remaining <= 0:
        st.toast("✅ Phase beendet!", icon="🎉")
        st.session_state.timer_running = False
    else:
        mins, secs = divmod(int(remaining), 60)
        phase_text = "🟥 ARBEIT" if st.session_state.timer_phase == "work" else "🟩 PAUSE"
        timer_placeholder.markdown(f"### {phase_text} – {mins:02d}:{secs:02d}")

# Webcam + YOLO
st.subheader("📹 Live Webcam mit YOLO-Erkennung")
webrtc_streamer(
    key="focusmate_yolo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.session_state.get("timer_running", False):
    time.sleep(0.3)
    st.rerun()

st.caption("YOLOv8n läuft live. Rote Warnung nur in der Arbeitsphase.")
