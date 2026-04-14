import streamlit as st
import time
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# YOLOv8n laden (vortrainiert)
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

class VideoProcessor:
    def __init__(self):
        self.model = yolo_model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # YOLO Inference
        results = self.model(img, conf=0.5, verbose=False)
        annotated = results[0].plot()  # malt Box + Label "cell phone"

        # Prüfen ob Handy erkannt wurde (COCO class 67)
        phone_detected = any(int(box.cls[0]) == 67 for box in results[0].boxes) if results[0].boxes else False

        # Rote Warnung nur in Arbeitsphase
        if phone_detected and st.session_state.get("timer_phase") == "work":
            cv2.putText(annotated, "HANDY ERKENNT! Leg es weg!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="FocusMate - YOLO Version", layout="wide")
st.title("🎯 FocusMate – Handy-Erkennung mit YOLOv8n")

st.markdown("**Nur Säule A:** YOLOv8n erkennt jedes Handy (COCO Klasse 67).")

# Sidebar Timer
st.sidebar.header("Pomodoro Timer")
col1, col2 = st.sidebar.columns(2)

if col1.button("▶️ Arbeitsphase (25 min)", type="primary"):
    st.session_state.timer_phase = "work"
    st.session_state.timer_start = time.time()
    st.session_state.timer_duration = 25 * 60
    st.session_state.timer_running = True

if col2.button("☕ Pause (5 min)"):
    st.session_state.timer_phase = "pause"
    st.session_state.timer_start = time.time()
    st.session_state.timer_duration = 5 * 60
    st.session_state.timer_running = True

if st.sidebar.button("⏹️ Timer stoppen"):
    st.session_state.timer_running = False

# Timer Anzeige
timer_placeholder = st.empty()

if st.session_state.get("timer_running", False):
    elapsed = time.time() - st.session_state.timer_start
    remaining = st.session_state.timer_duration - elapsed
    if remaining <= 0:
        st.toast("✅ Phase beendet!", icon="🎉")
        st.session_state.timer_running = False
    else:
        mins, secs = divmod(int(remaining), 60)
        phase_text = "🟥 ARBEIT" if st.session_state.timer_phase == "work" else "🟩 PAUSE"
        timer_placeholder.markdown(f"### {phase_text} – {mins:02d}:{secs:02d}")

# Live Webcam mit YOLO
st.subheader("📹 Live Webcam – Handy-Erkennung")
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

st.caption("YOLOv8n läuft live. In der Arbeitsphase wird bei Handy-Erkennung rot gewarnt.")
