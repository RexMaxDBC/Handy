import streamlit as st
import time
import av
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="Handy Detektor", layout="wide")

# --- Modelle laden (cached) ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_teachable():
    model = load_model("models/teachable/keras_model.h5", compile=False)
    with open("models/teachable/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

# --- Session State Initialisierung ---
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "timer_start" not in st.session_state:
    st.session_state.timer_start = None
if "timer_duration" not in st.session_state:
    st.session_state.timer_duration = 0
if "timer_phase" not in st.session_state:
    st.session_state.timer_phase = "work"

# --- Modelle laden ---
try:
    yolo_model = load_yolo()
    teachable_model, teachable_labels = load_teachable()
    st.success("✅ Modelle geladen!")
except Exception as e:
    st.error(f"Fehler beim Laden der Modelle: {e}")
    st.stop()

# --- VideoProcessor ---
class VideoProcessor:
    def __init__(self):
        self.yolo = yolo_model
        self.teachable = teachable_model
        self.labels = teachable_labels

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO Detection
        results = self.yolo(img, conf=0.5, verbose=False)
        annotated = results[0].plot()
        
        # Prüfen auf Handy (COCO class 67)
        yolo_phone = False
        if results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls[0]) == 67:
                    yolo_phone = True
                    break
        
        # Teachable Machine Prediction
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((224, 224))
        img_array = np.array(pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.teachable.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        tm_label = self.labels[class_idx]
        
        # Overlays
        color = (0, 255, 0) if "handy" in tm_label.lower() else (0, 0, 255)
        cv2.putText(annotated, f"Teachable: {tm_label} ({confidence:.2f})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if yolo_phone:
            cv2.putText(annotated, "YOLO: Handy erkannt!", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Warnung nur in Arbeitsphase
        if st.session_state.get("timer_phase") == "work" and (yolo_phone or "handy" in tm_label.lower()):
            cv2.putText(annotated, "⚠️ HANDY ERKANNT! Weiterarbeiten! ⚠️", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# --- UI: Sidebar Timer ---
st.sidebar.header("🍅 Pomodoro Timer")
col1, col2 = st.sidebar.columns(2)

if col1.button("📝 Arbeitsphase (25 min)", type="primary"):
    st.session_state.timer_phase = "work"
    st.session_state.timer_start = time.time()
    st.session_state.timer_duration = 25 * 60
    st.session_state.timer_running = True
    st.rerun()

if col2.button("☕ Pause (5 min)"):
    st.session_state.timer_phase = "pause"
    st.session_state.timer_start = time.time()
    st.session_state.timer_duration = 5 * 60
    st.session_state.timer_running = True
    st.rerun()

if st.sidebar.button("⏹️ Timer stoppen"):
    st.session_state.timer_running = False
    st.rerun()

# Timer Anzeige
timer_placeholder = st.sidebar.empty()

if st.session_state.get("timer_running", False) and st.session_state.get("timer_start"):
    elapsed = time.time() - st.session_state.timer_start
    remaining = st.session_state.timer_duration - elapsed
    
    if remaining <= 0:
        st.toast(f"✅ {st.session_state.timer_phase.capitalize()} beendet!", icon="🎉")
        st.session_state.timer_running = False
        st.rerun()
    else:
        mins, secs = divmod(int(remaining), 60)
        phase_icon = "📝" if st.session_state.timer_phase == "work" else "☕"
        phase_name = "ARBEIT" if st.session_state.timer_phase == "work" else "PAUSE"
        timer_placeholder.markdown(f"### {phase_icon} {phase_name}\n## {mins:02d}:{secs:02d}")
else:
    if st.session_state.timer_phase == "work":
        timer_placeholder.markdown("### 📝 Kein aktiver Timer\nStarte eine Arbeitsphase")
    else:
        timer_placeholder.markdown("### ☕ Kein aktiver Timer\nStarte eine Pause")

# --- Hauptbereich ---
st.title("📱 Handy-Erkennung mit KI")
st.markdown("---")

col_vid, col_info = st.columns([2, 1])

with col_vid:
    st.subheader("🎥 Live Kamera")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    ctx = webrtc_streamer(
        key="handy-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if not ctx.video_processor:
        st.info("📷 Bitte Kamera-Zugriff erlauben")

with col_info:
    st.subheader("ℹ️ Info")
    st.markdown("""
    **Zwei KI-Modelle gleichzeitig:**
    
    1. **YOLOv8** - Erkennt Handys (COCO Class 67)
    2. **Teachable Machine** - Eigene Klassifizierung
    
    **Timer-Funktion:**
    - Arbeitsphase: 25 min
    - Pause: 5 min
    - Warnung bei Handy in Arbeitsphase
    """)
    
    st.divider()
    
    if st.session_state.get("timer_phase") == "work" and st.session_state.get("timer_running"):
        st.success("🔴 Arbeitsphase AKTIV - Handy-Warnung eingeschaltet")
    elif st.session_state.get("timer_phase") == "pause":
        st.info("🟢 Pause - Keine Warnungen")
    else:
        st.warning("⏸️ Timer nicht aktiv - Starte eine Arbeitsphase")
