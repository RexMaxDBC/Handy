import streamlit as st
import time
import av
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ------------------- Modelle laden (cached) -------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # vortrainiert auf COCO

@st.cache_resource
def load_teachable():
    model = load_model("models/teachable/keras_model.h5", compile=False)
    with open("models/teachable/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

yolo_model = load_yolo()
teachable_model, teachable_labels = load_teachable()

# ------------------- VideoProcessor (zwei KI-Säulen live) -------------------
class VideoProcessor:
    def __init__(self):
        self.yolo = yolo_model
        self.teachable = teachable_model
        self.labels = teachable_labels

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Säule A: YOLOv8n
        results = self.yolo(img, conf=0.5, verbose=False)
        annotated = results[0].plot()  # zeichnet Box + "cell phone"

        # Prüfen, ob Handy (COCO class 67)
        yolo_phone = any(int(box.cls[0]) == 67 for box in results[0].boxes) if results[0].boxes else False

        # Säule B: Teachable Machine (ganzes Bild)
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((224, 224))
        img_array = np.array(pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.teachable.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        tm_label = self.labels[class_idx]

        # Overlays
        color = (0, 255, 0) if "mein Handy" in tm_label.lower() else (0, 0, 255)
        cv2.putText(annotated, f"Teachable: {tm_label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if yolo_phone:
            cv2.putText(annotated, "YOLO: Cell Phone (67) detected!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Rote Warnung nur in Arbeitsphase (falls du später erweiterst)
        if yolo_phone or "mein Handy" in tm_label.lower():
            cv2.putText(annotated, "HANDY ERKENNT!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="FocusMate", layout="wide")
st.title("🎯 FocusMate – KI-Pomodoro mit Webcam-Handy-Erkennung")

st.markdown("""
**So funktioniert's:**
- **Arbeitsphase (25 min)**: Rote Warnung + Overlay, wenn Handy erkannt wird
- **Pause (5 min)**: Keine Warnung
- Zwei KI-Säulen laufen **live** im Webcam-Stream
""")

# Sidebar: Timer-Steuerung
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

# Timer-Anzeige + Auto-Update
timer_placeholder = st.empty()

if st.session_state.get("timer_running", False) and st.session_state.get("timer_start"):
    elapsed = time.time() - st.session_state.timer_start
    remaining = st.session_state.timer_duration - elapsed

    if remaining <= 0:
        st.toast(f"✅ {st.session_state.timer_phase.capitalize()} beendet!", icon="🎉")
        st.session_state.timer_running = False
    else:
        mins, secs = divmod(int(remaining), 60)
        phase_text = "🟥 ARBEIT" if st.session_state.timer_phase == "work" else "🟩 PAUSE"
        timer_placeholder.markdown(f"### {phase_text} – {mins:02d}:{secs:02d}")
else:
    timer_placeholder.markdown("### Timer gestoppt – Starte eine Phase")

# Live Webcam + beide KI-Modelle
st.sub
