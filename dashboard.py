import streamlit as st
import cv2
import math
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="SpaceSense | Footfall & Occupancy Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Professional Enterprise Analytics CSS
st.markdown("""
    <style>
    /* Main Background & Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom KPI Card Styling */
    .kpi-card {
        background: linear-gradient(145deg, #161f33, #111827);
        border: 1px solid #26334d;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.78rem;
        margin-top: 8px;
        font-weight: 500;
    }
    .text-emerald { color: #10b981; }
    .text-amber { color: #f59e0b; }
    .text-sky { color: #38bdf8; }
    .text-rose { color: #f43f5e; }

    /* Header Bar */
    .top-header {
        background: #111827;
        border-bottom: 1px solid #1e293b;
        padding: 14px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load YOLO Model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# 4. Sidebar Configuration (Retail & Venue Analytics Controls)
with st.sidebar:
    st.markdown("## 📈 **SpaceSense AI**")
    st.caption("Enterprise Footfall & Venue Analytics")
    st.markdown("---")

    st.markdown("### 🎥 Video Feed Source")
    uploaded_file = st.file_uploader("Upload Footage (MP4, AVI)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with open("cctv.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Footage loaded into pipeline!")

    st.markdown("---")
    st.markdown("### ⚙️ Calibration & Zone Settings")
    counting_mode = st.selectbox(
        "Counting Axis",
        [
            "Smart Zone (Works on All Videos)",
            "Vertical Gate (Left ↔ Right)",
            "Horizontal Gate (Up ↔ Down)"
        ],
        index=0
    )
    line_pos = st.slider("Gate / Zone Position (%)", 15, 85, 50)
    zone_sensitivity = st.slider("Trigger Zone Width (px)", 15, 60, 35, help="Wider zone ensures fast-moving or crowded pedestrians are never missed.")
    max_capacity = st.number_input("Venue Max Capacity", min_value=10, max_value=500, value=50, step=10)
    conf_thresh = st.slider("Detection Confidence", 0.15, 0.70, 0.25, 0.05)

# 5. Top Executive Header
st.markdown("""
    <div class="top-header">
        <div>
            <h2 style="margin:0; padding:0; color:#f8fafc; font-size:1.45rem;">🏢 Commercial Footfall & Space Utilization Center</h2>
            <span style="color:#94a3b8; font-size:0.85rem;">Real-Time Pedestrian Flow, Zone Dwell & Capacity Compliance</span>
        </div>
        <div style="text-align:right;">
            <span style="background:#064e3b; color:#34d399; padding:6px 12px; border-radius:20px; font-size:0.78rem; font-weight:600;">
                ● PIPELINE ACTIVE
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# 6. Top KPI Row Placeholders (5 Meaningful Retail/Venue Metrics)
kpi_cols = st.columns(5)
kpi_in = kpi_cols[0].empty()
kpi_out = kpi_cols[1].empty()
kpi_live_crowd = kpi_cols[2].empty()
kpi_peak = kpi_cols[3].empty()
kpi_util = kpi_cols[4].empty()

# 7. Main Content Layout: Left (Video Stream) | Right (Live Analytics & Event Log)
main_col, right_col = st.columns([2.2, 1])

with main_col:
    st.markdown("##### 📹 Live Annotated Vision Feed")
    video_spot = st.empty()

with right_col:
    st.markdown("##### 📊 Space Capacity Gauge")
    capacity_bar = st.empty()
    st.markdown("##### 📈 Footfall Trend (Live Flow)")
    chart_spot = st.empty()
    st.markdown("##### 🕒 Recent Crossing Events")
    log_spot = st.empty()

# 8. Helper Function to Render Clean HTML KPI Cards
def render_kpi(placeholder, title, value, subtext, color_class):
    placeholder.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub {color_class}">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

# 9. Tracking & Counting State Initialization
cap = cv2.VideoCapture("cctv.mp4")

in_count = 0
out_count = 0
peak_crowd = 0

# id -> {"cx": int, "cy": int, "init_x": int, "init_y": int, "counted": bool, "frames_seen": int}
tracks = {}
next_id = 1
event_logs = []
trend_data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tracks.clear()
        continue

    frame_idx += 1
    frame = cv2.resize(frame, (860, 480))
    h, w, _ = frame.shape

    line_x = int(w * (line_pos / 100))
    line_y = int(h * (line_pos / 100))

    # Run YOLOv8 Inference on Person class (0)
    results = model(frame, classes=[0], conf=conf_thresh, verbose=False)
    detections = []

    if len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((cx, cy, x1, y1, x2, y2))

    live_headcount = len(detections)
    if live_headcount > peak_crowd:
        peak_crowd = live_headcount

    # Match Detections with Existing Centroid Tracks
    updated_tracks = {}
    for cx, cy, x1, y1, x2, y2 in detections:
        best_id = None
        min_dist = 110  # Generous matching radius for crowded scenes

        for tid, tdata in tracks.items():
            dist = math.hypot(cx - tdata["cx"], cy - tdata["cy"])
            if dist < min_dist and tid not in updated_tracks:
                min_dist = dist
                best_id = tid

        if best_id is None:
            best_id = next_id
            next_id += 1
            updated_tracks[best_id] = {
                "cx": cx, "cy": cy,
                "init_x": cx, "init_y": cy,
                "counted": False,
                "frames_seen": 1
            }
        else:
            prev = tracks[best_id]
            updated_tracks[best_id] = {
                "cx": cx, "cy": cy,
                "init_x": prev["init_x"], "init_y": prev["init_y"],
                "counted": prev["counted"],
                "frames_seen": prev["frames_seen"] + 1
            }

        t = updated_tracks[best_id]
        dx = cx - t["init_x"]
        dy = cy - t["init_y"]

        # --- BULLETPROOF COUNTING ENGINE ---
        if not t["counted"]:
            if counting_mode == "Vertical Gate (Left ↔ Right)":
                # Crossed vertical line OR entered vertical buffer zone with movement
                if (t["init_x"] < line_x and cx >= line_x) or (abs(cx - line_x) <= zone_sensitivity and dx > 8):
                    in_count += 1
                    t["counted"] = True
                    event_logs.insert(0, f"🟢 [{datetime.now().strftime('%H:%M:%S')}] ID #{best_id} — Entry (East Gate)")
                elif (t["init_x"] > line_x and cx <= line_x) or (abs(cx - line_x) <= zone_sensitivity and dx < -8):
                    out_count += 1
                    t["counted"] = True
                    event_logs.insert(0, f"🟠 [{datetime.now().strftime('%H:%M:%S')}] ID #{best_id} — Exit (West Gate)")

            elif counting_mode == "Horizontal Gate (Up ↔ Down)":
                if (t["init_y"] < line_y and cy >= line_y) or (abs(cy - line_y) <= zone_sensitivity and dy > 8):
                    in_count += 1
                    t["counted"] = True
                    event_logs.insert(0, f"🟢 [{datetime.now().strftime('%H:%M:%S')}] ID #{best_id} — Entry (North Gate)")
                elif (t["init_y"] > line_y and cy <= line_y) or (abs(cy - line_y) <= zone_sensitivity and dy < -8):
                    out_count += 1
                    t["counted"] = True
                    event_logs.insert(0, f"🟠 [{datetime.now().strftime('%H:%M:%S')}] ID #{best_id} — Exit (South Gate)")

            else:
                # SMART ZONE MODE: Counts both X/Y crossings AND confirmed active pedestrians inside the trigger band
                in_vertical_band = abs(cx - line_x) <= zone_sensitivity
                crossed_x_right = t["init_x"] < line_x and cx >= line_x
                crossed_x_left = t["init_x"] > line_x and cx <= line_x

                if crossed_x_right or (in_vertical_band and t["frames_seen"] >= 3 and dx >= 0):
                    in_count += 1
                    t["counted"] = True
                    event_logs.insert(0, f"🟢 [{datetime.now().strftime('%H:%M:%S')}] Visitor #{best_id} — Zone Inflow")
                elif crossed_x_left or (in_vertical_band and t["frames_seen"] >= 3 and dx < 0):
                    out_count += 1
                    t["counted"] = True
                    event_logs.insert(0, f"🟠 [{datetime.now().strftime('%H:%M:%S')}] Visitor #{best_id} — Zone Outflow")

        # Draw Clean Analytics Bounding Box (Teal if counted, Sky Blue if tracking)
        box_color = (16, 185, 129) if t["counted"] else (248, 189, 56)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"#{best_id} {'[COUNTED]' if t['counted'] else ''}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)

    tracks = updated_tracks

    # Draw Semi-Transparent Analytics Trigger Zone on Video
    overlay = frame.copy()
    if counting_mode == "Horizontal Gate (Up ↔ Down)":
        cv2.rectangle(overlay, (0, line_y - zone_sensitivity), (w, line_y + zone_sensitivity), (248, 189, 56), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 215, 255), 2)
    else:
        cv2.rectangle(overlay, (line_x - zone_sensitivity, 0), (line_x + zone_sensitivity, h), (248, 189, 56), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 215, 255), 2)

    # Top-Left HUD Overlay on Video
    cv2.rectangle(frame, (12, 12), (430, 52), (15, 23, 42), -1)
    cv2.putText(frame, f"IN: {in_count}  |  OUT: {out_count}  |  ACTIVE CROWD: {live_headcount}",
                (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)

    # Update 5 Top KPI Cards
    occupancy_pct = min(100, int((live_headcount / max_capacity) * 100))
    render_kpi(kpi_in, "📥 Total Entries (IN)", in_count, "↑ Cumulative Inflow", "text-emerald")
    render_kpi(kpi_out, "📤 Total Exits (OUT)", out_count, "↓ Cumulative Outflow", "text-amber")
    render_kpi(kpi_live_crowd, "👥 Live Zone Headcount", live_headcount, "Active people in frame", "text-sky")
    render_kpi(kpi_peak, "🔥 Peak Crowd Density", peak_crowd, "Session high watermark", "text-rose")
    render_kpi(kpi_util, "🏢 Space Utilization", f"{occupancy_pct}%", f"Target Limit: {max_capacity} pax", "text-emerald")

    # Update Right-Column Analytics every 5 frames for smooth UI
    if frame_idx % 5 == 0:
        status_label = "Optimal Flow" if occupancy_pct < 75 else "High Congestion"
        capacity_bar.progress(occupancy_pct / 100.0, text=f"{occupancy_pct}% Occupied ({status_label})")

        trend_data.append({"Inflow": in_count, "Outflow": out_count, "Live Crowd": live_headcount})
        if len(trend_data) > 40:
            trend_data.pop(0)
        chart_spot.line_chart(pd.DataFrame(trend_data), height=170)

        log_text = "\n\n".join(event_logs[:6]) if event_logs else "Waiting for pedestrian movement..."
        log_spot.info(log_text)

    # Render Video Frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_spot.image(frame_rgb, channels="RGB", use_container_width=True)

cap.release()