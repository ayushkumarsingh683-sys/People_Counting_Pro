# People_Counting_Pro
Real-time pedestrian footfall, bidirectional zone counting, and space occupancy analytics pipeline built with YOLOv8, OpenCV, FastAPI, and Streamlit.

 It is an enterprise-grade computer vision pipeline designed for commercial retail stores, transit hubs, and smart venues. It transforms raw CCTV video feeds into actionable business telemetry—tracking bidirectional pedestrian traffic (**IN / OUT**), **live zone headcount**, **peak crowd density**, and **real-time space capacity utilization**.

---

## 🏗️ System Architecture

```text
+-------------------------------------------------------------------------+
|                        VIDEO INGESTION LAYER                            |
|               CCTV Footage / Live Stream (cctv.mp4 / RTSP)              |
+------------------------------------+------------------------------------+
                                     |
                                     v  (Raw BGR Frames via OpenCV)
+-------------------------------------------------------------------------+
|                  COMPUTER VISION & TRACKING ENGINE (tracker.py)         |
|                                                                         |
|  1. YOLOv8n Detection    --> Filters Class 0 (Person) @ Conf >= 0.25    |
|  2. Centroid Extraction  --> Computes (cx, cy) & Euclidean matching     |
|  3. Occlusion Memory     --> Retains Track IDs for up to 12 missed frames|
|  4. Smart-Zone Trigger   --> Evaluates displacement (dx, dy) & gate band|
+------------------+-----------------------------------+------------------+
                   |                                   |
                   v (Annotated MJPEG Stream)          v (Live JSON Telemetry)
+------------------------------------+   +--------------------------------+
|     FASTAPI STREAMING SERVER       |   |       FASTAPI REST STATE       |
|       GET /video_feed (app.py)     |   |        GET /stats (app.py)     |
+------------------+-----------------+   +----------------+---------------+
                   |                                      |
                   +------------------+-------------------+
                                      | (Non-blocking HTTP Polling @ 1s)
                                      v
+-------------------------------------------------------------------------+
|               EXECUTIVE ANALYTICS DASHBOARD (dashboard.py)              |
|                                                                         |
|  • 5 Live KPI Cards: Inflow, Outflow, Active Headcount, Peak, % Util    |
|  • Real-Time Capacity Gauge & Congestion Status Indicator               |
|  • Timestamped Pedestrian Crossing Event Audit Log                      |
|  • Dynamic Video Uploader & Threshold Calibration Controls              |
+-------------------------------------------------------------------------+
```

---

## ✨ Key Technical Features

* **Smart-Zone Bidirectional Counting:** Replaces brittle 1-pixel line triggers with a configurable central trigger band (`±45px`). Tracks pedestrian displacement vectors ($\Delta x, \Delta y$) alongside boundary crossings to accurately capture diagonal and crowded street movement.
* **Crowd Occlusion Recovery:** Implements a 12-frame memory buffer (`missed` counter) for active track IDs, preventing duplicate counts or ID resets when pedestrians temporarily overlap in dense crowds.
* **Decoupled FastAPI + Streamlit Pipeline:** Uses non-blocking multipart MJPEG streaming (`/video_feed`) alongside a lightweight JSON telemetry endpoint (`/stats`) polled via isolated Streamlit fragments (`@st.fragment`) for zero-flicker 1-second UI updates.
* **5 Commercial Space KPIs:**
  1. **📥 Total Entries (IN):** Cumulative visitor inflow.
  2. **📤 Total Exits (OUT):** Cumulative visitor outflow.
  3. **👥 Live Zone Headcount:** Instantaneous active crowd detected in the current frame.
  4. **🔥 Peak Crowd Density:** Session high-watermark for bottleneck analysis.
  5. **🏢 Space Utilization (%):** Real-time ratio of active headcount against venue maximum capacity.

---

## 📂 Project Directory Structure

```text
spacesense-footfall-analytics/
│
├── app.py                 # FastAPI backend serving /video_feed (MJPEG) and /stats (JSON)
├── tracker.py             # Core YOLOv8n detection, Euclidean centroid tracker & Smart-Zone logic
├── dashboard.py           # Streamlit executive frontend with 1s live fragment polling
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes virtual environments, cache, and heavy media files
└── README.md              # Project documentation and architecture guide
```

---

## ⚙️ Installation & Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/spacesense-footfall-analytics.git](https://github.com/YOUR_USERNAME/spacesense-footfall-analytics.git)
cd spacesense-footfall-analytics
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add a Sample Video
Place any test pedestrian or CCTV video in the root folder and name it **`cctv.mp4`** (or upload a video directly from the Streamlit sidebar after launching).
*(Note: `yolov8n.pt` weights are automatically downloaded by Ultralytics on first run).*

### 4. Launch the System
Open **two terminals** in the project directory:

**Terminal 1 — Start the FastAPI Vision Server:**
```bash
uvicorn app:app --reload
```
*Server runs at `http://127.0.0.1:8000`*

**Terminal 2 — Start the Streamlit Analytics Dashboard:**
```bash
streamlit run dashboard.py
```
*Dashboard opens automatically at `http://localhost:8501`*

---

## 🧠 Counting & Tracking Methodology

1. **Detection:** Each frame is resized to `860x480` and passed through **YOLOv8n** constrained to `classes=[0]` (person).
2. **Centroid Association:** For every bounding box $(x_1, y_1, x_2, y_2)$, the center point $(c_x, c_y)$ is computed. Tracks are matched across frames using Euclidean distance:
   $$d = \sqrt{(c_x - p_x)^2 + (c_y - p_y)^2}$$
3. **State Transition & Visual Feedback:** Uncounted pedestrians are highlighted with **Amber** bounding boxes. Immediately upon crossing the virtual gate or confirming directional movement through the Smart Zone, the track state flips to `[COUNTED]`, the bounding box turns **Emerald Green**, and a timestamped entry/exit event is pushed to the audit log.

---

## 👨‍💻 Author

**Ayush Kumar Singh**  
*B.Tech in Computer Science & Engineering (Artificial Intelligence)*
