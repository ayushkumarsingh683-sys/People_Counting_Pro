import cv2
import math
from datetime import datetime
from ultralytics import YOLO

class PeopleCounterSystem:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.in_count = 0
        self.out_count = 0
        self.live_headcount = 0
        self.peak_crowd = 0
        # tid -> {"cx": int, "cy": int, "init_x": int, "init_y": int, "counted": bool, "seen": int, "missed": int}
        self.tracks = {}
        self.next_id = 1
        self.event_logs = []

    def process_frame(self, frame, line_y=None):
        frame = cv2.resize(frame, (860, 480))
        h, w, _ = frame.shape
        line_x = int(w * 0.5)
        zone_pad = 45  # 90px wide central trigger zone so crowded walkers are never missed

        results = self.model(frame, classes=[0], conf=0.25, verbose=False)
        detections = []

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, x1, y1, x2, y2))

        # Immediate frame count (never 0 when people are visible)
        self.live_headcount = len(detections)
        if self.live_headcount > self.peak_crowd:
            self.peak_crowd = self.live_headcount

        matched_tids = set()
        for cx, cy, x1, y1, x2, y2 in detections:
            best_id = None
            min_dist = 130

            for tid, tdata in self.tracks.items():
                if tid in matched_tids:
                    continue
                dist = math.hypot(cx - tdata["cx"], cy - tdata["cy"])
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "cx": cx, "cy": cy,
                    "init_x": cx, "init_y": cy,
                    "counted": False, "seen": 1, "missed": 0
                }
            else:
                t = self.tracks[best_id]
                t["cx"], t["cy"] = cx, cy
                t["seen"] += 1
                t["missed"] = 0

            matched_tids.add(best_id)
            t = self.tracks[best_id]

            # --- SMART ZONE + LINE CROSSING ENGINE ---
            if not t["counted"]:
                dx = cx - t["init_x"]
                dy = cy - t