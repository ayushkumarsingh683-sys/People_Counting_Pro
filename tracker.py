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

    def reset_stats(self):
        self.in_count = 0
        self.out_count = 0
        self.live_headcount = 0
        self.peak_crowd = 0
        self.tracks.clear()
        self.next_id = 1
        self.event_logs.clear()

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

        # Draw Smart-Zone shaded overlay & center trigger line
        overlay = frame.copy()
        cv2.rectangle(overlay, (line_x - zone_pad, 0), (line_x + zone_pad, h), (255, 191, 0), -1)
        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 215, 255), 2)
        cv2.line(frame, (line_x - zone_pad, 0), (line_x - zone_pad, h), (200, 150, 0), 1)
        cv2.line(frame, (line_x + zone_pad, 0), (line_x + zone_pad, h), (200, 150, 0), 1)

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
                    "cx": cx,
                    "cy": cy,
                    "init_x": cx,
                    "init_y": cy,
                    "counted": False,
                    "seen": 1,
                    "missed": 0,
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
                dy = cy - t["init_y"]
                in_trigger_zone = (line_x - zone_pad) <= cx <= (line_x + zone_pad)
                crossed_left_to_right = t["init_x"] < line_x and cx >= line_x
                crossed_right_to_left = t["init_x"] > line_x and cx <= line_x

                if t["seen"] >= 2 and (in_trigger_zone or crossed_left_to_right or crossed_right_to_left):
                    if dx > 12 or crossed_left_to_right:
                        self.in_count += 1
                        t["counted"] = True
                        self.event_logs.insert(0, {
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "Track ID": f"ID-{best_id}",
                            "Direction": "IN",
                            "Live Headcount": self.live_headcount,
                        })
                    elif dx < -12 or crossed_right_to_left:
                        self.out_count += 1
                        t["counted"] = True
                        self.event_logs.insert(0, {
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "Track ID": f"ID-{best_id}",
                            "Direction": "OUT",
                            "Live Headcount": self.live_headcount,
                        })

            # Keep latest 50 event logs in memory
            if len(self.event_logs) > 50:
                self.event_logs = self.event_logs[:50]

            # --- VISUAL BOUNDING BOX & STATUS OVERLAY ---
            if t["counted"]:
                box_color = (80, 200, 120)  # Emerald Green
                label = f"ID-{best_id} [COUNTED]"
            else:
                box_color = (0, 191, 255)   # Amber
                label = f"ID-{best_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (cx, cy), 4, box_color, -1)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
            )

        # --- OCCLUSION MEMORY CLEANUP (Retain IDs for up to 12 missed frames) ---
        stale_ids = []
        for tid, tdata in self.tracks.items():
            if tid not in matched_tids:
                tdata["missed"] += 1
                if tdata["missed"] > 12:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]

        return frame