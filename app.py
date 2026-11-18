import cv2
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from tracker import PeopleCounterSystem

app = FastAPI()

counter = PeopleCounterSystem()
VIDEO_PATH = "cctv.mp4"

@app.get("/stats")
def get_stats():
    return {
        "in_count": counter.in_count,
        "out_count": counter.out_count,
        "live_headcount": counter.live_headcount,
        "peak_crowd": counter.peak_crowd,
        "logs": counter.event_logs
    }

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            counter.tracks.clear()
            continue

        processed_frame, in_c, out_c = counter.process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        time.sleep(0.01)  # Allows /stats endpoint to respond instantly
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")