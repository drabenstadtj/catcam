import os
import time
import datetime
import logging
import subprocess
import threading
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
STREAM_FPS   = 15           # expected incoming FPS (used for pre-buffer sizing)
CAT_CLASS_ID = 15           # COCO class 15 = cat
MODEL_PATH   = "yolov8n.pt"
LOG_FILE     = "/data/detections.log"
CLIPS_DIR    = "/data/clips"
PRE_ROLL     = 30           # seconds of footage saved before first detection
POST_ROLL    = 30           # seconds of footage saved after last detection

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_frame_lock = threading.Lock()
_latest_frame: np.ndarray | None = None

_state_lock = threading.Lock()
_state = {
    "cat_detected": False,
    "count": 0,
    "stream_connected": False,
}


def _get_state() -> dict:
    with _state_lock:
        return dict(_state)


def _update_state(**kwargs) -> dict:
    with _state_lock:
        _state.update(kwargs)
        return dict(_state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _placeholder_frame() -> np.ndarray:
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for stream...", (80, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    return img


def _log_event(msg: str) -> None:
    os.makedirs("/data", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}\n"
    with open(LOG_FILE, "a") as fh:
        fh.write(entry)
    log.info(entry.strip())


def _encode_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Clip recording
# ---------------------------------------------------------------------------
def _new_clip_writer(path: str) -> cv2.VideoWriter:
    os.makedirs(CLIPS_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, float(STREAM_FPS), (FRAME_WIDTH, FRAME_HEIGHT))


# ---------------------------------------------------------------------------
# Background stream + inference thread
# ---------------------------------------------------------------------------
def _ffmpeg_cmd() -> list[str]:
    return [
        "ffmpeg", "-loglevel", "error",
        "-i", "udp://0.0.0.0:5000",
        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


def _stream_worker() -> None:
    global _latest_frame

    log.info("Loading YOLOv8 model (%s)...", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    log.info("Model ready.")

    frame_bytes = FRAME_WIDTH * FRAME_HEIGHT * 3

    # Rolling pre-buffer: store JPEG bytes to keep memory reasonable.
    # At ~20 KB/frame, 30 s × 15 fps = 450 frames ≈ 9 MB.
    pre_buffer: deque[bytes] = deque(maxlen=PRE_ROLL * STREAM_FPS)

    clip_writer: cv2.VideoWriter | None = None
    record_until: float = 0.0   # epoch time — keep recording until this

    while True:
        log.info("Starting FFmpeg — listening on udp://0.0.0.0:5000 ...")
        try:
            proc = subprocess.Popen(
                _ffmpeg_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            _update_state(stream_connected=True)
            socketio.emit("status", _get_state())

            prev_detected = False

            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    log.warning("Short read (%d bytes) — stream ended.", len(raw))
                    break

                frame = (
                    np.frombuffer(raw, dtype=np.uint8)
                    .reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    .copy()
                )

                # ---- inference ----
                results = model(frame, classes=[CAT_CLASS_ID], verbose=False)
                detected = False
                for r in results:
                    for box in r.boxes:
                        detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 60), 2)
                        cv2.putText(frame, f"cat  {conf:.0%}", (x1, max(y1 - 8, 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 60), 2, cv2.LINE_AA)

                # ---- state transitions ----
                if detected and not prev_detected:
                    with _state_lock:
                        _state["count"] += 1
                        _state["cat_detected"] = True
                        count_snap = _state["count"]
                    _log_event(f"Cat detected (session total: {count_snap})")
                    socketio.emit("status", _get_state())
                elif not detected and prev_detected:
                    _update_state(cat_detected=False)
                    socketio.emit("status", _get_state())

                prev_detected = detected

                # ---- clip recording ----
                # Always keep a compressed pre-roll buffer
                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                pre_buffer.append(jpeg_buf.tobytes())

                if detected:
                    # Extend (or start) the recording window
                    record_until = time.time() + POST_ROLL
                    if clip_writer is None:
                        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        clip_path = os.path.join(CLIPS_DIR, f"cat_{ts_str}.mp4")
                        clip_writer = _new_clip_writer(clip_path)
                        # Flush the pre-roll buffer first
                        for jpeg_bytes in pre_buffer:
                            decoded = cv2.imdecode(
                                np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                            if decoded is not None:
                                clip_writer.write(decoded)
                        log.info("Clip started: %s", clip_path)

                if clip_writer is not None:
                    clip_writer.write(frame)
                    if time.time() > record_until:
                        clip_writer.release()
                        clip_writer = None
                        _log_event(f"Clip saved: {clip_path}")
                        log.info("Clip saved: %s", clip_path)

                with _frame_lock:
                    _latest_frame = frame

            proc.wait()

        except Exception as exc:
            log.error("Stream error: %s", exc)
        finally:
            if clip_writer is not None:
                clip_writer.release()
                clip_writer = None
            _update_state(stream_connected=False, cat_detected=False)
            socketio.emit("status", _get_state())

        log.info("Reconnecting in 3 s...")
        time.sleep(3)


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def _mjpeg_stream():
    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else _placeholder_frame()
        jpeg = _encode_jpeg(frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(_get_state())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    threading.Thread(target=_stream_worker, daemon=True, name="stream-worker").start()
    socketio.run(app, host="0.0.0.0", port=8080, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
