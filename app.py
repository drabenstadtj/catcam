import os
import time
import datetime
import logging
import subprocess
import threading
from collections import deque

import requests

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_from_directory
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
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
DISCORD_MAX_MB  = 25        # Discord free tier file size limit

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


def _discord_notify(message: str, file_path: str | None = None) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        if file_path and os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb <= DISCORD_MAX_MB:
                with open(file_path, "rb") as fh:
                    requests.post(
                        DISCORD_WEBHOOK,
                        data={"content": message},
                        files={"file": (os.path.basename(file_path), fh, "video/mp4")},
                        timeout=30,
                    )
                return
            else:
                message += f"\n_(clip too large to attach: {size_mb:.0f} MB)_"
        requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=10)
    except Exception as exc:
        log.warning("Discord notify failed: %s", exc)


def _encode_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Clip recording — writes H.264/faststart directly via FFmpeg pipe so clips
# are immediately playable in browsers and Discord without a remux step.
# ---------------------------------------------------------------------------
class ClipWriter:
    """Writes BGR frames to an H.264 MP4 with proper faststart metadata.

    Two-pass approach:
      1. Pipe raw frames into FFmpeg → H.264 .tmp file (no faststart yet,
         because FFmpeg can't seek a pipe-fed output to rewrite the header).
      2. On release(): stream-copy the .tmp into the final .mp4 with
         +faststart, then delete the .tmp.

    This guarantees Discord and browsers can seek the file and show duration.
    The .tmp file is never listed on the clips page so partial clips don't appear.
    """

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path                # final .mp4 (written by release())
        self._tmp  = path + ".tmp"      # intermediate H.264 file
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-r", str(STREAM_FPS),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            self._tmp,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc.poll() is not None:   # process already dead
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def release(self) -> bool:
        """Finalise the clip: flush FFmpeg, then apply faststart via stream copy.

        Returns True if the clip was successfully saved, False otherwise.
        """
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        returncode = None
        try:
            _, stderr_data = self._proc.communicate(timeout=30)
            returncode = self._proc.returncode
            if stderr_data:
                log.warning("ClipWriter FFmpeg stderr: %s", stderr_data.decode(errors="replace").strip())
        except Exception as e:
            log.warning("ClipWriter FFmpeg communicate() failed: %s", e)
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                pass

        # Stream-copy into final path with moov atom at the front.
        # No re-encoding — this completes in under a second for most clips.
        if not os.path.exists(self._tmp) or os.path.getsize(self._tmp) == 0:
            log.warning("Clip encoder produced no output (exit code %s) — skipping %s", returncode, self.path)
            if os.path.exists(self._tmp):
                os.remove(self._tmp)
            return False

        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", self._tmp,
             "-c", "copy", "-movflags", "+faststart",
             self.path],
            timeout=120,
        )
        if r.returncode == 0:
            os.remove(self._tmp)
        else:
            log.warning("Faststart pass failed — falling back to non-faststart clip")
            if os.path.exists(self._tmp):
                os.rename(self._tmp, self.path)
        return True


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

    clip_writer: ClipWriter | None = None
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

                # ---- state: cat visible on screen ----
                if detected and not prev_detected:
                    _update_state(cat_detected=True)
                elif not detected and prev_detected:
                    _update_state(cat_detected=False)

                prev_detected = detected

                # ---- clip recording / session logic ----
                # Always push to the compressed pre-roll buffer (unannotated
                # frames for cleaner pre-roll would require a second copy; the
                # annotated frame is fine since detection boxes show context).
                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                pre_buffer.append(jpeg_buf.tobytes())

                if detected:
                    # Extend (or start) the recording window each time the cat
                    # is visible — brief disappearances don't end the session.
                    record_until = time.time() + POST_ROLL
                    if clip_writer is None:
                        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        clip_path = os.path.join(CLIPS_DIR, f"cat_{ts_str}.mp4")
                        clip_writer = ClipWriter(clip_path)
                        # Flush pre-roll buffer so clip starts before detection
                        for jpeg_bytes in pre_buffer:
                            decoded = cv2.imdecode(
                                np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                            if decoded is not None:
                                clip_writer.write(decoded)
                        with _state_lock:
                            _state["count"] += 1
                            count_snap = _state["count"]
                        _log_event(f"Session {count_snap} started — clip: {os.path.basename(clip_path)}")
                        threading.Thread(
                            target=_discord_notify,
                            args=(f"🐱 Cat detected! (session #{count_snap})",),
                            daemon=True,
                        ).start()

                if clip_writer is not None:
                    clip_writer.write(frame)
                    if time.time() > record_until:
                        saved = clip_writer.release()
                        clip_writer = None
                        if saved:
                            _log_event(f"Session {count_snap} ended — saved: {os.path.basename(clip_path)}")
                            threading.Thread(
                                target=_discord_notify,
                                args=(f"Session #{count_snap} ended — clip attached", clip_path),
                                daemon=True,
                            ).start()
                        else:
                            _log_event(f"Session {count_snap} ended — clip not saved (encoder error)")

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


@app.route("/clips")
def clips_page():
    clips = []
    if os.path.exists(CLIPS_DIR):
        for fname in sorted(os.listdir(CLIPS_DIR), reverse=True):
            if fname.endswith(".mp4"):
                path = os.path.join(CLIPS_DIR, fname)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                # Parse timestamp from filename: cat_YYYYMMDD_HHMMSS.mp4
                try:
                    ts = datetime.datetime.strptime(fname, "cat_%Y%m%d_%H%M%S.mp4")
                    label = ts.strftime("%d %b %Y  %H:%M:%S")
                except ValueError:
                    label = fname
                clips.append({"name": fname, "label": label, "size_mb": f"{size_mb:.1f}"})
    return render_template("clips.html", clips=clips)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    filename = os.path.basename(filename)   # prevent path traversal
    return send_from_directory(CLIPS_DIR, filename, mimetype="video/mp4")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    threading.Thread(target=_stream_worker, daemon=True, name="stream-worker").start()
    socketio.run(app, host="0.0.0.0", port=8080, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
