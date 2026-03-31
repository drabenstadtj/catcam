import os
import time
import datetime
import logging
import subprocess
import threading
from collections import deque

import asyncio

import discord
import requests


import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
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
PRE_ROLL     = int(os.environ.get("PRE_ROLL",  30))  # seconds before first detection
POST_ROLL    = int(os.environ.get("POST_ROLL", 30))  # seconds after last detection
INFER_EVERY  = int(os.environ.get("INFER_EVERY", 3))    # run YOLO on every Nth frame
CONF_THRESH  = float(os.environ.get("CONF_THRESH", 0.15)) # detection confidence threshold
DISCORD_WEBHOOK    = os.environ.get("DISCORD_WEBHOOK_URL", "")
DISCORD_BOT_TOKEN  = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = "1486458529417138356"
HWACCEL            = os.environ.get("HWACCEL", "qsv")      # qsv | cpu
INFER_DEVICE       = os.environ.get("INFER_DEVICE", "CPU")  # CPU | GPU (OpenVINO)
DISCORD_MAX_MB     = 25        # Discord free tier file size limit

# Feedback emojis — order matches CATS in train_classifier.py: bonnie, jinny, louise
_FEEDBACK_EMOJIS  = [("🩶", "bonnie"), ("🟤", "jinny"), ("⚫", "louise")]
_RETRAIN_AFTER    = 5   # retrain after this many new labeled samples arrive

_pending_feedback: list[dict] = []   # {message_id, snap_bytes, predicted, ts}
_pending_lock     = threading.Lock()
_new_sample_count = 0

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


def _discord_notify(message: str, file_path: str | None = None,
                    jpeg_bytes: bytes | None = None, predicted: str | None = None) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        if jpeg_bytes:
            # Use ?wait=true so Discord returns the message object (we need the ID for reactions)
            url = DISCORD_WEBHOOK + ("&wait=true" if "?" in DISCORD_WEBHOOK else "?wait=true")
            resp = requests.post(
                url,
                data={"content": message},
                files={"file": ("snapshot.jpg", jpeg_bytes, "image/jpeg")},
                timeout=10,
            )
            if DISCORD_BOT_TOKEN and predicted and resp.ok:
                msg_id = resp.json().get("id")
                if msg_id:
                    _add_reactions(msg_id)
                    with _pending_lock:
                        _pending_feedback.append({
                            "message_id": msg_id,
                            "snap_bytes":  jpeg_bytes,
                            "predicted":   predicted,
                            "ts":          time.time(),
                        })
            return
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


_bot_loop: asyncio.AbstractEventLoop | None = None


class _FeedbackBot(discord.Client):
    async def on_ready(self):
        log.info("Discord feedback bot ready (user: %s)", self.user)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.user.id:
            return
        msg_id = str(payload.message_id)
        emoji  = str(payload.emoji)
        cat    = dict(_FEEDBACK_EMOJIS).get(emoji)
        if not cat:
            return
        with _pending_lock:
            match = next((i for i in _pending_feedback if i["message_id"] == msg_id), None)
            if match:
                _pending_feedback.remove(match)
            else:
                return
        _save_feedback_sample(match["snap_bytes"], cat, match["predicted"])
        global _new_sample_count
        _new_sample_count += 1
        if _new_sample_count >= _RETRAIN_AFTER:
            _new_sample_count = 0
            threading.Thread(target=_retrain_classifier, daemon=True).start()


_feedback_bot: _FeedbackBot | None = None


def _add_reactions(message_id: str) -> None:
    if _feedback_bot is None or _bot_loop is None:
        return

    async def _do():
        try:
            ch  = _feedback_bot.get_channel(int(DISCORD_CHANNEL_ID)) or \
                  await _feedback_bot.fetch_channel(int(DISCORD_CHANNEL_ID))
            msg = await ch.fetch_message(int(message_id))
            for emoji, _ in _FEEDBACK_EMOJIS:
                await msg.add_reaction(emoji)
        except Exception as exc:
            log.warning("Failed to add reactions: %s", exc)

    asyncio.run_coroutine_threadsafe(_do(), _bot_loop)


def _run_discord_bot() -> None:
    global _feedback_bot, _bot_loop
    intents = discord.Intents.default()
    intents.reactions = True
    _feedback_bot = _FeedbackBot(intents=intents)
    _bot_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_bot_loop)
    _bot_loop.run_until_complete(_feedback_bot.start(DISCORD_BOT_TOKEN))


def _save_feedback_sample(snap_bytes: bytes, cat: str, predicted: str) -> None:
    folder = os.path.join("cat images", cat)
    os.makedirs(folder, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(folder, f"feedback_{ts}.jpg")
    with open(path, "wb") as fh:
        fh.write(snap_bytes)
    action = "confirmed" if cat == predicted else f"corrected {predicted} → {cat}"
    log.info("Feedback: %s (%s) — saved to %s", cat, action, path)


def _retrain_classifier() -> None:
    log.info("Retraining classifier with new feedback samples...")
    try:
        subprocess.run(
            ["python", "train_classifier.py"],
            check=True, timeout=300,
        )
        _load_classifier()
        log.info("Classifier retrained and reloaded.")
    except Exception as exc:
        log.warning("Retrain failed: %s", exc)


CLASSIFIER_PATH = "cat_classifier.pt"
_classifier       = None
_classifier_classes: list[str] = []
_classifier_tf    = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _load_classifier() -> None:
    global _classifier, _classifier_classes
    if not os.path.exists(CLASSIFIER_PATH):
        log.info("No cat classifier found at %s — cat identity disabled. "
                 "Run train_classifier.py to enable it.", CLASSIFIER_PATH)
        return
    checkpoint = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
    _classifier_classes = checkpoint["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(_classifier_classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _classifier = model
    log.info("Cat classifier loaded — classes: %s", _classifier_classes)


def _identify_cat(roi: np.ndarray) -> str:
    if _classifier is None or roi.size == 0:
        return "cat"
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    tensor = _classifier_tf(img).unsqueeze(0)
    with torch.no_grad():
        idx = _classifier(tensor).argmax(1).item()
    return _classifier_classes[idx]


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
        self._proc = self._start_ffmpeg()

    def _cpu_cmd(self) -> list:
        return [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-r", str(STREAM_FPS),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            "-f", "mp4", self._tmp,
        ]

    def _start_ffmpeg(self) -> subprocess.Popen:
        if HWACCEL == "qsv":
            cmd = [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-init_hw_device", "qsv=hw",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                "-r", str(STREAM_FPS),
                "-i", "pipe:0",
                "-vf", "format=nv12,hwupload=extra_hw_frames=64",
                "-c:v", "h264_qsv", "-global_quality", "23",
                "-f", "mp4", self._tmp,
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            import time as _time; _time.sleep(0.2)
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode(errors="replace").strip()
                log.warning("QSV init failed, falling back to CPU encoding: %s", stderr)
                proc = subprocess.Popen(self._cpu_cmd(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc
        return subprocess.Popen(self._cpu_cmd(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)

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
            stderr_data = self._proc.stderr.read()
            returncode = self._proc.wait(timeout=30)
            if stderr_data:
                log.warning("ClipWriter FFmpeg stderr: %s", stderr_data.decode(errors="replace").strip())
        except Exception as e:
            log.warning("ClipWriter FFmpeg wait failed: %s", e)
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
            ["ffmpeg", "-y", "-loglevel", "warning",
             "-i", self._tmp,
             "-c", "copy", "-movflags", "+faststart",
             self.path],
            capture_output=True,
            timeout=120,
        )
        if r.returncode == 0:
            os.remove(self._tmp)
        else:
            log.warning("Faststart pass failed (exit %d): %s — falling back to non-faststart clip",
                        r.returncode, r.stderr.decode(errors="replace").strip())
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

    log.info("Loading YOLOv8 model (%s) on device=%s...", MODEL_PATH, INFER_DEVICE)
    if INFER_DEVICE == "GPU":
        ov_path = MODEL_PATH.replace(".pt", "_openvino_model")
        if not os.path.exists(ov_path):
            log.info("Exporting model to OpenVINO format...")
            YOLO(MODEL_PATH).export(format="openvino")
        model = YOLO(ov_path, task="detect")
    else:
        model = YOLO(MODEL_PATH)
    log.info("Model ready.")

    frame_bytes = FRAME_WIDTH * FRAME_HEIGHT * 3

    # Rolling pre-buffer: store JPEG bytes to keep memory reasonable.
    # At ~20 KB/frame, 30 s × 15 fps = 450 frames ≈ 9 MB.
    pre_buffer: deque[bytes] = deque(maxlen=PRE_ROLL * STREAM_FPS)

    clip_writer: ClipWriter | None = None
    record_until: float = 0.0   # epoch time — keep recording until this
    session_start: float = 0.0
    cat_votes: dict = {}        # identity → detection count this session

    while True:
        log.info("Starting FFmpeg — listening on udp://0.0.0.0:5000 ...")
        try:
            proc = subprocess.Popen(
                _ffmpeg_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            _update_state(stream_connected=True)
            socketio.emit("status", _get_state())

            prev_detected = False
            detected = False
            frame_count = 0

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

                # ---- inference (throttled) ----
                frame_count += 1
                if frame_count % INFER_EVERY == 0:
                    results = model(frame, classes=[CAT_CLASS_ID], conf=CONF_THRESH, verbose=False)
                    detected = False
                    for r in results:
                        for box in r.boxes:
                            detected = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            roi = frame[y1:y2, x1:x2]
                            identity = _identify_cat(roi)
                            cat_votes[identity] = cat_votes.get(identity, 0) + 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 60), 2)
                            cv2.putText(frame, f"{identity}  {conf:.0%}", (x1, max(y1 - 8, 18)),
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
                        session_start = time.time()
                        cat_votes = {}
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
                        _, snap_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        identity_snap = max(cat_votes, key=cat_votes.get) if cat_votes else "cat"
                        threading.Thread(
                            target=_discord_notify,
                            args=(f"#{count_snap} {identity_snap} at {datetime.datetime.now().strftime('%H:%M:%S')}",),
                            kwargs={"jpeg_bytes": snap_buf.tobytes(), "predicted": identity_snap},
                            daemon=True,
                        ).start()

                if clip_writer is not None:
                    clip_writer.write(frame)
                    if time.time() > record_until:
                        saved = clip_writer.release()
                        identity = max(cat_votes, key=cat_votes.get) if cat_votes else "cat"
                        clip_writer = None
                        cat_votes = {}
                        if saved:
                            _log_event(f"Session {count_snap} ended — saved: {os.path.basename(clip_path)}")
                            threading.Thread(
                                target=_discord_notify,
                                args=(f"#{count_snap} {identity} at {datetime.datetime.now().strftime('%H:%M:%S')}, {round(time.time() - session_start)}s", clip_path),
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
    _load_classifier()
    threading.Thread(target=_stream_worker,  daemon=True, name="stream-worker").start()
    if DISCORD_BOT_TOKEN:
        threading.Thread(target=_run_discord_bot, daemon=True, name="discord-bot").start()
    socketio.run(app, host="0.0.0.0", port=8080, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
