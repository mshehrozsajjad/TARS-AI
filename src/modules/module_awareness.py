"""
module_awareness.py

24/7 ambient awareness system for TARS-AI.

Continuously captures frames from the camera to:
  - Detect and recognize faces (YuNet + SFace, on-device)
  - Track who is present (arrival/departure with hysteresis)
  - Periodically describe the scene via companion server BLIP

All context feeds into the LLM prompt so TARS always knows who's
in the room and what's happening — without being asked.

Face recognition shares the same known_faces.npz database as the
UI's FACE ID system, so faces trained in either place work in both.
"""

import os
import io
import time
import threading
import collections
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from modules.module_messageQue import queue_message
from modules.module_config import load_config

CONFIG = load_config()

# Global singleton
_awareness_instance = None


def get_awareness_manager():
    return _awareness_instance


# ── Model download (shared with module_ui_detections.py) ────────────────────

_ONNX_MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

# Paths — same as FaceRecognitionDetector in module_ui_detections.py
_MODELS_DIR = Path(__file__).parent / "UI" / "models"
_FACES_DIR = Path(__file__).parent.parent / "vision" / "faces"
_DB_FILE = _FACES_DIR / "known_faces.npz"


def _ensure_models():
    """Download ONNX models if missing."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in _ONNX_MODELS.items():
        dest = _MODELS_DIR / filename
        if not dest.exists():
            queue_message(f"AWARENESS: Downloading {filename}...")
            try:
                urlretrieve(url, str(dest))
                queue_message(f"AWARENESS: Downloaded {filename}")
            except Exception as e:
                queue_message(f"WARNING: Could not download {filename}: {e}")


# ── Headless Face Recognizer ────────────────────────────────────────────────

class HeadlessFaceRecognizer:
    """Face detection + recognition without pygame dependency.

    Uses OpenCV's YuNet (detection) and SFace (recognition) ONNX models.
    Shares the same known_faces.npz database as the UI's FaceRecognitionDetector.
    """

    COSINE_THRESHOLD = 0.363

    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.known_names = []
        self.known_embeddings = []
        self._lock = threading.Lock()

        try:
            import cv2
            _ensure_models()

            yunet_path = str(_MODELS_DIR / "face_detection_yunet_2023mar.onnx")
            sface_path = str(_MODELS_DIR / "face_recognition_sface_2021dec.onnx")

            if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
                queue_message("WARNING: Face recognition models not available")
                return

            self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
            self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
            self._load_database()
            queue_message(f"AWARENESS: Face recognizer loaded ({len(self.known_names)} known faces)")
        except ImportError:
            queue_message("WARNING: OpenCV (cv2) not available for face recognition")
        except Exception as e:
            queue_message(f"WARNING: Face recognizer init failed: {e}")

    def _load_database(self):
        """Load known faces from shared database."""
        _FACES_DIR.mkdir(parents=True, exist_ok=True)
        if _DB_FILE.exists():
            with self._lock:
                data = np.load(str(_DB_FILE), allow_pickle=True)
                self.known_names = data['names'].tolist()
                self.known_embeddings = [data[f'emb_{i}'] for i in range(len(self.known_names))]

    def reload_database(self):
        """Reload database from disk (picks up faces trained via UI)."""
        self._load_database()

    def detect_and_identify(self, frame_bgr):
        """Detect faces and identify them against known database.

        Args:
            frame_bgr: OpenCV BGR numpy array from camera.

        Returns:
            List of dicts: [{name, confidence, bbox}]
        """
        if self.detector is None or self.recognizer is None:
            return []

        import cv2

        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame_bgr)
        if faces is None:
            return []

        results = []
        with self._lock:
            for face in faces:
                bbox = face[:4].astype(int).tolist()
                aligned = self.recognizer.alignCrop(frame_bgr, face)
                embedding = self.recognizer.feature(aligned)

                name, score = self._identify(embedding)
                results.append({
                    "name": name,
                    "confidence": float(score),
                    "bbox": bbox,
                })

        return results

    def _identify(self, embedding):
        """Match embedding against known faces."""
        import cv2

        if not self.known_embeddings:
            return "UNKNOWN", 0.0

        best_name = "UNKNOWN"
        best_score = 0.0

        for name, known_emb in zip(self.known_names, self.known_embeddings):
            score = self.recognizer.match(
                embedding, known_emb, cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.COSINE_THRESHOLD:
            return "UNKNOWN", best_score

        return best_name, best_score


# ── Presence Tracker ────────────────────────────────────────────────────────

class PresenceTracker:
    """Tracks who is currently present with arrival/departure hysteresis."""

    ARRIVAL_THRESHOLD = 3     # detected in N out of last M frames to count as arrived
    ARRIVAL_WINDOW = 5        # last M frames to check
    ABSENCE_REENTRY_MIN = 600  # seconds absent before re-arrival triggers greeting (10 min)

    def __init__(self, departure_timeout=30):
        self.departure_timeout = departure_timeout
        # Per-person state: {name: {last_seen, first_seen, recent_hits: deque(bool), present: bool}}
        self._people = {}
        self._events = []  # [{type, name, time}]
        self._lock = threading.Lock()

    def update(self, detected_faces):
        """Update presence state based on detected faces.

        Args:
            detected_faces: List of {name, confidence, bbox} from recognizer.

        Returns:
            List of new events: [{type: "arrived"/"departed", name, time}]
        """
        now = time.time()
        seen_names = set()
        new_events = []

        with self._lock:
            # Mark who was seen this frame
            for face in detected_faces:
                name = face["name"]
                if name == "UNKNOWN":
                    continue
                seen_names.add(name)

                if name not in self._people:
                    self._people[name] = {
                        "last_seen": now,
                        "first_seen": now,
                        "recent_hits": collections.deque(maxlen=self.ARRIVAL_WINDOW),
                        "present": False,
                        "last_departed": 0,
                    }

                person = self._people[name]
                person["last_seen"] = now
                person["recent_hits"].append(True)

                # Check arrival: seen enough times in recent window
                if not person["present"]:
                    hits = sum(person["recent_hits"])
                    if hits >= self.ARRIVAL_THRESHOLD:
                        person["present"] = True
                        person["first_seen"] = now
                        absence = now - person.get("last_departed", 0)
                        event = {"type": "arrived", "name": name, "time": now}
                        # Only treat as notable arrival if absent for a while
                        if absence > self.ABSENCE_REENTRY_MIN:
                            event["notable"] = True
                        new_events.append(event)
                        self._events.append(event)

            # Mark misses for unseen people and check departures
            for name, person in list(self._people.items()):
                if name not in seen_names:
                    person["recent_hits"].append(False)

                    if person["present"] and (now - person["last_seen"]) > self.departure_timeout:
                        person["present"] = False
                        person["last_departed"] = now
                        event = {"type": "departed", "name": name, "time": now}
                        new_events.append(event)
                        self._events.append(event)

            # Trim event log to last 30 minutes
            cutoff = now - 1800
            self._events = [e for e in self._events if e["time"] > cutoff]

        return new_events

    def get_present(self):
        """Return list of currently present people with duration."""
        now = time.time()
        with self._lock:
            result = []
            for name, person in self._people.items():
                if person["present"]:
                    duration = now - person["first_seen"]
                    result.append({"name": name, "duration_seconds": duration})
            return result

    def get_recent_events(self, minutes=10):
        """Return recent arrival/departure events."""
        cutoff = time.time() - (minutes * 60)
        with self._lock:
            return [e for e in self._events if e["time"] > cutoff]


# ── Awareness Manager ───────────────────────────────────────────────────────

class AwarenessManager:
    """24/7 ambient awareness — face recognition + scene captioning."""

    def __init__(self, config, ui_manager=None):
        global _awareness_instance
        _awareness_instance = self

        self._config = config
        self._ui_manager = ui_manager
        self._running = False
        self._thread = None

        # Config
        awareness_cfg = config.get("AWARENESS", {})
        self._enabled = str(awareness_cfg.get("enabled", "false")).lower() == "true"
        self._face_interval = int(awareness_cfg.get("face_interval", 3))
        self._scene_interval = int(awareness_cfg.get("scene_interval", 60))
        self._server_url = awareness_cfg.get("server_url", "")
        self._proactive_greetings = str(awareness_cfg.get("proactive_greetings", "true")).lower() == "true"
        self._departure_timeout = int(awareness_cfg.get("departure_timeout", 30))

        # Components
        self._face_recognizer = None
        self._presence = PresenceTracker(departure_timeout=self._departure_timeout)
        self._scene_description = ""
        self._last_scene_time = 0
        self._camera = None

        # Quiet hours (reuse from drives config)
        drives_cfg = config.get("DRIVES", {})
        self._quiet_start = int(drives_cfg.get("quiet_start", 23))
        self._quiet_end = int(drives_cfg.get("quiet_end", 7))

        if self._enabled:
            self._face_recognizer = HeadlessFaceRecognizer()

    def start(self):
        """Start the background awareness thread."""
        if not self._enabled:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._awareness_loop,
            name="AwarenessThread",
            daemon=True,
        )
        self._thread.start()
        queue_message("LOAD: Awareness system started "
                      f"(face every {self._face_interval}s, scene every {self._scene_interval}s)")

    def stop(self):
        """Stop the awareness thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ── Main loop ───────────────────────────────────────────────────────

    def _awareness_loop(self):
        """Background loop with two cadences: face detection + scene captioning."""
        last_face_time = 0
        last_scene_time = 0

        while self._running:
            now = time.time()

            # Face detection cadence
            if now - last_face_time >= self._face_interval:
                last_face_time = now
                try:
                    self._do_face_detection()
                except Exception as e:
                    queue_message(f"WARNING: Awareness face detection failed: {e}")

            # Scene captioning cadence
            if self._server_url and now - last_scene_time >= self._scene_interval:
                last_scene_time = now
                try:
                    self._do_scene_captioning()
                except Exception as e:
                    queue_message(f"WARNING: Awareness scene captioning failed: {e}")

            time.sleep(0.5)

    def _capture_frame_bgr(self):
        """Capture a frame as a BGR numpy array via the CameraModule singleton.

        Uses capture_bytes() → JPEG decode, same path as the working /camera_feed endpoint.
        """
        import cv2

        try:
            from UI.module_ui_camera import CameraModule
            camera = CameraModule(640, 480)  # singleton — returns existing instance
            jpeg_bytes = camera.capture_bytes(timeout=2)
            if jpeg_bytes is None:
                return None
            jpg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _capture_jpeg(self, frame_bgr):
        """Encode a BGR frame as JPEG bytes."""
        import cv2
        _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()

    # ── Face detection ──────────────────────────────────────────────────

    def _do_face_detection(self):
        """Capture frame, detect/identify faces, update presence."""
        if self._face_recognizer is None:
            return

        frame = self._capture_frame_bgr()
        if frame is None:
            return

        # Periodically reload database to pick up faces trained via UI
        if int(time.time()) % 30 == 0:
            self._face_recognizer.reload_database()

        faces = self._face_recognizer.detect_and_identify(frame)

        # Update presence tracker
        events = self._presence.update(faces)

        # Handle events
        for event in events:
            if event["type"] == "arrived":
                queue_message(f"AWARENESS: {event['name']} arrived")
                # Proactive greeting for notable arrivals
                if event.get("notable") and self._proactive_greetings:
                    self._greet(event["name"])
                # Notify drives system
                self._notify_drives_presence()
            elif event["type"] == "departed":
                queue_message(f"AWARENESS: {event['name']} departed")
                self._notify_drives_presence()

    def _notify_drives_presence(self):
        """Notify the drives system about presence changes."""
        try:
            from modules.module_drives import get_drives_manager
            dm = get_drives_manager()
            if dm is not None and hasattr(dm, 'on_presence_change'):
                present = self._presence.get_present()
                dm.on_presence_change(present)
        except Exception:
            pass

    def _is_quiet_hours(self):
        """Check if current time is within quiet hours."""
        hour = datetime.now().hour
        if self._quiet_start > self._quiet_end:
            return hour >= self._quiet_start or hour < self._quiet_end
        else:
            return self._quiet_start <= hour < self._quiet_end

    def _greet(self, name):
        """Speak a proactive greeting when someone arrives."""
        if self._is_quiet_hours():
            return

        try:
            from modules.module_state import get_tars_state, TarsState
            if get_tars_state() != TarsState.STANDBY:
                return
        except Exception:
            return

        import random
        greetings = [
            f"Hey {name}, good to see you.",
            f"Oh, {name}. Welcome back.",
            f"{name}! I was wondering when you'd show up.",
            f"There you are, {name}.",
        ]
        line = random.choice(greetings)
        queue_message(f"AWARENESS: Greeting {name} — \"{line}\"")

        try:
            from modules.module_router import send
            send(line)
        except Exception as e:
            queue_message(f"WARNING: Awareness greeting failed: {e}")

    # ── Scene captioning ────────────────────────────────────────────────

    def _do_scene_captioning(self):
        """Capture frame and send to companion server for BLIP captioning."""
        frame = self._capture_frame_bgr()
        if frame is None:
            return

        jpeg = self._capture_jpeg(frame)

        try:
            import requests
            headers = {}
            api_key = os.environ.get('EXTERNAL_API_KEY', '')
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            files = {'image': ('frame.jpg', io.BytesIO(jpeg), 'image/jpeg')}
            response = requests.post(
                f"{self._server_url}/caption",
                files=files,
                headers=headers,
                timeout=15,
            )

            if response.status_code == 200:
                caption = response.json().get("caption", "")
                if caption:
                    self._scene_description = caption
                    self._last_scene_time = time.time()
            else:
                queue_message(f"WARNING: Scene caption server returned {response.status_code}")

        except requests.exceptions.ConnectionError:
            pass  # Server not running — silent failure
        except Exception as e:
            queue_message(f"WARNING: Scene captioning failed: {e}")

    # ── Public API ──────────────────────────────────────────────────────

    def get_present_people(self):
        """Return list of currently present people."""
        return self._presence.get_present()

    def get_scene_description(self):
        """Return the latest scene description."""
        return self._scene_description

    def get_awareness_context(self):
        """Build formatted context string for LLM prompt injection."""
        present = self._presence.get_present()
        events = self._presence.get_recent_events(minutes=10)
        scene = self._scene_description

        if not present and not scene and not events:
            return ""

        parts = []

        # People present
        if present:
            people_parts = []
            for p in present:
                duration = p["duration_seconds"]
                if duration < 60:
                    dur_str = "just arrived"
                elif duration < 3600:
                    dur_str = f"here for {int(duration / 60)} min"
                else:
                    dur_str = f"here for {duration / 3600:.1f} hours"
                people_parts.append(f"{p['name']} ({dur_str})")
            parts.append(f"People present: {', '.join(people_parts)}")

        # Scene description
        if scene:
            parts.append(f"Scene: {scene}")

        # Recent events
        arrival_names = [e["name"] for e in events if e["type"] == "arrived"]
        departure_names = [e["name"] for e in events if e["type"] == "departed"]
        if arrival_names:
            parts.append(f"Recently arrived: {', '.join(arrival_names)}")
        if departure_names:
            parts.append(f"Recently left: {', '.join(departure_names)}")

        if not parts:
            return ""

        return "[ENVIRONMENT AWARENESS] " + ". ".join(parts) + "."
