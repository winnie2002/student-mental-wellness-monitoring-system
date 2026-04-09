"""
Student Wellness Tracker — VS Code / Tkinter Desktop App
─────────────────────────────────────────────────────────
Features:
  • Batch enrollment  — add many students before any scanning
  • Photo capture     — one reference photo per student (for face matching)
  • Class scan        — single camera session detects & identifies enrolled faces
  • Auto face-match   — LBPH face recogniser maps detections → student records
  • Dashboard         — sorted lowest wellness first, colour-coded status
  • Individual report — full session history + sparkline trend per student
  • Persistent store  — students.json + face_db/ folder
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import json
import datetime
import threading
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
from collections import defaultdict

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tensorflow as tf                                              # noqa: F401
    from tensorflow.keras.models import load_model as keras_load_model  # type: ignore[attr-defined]
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras_load_model = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_FILE = "students.json"
FACE_DIR  = "face_db"

BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_INPUT  = "#21262d"
ACCENT    = "#58a6ff"
ACCENT2   = "#3fb950"
ACCENT3   = "#f78166"
ACCENT4   = "#d2a8ff"
ACCENT5   = "#e3b341"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
BORDER    = "#30363d"

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

WELLNESS_MAP: dict = {
    "Happy":    1.0,
    "Surprise": 0.7,
    "Neutral":  0.6,
    "Fear":     0.3,
    "Sad":      0.2,
    "Disgust":  0.1,
    "Angry":    0.0,
}

FACE_URL  = ("https://raw.githubusercontent.com/opencv/opencv/master/"
             "data/haarcascades/haarcascade_frontalface_default.xml")
MODEL_URL = ("https://raw.githubusercontent.com/oarriaga/face_classification/"
             "master/trained_models/emotion_models/"
             "fer2013_mini_XCEPTION.102-0.66.hdf5")


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(FACE_DIR, exist_ok=True)


def load_students() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_students(db: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2)


def face_image_path(sid: str) -> str:
    """Return path for the NEXT photo slot for this student (numbered)."""
    idx = 0
    while os.path.exists(os.path.join(FACE_DIR, f"{sid}_{idx}.jpg")):
        idx += 1
    return os.path.join(FACE_DIR, f"{sid}_{idx}.jpg")


def face_image_count(sid: str) -> int:
    """How many reference photos does this student already have?"""
    return sum(
        1 for f in os.listdir(FACE_DIR)
        if f.startswith(f"{sid}_") and f.endswith(".jpg")
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

class ModelLoader:
    face_cascade  = None
    emotion_model = None
    face_recogniser = None   # LBPH recogniser
    label_map: dict = {}     # int label → sid
    ready: bool  = False
    error: str   = ""

    @classmethod
    def load(cls, callback=None) -> None:
        def _worker() -> None:
            if not TF_AVAILABLE:
                cls.error = "TensorFlow not installed — run: pip install tensorflow"
                if callback:
                    callback()
                return
            try:
                # Haar cascade
                if not os.path.exists("haarcascade_frontalface_default.xml"):
                    urllib.request.urlretrieve(FACE_URL,
                                               "haarcascade_frontalface_default.xml")
                cls.face_cascade = cv2.CascadeClassifier(
                    "haarcascade_frontalface_default.xml"
                )

                # Emotion model
                if not os.path.exists("emotion_model.hdf5"):
                    urllib.request.urlretrieve(MODEL_URL, "emotion_model.hdf5")
                cls.emotion_model = keras_load_model(   # type: ignore[misc]
                    "emotion_model.hdf5", compile=False
                )

                # Build face recogniser from stored photos
                cls._rebuild_recogniser()
                cls.ready = True

            except Exception as exc:  # noqa: BLE001
                cls.error = str(exc)

            if callback:
                callback()

        threading.Thread(target=_worker, daemon=True).start()

    @classmethod
    def _rebuild_recogniser(cls) -> None:
        """Re-train LBPH from all photos currently in face_db/."""
        recogniser = cv2.face.LBPHFaceRecognizer_create()
        images, labels = [], []
        cls.label_map = {}
        label_idx = 0

        # Group files by sid (files are named sid_0.jpg, sid_1.jpg, ...)
        sid_files: dict = {}
        for fname in os.listdir(FACE_DIR):
            if not fname.endswith(".jpg"):
                continue
            # expect format:  SID_N.jpg
            parts = fname[:-4].rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            sid_key = parts[0]
            sid_files.setdefault(sid_key, []).append(
                os.path.join(FACE_DIR, fname)
            )

        for sid_key, paths in sid_files.items():
            for path in paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (100, 100))
                # Augment each photo
                for variant in _augment(img):
                    images.append(variant)
                    labels.append(label_idx)
            cls.label_map[label_idx] = sid_key
            label_idx += 1

        if images:
            recogniser.train(images, np.array(labels))
            cls.face_recogniser = recogniser
        else:
            cls.face_recogniser = None

    @classmethod
    def rebuild_recogniser_sync(cls) -> None:
        """Call from main thread after a new photo is saved."""
        threading.Thread(target=cls._rebuild_recogniser, daemon=True).start()


def _augment(gray: np.ndarray) -> list:
    """Return augmented variants.  All are histogram-equalised to match
    what identify_face() does at prediction time."""
    eq = cv2.equalizeHist(gray)
    variants = [eq, cv2.flip(eq, 1)]
    for alpha in (0.75, 0.9, 1.1, 1.25):
        adj = np.clip(eq.astype("float32") * alpha, 0, 255).astype("uint8")
        variants.append(adj)
    return variants


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_faces(frame: np.ndarray):
    """Return list of (x,y,w,h) bounding boxes."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = ModelLoader.face_cascade.detectMultiScale(gray, 1.3, 5)
    return gray, list(faces) if len(faces) else []


def identify_face(gray: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Return sid of best match or '' if unknown / no recogniser.

    LBPH confidence: lower = better match.
    We use histogram equalisation + a generous threshold (115) because
    lighting in a classroom varies a lot.  Multiple training photos per
    student (captured at enrolment) bring confidence scores down significantly.
    """
    if ModelLoader.face_recogniser is None:
        return ""
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (100, 100))
    roi = cv2.equalizeHist(roi)          # normalise lighting
    label, confidence = ModelLoader.face_recogniser.predict(roi)
    if confidence < 115:                 # generous but not too loose
        return ModelLoader.label_map.get(label, "")
    return ""


def predict_emotion(gray: np.ndarray, x: int, y: int, w: int, h: int):
    """Return (emotion_str, confidence_float) for a face ROI."""
    if not ModelLoader.ready or ModelLoader.emotion_model is None:
        return None, None
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))
    preds = ModelLoader.emotion_model.predict(roi, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return EMOTION_LABELS[idx], float(preds[idx])


def wellness_score(emotions: list) -> float:
    if not emotions:
        return 0.0
    return round(
        sum(WELLNESS_MAP.get(e, 0.5) for e in emotions) / len(emotions) * 100, 1
    )


def wellness_label(score: float):
    if score >= 80: return "THRIVING",       ACCENT2
    if score >= 60: return "GOOD",           ACCENT
    if score >= 40: return "OKAY",           ACCENT5
    return "NEEDS ATTENTION",                ACCENT3


# ══════════════════════════════════════════════════════════════════════════════
#  REUSABLE WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text: str, command=None,
                 bg: str = ACCENT, fg: str = BG_DARK,
                 width: int = 160, height: int = 38, radius: int = 10, **kw):
        try:
            parent_bg = parent.cget("bg")
        except tk.TclError:
            parent_bg = BG_DARK
        super().__init__(parent, width=width, height=height,
                         bg=parent_bg, highlightthickness=0, **kw)
        self._btn_bg = bg
        self._btn_fg = fg
        self._label  = text
        self._cmd    = command
        self._radius = radius
        self._bw, self._bh = width, height
        self._drawn  = False
        self.bind("<Map>",      self._on_map)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)

    def _on_map(self, _=None):
        if not self._drawn:
            self._drawn = True
            self._paint()

    def _paint(self, fill: str = ""):
        if not self.winfo_exists():
            return
        self.delete("all")
        c = fill or self._btn_bg
        r, w, h = self._radius, self._bw, self._bh
        self.create_arc(0,     0,     r*2,   r*2,   start=90,  extent=90,  fill=c, outline=c)
        self.create_arc(w-r*2, 0,     w,     r*2,   start=0,   extent=90,  fill=c, outline=c)
        self.create_arc(0,     h-r*2, r*2,   h,     start=180, extent=90,  fill=c, outline=c)
        self.create_arc(w-r*2, h-r*2, w,     h,     start=270, extent=90,  fill=c, outline=c)
        self.create_rectangle(r, 0, w-r, h,  fill=c, outline=c)
        self.create_rectangle(0, r, w,   h-r, fill=c, outline=c)
        self.create_text(w//2, h//2, text=self._label,
                         fill=self._btn_fg, font=("Consolas", 10, "bold"))

    def _on_click(self, _=None):
        if self._cmd: self._cmd()

    def _on_enter(self, _=None):
        if self._drawn: self._paint(fill=self._lighten(self._btn_bg))

    def _on_leave(self, _=None):
        if self._drawn: self._paint()

    def update_text(self, new_text: str):
        self._label = new_text
        if self._drawn: self._paint()

    @staticmethod
    def _lighten(hex_color: str) -> str:
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"


class StatusBar(tk.Label):
    def __init__(self, parent):
        super().__init__(parent, text="● Initialising models…",
                         fg=TEXT_SEC, bg=BG_DARK,
                         font=("Consolas", 9), anchor="w", padx=12)

    def set(self, msg: str, color: str = TEXT_SEC):
        self.config(text=msg, fg=color)


def card_frame(parent, **kw) -> tk.Frame:
    return tk.Frame(parent, bg=BG_CARD,
                    highlightbackground=BORDER, highlightthickness=1, **kw)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — HOME
# ══════════════════════════════════════════════════════════════════════════════

class HomePage(tk.Frame):
    def __init__(self, parent, switch_cb):
        super().__init__(parent, bg=BG_DARK)
        self._switch = switch_cb
        self._build()

    def _build(self):
        hero = tk.Frame(self, bg=BG_DARK)
        hero.pack(expand=True)

        tk.Label(hero, text="◈", fg=ACCENT, bg=BG_DARK,
                 font=("Consolas", 60)).pack(pady=(50, 0))
        tk.Label(hero, text="Student Wellness Tracker",
                 fg=TEXT_PRI, bg=BG_DARK,
                 font=("Georgia", 26, "bold")).pack()
        tk.Label(hero, text="Emotion-aware classroom intelligence",
                 fg=TEXT_SEC, bg=BG_DARK,
                 font=("Consolas", 11)).pack(pady=(4, 28))

        cards = tk.Frame(hero, bg=BG_DARK)
        cards.pack(pady=8)
        for icon, title, sub in [
            ("👥", "Batch Enroll",      "Register many students at once"),
            ("📸", "Reference Photo",   "One face photo per student"),
            ("🎥", "Class Scan",        "Identify all faces in one session"),
            ("📊", "Smart Dashboard",   "Sorted by wellness, lowest first"),
            ("📋", "Student Reports",   "Full history & trend per student"),
        ]:
            c = card_frame(cards, width=148, height=112)
            c.pack(side="left", padx=6)
            c.pack_propagate(False)
            tk.Label(c, text=icon,  bg=BG_CARD, font=("Arial", 20)).pack(pady=(12, 2))
            tk.Label(c, text=title, fg=TEXT_PRI, bg=BG_CARD,
                     font=("Consolas", 9, "bold")).pack()
            tk.Label(c, text=sub,   fg=TEXT_SEC, bg=BG_CARD,
                     font=("Consolas", 8), wraplength=130, justify="center").pack()

        row = tk.Frame(hero, bg=BG_DARK)
        row.pack(pady=28)
        RoundedButton(row, "Enroll Students",
                      command=lambda: self._switch("enroll"),
                      bg=ACCENT, width=180, height=40).pack(side="left", padx=8)
        RoundedButton(row, "Start Class Scan",
                      command=lambda: self._switch("scan"),
                      bg=ACCENT2, width=180, height=40).pack(side="left", padx=8)
        RoundedButton(row, "Dashboard",
                      command=lambda: self._switch("dashboard"),
                      bg=ACCENT4, fg=BG_DARK, width=140, height=40).pack(side="left", padx=8)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — BATCH ENROLL
# ══════════════════════════════════════════════════════════════════════════════

class EnrollPage(tk.Frame):
    """
    Batch enrollment:
      1. Fill form → Add to Queue
      2. For each queued student → Capture Photo (live camera)
      3. Save All → writes to students.json + face_db/
    """

    def __init__(self, parent, switch_cb, status_bar: StatusBar):
        super().__init__(parent, bg=BG_DARK)
        self._switch      = switch_cb
        self._status      = status_bar
        self._db: dict    = load_students()
        self._queue: list = []          # list of dicts before saving
        self._cap         = None
        self._running     = False
        self._after_id    = None
        self._photo_sid   = ""          # sid currently being photographed
        self._entries: dict = {}
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # ── top header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(hdr, text="Batch Enroll Students", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Georgia", 16, "bold")).pack(side="left")
        RoundedButton(hdr, "◀ Home",
                      command=lambda: self._switch("home"),
                      bg=BG_INPUT, fg=TEXT_PRI, width=90, height=28).pack(side="right")

        # ── main body
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # Left — form + queue
        left = tk.Frame(body, bg=BG_DARK, width=310)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Form card
        form = card_frame(left)
        form.pack(fill="x")
        tk.Label(form, text="Student Details", fg=TEXT_PRI, bg=BG_CARD,
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))

        for label, key in [("Full Name", "name"), ("Student ID", "sid"),
                            ("Class / Section", "cls")]:
            tk.Label(form, text=label, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Consolas", 9)).pack(anchor="w", padx=12, pady=(8, 0))
            e = tk.Entry(form, bg=BG_INPUT, fg=TEXT_PRI,
                         insertbackground=TEXT_PRI, relief="flat",
                         font=("Consolas", 11), bd=4)
            e.pack(fill="x", padx=12, pady=(2, 0))
            self._entries[key] = e

        btn_r = tk.Frame(form, bg=BG_CARD)
        btn_r.pack(fill="x", padx=12, pady=10)
        RoundedButton(btn_r, "+ Add to Queue",
                      command=self._add_to_queue,
                      bg=ACCENT, width=150, height=34).pack(side="left")
        RoundedButton(btn_r, "Clear",
                      command=self._clear_form,
                      bg=BG_INPUT, fg=TEXT_PRI, width=70, height=34).pack(side="right")

        # Queue list
        tk.Label(left, text="Enrollment Queue", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Consolas", 10, "bold")).pack(anchor="w", pady=(14, 4))
        q_wrap = card_frame(left)
        q_wrap.pack(fill="both", expand=True)
        self._queue_box = tk.Listbox(q_wrap, bg=BG_CARD, fg=TEXT_PRI,
                                      selectbackground=ACCENT, selectforeground=BG_DARK,
                                      relief="flat", font=("Consolas", 10),
                                      activestyle="none", bd=0)
        self._queue_box.pack(fill="both", expand=True, padx=4, pady=4)

        q_btns = tk.Frame(left, bg=BG_DARK)
        q_btns.pack(fill="x", pady=6)
        RoundedButton(q_btns, "Remove Selected",
                      command=self._remove_from_queue,
                      bg=ACCENT3, fg=BG_DARK, width=150, height=30).pack(side="left")
        self._save_btn = RoundedButton(q_btns, "Save All →",
                                        command=self._save_all,
                                        bg=ACCENT2, width=110, height=30)
        self._save_btn.pack(side="right")

        # Right — photo capture
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="Capture Reference Photo", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Georgia", 14, "bold")).pack(anchor="w")
        tk.Label(right,
                 text="Select a student from the queue, then capture their face photo.",
                 fg=TEXT_SEC, bg=BG_DARK, font=("Consolas", 9)).pack(anchor="w", pady=(2, 8))

        self._cam_label = tk.Label(right, bg=BG_CARD,
                                    highlightbackground=BORDER, highlightthickness=1)
        self._cam_label.pack(fill="both", expand=True)
        tk.Label(self._cam_label, text="Camera feed appears here",
                 fg=TEXT_SEC, bg=BG_CARD,
                 font=("Consolas", 11)).place(relx=.5, rely=.5, anchor="center")

        self._photo_status = tk.Label(right, text="No student selected",
                                       fg=TEXT_SEC, bg=BG_DARK,
                                       font=("Consolas", 9))
        self._photo_status.pack(anchor="w", pady=(4, 0))

        # Photo count progress
        self._photo_count_label = tk.Label(
            right,
            text="Photos captured: 0 / 8  (capture at least 5 for good accuracy)",
            fg=TEXT_SEC, bg=BG_DARK, font=("Consolas", 8)
        )
        self._photo_count_label.pack(anchor="w")

        cam_row = tk.Frame(right, bg=BG_DARK)
        cam_row.pack(pady=6)
        RoundedButton(cam_row, "📷  Open Camera",
                      command=self._open_camera,
                      bg=ACCENT, width=150, height=34).pack(side="left", padx=4)
        self._snap_btn = RoundedButton(cam_row, "✔  Capture Photo",
                                        command=self._capture_photo,
                                        bg=ACCENT2, width=160, height=34)
        self._snap_btn.pack(side="left", padx=4)
        RoundedButton(cam_row, "■  Stop Camera",
                      command=self._stop_camera,
                      bg=BG_INPUT, fg=TEXT_PRI, width=130, height=34).pack(side="left", padx=4)

        # Already enrolled
        tk.Label(right, text="Already Enrolled", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Consolas", 10, "bold")).pack(anchor="w", pady=(12, 2))
        enrolled_wrap = card_frame(right, height=100)
        enrolled_wrap.pack(fill="x")
        enrolled_wrap.pack_propagate(False)
        self._enrolled_box = tk.Listbox(enrolled_wrap, bg=BG_CARD, fg=TEXT_SEC,
                                         relief="flat", font=("Consolas", 9),
                                         activestyle="none", bd=0)
        self._enrolled_box.pack(fill="both", expand=True, padx=4, pady=2)
        self._refresh_enrolled()

    # ── queue helpers ─────────────────────────────────────────────────────────

    def _add_to_queue(self):
        name = self._entries["name"].get().strip()
        sid  = self._entries["sid"].get().strip()
        cls  = self._entries["cls"].get().strip()
        if not name or not sid:
            messagebox.showerror("Missing", "Name and Student ID are required.")
            return
        if sid in self._db:
            if not messagebox.askyesno("Exists",
                                        f"ID '{sid}' already enrolled. Update info?"):
                return
        # Check if already in queue
        for q in self._queue:
            if q["sid"] == sid:
                messagebox.showinfo("Duplicate", f"'{sid}' is already in the queue.")
                return
        self._queue.append({"name": name, "sid": sid, "class": cls, "photo": False})
        self._refresh_queue()
        self._clear_form()

    def _clear_form(self):
        for e in self._entries.values():
            e.delete(0, "end")

    def _remove_from_queue(self):
        sel = self._queue_box.curselection()
        if not sel:
            return
        del self._queue[sel[0]]
        self._refresh_queue()

    def _refresh_queue(self):
        self._queue_box.delete(0, "end")
        for q in self._queue:
            mark = "✔" if q["photo"] else "○"
            self._queue_box.insert("end", f"{mark}  {q['name']}  [{q['sid']}]")

    def _refresh_enrolled(self):
        self._enrolled_box.delete(0, "end")
        for sid, info in self._db.items():
            self._enrolled_box.insert("end", f"{info['name']}  [{sid}]")

    def _save_all(self):
        if not self._queue:
            messagebox.showinfo("Empty", "No students in the queue.")
            return
        no_photo = [q for q in self._queue if not q["photo"]]
        if no_photo:
            names = ", ".join(q["name"] for q in no_photo)
            if not messagebox.askyesno("Missing Photos",
                                        f"These students have no photo: {names}\n"
                                        "Save anyway? (Face recognition won't work for them.)"):
                return
        for q in self._queue:
            sid = q["sid"]
            if sid not in self._db:
                self._db[sid] = {"name": q["name"], "class": q["class"], "sessions": []}
            else:
                self._db[sid]["name"]  = q["name"]
                self._db[sid]["class"] = q["class"]
        save_students(self._db)
        ModelLoader.rebuild_recogniser_sync()
        saved = len(self._queue)
        self._queue.clear()
        self._refresh_queue()
        self._refresh_enrolled()
        self._status.set(f"✔ {saved} student(s) saved.", ACCENT2)
        messagebox.showinfo("Saved", f"{saved} student(s) enrolled successfully!")

    # ── camera / photo capture ────────────────────────────────────────────────

    def _selected_queue_student(self):
        sel = self._queue_box.curselection()
        if not sel:
            messagebox.showinfo("Select", "Select a student from the queue first.")
            return None
        return self._queue[sel[0]]

    def _open_camera(self):
        stu = self._selected_queue_student()
        if not stu:
            return
        self._photo_sid = stu["sid"]
        self._photo_status.config(
            text=f"Capturing photos for: {stu['name']}  [{stu['sid']}]",
            fg=ACCENT
        )
        self._update_photo_count()
        if self._running:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera", "Cannot open webcam.")
            return
        self._running = True
        self._tick()

    def _update_photo_count(self):
        if not self._photo_sid:
            return
        n = face_image_count(self._photo_sid)
        color = ACCENT2 if n >= 5 else ACCENT5 if n >= 3 else TEXT_SEC
        self._photo_count_label.config(
            text=f"Photos captured: {n} / 8  "
                 f"({'✔ Good accuracy' if n >= 5 else 'capture more for better accuracy'})",
            fg=color
        )

    def _tick(self):
        if not self._running or self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            # Draw face box preview
            if ModelLoader.face_cascade is not None:
                gray, faces = detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (88, 166, 255), 2)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            lw    = max(self._cam_label.winfo_width(),  320)
            lh    = max(self._cam_label.winfo_height(), 240)
            img.thumbnail((lw, lh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._cam_label.config(image=photo)
            self._cam_label.image = photo
            self._cam_label._frame = frame  # store latest for capture

        self._after_id = self.after(30, self._tick)

    def _capture_photo(self):
        if not self._running:
            messagebox.showinfo("Camera", "Open the camera first.")
            return
        if not self._photo_sid:
            messagebox.showinfo("Select", "Select a student in the queue first.")
            return
        frame = getattr(self._cam_label, "_frame", None)
        if frame is None:
            return

        if ModelLoader.face_cascade is not None:
            gray, faces = detect_faces(frame)
            if not faces:
                messagebox.showwarning("No Face",
                                       "No face detected — adjust your position and try again.")
                return
            x, y, w, h = faces[0]
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            # Save raw (not equalised) — _augment handles equalisation at train time
            path = face_image_path(self._photo_sid)
            cv2.imwrite(path, face_roi)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(face_image_path(self._photo_sid), gray)

        # Count how many photos now
        n = face_image_count(self._photo_sid)

        # Mark photo done in queue (first photo sets the flag)
        for q in self._queue:
            if q["sid"] == self._photo_sid:
                q["photo"] = True
        self._refresh_queue()
        self._update_photo_count()

        hint = "  (move slightly, capture more angles!)" if n < 5 else "  ✔ Enough for good accuracy"
        self._photo_status.config(
            text=f"Photo #{n} saved for [{self._photo_sid}]{hint}",
            fg=ACCENT2 if n >= 5 else ACCENT5
        )
        # Retrain incrementally in background after each new photo
        ModelLoader.rebuild_recogniser_sync()

    def _stop_camera(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_hide(self):
        self._stop_camera()

    def on_show(self):
        self._db = load_students()
        self._refresh_enrolled()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — CLASS SCAN
# ══════════════════════════════════════════════════════════════════════════════

class ScanPage(tk.Frame):
    """
    Single camera session — detects faces, identifies enrolled students,
    records emotion per identified student.  Unknown faces shown as '?'.
    After stopping, each identified student gets a session record saved.
    """

    def __init__(self, parent, switch_cb, status_bar: StatusBar):
        super().__init__(parent, bg=BG_DARK)
        self._switch = switch_cb
        self._status = status_bar
        self._cap    = None
        self._running = False
        self._after_id = None
        # { sid: [emotion, emotion, ...] } accumulated during session
        self._session_emotions: dict = defaultdict(list)
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(hdr, text="Class Scan", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Georgia", 16, "bold")).pack(side="left")
        RoundedButton(hdr, "◀ Home",
                      command=lambda: self._switch("home"),
                      bg=BG_INPUT, fg=TEXT_PRI, width=90, height=28).pack(side="right")

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # Camera area
        self._cam_label = tk.Label(body, bg=BG_CARD,
                                    highlightbackground=BORDER, highlightthickness=1)
        self._cam_label.pack(side="left", fill="both", expand=True)
        tk.Label(self._cam_label, text="Click 'Start Scan' to begin",
                 fg=TEXT_SEC, bg=BG_CARD,
                 font=("Consolas", 11)).place(relx=.5, rely=.5, anchor="center")

        # Right side — live detections panel
        right = tk.Frame(body, bg=BG_DARK, width=280)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Live Detections", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Consolas", 11, "bold")).pack(anchor="w", pady=(0, 6))

        det_wrap = card_frame(right)
        det_wrap.pack(fill="both", expand=True)
        self._det_text = tk.Text(det_wrap, bg=BG_CARD, fg=TEXT_PRI,
                                  relief="flat", font=("Consolas", 9),
                                  state="disabled", wrap="word")
        self._det_text.pack(fill="both", expand=True, padx=4, pady=4)

        tk.Label(right, text="Session Summary", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Consolas", 11, "bold")).pack(anchor="w", pady=(10, 4))
        summ_wrap = card_frame(right, height=140)
        summ_wrap.pack(fill="x")
        summ_wrap.pack_propagate(False)
        self._summ_text = tk.Text(summ_wrap, bg=BG_CARD, fg=TEXT_PRI,
                                   relief="flat", font=("Consolas", 9),
                                   state="disabled", wrap="word")
        self._summ_text.pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = tk.Frame(self, bg=BG_DARK)
        ctrl.pack(pady=8)
        RoundedButton(ctrl, "▶  Start Scan",
                      command=self._start,
                      bg=ACCENT2, width=140, height=36).pack(side="left", padx=6)
        RoundedButton(ctrl, "■  Stop & Save",
                      command=self._stop_and_save,
                      bg=ACCENT3, fg=BG_DARK, width=140, height=36).pack(side="left", padx=6)
        RoundedButton(ctrl, "Dashboard →",
                      command=lambda: self._switch("dashboard"),
                      bg=ACCENT4, fg=BG_DARK, width=140, height=36).pack(side="left", padx=6)

    # ── logic ─────────────────────────────────────────────────────────────────

    def _start(self):
        if not ModelLoader.ready:
            messagebox.showwarning("Not Ready", "Models still loading — please wait.")
            return
        if self._running:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera", "Cannot open webcam.")
            return
        self._running = True
        self._session_emotions.clear()
        self._status.set("● Class scan running…", ACCENT2)
        self._tick()

    def _tick(self):
        if not self._running or self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            gray, faces = detect_faces(frame)
            det_lines = []

            for (x, y, w, h) in faces:
                sid           = identify_face(gray, x, y, w, h)
                emotion, conf = predict_emotion(gray, x, y, w, h)

                if sid:
                    db    = load_students()
                    sname = db.get(sid, {}).get("name", sid)
                    label = f"{sname}"
                    color = (63, 185, 80)   # green
                    if emotion:
                        self._session_emotions[sid].append(emotion)
                        det_lines.append(f"  {sname}: {emotion} {conf*100:.0f}%")
                else:
                    label = "?"
                    color = (247, 129, 102)   # orange-red
                    if emotion:
                        det_lines.append(f"  Unknown: {emotion}")

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if emotion:
                    cv2.putText(frame, emotion, (x, y+h+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 168, 255), 1)

            # Update live detections
            self._set_text(self._det_text,
                           "\n".join(det_lines) if det_lines else "  Searching for faces…")

            # Update summary
            self._update_summary()

            # Show frame
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            lw    = max(self._cam_label.winfo_width(),  400)
            lh    = max(self._cam_label.winfo_height(), 300)
            img.thumbnail((lw, lh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._cam_label.config(image=photo)
            self._cam_label.image = photo

        self._after_id = self.after(30, self._tick)

    def _update_summary(self):
        lines = []
        for sid, emotions in self._session_emotions.items():
            ws        = wellness_score(emotions)
            lbl, _col = wellness_label(ws)
            db        = load_students()
            name      = db.get(sid, {}).get("name", sid)
            lines.append(f"  {name}: {ws}%  [{lbl}]  ({len(emotions)} frames)")
        self._set_text(self._summ_text,
                       "\n".join(lines) if lines else "  No students identified yet.")

    def _stop_and_save(self):
        if not self._running:
            return
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None

        if not self._session_emotions:
            self._status.set("● Scan stopped — no students were identified.", TEXT_SEC)
            messagebox.showinfo("No Data",
                                "No enrolled students were identified during this scan.")
            return

        db = load_students()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        saved_names = []
        for sid, emotions in self._session_emotions.items():
            if sid not in db:
                continue
            ws          = wellness_score(emotions)
            lbl, _color = wellness_label(ws)
            session = {
                "timestamp": ts,
                "emotions":  emotions,
                "wellness":  ws,
                "status":    lbl,
            }
            db[sid]["sessions"].append(session)
            saved_names.append(f"{db[sid]['name']} ({ws}%)")

        save_students(db)
        self._status.set(
            f"✔ Session saved for {len(saved_names)} student(s): "
            + ", ".join(saved_names), ACCENT2
        )
        messagebox.showinfo("Session Saved",
                            f"Session recorded for {len(saved_names)} student(s):\n"
                            + "\n".join(saved_names))
        self._session_emotions.clear()
        self._set_text(self._summ_text, "  Session saved.")

    @staticmethod
    def _set_text(widget: tk.Text, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_hide(self):
        if self._running:
            self._stop_and_save()

    def on_show(self):
        self._session_emotions.clear()
        self._set_text(self._det_text, "  Waiting to start…")
        self._set_text(self._summ_text, "  No session yet.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

class DashboardPage(tk.Frame):
    def __init__(self, parent, switch_cb):
        super().__init__(parent, bg=BG_DARK)
        self._switch   = switch_cb
        self._db: dict = {}
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=16, pady=12)
        tk.Label(hdr, text="Class Dashboard", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Georgia", 18, "bold")).pack(side="left")
        RoundedButton(hdr, "◀ Home",
                      command=lambda: self._switch("home"),
                      bg=BG_INPUT, fg=TEXT_PRI, width=90, height=30).pack(side="right", padx=4)
        RoundedButton(hdr, "Start Scan",
                      command=lambda: self._switch("scan"),
                      bg=ACCENT2, width=110, height=30).pack(side="right", padx=4)
        RoundedButton(hdr, "Enroll",
                      command=lambda: self._switch("enroll"),
                      bg=ACCENT, width=90, height=30).pack(side="right", padx=4)
        tk.Label(hdr, text="Sorted: lowest wellness first",
                 fg=TEXT_SEC, bg=BG_DARK,
                 font=("Consolas", 8)).pack(side="right", padx=12)

        # Summary strip
        self._summary = card_frame(self)
        self._summary.pack(fill="x", padx=16, pady=4)

        # Table
        tbl = tk.Frame(self, bg=BG_DARK)
        tbl.pack(fill="both", expand=True, padx=16, pady=4)

        cols   = ("Name", "ID", "Class", "Sessions",
                  "Last Scan", "Last Wellness", "Avg Wellness", "Status")
        widths = (145,     80,   90,      70,
                  130,      100,           105,            130)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                         background=BG_CARD, foreground=TEXT_PRI,
                         fieldbackground=BG_CARD, rowheight=26,
                         font=("Consolas", 10))
        style.configure("Dark.Treeview.Heading",
                         background=BG_INPUT, foreground=ACCENT,
                         font=("Consolas", 10, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG_DARK)])

        self._tree = ttk.Treeview(tbl, columns=cols,
                                   show="headings", style="Dark.Treeview")
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(tbl, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<Double-1>", self._open_report)

        tk.Label(self, text="Double-click a student to open their Individual Report",
                 fg=TEXT_SEC, bg=BG_DARK, font=("Consolas", 8)).pack(pady=(2, 4))

    def on_show(self):
        self._db = load_students()
        self._populate()

    def _populate(self):
        for row in self._tree.get_children():
            self._tree.delete(row)

        total = len(self._db)
        thriving = good = concern = 0

        # Build rows with sort key = last wellness (ascending — lowest first)
        rows = []
        for sid, info in self._db.items():
            sessions = info.get("sessions", [])
            n = len(sessions)
            if sessions:
                last    = sessions[-1]
                ts, ws, st = last["timestamp"], last["wellness"], last["status"]
                avg_ws  = round(sum(s["wellness"] for s in sessions) / n, 1)
            else:
                ts, ws, st, avg_ws = "—", 999, "—", 999   # unscanned goes to end

            rows.append((ws, sid, info, n, ts, ws, avg_ws, st))

            if st == "THRIVING":          thriving += 1
            elif st in ("GOOD", "OKAY"):  good     += 1
            elif st == "NEEDS ATTENTION": concern  += 1

        rows.sort(key=lambda r: r[0])   # sort by last wellness ascending

        for (sort_key, sid, info, n, ts, ws, avg_ws, st) in rows:
            tag = st.lower().replace(" ", "_") if st != "—" else "none"
            disp_ws  = f"{ws}%"  if ws  != 999 else "—"
            disp_avg = f"{avg_ws}%" if avg_ws != 999 else "—"
            self._tree.insert("", "end", iid=sid,
                               values=(info["name"], sid, info.get("class", ""),
                                       n, ts, disp_ws, disp_avg, st),
                               tags=(tag,))

        self._tree.tag_configure("thriving",        foreground=ACCENT2)
        self._tree.tag_configure("good",            foreground=ACCENT)
        self._tree.tag_configure("okay",            foreground=ACCENT5)
        self._tree.tag_configure("needs_attention", foreground=ACCENT3)

        # Summary strip
        for w in self._summary.winfo_children():
            w.destroy()
        for label, val, color in [
            ("Total Students", total,    TEXT_PRI),
            ("Thriving",       thriving, ACCENT2),
            ("Good / Okay",    good,     ACCENT),
            ("Need Attention", concern,  ACCENT3),
        ]:
            cell = tk.Frame(self._summary, bg=BG_CARD)
            cell.pack(side="left", expand=True, pady=8)
            tk.Label(cell, text=str(val), fg=color, bg=BG_CARD,
                     font=("Georgia", 20, "bold")).pack()
            tk.Label(cell, text=label, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Consolas", 8)).pack()

    def _open_report(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        # iid was set to the actual sid string when inserting — always reliable
        sid = sel[0]
        try:
            ReportWindow(self, sid)
        except Exception as exc:
            import traceback
            messagebox.showerror(
                "Report Error",
                f"Could not open report for '{sid}':\n\n{traceback.format_exc()}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  POPUP — INDIVIDUAL STUDENT REPORT
# ══════════════════════════════════════════════════════════════════════════════

class ReportWindow(tk.Toplevel):
    """
    Individual student report popup.
    Always reloads from disk on open — guaranteed to show latest sessions.
    """

    def __init__(self, parent, sid: str):
        super().__init__(parent)
        self._sid = sid

        # ── always reload fresh from disk ─────────────────────────────────────
        db         = load_students()
        self._info = db.get(sid, {})
        self._sessions = self._info.get("sessions", [])

        name = self._info.get("name", sid)
        self.title(f"Report — {name}")
        self.geometry("900x660")
        self.minsize(700, 500)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        try:
            self._build()
        except Exception as exc:
            import traceback
            tk.Label(self, text=f"Error building report:\n{traceback.format_exc()}",
                     fg=ACCENT3, bg=BG_DARK, justify="left",
                     font=("Consolas", 9), wraplength=860).pack(padx=16, pady=16)
            return

        # Draw sparkline after window is fully rendered
        self.after(100, self._draw_sparkline)

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self):
        sessions = self._sessions
        name     = self._info.get("name", self._sid)
        cls      = self._info.get("class", "—")
        n        = len(sessions)

        # Title bar
        hdr = tk.Frame(self, bg=BG_CARD,
                       highlightbackground=BORDER, highlightthickness=1, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  {name}", fg=TEXT_PRI, bg=BG_CARD,
                 font=("Georgia", 15, "bold")).pack(side="left", padx=10)
        tk.Label(hdr, text=f"ID: {self._sid}     Class: {cls}",
                 fg=TEXT_SEC, bg=BG_CARD,
                 font=("Consolas", 10)).pack(side="left", padx=4)
        RoundedButton(hdr, "↺  Refresh",
                      command=self._refresh,
                      bg=BG_INPUT, fg=TEXT_PRI,
                      width=100, height=28).pack(side="right", padx=10)

        # Summary stat cards
        if n > 0:
            scores   = [s["wellness"] for s in sessions]
            avg_ws   = round(sum(scores) / n, 1)
            last_ws  = scores[-1]
            best_ws  = max(scores)
            worst_ws = min(scores)
            lbl, col = wellness_label(last_ws)
        else:
            scores = []
            avg_ws = best_ws = worst_ws = last_ws = 0.0
            lbl, col = "NO DATA", TEXT_SEC

        stat_row = tk.Frame(self, bg=BG_DARK)
        stat_row.pack(fill="x", padx=16, pady=(10, 6))
        for label, val, color in [
            ("Sessions",   str(n),          TEXT_PRI),
            ("Last Score", f"{last_ws}%",   col),
            ("Average",    f"{avg_ws}%",    ACCENT),
            ("Best",       f"{best_ws}%",   ACCENT2),
            ("Worst",      f"{worst_ws}%",  ACCENT3),
            ("Status",     lbl,             col),
        ]:
            c = card_frame(stat_row, width=120, height=60)
            c.pack(side="left", padx=4)
            c.pack_propagate(False)
            tk.Label(c, text=val,   fg=color,   bg=BG_CARD,
                     font=("Georgia", 14, "bold")).pack(pady=(8, 0))
            tk.Label(c, text=label, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Consolas", 8)).pack()

        # Sparkline canvas — drawn after window maps via self.after(100,...)
        spark_outer = card_frame(self)
        spark_outer.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(spark_outer, text="Wellness Trend  (oldest → newest)",
                 fg=TEXT_PRI, bg=BG_CARD,
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=(6, 2))
        self._spark = tk.Canvas(spark_outer, bg=BG_CARD, height=90,
                                highlightthickness=0)
        self._spark.pack(fill="x", padx=12, pady=(0, 8))
        # also redraw on resize
        self._spark.bind("<Configure>", lambda _: self._draw_sparkline())

        # Session history label
        tk.Label(self, text="Session History  (newest first)",
                 fg=TEXT_PRI, bg=BG_DARK,
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=16, pady=(2, 2))

        # Session table
        tbl_frame = tk.Frame(self, bg=BG_DARK)
        tbl_frame.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        # Use a unique style name per window to avoid collision
        style_name = f"Rep{id(self)}.Treeview"
        st = ttk.Style()
        st.configure(style_name,
                     background=BG_CARD, foreground=TEXT_PRI,
                     fieldbackground=BG_CARD, rowheight=24,
                     font=("Consolas", 9))
        st.configure(f"{style_name}.Heading",
                     background=BG_INPUT, foreground=ACCENT,
                     font=("Consolas", 9, "bold"), relief="flat")
        st.map(style_name,
               background=[("selected", ACCENT)],
               foreground=[("selected", BG_DARK)])

        cols   = ("#", "Date & Time", "Wellness", "Status", "Emotion Breakdown", "Frames")
        widths = (36,   145,           76,          128,      310,                  55)

        self._tree = ttk.Treeview(tbl_frame, columns=cols,
                                   show="headings", style=style_name)
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            anchor = "w" if col == "Emotion Breakdown" else "center"
            self._tree.column(col, width=w, anchor=anchor)

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",
                             command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        self._tree.tag_configure("thriving",        foreground=ACCENT2)
        self._tree.tag_configure("good",            foreground=ACCENT)
        self._tree.tag_configure("okay",            foreground=ACCENT5)
        self._tree.tag_configure("needs_attention", foreground=ACCENT3)

        self._populate_table()

        # Footer
        foot = tk.Frame(self, bg=BG_DARK)
        foot.pack(fill="x", padx=16, pady=(0, 8))
        RoundedButton(foot, "Export as TXT",
                      command=self._export,
                      bg=BG_INPUT, fg=TEXT_PRI,
                      width=150, height=28).pack(side="right")

    def _populate_table(self):
        """Fill the session table from self._sessions."""
        for row in self._tree.get_children():
            self._tree.delete(row)
        for i, s in enumerate(reversed(self._sessions), 1):
            emotions = s.get("emotions", [])
            counts   = {e: emotions.count(e) for e in EMOTION_LABELS
                        if emotions.count(e) > 0}
            emo_str  = "  ".join(f"{e}×{c}" for e, c in counts.items())
            tag      = s["status"].lower().replace(" ", "_")
            self._tree.insert("", "end",
                               values=(i, s["timestamp"],
                                       f"{s['wellness']}%",
                                       s["status"],
                                       emo_str,
                                       len(emotions)),
                               tags=(tag,))

    def _refresh(self):
        """Reload from disk and repopulate everything."""
        db             = load_students()
        self._info     = db.get(self._sid, {})
        self._sessions = self._info.get("sessions", [])
        # Rebuild the whole window
        for widget in self.winfo_children():
            widget.destroy()
        self._build()
        self.after(100, self._draw_sparkline)

    # ── sparkline ─────────────────────────────────────────────────────────────

    def _draw_sparkline(self):
        canvas = self._spark
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        sessions = self._sessions

        if not sessions:
            canvas.create_text(
                canvas.winfo_width() // 2, 45,
                text="No session data yet",
                fill=TEXT_SEC, font=("Consolas", 10)
            )
            return

        scores = [s["wellness"] for s in sessions]
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 20 or H < 20:
            return

        PAD_X, PAD_Y = 30, 12

        def sx(i):
            if len(scores) == 1:
                return W // 2
            return int(PAD_X + i * (W - 2 * PAD_X) / (len(scores) - 1))

        def sy(v):
            return int(PAD_Y + (100.0 - v) / 100.0 * (H - 2 * PAD_Y))

        # Reference lines + labels
        for threshold, band_label in [(80, "80"), (60, "60"), (40, "40")]:
            yy = sy(threshold)
            canvas.create_line(PAD_X, yy, W - PAD_X, yy,
                                fill=BORDER, dash=(4, 4))
            canvas.create_text(PAD_X - 4, yy, text=band_label,
                                fill=TEXT_SEC, anchor="e",
                                font=("Consolas", 7))

        # Line connecting dots
        pts = [(sx(i), sy(v)) for i, v in enumerate(scores)]
        if len(pts) > 1:
            canvas.create_line(
                *[c for pt in pts for c in pt],
                fill=ACCENT, width=2, smooth=True
            )

        # Coloured dots
        for i, (px, py) in enumerate(pts):
            _, dot_col = wellness_label(scores[i])
            canvas.create_oval(px - 5, py - 5, px + 5, py + 5,
                                fill=dot_col, outline=dot_col)
            # Score label above dot
            canvas.create_text(px, py - 10, text=f"{scores[i]:.0f}",
                                fill=dot_col, font=("Consolas", 7))

    # ── export ────────────────────────────────────────────────────────────────

    def _export(self):
        # Re-read from disk for export too
        db       = load_students()
        info     = db.get(self._sid, self._info)
        sessions = info.get("sessions", [])
        name     = info.get("name", self._sid)

        fname = f"report_{self._sid}_{datetime.datetime.now():%Y%m%d_%H%M}.txt"
        lines = [
            "Student Wellness Report",
            "=" * 50,
            f"Name     : {name}",
            f"ID       : {self._sid}",
            f"Class    : {info.get('class', '—')}",
            f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
            "=" * 50,
            "",
        ]
        if sessions:
            scores = [s["wellness"] for s in sessions]
            lines += [
                f"Total Sessions : {len(sessions)}",
                f"Last Wellness  : {scores[-1]}%",
                f"Average        : {round(sum(scores) / len(scores), 1)}%",
                f"Best           : {max(scores)}%",
                f"Worst          : {min(scores)}%",
                "",
                "Session Log (newest first):",
                "-" * 50,
            ]
            for i, s in enumerate(reversed(sessions), 1):
                emotions = s.get("emotions", [])
                counts   = {e: emotions.count(e) for e in EMOTION_LABELS
                            if emotions.count(e) > 0}
                emo_str  = ", ".join(f"{e}x{c}" for e, c in counts.items())
                lines.append(
                    f"#{i:02d}  {s['timestamp']}  |  "
                    f"{s['wellness']}%  [{s['status']}]  |  {emo_str}"
                )
        else:
            lines.append("No sessions recorded yet.")

        try:
            with open(fname, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            messagebox.showinfo("Exported", f"Report saved as:\n{fname}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Wellness Tracker")
        self.geometry("1180x720")
        self.minsize(960, 620)
        self.configure(bg=BG_DARK)

        self._pages: dict  = {}
        self._current: str = ""

        self._build_nav()
        self._build_pages()
        self._build_status()
        self._show("home")

        ModelLoader.load(callback=self._on_models_ready)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_nav(self):
        nav = tk.Frame(self, bg=BG_CARD, height=46,
                       highlightbackground=BORDER, highlightthickness=1)
        nav.pack(fill="x")
        nav.pack_propagate(False)
        tk.Label(nav, text="◈ WellnessTracker", fg=ACCENT, bg=BG_CARD,
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16)

        for label, page in [("Home",         "home"),
                             ("Enroll",       "enroll"),
                             ("Class Scan",   "scan"),
                             ("Dashboard",    "dashboard")]:
            tk.Button(nav, text=label,
                      command=lambda p=page: self._show(p),
                      bg=BG_CARD, fg=TEXT_SEC, relief="flat",
                      font=("Consolas", 10), bd=0, padx=14,
                      activebackground=BG_DARK, activeforeground=ACCENT,
                      cursor="hand2").pack(side="left")

    def _build_pages(self):
        self._container = tk.Frame(self, bg=BG_DARK)
        self._container.pack(fill="both", expand=True)
        self._status_bar = StatusBar(self)

        self._pages["home"]      = HomePage(self._container, self._show)
        self._pages["enroll"]    = EnrollPage(self._container, self._show,
                                               self._status_bar)
        self._pages["scan"]      = ScanPage(self._container, self._show,
                                             self._status_bar)
        self._pages["dashboard"] = DashboardPage(self._container, self._show)

        for page in self._pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _build_status(self):
        self._status_bar.pack(fill="x", side="bottom")

    def _show(self, name: str):
        if self._current:
            cur = self._pages.get(self._current)
            if cur and hasattr(cur, "on_hide"):
                cur.on_hide()
        self._current = name
        page = self._pages[name]
        page.lift()
        if hasattr(page, "on_show"):
            page.on_show()

    def _on_models_ready(self):
        self.after(0, self._update_status)

    def _update_status(self):
        if ModelLoader.ready:
            self._status_bar.set("✔ Models loaded — ready to scan.", ACCENT2)
        else:
            self._status_bar.set(f"✘ Model error: {ModelLoader.error}", ACCENT3)

    def _on_close(self):
        for page in self._pages.values():
            if hasattr(page, "on_hide"):
                page.on_hide()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
