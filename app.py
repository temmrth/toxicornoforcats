import os
import json
import sqlite3
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

load_model = None
try:
    from tensorflow.keras.models import load_model as _lm
    load_model = _lm
except Exception:
    try:
        from keras.models import load_model as _lm2
        load_model = _lm2
    except Exception:
        load_model = None

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key_change_me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "model_classes.json")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("toxic_cat_app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = RotatingFileHandler(LOG_PATH, maxBytes=512_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

logger.info("App booting…")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

_last_cleanup_ts = 0.0

# delete old uploads
def cleanup_uploads(max_age_hours: int = 24) -> int:
    """Delete uploaded images older than max_age_hours. Returns number deleted."""
    deleted = 0
    try:
        cutoff = time.time() - (max_age_hours * 3600)
        for name in os.listdir(UPLOAD_DIR):
            path = os.path.join(UPLOAD_DIR, name)
            if not os.path.isfile(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    deleted += 1
            except Exception:
                logger.exception("Failed deleting upload: %s", path)
        if deleted:
            logger.info("Cleanup uploads: deleted=%s (older than %sh)", deleted, max_age_hours)
    except Exception:
        logger.exception("Cleanup uploads failed")
    return deleted

cleanup_uploads()

MODEL = None
CLASSES = ["aloe_vera", "lily", "monstera"]

TOX = {
    "aloe_vera": "Toxic",
    "lily": "Toxic",
    "monstera": "Toxic",
}

# get utc time
def now() -> str:
    return datetime.utcnow().isoformat()

# open db connection
def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

# create db tables
def init_db() -> None:
    con = get_db()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS plants(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            is_toxic INTEGER NOT NULL CHECK(is_toxic IN (0,1)),
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            method TEXT NOT NULL,
            result_label TEXT NOT NULL,
            confidence INTEGER,
            image_filename TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    con.commit()
    con.close()

# check file type
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# load class names
def load_classes() -> None:
    """Loads class order used by the trained model, if a json exists."""
    global CLASSES
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                CLASSES = data
        except Exception:
            pass

# load ml model
def load_ml() -> None:
    global MODEL
    if MODEL is not None:
        return

    load_classes()

    if load_model is None:
        MODEL = None
        return

    if os.path.exists(MODEL_PATH):
        try:
            MODEL = load_model(MODEL_PATH)
        except Exception:
            MODEL = None
    else:
        MODEL = None

# save check history
def add_history(username: str, query: str, method: str, label: str, conf=None, img=None) -> None:
    con = get_db()
    con.execute(
        """
        INSERT INTO history(username, query, method, result_label, confidence, image_filename, created_at)
        VALUES(?,?,?,?,?,?,?)
        """,
        (username, query, method, label, conf, img, now()),
    )
    con.commit()
    con.close()

# read recent checks
def get_recent(username: str, limit: int = 6):
    if not username:
        return []

    con = get_db()
    rows = con.execute(
        """
        SELECT query, result_label, created_at
        FROM history
        WHERE username=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (username, limit),
    ).fetchall()
    con.close()

    return [
        {"query": r["query"], "label": r["result_label"], "created_at": r["created_at"]}
        for r in rows
    ]

# lookup plant label
def text_check(name: str) -> str:
    plant = (name or "").strip().lower()
    if not plant:
        return "Unknown"

    con = get_db()
    row = con.execute(
        "SELECT is_toxic FROM plants WHERE LOWER(name)=? LIMIT 1",
        (plant,),
    ).fetchone()
    con.close()

    if row is None:
        return "Unknown"
    return "Toxic" if int(row["is_toxic"]) == 1 else "Non-toxic"

# predict from image
def predict_image(path: str):
    """Return (class_name, confidence_percent, label)."""
    load_ml()
    if MODEL is None:
        return None, None, "Unknown"

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        logger.exception("Image open failed: %s", path)
        return None, None, "Unknown"

    try:
        img = img.resize((224, 224))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = MODEL.predict(arr, verbose=0)
        idx = int(np.argmax(pred))
        conf = int(round(float(np.max(pred)) * 100))

        cls = CLASSES[idx] if idx < len(CLASSES) else "unknown"
        label = TOX.get(cls, "Unknown")
        return cls, conf, label
    except Exception:
        logger.exception("predict_image failed")
        return None, None, "Unknown"

init_db()

# run before request
@app.before_request
# run quick cleanup
def _periodic_maintenance():
    """Lightweight periodic maintenance."""
    global _last_cleanup_ts
    now_ts = time.time()
    if now_ts - _last_cleanup_ts > 600:
        _last_cleanup_ts = now_ts
        cleanup_uploads(max_age_hours=24)

# web endpoint
@app.route("/")
# render home page
def index():
    user = session.get("user")
    history = get_recent(user, 6) if user else []

    plant = session.pop("last_plant", None)
    label = session.pop("last_label", None)
    conf = session.pop("last_conf", None)

    result_text = session.pop("last_result", None)
    result_label = session.pop("last_result_label", None)

    auth_error = session.pop("auth_error", None)

    if plant and label and result_text is None:
        if conf is None:
            result_text = f"{plant} — {label}"
        else:
            result_text = f"{plant} — {label} ({conf}%)"
        result_label = label

    return render_template(
        "index.html",
        history=history,
        plant=plant,
        label=label,
        conf=conf,
        result_text=result_text,
        result_label=result_label,
        auth_error=auth_error,
        user=user,
    )

# web endpoint
@app.route("/check", methods=["POST"])
# handle text check
def check():
    plant_name = (request.form.get("plant_name") or "").strip()
    if not plant_name:
        return redirect("/")

    label = text_check(plant_name)

    session["last_plant"] = plant_name
    session["last_label"] = label
    session["last_conf"] = None
    session["last_result"] = f"{plant_name} — {label}"
    session["last_result_label"] = label

    user = session.get("user")
    if user:
        add_history(user, plant_name, "text", label)

    return redirect("/")

# web endpoint
@app.route("/upload", methods=["POST"])
# handle image check
def upload():
    user = session.get("user")

    f = request.files.get("file")
    filename0 = f.filename if f else None
    logger.info('Image upload: user=%s filename=%s', session.get('user'), filename0)
    if not filename0:
        session["last_result"] = "No file selected"
        session["last_result_label"] = "Unknown"
        return redirect("/")

    if not allowed_file(filename0):
        session["last_result"] = "Bad file type"
        session["last_result_label"] = "Unknown"
        return redirect("/")

    safe_name = secure_filename(filename0)
    base, ext = os.path.splitext(safe_name)
    filename = f"{base}_{int(datetime.utcnow().timestamp())}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    f.save(path)

    cls, conf, label = predict_image(path)

    session["last_plant"] = cls or "unknown"
    session["last_label"] = label
    session["last_conf"] = conf
    session["last_result"] = f"{cls or 'unknown'} — {label} ({conf if conf is not None else 0}%)"
    session["last_result_label"] = label

    if user:
        add_history(user, f"Image: {cls or 'unknown'}", "image", label, conf, filename)

    return redirect("/")

# web endpoint
@app.route("/register", methods=["POST"])
# create new account
def register():
    u = (request.form.get("username") or "").strip()
    p = (request.form.get("password") or "").strip()
    if not u or not p:
        session["auth_error"] = "Username and password required"
        return redirect("/")

    ph = generate_password_hash(p)

    con = get_db()
    try:
        con.execute(
            "INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
            (u, ph, now()),
        )
        con.commit()
        session["user"] = u
    except sqlite3.IntegrityError:
        session["auth_error"] = "Username already exists"
    finally:
        con.close()

    return redirect("/")

# web endpoint
@app.route("/login", methods=["POST"])
# sign in user
def login():
    u = (request.form.get("username") or "").strip()
    p = (request.form.get("password") or "").strip()
    if not u or not p:
        session["auth_error"] = "Username and password required"
        return redirect("/")

    con = get_db()
    row = con.execute(
        "SELECT password_hash FROM users WHERE username=? LIMIT 1",
        (u,),
    ).fetchone()
    con.close()

    if row and check_password_hash(row["password_hash"], p):
        session["user"] = u
    else:
        session["auth_error"] = "Invalid username or password"

    return redirect("/")

# web endpoint
@app.route("/logout")
# sign out user
def logout():
    logger.info('Logout: user=%s', session.get('user'))
    session.pop("user", None)
    return redirect("/")

# web endpoint
@app.route("/profile")
# web endpoint
@app.route("/stats_data")
# send stats json
def stats_data():
    """Return per-user analytics for charts in the profile panel."""
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "reason": "not_logged_in"})

    con = get_db()

    rows = con.execute(
        """
        SELECT result_label, COUNT(*) as cnt
        FROM history
        WHERE username=?
        GROUP BY result_label
        """,
        (user,),
    ).fetchall()
    label_counts = {r["result_label"]: int(r["cnt"]) for r in rows}

    rows = con.execute(
        """
        SELECT query, COUNT(*) as cnt
        FROM history
        WHERE username=?
        GROUP BY query
        ORDER BY cnt DESC
        LIMIT 5
        """,
        (user,),
    ).fetchall()
    top_queries = [{"query": r["query"], "count": int(r["cnt"])} for r in rows]

    rows = con.execute(
        """
        SELECT confidence
        FROM history
        WHERE username=? AND confidence IS NOT NULL
        """,
        (user,),
    ).fetchall()
    conf_vals = [int(r["confidence"]) for r in rows if r["confidence"] is not None]

    bins = [0] * 11
    for v in conf_vals:
        if v < 0:
            continue
        if v >= 100:
            bins[10] += 1
        else:
            bins[v // 10] += 1
    conf_bins = [{"bin": f"{i*10}-{i*10+9}" if i < 10 else "100", "count": bins[i]} for i in range(11)]

    rows = con.execute(
        """
        SELECT SUBSTR(created_at, 1, 10) as day, COUNT(*) as cnt
        FROM history
        WHERE username=?
        GROUP BY day
        ORDER BY day DESC
        LIMIT 14
        """,
        (user,),
    ).fetchall()
    per_day = [{"day": r["day"], "count": int(r["cnt"])} for r in rows][::-1]

    con.close()

    return jsonify({
        "ok": True,
        "label_counts": label_counts,
        "top_queries": top_queries,
        "conf_bins": conf_bins,
        "per_day": per_day,
        "total_checks": sum(label_counts.values()) if label_counts else 0
    })

# profile redirect
def profile():
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)