# app_server.py — FINAL, FIXED, STABLE VERSION

import os
import time
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from datetime import datetime
from collections import deque
import threading

from model import NIDSModel

# ---------------------------------------------------------
# Try loading REAL EMBER extractor
# ---------------------------------------------------------
try:
    import ember_extractor2 as ember_extractor
    HAVE_EMBER_EXTRACTOR = True
    print("🔥 Using REAL EMBER feature extractor (ember_extractor.py)")
except:
    HAVE_EMBER_EXTRACTOR = False
    print("⚠️ ember_extractor.py not found — using upgraded fallback extractor.")

# ---------------------------------------------------------
# Flask setup
# ---------------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLIENT2_URL = os.environ.get("CLIENT2_URL", "http://127.0.0.1:5003")

LOGS = deque(maxlen=2000)
LOG_LOCK = threading.Lock()

model = NIDSModel()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def shannon_entropy(arr):
    if len(arr) == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256).astype(np.float32)
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# =========================================================
# UPGRADED FALLBACK FEATURE EXTRACTOR (FIXED)
# =========================================================
def fallback_extract_features(filepath: str) -> np.ndarray:
    ORIGINAL_DIM = 2381
    vec = np.zeros(ORIGINAL_DIM, dtype=np.float32)

    # ----------------------------- read file safely
    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except:
        return vec

    size = len(data)
    vec[0] = float(size)

    arr = np.frombuffer(data, dtype=np.uint8)

    # ----------------------------- byte histogram (1–256)
    hist = np.bincount(arr, minlength=256).astype(np.float32)
    hist /= max(1, hist.sum())
    vec[1:257] = hist

    # ----------------------------- entropy histogram (257–272)
    ent_list = []
    window = 2048
    for i in range(0, size, window):
        chunk = arr[i:i+window]
        ent_list.append(shannon_entropy(chunk))

    ent_hist = np.histogram(ent_list, bins=16, range=(0, 8))[0].astype(np.float32)
    ent_hist /= max(1, ent_hist.sum())
    vec[257:273] = ent_hist

    # ----------------------------- string features (273–276)
    import re
    strings = re.findall(rb"[ -~]{4,}", data)
    n_str = len(strings)
    long_str = sum(1 for s in strings if len(s) > 20)
    avg_len = (sum(len(s) for s in strings) / n_str) if n_str else 0
    max_len = max((len(s) for s in strings), default=0)

    vec[273] = n_str
    vec[274] = long_str
    vec[275] = avg_len
    vec[276] = max_len

    # ----------------------------- PE FEATURES (277–289)
    try:
        import pefile
        pe = pefile.PE(filepath, fast_load=True)

        vec[277] = len(pe.sections)

        sec_sizes = []
        sec_entropy = []
        for s in pe.sections:
            sec_data = s.get_data()
            sec_sizes.append(len(sec_data))
            sec_entropy.append(shannon_entropy(np.frombuffer(sec_data, dtype=np.uint8)))

        if sec_sizes:
            vec[278] = float(np.mean(sec_sizes))
            vec[279] = float(np.var(sec_sizes))
            vec[280] = float(np.max(sec_sizes))

        if sec_entropy:
            vec[281] = float(np.mean(sec_entropy))
            vec[282] = float(np.var(sec_entropy))
            vec[283] = float(np.max(sec_entropy))

        imp_funcs = 0
        imp_dlls = 0
        try:
            if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                imp_dlls = len(pe.DIRECTORY_ENTRY_IMPORT or [])
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    imp_funcs += len(entry.imports or [])
        except:
            pass

        vec[284] = imp_funcs
        vec[285] = imp_dlls

        exp_count = 0
        try:
            if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") and pe.DIRECTORY_ENTRY_EXPORT:
                exp_count = len(pe.DIRECTORY_ENTRY_EXPORT.symbols or [])
        except:
            pass

        vec[286] = exp_count

        try:
            vec[287] = float(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
        except:
            vec[287] = 0.0

        try:
            end_raw = max(s.PointerToRawData + s.SizeOfRawData for s in pe.sections)
            vec[288] = float(max(0, size - end_raw))
        except:
            vec[288] = 0.0

        # resource count
        r_count = 0
        try:
            if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE") and pe.DIRECTORY_ENTRY_RESOURCE:
                r_count = len(pe.DIRECTORY_ENTRY_RESOURCE.entries or [])
        except:
            pass

        vec[289] = float(r_count)

    except:
        pass  # non-PE, keep defaults

    # ----------------------------- n-grams (291–354)
    ngram_start = 291
    buckets = 64
    idx = arr // (256 // buckets)
    ngram = np.bincount(idx, minlength=buckets).astype(np.float32)
    ngram /= max(1, ngram.sum())
    vec[ngram_start:ngram_start + buckets] = ngram

    # ----------------------------- safe filler (355–2380)
    f_start = 355
    base_vals = [
        float(size),
        float(arr.mean() if size else 0),
        float(arr.var() if size else 0),
        float(n_str),
        float(long_str),
    ]

    j = 0
    for i in range(f_start, ORIGINAL_DIM):
        a = base_vals[j % len(base_vals)]
        vec[i] = float((a * 1.001) + j * 0.00001)
        j += 1

    return vec


# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def dashboard():
    return render_template("server_index.html", server_name="NIDS-LightGBM")


@app.post('/ingest')
def ingest():
    src_ip = request.form.get("src_ip", "").strip()
    dst_port = int(request.form.get("dst_port", 80))
    method = request.form.get("method", "GET")
    meta = request.form.get("meta", "")

    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ------------------------ extract features
    try:
        if HAVE_EMBER_EXTRACTOR:
            full_feats = ember_extractor.extract_ember_features(filepath)
        else:
            full_feats = fallback_extract_features(filepath)

        reduced_feats = model.preprocess(full_feats)

    except Exception as e:
        safe_delete(filepath)
        return jsonify({"ok": False, "error": f"Feature extraction failed: {e}"}), 500

    # ------------------------ prediction
    try:
        label, prob = model.predict(reduced_feats)
    except Exception as e:
        safe_delete(filepath)
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500

    verdict = "malware" if label == 1 else "benign"
    allowed = (label == 0)

    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "src_ip": src_ip,
        "dst_port": dst_port,
        "method": method,
        "filename": filename,
        "prob_malware": round(prob, 4),
        "verdict": verdict,
        "allowed": allowed,
    }

    # ------------------------ forwarding if benign
    if allowed:
        try:
            with open(filepath, "rb") as f:
                requests.post(
                    f"{CLIENT2_URL}/receive_file",
                    data={"src_ip": src_ip, "dst_port": dst_port, "method": method, "meta": meta},
                    files={"file": (filename, f, "application/octet-stream")},
                    timeout=10
                )
            record["forwarded_to_client2"] = True
        except:
            record["forwarded_to_client2"] = False

    else:
        record["reason"] = "Malware detected (LightGBM)"

    safe_delete(filepath)

    with LOG_LOCK:
        LOGS.appendleft(record)

    return jsonify({"ok": True, "record": record})


def safe_delete(path):
    """Windows-safe file deletion."""
    try:
        os.remove(path)
    except PermissionError:
        time.sleep(0.2)
        try:
            os.remove(path)
        except:
            pass


@app.get("/logs")
def logs():
    return jsonify({"ok": True, "items": list(LOGS)})


@app.post("/debug_features")
def debug_features():
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "No file"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        feats = fallback_extract_features(path) if not HAVE_EMBER_EXTRACTOR else ember_extractor.extract_ember_features(path)
        reduced = model.preprocess(feats)

        safe_delete(path)

        return jsonify({
            "ok": True,
            "len_full": len(feats),
            "len_reduced": len(reduced),
            "full_head_50": [float(x) for x in feats[:50]],
            "reduced_head_50": [float(x) for x in reduced[:50]]
        })

    except Exception as e:
        safe_delete(path)
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
