# client1.py
import os
from flask import Flask, render_template, request
import requests as http

app = Flask(__name__)
# URL of your NIDS server
NIDS_SERVER_URL = os.environ.get("NIDS_SERVER_URL", "http://127.0.0.1:5000")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        # file from form
        f = request.files.get("file")
        if not f or f.filename.strip() == "":
            result = {"ok": False, "error": "No file uploaded"}
        else:
            # metadata fields
            src_ip = request.form.get("src_ip") or ""
            dst_port = request.form.get("dst_port") or ""
            method = request.form.get("method") or "GET"
            meta = request.form.get("meta") or ""

            # read file bytes to forward
            try:
                file_bytes = f.read()
                files = {"file": (f.filename, file_bytes, f.mimetype or "application/octet-stream")}
                data = {
                    "src_ip": src_ip,
                    "dst_port": dst_port,
                    "method": method,
                    "meta": meta,
                }
                r = http.post(f"{NIDS_SERVER_URL}/ingest", data=data, files=files, timeout=20)
                result = r.json()
            except Exception as e:
                result = {"ok": False, "error": str(e)}

    return render_template("index.html", server_url=NIDS_SERVER_URL, result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
