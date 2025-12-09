from flask import Flask, request, jsonify, render_template_string, send_from_directory
from collections import deque
from datetime import datetime
import threading, os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Store received files + metadata
RECEIVED = deque(maxlen=1000)
LOCK = threading.Lock()

UPLOAD_FOLDER = "received_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Client 2 - Received Files</title>
        <meta http-equiv="refresh" content="3">
        <style>
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
          th { background: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Client 2 - Received Files</h1>
        <table>
            <tr>
              <th>Timestamp</th>
              <th>Filename</th>
              <th>Size</th>
              <th>Src IP</th>
              <th>Dst Port</th>
              <th>Method</th>
              <th>Meta</th>
              <th>Download</th>
            </tr>
            {% for item in files %}
            <tr>
                <td>{{item.ts}}</td>
                <td>{{item.filename}}</td>
                <td>{{item.size}}</td>
                <td>{{item.src_ip}}</td>
                <td>{{item.dst_port}}</td>
                <td>{{item.method}}</td>
                <td>{{item.meta}}</td>
                <td><a href="/download/{{item.filename}}">Download</a></td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """, files=list(RECEIVED))


@app.post("/receive_file")
def receive_file():
    """Receive forwarded file from server"""
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "No file received"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "filename": filename,
        "size": os.path.getsize(filepath),
        "src_ip": request.form.get("src_ip", ""),
        "dst_port": request.form.get("dst_port", ""),
        "method": request.form.get("method", ""),
        "meta": request.form.get("meta", ""),
    }
    with LOCK:
        RECEIVED.appendleft(record)

    return jsonify({"ok": True, "received": record})

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=True)   # client2.py