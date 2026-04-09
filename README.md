# Demo Network Intrusion Detection System (Python, Flask, ML)

This is a **prototype network simulation** demo that lets you simulate traffic from a client UI and watch a server-side NIDS classify each request as **harmless** or **harmful** in real time.

## Structure

```text
nids-project/
├── client/                 # Flask client app with a form UI
│   ├── app.py
│   ├── requirements.txt
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       └── index.html
└── server/                 # Flask server app with ML model + dashboard
    ├── app.py
    ├── model.py
    ├── train_model.py      # Optional: generate model.pkl (synthetic data)
    ├── requirements.txt
    ├── static/
    │   └── styles.css
    └── templates/
        └── index.html
```

## Features

- **Separate Client & Server**
  - **Client**: Choose attributes (Source IP, Metadata, Payload, Port) and send a request.
  - **Server**: Receives, classifies (ML/heuristic), and **logs** each request.
- **ML Module**
  - `server/model.py` exposes `NIDSModel` using a LightGBM model **if** `model.pkl` exists.
  - If no model is present, it falls back to a **heuristic** scorer (weighted logistic).
  - Features include: blacklisted IP, risky port, payload/meta length, simple SQLi/XSS signatures, HTTP method one-hot.
- **Real-time Logs**
  - Server dashboard polls `/logs` every 1.5s to show the latest classification results.

## Quickstart

Open two terminals:

### 1) Start the server
```bash
cd server
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
# (Optional) Train a tiny model on synthetic data:
python train_model.py
python app.py  # listens on http://127.0.0.1:5001
```

### 2) Start the client
```bash
cd client
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Point client to your server (default: http://127.0.0.1:5001)
# export NIDS_SERVER_URL="http://127.0.0.1:5001"   # PowerShell: $env:NIDS_SERVER_URL="http://127.0.0.1:5001"
python app.py  # opens http://127.0.0.1:5002
```

Visit the client at **http://127.0.0.1:5002**, submit **safe** values first (e.g., IP `192.168.1.10`, port `80/443`, method `GET`, normal payload), then try **blacklisted** or suspicious values (e.g., IP `66.66.66.66`, port `22/23/445/3389/1433/5900`, payload containing SQLi/XSS). Watch the server dashboard at **http://127.0.0.1:5001** update live.

## Notes

- This is a **demo** (educational) NIDS, not production security software.
- Replace blacklists/signatures with your sources; add persistence (DB) if needed.
- For true real time, consider **Server-Sent Events** or websockets; this demo uses polling for simplicity.
