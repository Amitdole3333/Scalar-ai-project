"""Minimal web interface for HF Spaces (port 7860).

This keeps the Docker container alive so HF Spaces shows 'Running'.
It also exposes OpenEnv-compatible HTTP API endpoints:
  POST /reset   — reset the environment (accepts JSON body with task_name)
  POST /step    — take one step (accepts JSON body with action dict)
  POST /state   — return current environment state
  POST /close   — close the environment
  GET  /        — dashboard HTML page

The actual inference can also run via CLI: python inference.py
"""

import http.server
import json
import os
import subprocess
import sys
import threading
import traceback
from urllib.parse import parse_qs, urlparse

# Import the environment
from env import HospitalEnv

PORT = int(os.environ.get("PORT", 7860))

# Shared environment instance (thread-safe for single-threaded http.server)
_env = HospitalEnv(seed=42)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital ER Triage - OpenEnv AI Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #1a1a3e, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #ff6b6b, #ffa07a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle { color: #888; font-size: 0.95rem; margin-bottom: 2rem; }
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }
        .card h2 { color: #ffa07a; font-size: 1.2rem; margin-bottom: 1rem; }
        .scores-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        .score-item {
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .score-item .label { font-size: 0.85rem; color: #888; text-transform: uppercase; }
        .score-item .value { font-size: 1.8rem; font-weight: 700; color: #4ade80; margin: 0.3rem 0; }
        .score-item .steps { font-size: 0.8rem; color: #666; }
        .env-vars { font-family: monospace; font-size: 0.85rem; }
        .env-vars .row { display: flex; gap: 1rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .env-vars .key { color: #ffa07a; min-width: 140px; }
        .env-vars .val { color: #4ade80; }
        .tag {
            display: inline-block;
            background: rgba(74,222,128,0.15);
            color: #4ade80;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
        }
        code {
            background: rgba(255,255,255,0.08);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        .status { color: #4ade80; font-weight: 600; }
        a { color: #ffa07a; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .footer { text-align: center; color: #555; font-size: 0.8rem; margin-top: 2rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hospital ER Triage</h1>
        <p class="subtitle">OpenEnv-compatible AI Training Environment &mdash; Meta Hackathon Submission</p>

        <div class="card">
            <h2>Baseline Scores (Rule-Based Agent)</h2>
            <div class="scores-grid">
                <div class="score-item">
                    <div class="label">Easy</div>
                    <div class="value">0.900</div>
                    <div class="steps">3 steps</div>
                </div>
                <div class="score-item">
                    <div class="label">Medium</div>
                    <div class="value">0.885</div>
                    <div class="steps">5 steps</div>
                </div>
                <div class="score-item">
                    <div class="label">Hard</div>
                    <div class="value">0.767</div>
                    <div class="steps">6 steps</div>
                </div>
                <div class="score-item">
                    <div class="label">Average</div>
                    <div class="value" style="color:#ffa07a;">0.851</div>
                    <div class="steps">&mdash;</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Environment Status</h2>
            <p><span class="status">&#9679; Running</span> &mdash; Docker container active on port 7860</p>
            <br>
            <div class="env-vars">
                <div class="row"><span class="key">API_BASE_URL</span><span class="val">ENDPOINT</span></div>
                <div class="row"><span class="key">MODEL_NAME</span><span class="val">MODEL</span></div>
                <div class="row"><span class="key">HF_TOKEN</span><span class="val">CONFIGURED</span></div>
            </div>
        </div>

        <div class="card">
            <h2>12 Real-World Challenges</h2>
            <p style="margin-bottom:0.5rem">
                <span class="tag">Patient Triage</span>
                <span class="tag">Priority Queue</span>
                <span class="tag">Delay Penalty</span>
                <span class="tag">Resource Limits</span>
                <span class="tag">Mass Casualty</span>
                <span class="tag">Ambulance</span>
                <span class="tag">Doctor Assignment</span>
                <span class="tag">Dynamic Events</span>
                <span class="tag">New Arrivals</span>
                <span class="tag">Transfer</span>
                <span class="tag">Misleading Symptoms</span>
                <span class="tag">Patient Escalation</span>
            </p>
        </div>

        <div class="card">
            <h2>Stdout Format</h2>
            <pre style="color:#ccc; font-size:0.8rem; overflow-x:auto;">
[START] task=easy env=hospital-er-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=2.20 done=false error=null
[STEP] step=2 action={...} reward=2.20 done=false error=null
[STEP] step=3 action={...} reward=2.20 done=true error=null
[END] success=true steps=3 score=0.900 rewards=2.20,2.20,2.20</pre>
        </div>

        <div class="card">
            <h2>Links</h2>
            <p>
                <a href="https://github.com/Amitdole3333/Scalar-ai-project" target="_blank">GitHub Repository</a>
                &nbsp;|&nbsp;
                <a href="https://huggingface.co/spaces/Amit9373/hospital-er-triage/tree/main" target="_blank">View Files</a>
            </p>
        </div>

        <p class="footer">Hospital ER Triage &middot; OpenEnv Hackathon &middot; Pydantic + OpenAI Client</p>
    </div>
</body>
</html>
"""


def _send_json(handler, data, status=200):
    """Helper to send a JSON response."""
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # API endpoints also accept GET for convenience
        if path == "/state":
            try:
                state = _env.state()
                _send_json(self, state)
            except Exception as e:
                _send_json(self, {"error": str(e)}, status=500)
            return

        # Default: serve the dashboard HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        token_status = "SET" if os.environ.get("HF_TOKEN") else "MISSING"

        page = HTML_PAGE.replace("ENDPOINT", api_base).replace("MODEL", model).replace("CONFIGURED", token_status)
        self.wfile.write(page.encode("utf-8"))

    def do_POST(self):
        """Handle OpenEnv API POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            body = {}

        try:
            if path == "/reset":
                task_name = body.get("task_name", body.get("task", "easy"))
                obs = _env.reset(task_name=task_name)
                _send_json(self, obs)

            elif path == "/step":
                action_dict = body.get("action", body)
                obs, reward, done, info = _env.step(action_dict)
                _send_json(self, {
                    "observation": obs,
                    "reward": reward,
                    "done": done,
                    "info": info,
                })

            elif path == "/state":
                state = _env.state()
                _send_json(self, state)

            elif path == "/close":
                _env.close()
                _send_json(self, {"status": "closed"})

            elif path == "/":
                # Root POST — treat as reset (OpenEnv compatibility)
                task_name = body.get("task_name", body.get("task", "easy"))
                obs = _env.reset(task_name=task_name)
                _send_json(self, obs)

            else:
                _send_json(self, {"error": f"Unknown endpoint: {path}"}, status=404)

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _send_json(self, {"error": str(e)}, status=500)

    def log_message(self, format, *args):
        # Log to stderr for debugging (but keep it minimal)
        print(f"[HTTP] {args[0]}", file=sys.stderr, flush=True)


def main():
    print(f"Starting web server on port {PORT}...")
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Server running at http://0.0.0.0:{PORT}")
    print(f"OpenEnv API endpoints: POST /reset, /step, /state, /close")
    server.serve_forever()


if __name__ == "__main__":
    main()
