import os
import sys


server_dir = os.path.join(os.path.dirname(__file__), "server")
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

# Import the 'app' object from server/flask_server.py
try:
    from flask_server import app
except ImportError:
    from server.flask_server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"SMARTCARE FLASK SERVER\n+ Starting SmartCare Flask Server on port {port}...")
    app.run(host="0.0.0.0", port=port)