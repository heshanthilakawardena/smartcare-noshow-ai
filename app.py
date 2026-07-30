import subprocess
import os
import sys


def start_server():
    server_path = os.path.join(
        "server",
        "Flask_server.py"
    )
    print("SMARTCARE FLASK SERVER\n" \
    "+ Starting SmartCare Flask Server...")
    subprocess.run(
        [
            sys.executable,
            server_path
        ]
    )

if __name__ == "__main__":

    start_server()