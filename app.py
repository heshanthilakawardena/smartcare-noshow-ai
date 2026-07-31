import subprocess
import os
import sys

def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(
        base_path,
        relative_path
    )


def start_server():

    server_path = resource_path(
        os.path.join(
            "server",
            "flask_server.py"
        )
    )

    print(
        "SMARTCARE FLASK SERVER\n"
        "+ Starting SmartCare Flask Server..."
    )

    subprocess.run(
        [
            sys.executable,
            server_path
        ]
    )


if __name__ == "__main__":

    start_server()