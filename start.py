"""Bootstrap the project environment and run both local services."""
import os
from pathlib import Path
import subprocess
import sys
import time
import venv
from urllib.request import urlopen


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Требуется Python 3.11 или новее.")
    root = Path(__file__).resolve().parent
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.create(root / ".venv", with_pip=True)
    subprocess.run([python, "-m", "pip", "install", "-r", "requirements.txt"], cwd=root, check=True)
    service = subprocess.Popen([python, "-m", "clinical_mcp.server", "--sse"], cwd=root)
    try:
        for _ in range(120):
            if service.poll() is not None:
                raise SystemExit("Не удалось запустить SSE-сервис. Проверьте порт 8000.")
            try:
                with urlopen("http://127.0.0.1:8000/sse", timeout=0.3) as response:
                    if response.headers.get_content_type() == "text/event-stream":
                        break
            except OSError:
                time.sleep(0.25)
        else:
            raise SystemExit("SSE-сервис не ответил на порту 8000.")
        return subprocess.call([python, "-m", "streamlit", "run", "app.py"], cwd=root)
    finally:
        if service.poll() is None:
            service.terminate()
            service.wait()


if __name__ == "__main__":
    raise SystemExit(main())
