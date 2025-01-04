import sys
import os

from app import setup_application

from torch import cuda

if __name__ == "__main__":
    app, overlay, setup, master, worker = setup_application(path=os.path.dirname(os.path.abspath(__file__)), cuda=cuda.is_available())
    sys.exit(app.exec())
