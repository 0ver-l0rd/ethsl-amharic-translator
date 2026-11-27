"""
python app/web
==============
Starts the translator's web page (app/web_app.py) and opens it in the
browser. The model is models/finetuned/best_model.h5; if it isn't there, the
newest best_model*.h5 in Downloads (where Telegram saves files) or in the repo
folder is copied there first.

Options are those of app/web_app.py, e.g.:
  python app/web --port 8000
  python app/web --public
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.retrain import require_live_model

if __name__ == "__main__":
    print(f"Model: {require_live_model()}")
    from app import web_app       # slow (Gradio, TensorFlow): after the model check
    web_app.main(open_browser=True)
