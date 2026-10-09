"""Launch the playable demo: python scripts/demo.py [--checkpoint models/connect4_az.pt] [--port 8000]"""
import argparse, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo.server import main

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(ROOT / "models" / "connect4_az.pt"))
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
main(a.checkpoint, a.port)
