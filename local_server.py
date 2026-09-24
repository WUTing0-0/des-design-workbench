"""Serve the workbench and expose the frozen B4 model from the paper release."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import math
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from property_inference import PropertyResolver

ROOT = Path(__file__).resolve().parent
RELEASE = Path(os.environ.get("DES_WORKFLOW_ROOT", r"C:\code2026\DES_Paper_Release")).resolve()
PROJECT = Path(os.environ.get("DES_PROJECT_ROOT", r"C:\code2026")).resolve()
MODULE_PATH = RELEASE / "77_salt_tm_role_context_propagation_v1" / "run_dense_grid_chunk_v76.py"
PACK_PATH = RELEASE / "77_salt_tm_role_context_propagation_v1" / "benchmark_model" / "FULL_CANONICAL_B4_B5_PACK.joblib"

os.environ.setdefault("DES_WORKFLOW_ROOT", str(RELEASE))
os.environ.setdefault("DES_PROJECT_ROOT", str(PROJECT))
spec = importlib.util.spec_from_file_location("des_dense_grid", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
pack = joblib.load(PACK_PATH)
property_resolver = PropertyResolver()


def predict(payload: dict) -> list[float]:
    if payload.get('activityMode') not in {'sigma', 'gamma'}:
        raise ValueError('B4 requires non-ideal features. Ideal SLE is available as a physical reference only.')
    from rdkit import Chem
    for field in ['smilesA', 'smilesB']:
        if not payload.get(field) or Chem.MolFromSmiles(payload[field]) is None:
            raise ValueError('Two valid component SMILES are required.')
    for field in ['tm1', 'tm2', 'h1', 'h2']:
        if not math.isfinite(float(payload[field])) or float(payload[field]) <= 0:
            raise ValueError('Pure properties must be finite and positive.')
    points_in = payload.get('points', [])
    if not 2 <= len(points_in) <= 2000:
        raise ValueError('Supply 2–2,000 composition points.')
    for point in points_in:
        if not 0 < float(point['x']) < 1 or point.get('t') is None or not math.isfinite(float(point['t'])):
            raise ValueError('Model inference requires finite physical temperatures at every supplied composition.')
    base = {
        "DES_ID": "user_submission", "System_Type": payload.get("systemType", "Type V"),
        "Component1": payload.get("nameA", "Component A"), "Component2": payload.get("nameB", "Component B"),
        "SMILES1": payload.get("smilesA", ""), "SMILES2": payload.get("smilesB", ""),
        "T1": float(payload["tm1"]), "T2": float(payload["tm2"]),
        "H1": float(payload["h1"]), "H2": float(payload["h2"]),
    }
    rows = [{**base, "X1": float(p["x"]), "X2": 1-float(p["x"]), "T_BA": float(p["t"]),
             "hardmax_branch1_K": p.get("b1"), "hardmax_branch2_K": p.get("b2")} for p in payload["points"]]
    points = pd.DataFrame(rows)
    data, block = module.build_b4_block(points)
    numeric = pack["imputer"].transform(block.reindex(columns=pack["numeric_columns"]))
    _, text_cols = module.normalize_library_columns(data, "ratio")
    z = pack["svd"].transform(pack["vectorizer"].transform(module.component_text(data, text_cols)))
    X = pack["scaler"].transform(np.hstack([numeric, z]))
    return [float(v) for v in pack["direct_model"].predict(X)]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "dist"), **kwargs)

    def send_json(self, status: int, value: dict):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self.send_json(200, {
                "status": "ready", "model": "frozen B4 direct",
                "properties": property_resolver.ready_summary,
            })
        return super().do_GET()

    def do_POST(self):
        if self.path not in {"/api/predict", "/api/properties"}:
            return self.send_json(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 2_000_000:
                raise ValueError('Request must contain at most 2 MB of JSON.')
            payload = json.loads(self.rfile.read(size))
            if self.path == "/api/properties":
                return self.send_json(200, property_resolver.resolve(
                    payload.get("smiles", ""), payload.get("componentClass")
                ))
            return self.send_json(200, {"predictions": predict(payload)})
        except Exception as exc:
            return self.send_json(400, {"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    host, port = "127.0.0.1", int(os.environ.get("PORT", "4173"))
    print(f"DES Design Workbench: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
