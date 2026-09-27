"""Experiment-first pure-property resolution for the DES workbench.

The resolver follows the paper freeze: exact, curated experimental values take
priority; missing values are supplied by the corresponding production model.
Model objects are loaded lazily because the molecular encoders and TabPFN are
substantially heavier than a database lookup.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


RELEASE = Path(os.environ.get("DES_WORKFLOW_ROOT", r"C:\code2026\DES_Paper_Release")).resolve()
PROJECT = Path(os.environ.get("DES_PROJECT_ROOT", r"C:\code2026")).resolve()
P75 = RELEASE / "75_legacy_architecture_rebuild_v1"
P76 = RELEASE / "76_experiment_first_fullchain_v1"
P77 = RELEASE / "77_salt_tm_role_context_propagation_v1"
LEGACY = P75 / "legacy_source"
OLD = PROJECT / "outputs_v3" / "outputs_v3"
CHEMBERTA = PROJECT / "models" / "models" / "ChemBERTa-zinc-base-v1"
TABPFN_CKPT = PROJECT / "数据" / "数据" / "训练" / "tabpfn-v2.6-regressor-v2.6_default.ckpt"
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

METRICS = {
    "neutral": {"tm": (22.0788898468, "four-layer GIN"), "hfus": (10.9468774851, "TabPFN")},
    "salt": {"tm": (24.999, "role-aware G-TabPFN"), "hfus": (8.7124440256, "TabPFN")},
}


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        raise ValueError("The SMILES string could not be parsed.")
    return Chem.MolToSmiles(mol, canonical=True)


def salt_parts(canonical: str) -> tuple[str, str] | None:
    parts = canonical.split(".")
    if len(parts) < 2:
        return None
    scored = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        scored.append((part, charge))
    positives = [p for p, q in scored if q > 0]
    negatives = [p for p, q in scored if q < 0]
    if len(positives) == 1 and len(negatives) == 1 and len(parts) == 2:
        return positives[0], negatives[0]
    return None


class PropertyResolver:
    def __init__(self):
        master_path = P77 / "property_master" / "PURE_PROPERTY_MASTER_EXPERIMENT_FIRST.csv"
        self.portable = not master_path.exists()
        if self.portable:
            master_path = Path(__file__).resolve().parent / "portable_data" / "reviewed_property_examples.csv"
        self.master = pd.read_csv(master_path)
        self.master["canonical_key"] = self.master.component_canonical.map(canonicalize)
        self.lookup = {row.canonical_key: row for row in self.master.itertuples(index=False)}
        self.lock = threading.RLock()
        self._bert = None
        self._feature_gin = None
        self._tm_gin = None
        self._tab_models: dict[str, tuple[dict, object]] = {}
        self._recipes: dict[str, dict] = {}
        self._Config = None
        self._PretrainModel = None
        self._smiles_to_graph = None
        self._compute_raw_features = None

    @property
    def ready_summary(self) -> dict:
        return {
            "status": "ready",
            "database_rows": len(self.master),
            "distribution": "portable example library" if self.portable else "full reviewed research library",
            "inference_available": not self.portable,
            "policy": "experiment-first; model only when missing",
            "models": {
                "neutral_tm": "four-layer GIN, MAE 22.08 K",
                "neutral_hfus": "TabPFN, MAE 10.95 kJ mol-1",
                "salt_tm": "role-aware G-TabPFN, MAE 25.00 K",
                "salt_hfus": "TabPFN, MAE 8.71 kJ mol-1",
            },
        }

    def _load_encoders(self):
        if self._bert is not None:
            return
        import torch
        if str(LEGACY) not in sys.path:
            sys.path.insert(0, str(LEGACY))
        from config import Config
        from data.feature_extractor import compute_raw_features
        from data.mol_graph import smiles_to_graph
        from models.chemberta import ThermalBERTa
        from models.full_model_v3 import PretrainModel
        bert = ThermalBERTa(str(CHEMBERTA)).eval()
        bert.load_state_dict(torch.load(OLD / "thermal_bert.pt", map_location="cpu", weights_only=True), strict=True)
        gin = PretrainModel(Config()).eval()
        gin.load_state_dict(torch.load(OLD / "gin_melt_encoder.pt", map_location="cpu", weights_only=True), strict=True)
        self._bert, self._feature_gin = bert, gin
        self._Config, self._PretrainModel = Config, PretrainModel
        self._smiles_to_graph, self._compute_raw_features = smiles_to_graph, compute_raw_features

    def _recipe(self, task: str, path: Path) -> dict:
        if task not in self._recipes:
            self._recipes[task] = joblib.load(path)
        return self._recipes[task]

    def _raw_features(self, smiles: list[str]) -> dict[str, np.ndarray]:
        import torch
        from torch_geometric.loader import DataLoader
        self._load_encoders()
        with rdBase.BlockLogs():
            morgan, rdkit = self._compute_raw_features(smiles)
        bert = self._bert.extract_cls(smiles, torch.device("cpu"), batch_size=min(16, len(smiles)))
        graphs = [self._smiles_to_graph(s) for s in smiles]
        if any(g is None for g in graphs):
            raise ValueError("A molecular graph could not be generated.")
        gin_rows = []
        with torch.no_grad():
            for graph_batch in DataLoader(graphs, batch_size=min(32, len(graphs))):
                gin_rows.append(self._feature_gin.encoder(
                    graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
                ).cpu().numpy())
        return {"morgan": morgan, "rdkit": rdkit, "bert": bert, "gin": np.concatenate(gin_rows)}

    def _neutral_tm(self, smiles: str) -> float:
        import torch
        from torch_geometric.loader import DataLoader
        self._load_encoders()
        if self._tm_gin is None:
            manifest = pd.read_json(P76 / "models" / "organic_tm_gin_full_manifest.json", typ="series")
            model = self._PretrainModel(self._Config()).eval()
            model.load_state_dict(torch.load(
                P76 / "models" / "organic_tm_gin_full" / "model_state.pt",
                map_location="cpu", weights_only=True,
            ))
            self._tm_gin = (model, float(manifest.target_mean), float(manifest.target_std))
        model, mean, std = self._tm_gin
        graph = self._smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Cannot build a molecular graph for {smiles!r}")
        graph = next(iter(DataLoader([graph], batch_size=1)))
        with torch.no_grad():
            value = float(model(graph).cpu().numpy().ravel()[0] * std + mean)
        return value

    def _tabpfn(self, task: str, query: np.ndarray) -> float:
        if task not in self._tab_models:
            from tabpfn import TabPFNRegressor
            path = {
                "neutral_hfus": P76 / "models" / "organic_hfus_production_recipe.joblib",
                "salt_hfus": P76 / "models" / "salt_hfus_production_recipe.joblib",
                "salt_tm": P77 / "models" / "salt_tm_role_context_production_recipe.joblib",
            }[task]
            recipe = self._recipe(task, path)
            estimators = int(os.environ.get("DES_TABPFN_ESTIMATORS", recipe["n_estimators"]))
            model = TabPFNRegressor(
                device=os.environ.get("DES_TABPFN_DEVICE", "cpu"),
                n_estimators=estimators,
                model_path=str(TABPFN_CKPT), random_state=int(recipe["seed"]),
                n_preprocessing_jobs=1,
            )
            model.fit(recipe["context_X"], recipe["context_y_normalized"])
            self._tab_models[task] = (recipe, model)
        recipe, model = self._tab_models[task]
        normalized = float(np.asarray(model.predict(query.astype(np.float32))).ravel()[0])
        return normalized * float(recipe["target_std"]) + float(recipe["target_mean"])

    @staticmethod
    def _tab_blocks(raw: dict[str, np.ndarray], recipe: dict) -> np.ndarray:
        pipes = recipe["preprocessing"]
        blocks = []
        for key in ("morgan", "rdkit", "bert"):
            values = np.where(np.isfinite(raw[key]), raw[key], np.nan)
            blocks.append(pipes[key].transform(values).astype(np.float32))
        return np.concatenate(blocks, axis=1)

    def _model_value(self, component_class: str, prop: str, canonical: str, parts: tuple[str, str] | None) -> float:
        if component_class == "neutral" and prop == "tm":
            return self._neutral_tm(canonical)
        if component_class == "neutral":
            recipe = self._recipe("neutral_hfus", P76 / "models" / "organic_hfus_production_recipe.joblib")
            raw = self._raw_features([canonical])
            query = self._tab_blocks(raw, recipe)
            return self._tabpfn("neutral_hfus", query)
        if parts is None:
            raise ValueError("Salt inference requires one dot-separated cation and one anion in the SMILES.")
        raw = self._raw_features(list(parts))
        task = "salt_tm" if prop == "tm" else "salt_hfus"
        path = (P77 / "models" / "salt_tm_role_context_production_recipe.joblib") if prop == "tm" else (P76 / "models" / "salt_hfus_production_recipe.joblib")
        recipe = self._recipe(task, path)
        tab = self._tab_blocks(raw, recipe)
        if prop == "tm":
            joined = np.c_[np.c_[raw["gin"][0:1], tab[0:1]], np.c_[raw["gin"][1:2], tab[1:2]]]
            query = recipe["preprocessing"]["pair"].transform(joined).astype(np.float32)
        else:
            query = ((tab[0:1] + tab[1:2]) / 2).astype(np.float32)
        return self._tabpfn(task, query)

    def lookup_only(self, smiles: str, force_class: str | None = None) -> dict:
        """Return reviewed experimental values without silently imputing gaps."""
        canonical = canonicalize(smiles)
        parts = salt_parts(canonical)
        inferred_class = "salt" if parts else "neutral"
        if force_class in {"neutral", "salt"} and force_class != inferred_class:
            if force_class == "salt":
                raise ValueError("The selected salt class needs a dot-separated cation and anion SMILES.")
            raise ValueError("The structure contains a charged ion pair; choose salt.")
        component_class = force_class or inferred_class
        record = self.lookup.get(canonical)
        result = {
            "canonical_smiles": canonical,
            "component_class": component_class,
            "matched_database": record is not None,
            "name": (getattr(record, "component_name", None) if record is not None else None),
            "properties": {},
        }
        for prop, exp_col, unit in (
            ("tm", "Tm_experimental", "K"),
            ("hfus", "Hfus_experimental", "kJ mol-1"),
        ):
            value = getattr(record, exp_col, np.nan) if record is not None else np.nan
            result["properties"][prop] = {
                "available": bool(pd.notna(value)),
                "value": float(value) if pd.notna(value) else None,
                "unit": unit,
                "origin": "experimental" if pd.notna(value) else "missing",
            }
        result["complete_experimental"] = all(
            item["available"] for item in result["properties"].values()
        )
        result["requires_user_choice"] = not result["complete_experimental"]
        if component_class == "salt":
            result["ions"] = {"cation": parts[0], "anion": parts[1]}
            result["warning"] = "Unseen anions are the main transfer limitation for salt properties."
        return result

    def resolve(self, smiles: str, force_class: str | None = None) -> dict:
        with self.lock:
            canonical = canonicalize(smiles)
            parts = salt_parts(canonical)
            inferred_class = "salt" if parts else "neutral"
            if force_class in {"neutral", "salt"} and force_class != inferred_class:
                if force_class == "salt":
                    raise ValueError("The selected salt class needs a dot-separated cation and anion SMILES.")
                raise ValueError("The structure contains a charged ion pair; choose salt.")
            component_class = force_class or inferred_class
            record = self.lookup.get(canonical)
            result = {
                "canonical_smiles": canonical,
                "component_class": component_class,
                "matched_database": record is not None,
                "name": (getattr(record, "component_name", None) if record is not None else None),
                "properties": {},
            }
            for prop, exp_col, final_col, unit in (
                ("tm", "Tm_experimental", "Tm_final", "K"),
                ("hfus", "Hfus_experimental", "Hfus_final", "kJ mol-1"),
            ):
                experimental = getattr(record, exp_col, np.nan) if record is not None else np.nan
                if pd.notna(experimental):
                    value, origin, model = float(experimental), "experimental", None
                elif record is not None and pd.notna(getattr(record, final_col, np.nan)):
                    value, origin = float(getattr(record, final_col)), "model"
                    model = METRICS[component_class][prop][1]
                else:
                    if self.portable:
                        raise ValueError(
                            "This compound is not in the portable reviewed example library. "
                            "Enter measured pure properties manually, or install the separately licensed full model bundle."
                        )
                    value = float(self._model_value(component_class, prop, canonical, parts))
                    origin, model = "model", METRICS[component_class][prop][1]
                mae = None if origin == "experimental" else METRICS[component_class][prop][0]
                result["properties"][prop] = {
                    "value": value, "unit": unit, "origin": origin,
                    "model": model, "reference_mae": mae,
                }
            if component_class == "salt":
                result["ions"] = {"cation": parts[0], "anion": parts[1]}
                result["warning"] = "Unseen anions are the main transfer limitation for salt properties."
            return result
