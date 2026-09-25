"""Build the validated thymol–octanoic-acid example from frozen artifacts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RELEASE = PROJECT / "DES_Paper_Release" / "77_salt_tm_role_context_propagation_v1"
PROFILES = PROJECT / "DES百万库_SLE流程" / "generated_sigma_profiles"
OUT = ROOT / "examples" / "thymol_octanoic_acid"
PORTABLE = ROOT / "portable_data"
DIST_EXAMPLE = ROOT / "dist" / "examples"

COMPONENTS = [
    {
        "id": "thymol", "name": "Thymol",
        "smiles": "Cc1ccc(C(C)C)c(O)c1", "profile_stem": "Thymol_opt",
    },
    {
        "id": "octanoic_acid", "name": "Octanoic acid",
        "smiles": "CCCCCCCC(=O)O", "profile_stem": "Fatty_Acid_C8_opt",
    },
]


def component_payload(row, spec):
    meta = json.loads((PROFILES / f"{spec['profile_stem']}.json").read_text(encoding="utf-8"))
    p = np.load(PROFILES / f"{spec['profile_stem']}.npz")["p_sigma"].astype(float)
    p /= p.sum()
    sigma = np.linspace(-0.025, 0.025, 51)
    return {
        **spec,
        "component_class": "neutral",
        "properties": {
            "tm_K": float(row.Tm_experimental),
            "hfus_kJ_mol": float(row.Hfus_experimental),
            "tm_source": str(row.Tm_source_quality),
            "hfus_source": str(row.Hfus_source_quality),
        },
        "profile": {
            "name": spec["profile_stem"],
            "sigma": sigma.tolist(),
            "p_sigma": p.tolist(),
            "area_A2": float(meta["area_ang2"]),
            "volume_A3": float(meta["volume_ang3"]),
            "method": "ORCA CPCM surface; sigma = surface charge / segment area",
            "grid": "-0.025 to 0.025 e A^-2, 0.001 spacing",
        },
    }


def main():
    for folder in (OUT, PORTABLE, DIST_EXAMPLE):
        folder.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(RELEASE / "property_master" / "PURE_PROPERTY_MASTER_EXPERIMENT_FIRST.csv")
    items = []
    selected_rows = []
    for spec in COMPONENTS:
        row = master.loc[master.component_canonical.eq(spec["smiles"])].iloc[0]
        selected_rows.append(row)
        item = component_payload(row, spec)
        items.append(item)
        pd.DataFrame({"sigma": item["profile"]["sigma"], "p_sigma": item["profile"]["p_sigma"]}).to_csv(
            OUT / f"{spec['id']}_sigma_profile.csv", index=False
        )
        shutil.copy2(PROFILES / f"{spec['profile_stem']}.json", OUT / f"{spec['id']}_sigma_metadata.json")

    payload = {
        "example_id": "thymol_octanoic_acid_v1",
        "title": "Validated example: thymol + octanoic acid",
        "selection_reason": (
            "Both components have reviewed experimental melting points and fusion enthalpies, "
            "both have computed sigma profiles, and the canonical ten-point pair-disjoint subset "
            "has B4 MAE 2.47 K (maximum absolute error 5.85 K)."
        ),
        "benchmark_note": "Pair-specific held-out performance; not the global model accuracy.",
        "components": items,
        "default_activity": "library",
        "default_model": "B4",
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    (OUT / "example.json").write_text(text, encoding="utf-8")
    (DIST_EXAMPLE / "thymol_octanoic_acid.json").write_text(text, encoding="utf-8")

    subset = pd.DataFrame(selected_rows)
    subset.to_csv(PORTABLE / "reviewed_property_examples.csv", index=False)
    (PORTABLE / "profiles").mkdir(exist_ok=True)
    for spec in COMPONENTS:
        shutil.copy2(PROFILES / f"{spec['profile_stem']}.json", PORTABLE / "profiles" / f"{spec['profile_stem']}.json")
        shutil.copy2(PROFILES / f"{spec['profile_stem']}.npz", PORTABLE / "profiles" / f"{spec['profile_stem']}.npz")

    oof = pd.read_csv(RELEASE / "benchmark_model" / "UNIFIED_HARDMAX_ABLATION_OOF.csv")
    key = "smi:CCCCCCCC(=O)O||smi:Cc1ccc(C(C)C)c(O)c1"
    validation = oof[
        oof.canonical_pair.eq(key)
        & oof.model.eq("B4_direct_plus_nonideal")
        & oof.source_row.ge(2392)
    ].copy().sort_values("source_row")
    validation["absolute_error_K"] = (validation.observed_K - validation.predicted_K).abs()
    validation.to_csv(OUT / "pair_disjoint_validation.csv", index=False)

    models = ROOT / "runtime_models"
    models.mkdir(exist_ok=True)
    shutil.copy2(RELEASE / "benchmark_model" / "FULL_CANONICAL_B3_PACK.joblib", models / "FULL_CANONICAL_B3_PACK.joblib")
    shutil.copy2(RELEASE / "benchmark_model" / "FULL_CANONICAL_B4_B5_PACK.joblib", models / "FULL_CANONICAL_B4_B5_PACK.joblib")


if __name__ == "__main__":
    main()
