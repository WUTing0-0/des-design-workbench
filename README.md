# DES Design Workbench

This interface calculates a complete binary composition curve while keeping experimental inputs, model estimates and physical assumptions visibly separate.

## User workflow

1. Enter two components and their melting temperatures and fusion enthalpies.
2. Mark every pure property as experimental or model-estimated.
3. Choose the liquid-phase route:
   - ideal fallback, with both activity coefficients fixed to one;
   - upload a precomputed curve with columns `x,gamma1,gamma2`;
   - upload two standardized sigma-profile CSV files with columns `sigma,p_sigma`, plus the ORCA-derived molecular area and volume. The browser then evaluates the same COSMO-SAC segment and combinatorial equations used in the frozen workflow at 298.15 K.
4. Search the full composition range from 0.02 to 0.98 and inspect the physical liquidus minimum.
5. When the local service is connected, apply the frozen B4 correction and report 90% marginal calibration ranges for minimum temperature and composition. These ranges describe coverage across comparable unseen pairs; they are not guarantees for an individual chemistry.
6. Export results and record new measurements in a local update queue. Contributions are reviewed for a versioned six- or twelve-month release; the published model is never retrained silently online.

## Run the full local version

The service expects the frozen release at `C:\code2026\DES_Paper_Release`. Set `DES_WORKFLOW_ROOT` and `DES_PROJECT_ROOT` if it is stored elsewhere, then run:

```text
python local_server.py
```

Open `http://127.0.0.1:4173`. The B4 status changes from `offline` to `ready` when the archived model pack has loaded.

Required packages are `numpy`, `pandas`, `scikit-learn`, `joblib` and `rdkit`, using versions compatible with the archived model manifest.

## Scientific boundaries

- The public browser version calculates physical SLE references. It does not fake model correction when the frozen Python service is absent.
- A low-temperature prediction is not proof of DES formation, novelty, safety, single-phase stability or experimental reproducibility.
- Experimental pure-component values should replace model estimates whenever identity, solid form, units and transition meaning have been checked.
- Community measurements enter a review queue and become eligible only for a later versioned model release with new hashes, metrics and release notes.
