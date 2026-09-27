# DES Design Workbench

A provenance-aware interface for exploring binary liquidus predictions.

This repository and its portable download are public. No GitHub access approval is required. The separate scientific archive is still undergoing third-party data redistribution review.

The currently implemented local calculation is:

`SMILES → reviewed property/profile lookup → explicit choice for missing inputs → SLE/reference curve → B3 or B4 correction`

## What the interface does

1. Accepts two molecular structures as SMILES.
2. Canonicalizes each structure and searches the frozen, experiment-first pure-property table.
3. Preserves reviewed experimental melting points and fusion enthalpies. A library search never fills a gap silently; the user must enter a value or explicitly request one of the four frozen property models:
   - organic-compound melting point: four-layer GIN;
   - organic-compound fusion enthalpy: TabPFN;
   - salt melting point: role-aware cation/anion G-TabPFN;
   - salt fusion enthalpy: TabPFN.
4. Searches the canonical sigma-profile inventory. When both components are present, their profiles are displayed and selected automatically. Otherwise the interface waits for the user to choose ideal coefficients, upload `gamma(x)`, or upload two profiles.
5. Calculates the binary hard-max SLE curve from `x = 0.02–0.98`. The frozen B3 direct model corrects the ideal-SLE route; the frozen B4 direct model corrects non-ideal routes.
6. Reports the curve and input provenance. Formal applicability and optimized-target interval deployment are not yet integrated for arbitrary user inputs; the interface explicitly says **Not assessed** instead of applying an obsolete pooled interval.

The hosted interface performs physical browser calculations and automatically connects to a local service at `127.0.0.1:4173` when it is running. Reviewed-library search, missing-property inference and B3/B4 correction remain local because the frozen scientific models, reviewed database and third-party foundation weights are not embedded in the hosted site.

The current paper data and supplementary-file versions are tracked in the scientific repository's [release status](https://github.com/WUTing0-0/des-physics-ml-atlas/blob/main/docs/RELEASE_STATUS_20260927.md). Its calibrated Atlas intervals should not be applied automatically to arbitrary inputs in this interface. The organic-compound property tasks retain the source-dataset definitions; they are not a claim that every source structure is strictly non-ionic.

## Run the full local workflow

For a quick demonstration, click **Load validated example**. The bundled thymol–octanoic-acid case contains four reviewed experimental pure properties and two computed sigma profiles. Its canonical ten-point pair-disjoint subset has B4 MAE 2.47 K (maximum absolute error 5.85 K); this is pair-specific held-out evidence, not a global accuracy claim. For your own system, enter four measured properties or use the full research installation to resolve missing values from SMILES.

The browser uses the workflow's simplified COSMO-based electrostatic/size–shape implementation when sigma profiles are supplied; it does not implement every standard COSMO-SAC variant. Ideal `gamma = 1` is corrected with B3, whose inputs are structure, composition, pure properties and ideal-SLE features. It is not mislabeled as B4, because B4 additionally requires non-ideal features.

Create a Python environment compatible with `requirements-local.txt`. The tested Windows environment uses Python 3.12, PyTorch 2.6 CPU, RDKit 2026.03.4, scikit-learn 1.5.2 and the TabPFN 7.1.1 runtime with the frozen v2.6 regression checkpoint. Then set these optional paths if your release is not stored in the defaults:

```text
DES_WORKFLOW_ROOT=C:\code2026\DES_Paper_Release
DES_PROJECT_ROOT=C:\code2026
DES_TABPFN_DEVICE=cpu
```

Start `.venv\Scripts\python.exe local_server.py` and open `http://127.0.0.1:4173`. The hosted page can also connect to the running local service.

The first unseen-molecule inference is slow on CPU because the molecular encoders and TabPFN context are loaded once. A CUDA environment is recommended for repeated arbitrary-structure inference. Exact database matches return immediately. The frozen scientific setting uses 16 TabPFN estimators; `DES_TABPFN_ESTIMATORS` exists only for smoke testing and must not be used for reported results.

## Model dependencies

The portable archive includes the frozen B3/B4 mixture models. Third-party foundation checkpoints and restricted source datasets are not included. `MODEL_MANIFEST.json` records the model paths, reference metrics and SHA-256 hashes needed for the full research installation. Obtain third-party dependencies from their approved distribution channels.

## Portable download

`release_assets/DES_Design_Workbench_Portable_v0.3.0.zip` is built as a standalone local package. It does not depend on ChatGPT: users unzip it, run `INSTALL_AND_START_WINDOWS.bat` (or `install_and_start_linux.sh`), and open `http://127.0.0.1:4173`. The archive includes the validated example and frozen B3/B4 phase-equilibrium models. Version 0.3.0 separates the calibrated-range field from the pair-specific example check, exports both a report and a composition-curve CSV, and prevents the portable interface from offering unavailable pure-property inference. Third-party molecular foundation models are deliberately not redistributed; outside the bundled reviewed example, users can supply measured pure properties and activity inputs manually. See `PORTABLE_README_CN.md`.

The intended user journey and the remaining usability priorities are recorded in [`docs/USER_WORKFLOW_AND_RELEASE_PLAN_CN.md`](docs/USER_WORKFLOW_AND_RELEASE_PLAN_CN.md).

## Input formats

- Salt structures must contain one dot-separated cation and one anion. Ion roles are assigned from formal charges; ambiguous multi-fragment structures are rejected rather than guessed.
- Activity curve CSV: `x,gamma1,gamma2`.
- Sigma profile CSV: `sigma,p_sigma` on the same 51-point grid for both components, plus molecular area and volume.
- Temperatures are Kelvin; fusion enthalpies are kJ mol⁻¹.

## Scientific boundaries

- Experimental pure-component values are used only after identity, units, solid form and transition meaning have been reviewed.
- Model MAEs are population-level reference errors, not molecule-specific confidence probabilities.
- User-input B3/B4 outputs are exploratory. Manuscript-calibrated ranges must not be applied until profile identity, input regime, numerical eligibility and production applicability have been checked. The website does not yet perform that complete deployment check.
- A low predicted liquidus temperature is not evidence of DES formation, novelty, safety, single-phase stability or experimental reproducibility.
- Submitted measurements enter a review queue. The published model is updated only in a new versioned release; it is never retrained silently online.

## Repository layout

- `dist/`: static browser interface and physical calculations.
- `local_server.py`: local HTTP service for library lookup and B3/B4 correction.
- `property_inference.py`: experiment-first lookup and unified four-model inference interface.
- `MODEL_MANIFEST.json`: frozen artifact contract and hashes.
- `requirements-local.txt`: local runtime requirements.
