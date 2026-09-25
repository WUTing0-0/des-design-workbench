# DES Design Workbench

A provenance-aware interface for exploring binary liquidus predictions. The currently implemented local calculation is:

`SMILES → reviewed property/profile lookup → explicit choice for missing inputs → SLE/reference curve → B3 or B4 correction`

## What the interface does

1. Accepts two molecular structures as SMILES.
2. Canonicalizes each structure and searches the frozen, experiment-first pure-property table.
3. Preserves reviewed experimental melting points and fusion enthalpies. A library search never fills a gap silently; the user must enter a value or explicitly request one of the four frozen property models:
   - neutral melting point: four-layer GIN;
   - neutral fusion enthalpy: TabPFN;
   - salt melting point: role-aware cation/anion G-TabPFN;
   - salt fusion enthalpy: TabPFN.
4. Searches the canonical sigma-profile inventory. When both components are present, their profiles are displayed and selected automatically. Otherwise the interface waits for the user to choose ideal coefficients, upload `gamma(x)`, or upload two profiles.
5. Calculates the binary hard-max SLE curve from `x = 0.02–0.98`. The frozen B3 direct model corrects the ideal-SLE route; the frozen B4 direct model corrects non-ideal routes.
6. Reports the curve and input provenance. Formal applicability and optimized-target interval deployment are not yet integrated for arbitrary user inputs; the interface explicitly says **Not assessed** instead of applying an obsolete pooled interval.

The hosted interface performs physical browser calculations and automatically connects to a local service at `127.0.0.1:4173` when it is running. Reviewed-library search, missing-property inference and B3/B4 correction remain local because the frozen scientific models, reviewed database and third-party foundation weights are not embedded in the hosted site.

## Run the full local workflow

For a quick browser demonstration, click **Load example**, then inspect the composition curve. Example inputs are illustrative, not reviewed experimental measurements. For your own system, enter four measured properties, or start the local service to resolve missing values from SMILES. Activity inputs and optional model settings are collapsed under **Activity coefficients and model options**. Pure physical prediction remains available without a model server.

The browser uses the workflow's simplified COSMO-based electrostatic/size–shape implementation when sigma profiles are supplied; it does not implement every standard COSMO-SAC variant. Ideal `gamma = 1` is corrected with B3, whose inputs are structure, composition, pure properties and ideal-SLE features. It is not mislabeled as B4, because B4 additionally requires non-ideal features.

Create a Python environment compatible with `requirements-local.txt`. The tested Windows environment uses Python 3.12, PyTorch 2.6 CPU, RDKit 2026.03.4, scikit-learn 1.5.2 and the TabPFN 7.1.1 runtime with the frozen v2.6 regression checkpoint. Then set these optional paths if your release is not stored in the defaults:

```text
DES_WORKFLOW_ROOT=C:\code2026\DES_Paper_Release
DES_PROJECT_ROOT=C:\code2026
DES_TABPFN_DEVICE=cpu
```

Start `local_server.py` and open `http://127.0.0.1:4173`.

The first unseen-molecule inference is slow on CPU because the molecular encoders and TabPFN context are loaded once. A CUDA environment is recommended for repeated arbitrary-structure inference. Exact database matches return immediately. The frozen scientific setting uses 16 TabPFN estimators; `DES_TABPFN_ESTIMATORS` exists only for smoke testing and must not be used for reported results.

## Required private/release artifacts

The repository intentionally excludes raw licensed datasets, pretrained foundation checkpoints and frozen model binaries. `MODEL_MANIFEST.json` records their expected release-relative locations, reference metrics and SHA-256 hashes. Publication archives should supply these artifacts through the paper data repository, subject to their original licenses.

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
