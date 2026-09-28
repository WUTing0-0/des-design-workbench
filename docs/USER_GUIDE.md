# DES Design Workbench user guide

Workbench calculates composition-dependent liquidus curves for a binary mixture using pure-component properties and activity inputs. The supplied frozen models can correct the ideal and non-ideal reference curves.

## Run the example

1. Download and extract the portable package linked in the repository README. Follow the installation instructions to start the local service.
2. Open `http://127.0.0.1:4173` and select **Load validated example**.
3. Review the four experimental pure-component properties and the two sigma profiles for thymol–octanoic acid, then run the calculation.
4. Inspect the composition curve and export the composition-resolved CSV and complete JSON report.

The example's held-out error describes this specific pair. It is not an uncertainty estimate for other mixtures.

## Calculate your own mixture

1. Enter the names and SMILES of both components. Check component identities and salt forms.
2. Search for available experimental properties and sigma profiles. Review the sources before accepting a match.
3. Supply each component's melting point in K and fusion enthalpy in kJ mol⁻¹. In the portable edition, missing properties outside the bundled library must be entered manually. The full research installation can use property models when their dependencies and appropriately licensed weights are installed.
4. Select an activity route: ideal coefficients (`gamma = 1`), an uploaded `gamma(x)` curve, or two sigma profiles.
5. Run the composition curve and model correction. The ideal route uses B3; routes with non-ideal features use B4.
6. Export the report to retain input sources, model versions and the calculation route.

Activity-curve CSV files require columns `x,gamma1,gamma2`. Sigma-profile CSV files require `sigma,p_sigma`, a common 51-point grid for both components, and molecular area and volume inputs.

## Interpret the output

Use the curves to compare compositions and plan experimental measurements. The paper's complete applicability assessment and optimized-target calibration are not deployed for arbitrary new inputs. Their reliability is therefore displayed as **Not assessed**; the interface does not claim validated 90% coverage for these calculations. A low predicted temperature alone does not establish DES formation or phase stability.

New measurements can be exported for source review and later versioned updates. Running a calculation does not retrain the published models online.
