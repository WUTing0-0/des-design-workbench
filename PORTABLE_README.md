# DES Design Workbench: local installation

The application runs on your computer at `http://127.0.0.1:4173`. It does not require ChatGPT.

## Windows

1. Extract the portable package to a folder such as `D:\DES_Design_Workbench`.
2. Install Python 3.11 or 3.12 from python.org. Include the Python launcher and select **Add Python to PATH**.
3. Double-click `INSTALL_AND_START_WINDOWS.bat`. The first launch installs the Python dependencies and requires internet access. Subsequent launches use the installed environment.
4. Keep the application terminal open. Open `http://127.0.0.1:4173` in a browser and select **Load validated example**.
5. To stop the service, press Ctrl+C in its terminal.

## Ubuntu / Linux

From the extracted directory, run `bash install_and_start_linux.sh`, then open `http://127.0.0.1:4173`.

## Included functionality

The package includes experimental pure-component properties and two sigma profiles for thymol–octanoic acid, along with frozen B3/B4 mixture models. For other mixtures, supply measured melting points and fusion enthalpies, and select ideal activity coefficients, upload an activity curve, or supply two sigma profiles. Export the complete report and composition-resolved CSV after calculation.

Third-party foundation checkpoints are not bundled. Missing pure-component properties outside the example library require manual input unless the full research environment and appropriately licensed property-model dependencies are installed. Formal calibrated intervals are not supplied for arbitrary new mixtures without the corresponding applicability and calibration checks.

See the [user guide](docs/USER_GUIDE.md) for input formats and interpretation. Predictions support experimental screening; they do not establish DES formation, phase stability, safety or novelty.
