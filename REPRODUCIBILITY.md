# Reproducibility index

1. Install the pinned Python environment from requirements.txt.
2. Run the NP038, NP039, NP040 and NP041 verification scripts under code/script.
3. Recompute the score tables from the processed arrays in data/processed and results.
4. Check release_manifest.json before using any result.

The package contains processed aggregate/feature/label arrays and frozen
predictions. It does not contain raw turbine-level SCADA, coordinates,
operational-status records, API credentials or private source responses.
Anonymous readers can clone this repository and run the verification scripts
without an account.
