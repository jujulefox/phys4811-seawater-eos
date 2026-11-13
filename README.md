# Learning the seawater equation of state with ML

Final project for PHYS4811 (Machine Learning in Physics), HKUST, Fall 2026.
Author: Juliette Renard

We train models to predict seawater density from salinity, temperature
and pressure. The true density comes from TEOS-10 (gsw package).
We compare linear regression, an MLP and LightGBM, and test how well
they extrapolate to depths not seen in training.

## Files

- clean_calcofi.py: reads the raw CalCOFI bottle data and writes data/calcofi_clean.csv
- seawater_eos.ipynb: training, comparison and extrapolation test
- requirements.txt: Python packages used

## Data

Raw data: CalCOFI bottle database, https://calcofi.org/data/oceanographic-data/bottle-database/
Downloaded on 28 October 2026

The raw data is not in this repo. Download it and put it in data/.

## How to run

1. pip install -r requirements.txt
2. python clean_calcofi.py
3. Open seawater_eos.ipynb and run all cells.
