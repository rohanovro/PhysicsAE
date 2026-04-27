#!/usr/bin/env python
"""
run_all.py  –  Complete end-to-end pipeline for PhysicsAE.

Usage
-----
    python scripts/run_all.py

What it does
------------
1. Load and preprocess the Paderborn bearing dataset
2. Train PhysicsAE  (with physics loss)
3. Train Baseline AE  (without physics loss)
4. Train Deep SVDD
5. Evaluate all models and print + save a results table
6. Save model checkpoints to results/models/
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from sklearn.svm import OneClassSVM

from config import CONFIG, FAULT_FREQUENCIES
from data_loader import load_paderborn_data
from evaluate import (
    build_test_arrays,
    evaluate_all_models,
    evaluate_model,
    get_anomaly_scores,
    print_results_table,
    save_results_csv,
)
from train import train_physicsae, train_deep_svdd
from utils import save_model, count_parameters


ZIP_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'paderborn-db.zip')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    print("=" * 70)
    print("  PHYSICSAE  –  Complete Pipeline")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading Paderborn dataset ...")
    X_train, X_val, X_fault, scaler = load_paderborn_data(ZIP_PATH, CONFIG)

    # ── 2. Train PhysicsAE ────────────────────────────────────────────────────
    print("\n[2/5] Training PhysicsAE ...")
    physics_model = train_physicsae(
        X_train, X_val, FAULT_FREQUENCIES, CONFIG,
        use_physics=True, seed=42, verbose=True,
    )
    save_model(physics_model, os.path.join(RESULTS_DIR, 'models', 'physicsae.pt'))
    print(f"  Parameters: {count_parameters(physics_model):,}")

    # ── 3. Train Baseline AE ──────────────────────────────────────────────────
    print("\n[3/5] Training Baseline AE ...")
    baseline_model = train_physicsae(
        X_train, X_val, FAULT_FREQUENCIES, CONFIG,
        use_physics=False, seed=42, verbose=True,
    )
    save_model(baseline_model, os.path.join(RESULTS_DIR, 'models', 'baseline_ae.pt'))

    # ── 4. Train Deep SVDD ────────────────────────────────────────────────────
    print("\n[4/5] Training Deep SVDD ...")
    svdd_model = train_deep_svdd(X_train, X_val, CONFIG, seed=42, verbose=True)
    save_model(svdd_model, os.path.join(RESULTS_DIR, 'models', 'deep_svdd.pt'))

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating ...")
    X_test, y_test = build_test_arrays(X_val, X_fault)

    models = {
        'PhysicsAE':  physics_model,
        'Baseline AE': baseline_model,
        'Deep SVDD':   svdd_model,
    }

    df = evaluate_all_models(models, X_test, y_test, CONFIG)
    print_results_table(df)
    save_results_csv(df, os.path.join(RESULTS_DIR, 'tables', 'main_results.csv'))

    print("\nDone! All outputs saved in results/")


if __name__ == '__main__':
    main()
