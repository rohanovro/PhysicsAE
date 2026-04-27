# PhysicsAE: Physics-Guided Self-Supervised Autoencoder for Bearing Fault Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()

This repository contains the official implementation of **PhysicsAE**, a physics-guided self-supervised autoencoder for bearing fault detection, as presented in:

> **Physics-Guided Self-Supervised Autoencoder for Bearing Fault Detection Using Harmonic Frequency Constraints and Frequency-Aware Explainability**  
> *Submitted to Measurement, Elsevier*

---

## 📊 Key Results

| Model | AUC | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| **PhysicsAE** | **0.9052** | **0.7358** | **0.9960** | **0.5834** |
| AE Baseline | 0.8853 | 0.7282 | 0.9857 | 0.5774 |
| One-Class SVM | 0.9588 | 0.7380 | 0.9989 | 0.5851 |
| Deep SVDD | 0.5963 | 0.5500 | 0.5400 | 0.5600 |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/PhysicsAE.git
cd PhysicsAE
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Paderborn dataset

Download from: https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/

Place `paderborn-db.zip` in the `data/` folder.

### 4. Run the complete pipeline

```bash
python scripts/run_all.py
```

---

## 📁 Repository Structure

```
PhysicsAE/
├── src/               # Core source code (models, training, data loading)
├── notebooks/         # Jupyter notebooks for exploration and figures
├── scripts/           # End-to-end pipeline runner
├── data/              # Place your downloaded dataset here
├── results/           # Auto-generated figures, tables, model checkpoints
└── tests/             # Unit tests
```

---

## 📈 Reproducibility

All paper results are reproducible by running:

```bash
python scripts/run_all.py
```

Expected output:
- Figures in `results/figures/`
- Tables in `results/tables/`
- Model checkpoints in `results/models/`

---

## 📝 Citation

```bibtex
@article{rohan2025physicsae,
  title={Physics-Guided Self-Supervised Autoencoder for Bearing Fault Detection},
  author={Rohan, Mahmudul Hasan},
  journal={Measurement},
  year={2025}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
