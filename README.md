# AI-Fiber-NLC

> **Open-source framework for AI-based nonlinear compensation in optical fiber systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/actions/workflows/test.yml/badge.svg)](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/actions/workflows/test.yml)

---

## 📄 Paper

**Limits of AI-Based Nonlinear Compensation in EDFA-Amplified Coherent Optical Fiber Systems**

A comprehensive benchmark study evaluating AI-NLC across 17 launch power points (−3 to +10 dBm) and two transmission distances (800 km and 1600 km).

**Key finding:** Neither DBP nor AI-NLC achieves meaningful Q-factor improvement over a minimal DSP chain (EDC + CPR) in standard EDFA-amplified systems. The bottleneck is accumulated ASE noise, a physical noise floor that no algorithmic approach can overcome.

- **Paper source:** [`paper/paper.tex`](paper/paper.tex)
- **Citation:** [`CITATION.cff`](CITATION.cff)

---

## Quick Start

### Install

```bash
git clone https://github.com/AI-Fiber-NLC/AI-Fiber-NLC.git
cd AI-Fiber-NLC
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Generate Simulation Data

```bash
python scripts/generate_data.py --scenarios mvb1
```

### Run DBP Benchmark

```bash
python scripts/run_dbp_benchmark.py --scenario mvb1 --power-index 4
```

### Train MLP-NLC (DSP chain integrated)

```bash
python scripts/train_mlp_dsp.py --scenario mvb1 --power-index 4 --epochs 50
```

---

## Benchmark Results Summary

| Scene | Distance | Power Range | Baseline Q | DBP Q | Improvement |
|-------|----------|-------------|-----------|-------|------------|
| MVB-1 | 800 km | −3 to +5 dBm | +1.91 to +2.20 | +1.85 to +2.20 | −0.27 to +0.03 dB |
| MVB-4 | 800 km | +7 to +10 dBm | +2.01 to +2.07 | +2.01 to +2.09 | −0.01 to +0.02 dB |
| MVB-5 | 1600 km | 0 to +3 dBm | +1.92 to +2.16 | +1.92 to +2.19 | −0.04 to +0.03 dB |

**DBP never shows meaningful improvement (>0.05 dB) in any tested regime.**

---

## Project Structure

```
├── paper/                 # Academic paper (LaTeX)
│   └── paper.tex
├── src/
│   ├── benchmark/         # Protocol: scenes, scoring, validation
│   ├── data/              # Simulation (OptiCommPy) and data loading
│   ├── dsp/               # DSP receiver chain (EDC, CR, CPR)
│   ├── models/            # NLC models (DBP, MLP, CNN)
│   └── training/          # Training utilities
├── data/raw/              # Pre-generated simulation data
│   ├── MVB-1/             # 16QAM, 800 km, −3 to +5 dBm
│   ├── MVB-3/             # 64QAM+PCS, 800 km
│   ├── MVB-4/             # 16QAM, 800 km, +7 to +10 dBm
│   └── MVB-5/             # 16QAM, 1600 km, 0 to +3 dBm
├── scripts/               # CLI tools
├── notebooks/             # Colab demo
├── tests/                 # 79 unit tests
└── models/                # Pre-trained checkpoints
```

---

## Available Scenes

| Scene | Modulation | Distance | Spans | Power Range | Status |
|-------|-----------|----------|-------|-------------|--------|
| MVB-1 | 16QAM | 800 km | 10 | −3 to +5 dBm | ✅ Data available |
| MVB-3 | 64QAM+PCS | 800 km | 10 | −3 to +5 dBm | ✅ Data available |
| MVB-4 | 16QAM | 800 km | 10 | +7 to +10 dBm | ✅ Data available |
| MVB-5 | 16QAM | 1600 km | 20 | 0 to +3 dBm | ✅ Data available |

---

## License

[MIT License](LICENSE)

## Citing

If you use this framework, please cite:

```bibtex
@misc{ai-fiber-nlc-2026,
  title = {Limits of {AI}-Based Nonlinear Compensation in {EDFA}-Amplified Coherent Optical Fiber Systems},
  author = {{AI-Fiber-NLC Contributors}},
  year = {2026},
  howpublished = {\url{https://github.com/AI-Fiber-NLC/AI-Fiber-NLC}},
}
```
