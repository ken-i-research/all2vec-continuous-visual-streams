# Continuous Predictive Representations for Dynamic Visual Streams (ALL2Vec)

**Author:** Ken I.  
**Email:** ken.i.research@gmail.com  
**Paper:** https://doi.org/10.5281/zenodo.17513405

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17513405.svg)](https://doi.org/10.5281/zenodo.17513405)

## Overview

This repository contains the implementation of ALL2Vec, a framework for 
continuous processing of visual streams through predictive state spaces.

## Key Features

- **Rapid Self-Organization:** Converges in <1 minute from random initialization
- **Long-Term Stability:** Operates for 3+ hours without drift
- **Cross-Domain Generalization:** Transfers immediately across visual domains
- **Consumer Hardware:** Runs on RTX 5060 at 9-10 Hz

## Quick Start

### Installation
```bash
git clone https://github.com/ken-i-research/all2vec-continuous-visual-streams.git
cd all2vec-continuous-visual-streams
pip install -r requirements.txt
```

### Run the Webcam Demo

From the repository root:
```bash
python -m all2vec.train
```

Or directly:
```bash
python src/all2vec/train.py
```

The system will:
1. Initialize random states
2. Self-organize within ~1 minute
3. Display real-time PCA visualization of attractor dynamics
4. Continue running with stable tracking

Press `Ctrl+C` to stop.

## Repository Structure
```
├── paper/
│   └── all2vec.pdf           # Paper (will be added post-Zenodo)
├── src/
│   └── all2vec/
│       ├── __init__.py
│       ├── model.py           # Core architecture
│       ├── train.py           # Training loop with webcam demo
│       └── visualize.py       # Real-time visualization
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.13+
- NVIDIA GPU (tested on RTX 5060)
- CUDA 12.8
- Webcam (for demo)

See [requirements.txt](requirements.txt) for full dependencies.

## System Behavior

The system demonstrates three key properties described in the paper:

1. **Rapid Convergence:** Random initialization → stable attractors in <1 minute
2. **Long-Term Stability:** Continuous operation for 3+ hours without drift
3. **Domain Generalization:** Works across different visual inputs without retraining

## Citation
```bibtex
@article{ken2025all2vec,
  title={ALL2Vec: Continuous Predictive Representations for Dynamic Visual Streams},
  author={Ken I.},
  year={2025},
  note={Preprint},
  doi={10.5281/zenodo.17513405},
  url={https://doi.org/10.5281/zenodo.17513405}
}
```

## License

- **Paper:** CC BY 4.0
- **Code:** MIT License

## Contact

Ken I.  
Email: ken.i.research@gmail.com

---

**Status:** 📝 Preprint available | 💻 Code released | ⭐ Star for updates