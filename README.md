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

**Note:** Full implementation code will be released following paper review and publication. 
The repository currently contains project documentation and will be updated with:

- Complete training implementation
- Real-time visualization tools
- Pre-trained model checkpoints
- Usage examples and tutorials

Star/watch this repository to be notified of updates.

### Planned Usage (Preview)
```bash
# Installation (coming soon)
git clone https://github.com/ken-i-research/all2vec-continuous-visual-streams.git
cd all2vec-continuous-visual-streams
pip install -r requirements.txt

# Real-time webcam processing with visualization
python src/run_continuous.py --webcam --visualize

# Process video file
python src/run_continuous.py --video path/to/video.mp4
```

For questions or early access requests, contact: ken.i.research@gmail.com

## Repository Structure
```
├── paper/
│   └── all2vec.pdf           # Paper (will be added post-Zenodo)
├── src/                      # Source code (coming soon)
│   ├── model.py             # Core architecture
│   ├── train.py             # Training loop
│   └── visualize.py         # Real-time visualization
├── examples/                 # Usage examples (coming soon)
├── assets/                   # Demo visualizations (coming soon)
├── requirements.txt
└── README.md
```

## Requirements
```
torch==2.7.1; python_version >= "3.13" and python_version < "3.14"
torchvision==0.22.1; python_version >= "3.13" and python_version < "3.14"
torchaudio==2.7.1; python_version >= "3.13" and python_version < "3.14"

opencv-python
numpy
matplotlib
scikit-learn
pillow
```

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
- **Code:** MIT License (to be released)

## Contact

Ken I.  
Email: ken.i.research@gmail.com

---

**Status:** 📝 Paper submitted | 💻 Code preparation in progress | ⭐ Watch for updates