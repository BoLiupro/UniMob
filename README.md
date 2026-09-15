<div align="center">

# UniMob
### All Cities are Equal: A Unified Human Mobility Generation Model Enabled by LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2602.19694-b31b1b.svg)](https://arxiv.org/abs/2602.19694)
[![PDF](https://img.shields.io/badge/PDF-Download-blue)](https://arxiv.org/pdf/2602.19694)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)](https://arxiv.org/abs/2602.19694)

**Bo Liu · Tong Li · Zhu Xiao · Ruihui Li · Geyong Min · Zhuo Tang · Kenli Li**

</div>

---

## Overview

**UniMob** is a unified human mobility generation framework designed to generalize across cities with different spatial structures and data availability. It combines LLM-based semantic planning, unified spatial representation learning, and diffusion-based trajectory generation.

<p align="center">
  <img src="framework.png" width="92%" alt="UniMob framework" />
</p>

## Highlights

- **LLM-powered travel planner** for high-level temporal and semantic travel intentions.
- **Unified spatial embedding** that maps heterogeneous city regions into a shared representation space.
- **Diffusion-based mobility generator** for realistic and diverse trajectory generation.
- **Cross-city generalization** evaluated in zero-shot and few-shot settings.
- Experiments on **two real-world datasets covering five cities**, with improvements of more than 30% across multiple metrics reported in the paper.

## Paper & Download

- **arXiv:** https://arxiv.org/abs/2602.19694
- **PDF:** https://arxiv.org/pdf/2602.19694
- **Source:** https://arxiv.org/src/2602.19694
- **Status:** Under review.

## Framework

UniMob contains three main components:

1. **LLM-based Travel Planner** — derives temporally aware and semantically meaningful travel plans from contextual information.
2. **Unified Spatial Embedding** — projects regions from different cities into a shared latent space and supports city-specific decoding.
3. **Diffusion-based Mobility Generator** — generates mobility sequences conditioned on travel plans and initial states.

## Repository Structure

```text
UniMob/
├── data/                 # Data files / download instructions
├── dataset/              # Data preprocessing
├── model/                # Model definitions
├── run/
│   ├── pretrain/
│   │   ├── LLM/          # Travel planner
│   │   ├── Encoder/      # Unified spatial encoder
│   │   └── Diffusion/    # Mobility generator
│   └── tuning/           # Target-city adaptation
├── trainer/              # Training loops
├── args.py
├── util.py
└── README.md
```

## Datasets

The experiments use real-world human mobility datasets covering multiple cities, including private-car GPS trajectories, mobile-phone location data, and telecom mobility data. Please follow the original data providers' licenses and privacy requirements when reproducing the experiments.

## Citation

```bibtex
@article{liu2026unimob,
  title   = {All Cities are Equal: A Unified Human Mobility Generation Model Enabled by LLMs},
  author  = {Liu, Bo and Li, Tong and Xiao, Zhu and Li, Ruihui and Min, Geyong and Tang, Zhuo and Li, Kenli},
  journal = {arXiv preprint arXiv:2602.19694},
  year    = {2026}
}
```

## Contact

For questions or collaboration, please open an issue or contact **Bo Liu** at `liubo317@hnu.edu.cn`.
