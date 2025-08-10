# All Cities are Equal: A Unified Human Mobility Generation Model Enabled by Large Language Models

This repository contains the official implementation of **UniMob**, a unified framework for **synthetic human mobility generation across cities**, integrating Large Language Models (LLMs), a spatio-temporal encoder, and a diffusion-based mobility generator.

## 📖 Framework Overview

UniMob consists of three core components:

1. **LLM-based Travel Planner**  
  Fine-tunes an LLM to infer **temporal** and **semantic** travel plans from contextual inputs (current time and POI distribution), providing high-level intent guidance for mobility generation.
  
2. **Unified Spatial Embedding**  
  Encodes diverse city spatial structures into a **shared latent space** via a spatio-temporal encoder, and decodes them back to city-specific identifiers through a lightweight decoder for cross-city generalization.
  
3. **Diffusion-based Mobility Generator**  
  Uses a conditional **Diffusion Transformer (DiT)** to generate mobility sequences in the unified space, guided by travel plans and start conditions, producing realistic and diverse trajectories.
  

<div align="center">
<img src="docs/framework.png" width="700">
</div>

---

## 📂 Project Structure

```
.
├── data/                 # Datasets (see download links below)
├── dataset/              # Dataset processing scripts
├── model/                # Model definitions
├── run/
│   ├── pretrain/
│   │   ├── LLM/          # LLM travel planner fine-tuning
│   │   │   └── train_LLM.ipynb
│   │   ├── Encoder/      # Spatio-temporal encoder pretraining
│   │   │   ├── train_Encoder.ipynb
│   │   │   └── test_Encoder.ipynb
│   │   └── Diffusion/    # Diffusion mobility generator pretraining
│   │       └── train_Diffusion.py
│   └── tuning/           # Target-city adaptation
│       └── tune_Projector.ipynb
├── trainer/              # Training loops
├── args.py               # Argument parser
├── util.py               # Utility functions
└── README.md
```

---

## 📊 Datasets

We use two real-world human mobility datasets covering five cities:

1. **Private Car GPS dataset** (Guangzhou, Shenzhen, Changsha)  
  [CoPB dataset](https://anonymous.4open.science/r/CoPB)  
  Derived from in-vehicle GPS devices, mapped to discrete urban regions.
  
2. **Mobile Phone Location dataset** (Beijing, Shenzhen, Shanghai)  
  [Rutgers dataset](https://www.cs.rutgers.edu/~dz220/data.html)  
  Collected from GPS and cell-tower localization.
  
3. **Telecom-based Mobility dataset**  
  [Telecom dataset](https://wangshangguang.github.io/telecom_dataset/)  
  Includes large-scale POI and mobility records.
  

POIs are mapped to high-level semantic categories, and each region is assigned a POI distribution vector for semantic context.

---

## 🚀 Training Pipeline

### **Step 1 – Fine-tune the LLM Travel Planner**

Generates temporal & semantic travel plans.

```bash
# Notebook execution
run/pretrain/LLM/train_LLM.ipynb
```

### **Step 2 – Pre-train the Spatio-temporal Encoder**

Encodes POI + spatial features into a unified latent space.

```bash
run/pretrain/Encoder/train_Encoder.ipynb
```

### **Step 3 – Pre-train the Diffusion Mobility Generator**

Trains DiT to generate latent mobility sequences.

```bash
python run/pretrain/Diffusion/train_Diffusion.py
```

### **Step 4 – Fine-tune for Target City Adaptation**

Trains a lightweight decoder/projector with small labeled data from target city.

```bash
run/tuning/tune_Projector.ipynb
```

---

## 📜 Citation

If you use this code, please cite our paper:

```bibtex
@article{unimob2025,
  title={All Cities are Equal: A Unified Human Mobility Generation Model Enabled by Large Language Models},
  author={Your Name and Others},
  year={2025},
  journal={Proceedings of ...}
}
```
