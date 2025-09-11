# G-PROG: modeling glaucoma progression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Code, sample data, and figures accompanying our article:

> **Prediction of structural glaucoma progression from baseline fundus photographs using deep learning: a retrospective multicentre study**  
> Authors: Hemelings, R. et al.  
> Submitted to *Journal Name*, 2025

---

## 🔍 Overview

Glaucoma is a leading cause of irreversible blindness.  
**G-PROG** provides a reproducible pipeline to:

- Train and evaluate progression models for longitudinal glaucoma data  
- Generate *progplots* (progression plots) for visualization  
- Reproduce figures and metrics reported in our paper  

The repository is designed for transparency, reproducibility, and extensibility.

---

## 📂 Repository Structure

```yaml
G-PROG/
├─ src/
├─ scripts/
├─ notebooks/
├─ data/
│ ├─ sample_images/
│ ├─ metadata/
│ └─ raw/
├─ results/
│ ├─ progplots/
│ └─ metrics/
├─ docs/
├─ environment.yml
├─ requirements.txt
└─ LICENSE # License for code (and data license separately if needed)
```

## ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/rubenhx/G-PROG.git
cd G-PROG
```
Set up the environment (conda):

```bash
conda env create -f environment.yml
conda activate gprog
```

Or install with pip:

```bash
pip install -r requirements.txt
```

## 📊 Data

This project is based on the GRAPE dataset:

> Huang, X., et al. (2023). 
> GRAPE: A multi-modal glaucoma dataset of follow-up visual field and fundus images for glaucoma management. 
> Scientific Data, 10: 561. 
> https://springernature.figshare.com/collections/GRAPE_A_multi-modal_glaucoma_dataset_of_follow-up_visual_field_and_fundus_images_for_glaucoma_management/6406319/1 

Source Licenses

CFPs (color fundus photographs): released under CC0 (public domain).

Visual field & clinical data: released under CC0 (public domain).

Other dataset components (e.g., ROI, annotations): provided under CC BY 4.0.

Derived Data

In this work, we generated optic disc–centered 30° crops from the CFPs.
Because CFPs are under CC0, these crops are also free of redistribution restrictions.
However, please cite the original GRAPE dataset when using them:

```bibtex
@article{huang2023grape,
  title   = {GRAPE: A multi-modal glaucoma dataset of follow-up visual field and fundus images for glaucoma management},
  author  = {Huang, X. and others},
  journal = {Scientific Data},
  volume  = {10},
  number  = {561},
  year    = {2023},
  publisher = {Nature Publishing Group},
  doi     = {10.1038/s41597-023-02424-4}
}
```

Access

Sample crops: small anonymized subset included under data/sample_images/ (via Git LFS).

Full dataset: not hosted here — download directly from the Figshare collection
 or use DVC/git-annex configuration provided in this repo to manage access.

## 🚀 Usage

Train a model:
```bash
python scripts/train.py --config configs/train.yaml
```

Evaluate performance:
```bash
python scripts/eval.py --checkpoint checkpoints/model.pt
```

Generate progression plots:
```bash
python scripts/make_progplots.py --out results/progplots/
```

## 📈 Results

Plots are saved under results/progplots/.

## 📜 Citation

If you use G-PROG, please cite:

```bibtex
@article{hemelings2025gprog,
  title   = {Prediction of structural glaucoma progression from baseline fundus photographs using deep learning: a retrospective multicentre study},
  author  = {Your Name and Others},
  journal = {Journal Name},
  year    = {2025},
  doi     = {}
}
```

## ⚖️ License

Code: MIT License

Data: Sample images are released under CC BY-NC 4.0

Access to the full dataset is restricted and requires appropriate agreements.

## 🙏 Acknowledgements

This project was supported by …
We thank collaborators and institutions who contributed to data collection and analysis.
