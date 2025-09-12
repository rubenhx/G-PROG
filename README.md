# G-PROG: modeling glaucoma progression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Code, sample data, and figures accompanying our article:

> **Prediction of structural glaucoma progression from baseline fundus photographs using deep learning: a retrospective multicentre study**  
> Authors: Hemelings, R. et al.  
> Submitted to *Journal Name*, 2025

---

## Overview

Glaucoma is a leading cause of irreversible blindness. Identifying patients at risk of rapid disease progression is critical to preventing vision loss.

**G-PROG** provides a reproducible pipeline to:

- Train and evaluate progression models for longitudinal glaucoma data  
- Generate *progplots* (progression plots) for visualization  
- Reproduce selected figures and metrics reported in our paper  

The repository is designed for transparency, reproducibility, and extensibility.

---

![Example of glaucoma progressor and G-PROG output](fig1.jpg)

## Repository Structure

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
└─ LICENSE
```

## Installation

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

## Data

This project conducted external testing on the publicly available GRAPE dataset:

> Huang, X., et al. (2023).  
> GRAPE: A multi-modal glaucoma dataset of follow-up visual field and fundus images for glaucoma management.  
> Scientific Data, 10: 561.  
> https://springernature.figshare.com/collections/GRAPE_A_multi-modal_glaucoma_dataset_of_follow-up_visual_field_and_fundus_images_for_glaucoma_management/6406319/1  

### Source Licenses

CFPs (color fundus photographs): released under CC0 (public domain).  
Visual field & clinical data: released under CC0 (public domain).

**Derived Data**

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

**Access**

Sample crops: small anonymized subset included under data/sample_images/
Full dataset: not hosted here — download directly from the Figshare collection

## Usage

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

## Results

Plots are saved under results/progplots/

## Citation

If you use G-PROG, please cite:

```bibtex
@article{hemelings2025gprog,
  title   = {Prediction of structural glaucoma progression from baseline fundus photographs using deep learning: a retrospective multicentre study},
  author  = {Hemelings, R. and others},
  journal = {Journal Name},
  year    = {2025},
  doi     = {}
}
```

## License

Code: MIT License
Data: Sample images are released under CC BY-NC 4.0
Access to the other datasets used in our project is restricted and requires appropriate agreements.

## Acknowledgements

We thank all research partners that made this international collaboration possible. 
This work was funded by grants from the National Medical Research Council (OFLCG/004c/2018-00; MOH-000249-00; MOH-000647-00; MOH-001001-00; MOH-001015-00; MOH-000500-00; MOH-000707-00; MOH-001072-06; MOH-001286-00), National Research Foundation Singapore (NRF2019-THE002-0006 and NRF-CRP24-2020-0001), Agency for Science, Technology and Research (A20H4b0141) and the Singapore Eye Research Institute & Nanyang Technological University (SERI-NTU Advanced Ocular Engineering (STANCE) Program).
The study was supported by the Competitive Research Funding of the Pirkanmaa Wellbeing Services County for AT (grant no. 9AA076). LUX –foundation for glaucoma research for AT and HU-J. State funding for university-level health research, Tampere University Hospital, Wellbeing services county of Pirkanmaa (T63464) for HU-J and Tampere University Hospital Support Foundation, Tampere University Hospital, Wellbeing services county of Pirkanmaa (T64124) for HU-J.
We thank the administrators from Istekki Oy for the technical support and retrieval of figures and graphs.
The BEGONIA project was funded by the BOCSS (Belgian Ophthalmology Cooperation in Clinical Sciences) initiative hosted by the FRO (Funds for Research in Ophthalmology). 

