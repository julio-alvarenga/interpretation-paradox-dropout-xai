# The Interpretation Paradox in Educational Data Mining

Companion code for the article **"The Interpretation Paradox in Educational Data Mining: A Comparative XAI Audit of Supervised and One-Class Dropout Predictors"** (J. C. N. Alvarenga and C. Z. Resende, Federal Institute of Espirito Santo, Ifes).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/julio-alvarenga/interpretation-paradox-dropout-xai/blob/main/Notebook_Interpretation_Paradox_XAI_Audit_EN.ipynb)

The notebook compares supervised dropout predictors (Random Forest with six resampling strategies and cost-sensitive XGBoost) with One-Class detectors (OC-SVM, Isolation Forest, LOF, Autoencoder), tests the differences statistically, audits SHAP and LIME explanations of XGBoost and Isolation Forest, and runs an exploratory equity audit with a two-layer simulation.

## Repository contents

| File | Description |
|---|---|
| `interpretation_paradox_dropout_xai.ipynb` | Full pipeline, with the outputs of the run reported in the article |
| `requirements.txt` | Library versions used |
| `data/README.md` | How to obtain the dataset (not redistributed) |
| `CITATION.cff`, `LICENSE` | Citation metadata and license (MIT for the code) |

## How to run

**Google Colab.** Open the notebook with the badge above, upload the dataset to the session as `data/dataset.csv` (create a `data` folder), and run all cells. The first cell installs pinned versions of Optuna, SHAP, LIME, XGBoost and imbalanced-learn.

**Locally.**
```bash
git clone https://github.com/julio-alvarenga/interpretation-paradox-dropout-xai.git
cd interpretation-paradox-dropout-xai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
# place the dataset at data/dataset.csv (see data/README.md)
jupyter notebook interpretation_paradox_dropout_xai.ipynb
```
In a local run the Google Drive cell prints a notice and is skipped. Paths can be changed with the environment variables `DATASET_PATH` and `OUTPUT_DIR`. Setting `MODO_TESTE_RAPIDO=1` runs a reduced version (3 trials, 2 seeds) for smoke tests; its results differ from the article.

## Notebook structure

| Cells | Content |
|---|---|
| 1-4 | Dependencies, configuration, imports and determinism, hardware check |
| 5-9 | Raw data inspection (SHA-256 check), preprocessing and 60/20/20 split, exploratory analysis, evaluation functions, feature-space projection |
| 10-11 | Supervised models (Optuna, `ImbPipeline`); One-Class detectors on the top-20 feature subspace |
| 12-13 | Results table, ablation, ROC/PR curves; paired bootstrap, DeLong, McNemar with Holm correction, calibration and decision curves |
| 14-15 | Global SHAP audit; local LIME audit (Jaccard) |
| 16 | Equity audit and two-layer hybrid system |
| R1-R7 | Additional analyses: sanity checks, matched-feature XAI controls, sensitivity scenarios (early information, one-hot), equity with bootstrap intervals, reproducibility tables, regenerated figures, selected Random Forest hyperparameters |

## Reference results

With the recorded environment, the test-set MCC values are 0.7530 (Random Forest + SMOTE), 0.7292 (cost-sensitive XGBoost), 0.6584 (OC-SVM), 0.5855 (Isolation Forest) and 0.6131 (LOF). Use them to check that a new run reproduces the study.

## Reproducibility notes

- Global seed 42; Optuna uses `TPESampler(seed=42)`; the split is stratified 60/20/20; the scaler is fitted on the training set only.
- Results were produced on Google Colab (Python 3.13.16) with the versions in `requirements.txt`. Different library versions, hardware or TensorFlow GPU kernels can change the Autoencoder and the stochastic LIME and bootstrap results slightly.
- Hyperparameters selected for every model are in the notebook (cells R5 and R7) and in the article.

## Data and citation

The dataset comes from Realinho et al. (see `data/README.md`). If you use this code, please cite the article (see `CITATION.cff`).

## License

MIT for the code (see `LICENSE`). The dataset follows the license of its original record.
