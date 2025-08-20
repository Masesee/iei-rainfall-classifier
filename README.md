# Indigenous Rainfall Prediction (IEI) – Baseline LGBM Pipeline

End-to-end, reproducible pipeline to classify 12–24 hour rainfall intensity (Heavy, Moderate, Small, No-Rain) from Indigenous Ecological Indicators (IEIs). The system is designed for a GitHub ↔ Kaggle workflow with deterministic training, explainability, and stratified group CV.

## Quick Start

1. Create and activate a Python environment (optional):
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place data:
- Put `train.csv` and `test.csv` in `data/`.
- Ensure columns match config (`configs/baseline_lgbm.yaml`).

4. Run the training notebook end-to-end:
- Open `notebooks/02_Training_Pipeline.ipynb` and run all cells to reproduce training, validation (OOF Macro F1), and `submission.csv` generation.

## Project Structure
```
indigenous-rainfall-prediction/
  configs/
    baseline_lgbm.yaml
  data/
    train.csv  # not tracked
    test.csv   # not tracked
  models/
  notebooks/
    01_EDA.ipynb
    02_Training_Pipeline.ipynb
  src/
    preprocess.py
    features.py
    train.py
    predict.py
  submissions/
  requirements.txt
  .gitignore
  README.md
```

## Reproducibility
- Global seeds set in training.
- Deterministic LightGBM settings where applicable.
- All artifacts (models, TF-IDF vectorizer, encoders, label mapping, feature columns) are saved under `models/`.

## Explainability
- The training notebook includes a SHAP example for a trained fold model to explain predictions.

## Kaggle Workflow
- Commit this repo to GitHub.
- On Kaggle, create a new Notebook linked to the GitHub repo.
- Run `02_Training_Pipeline.ipynb` (or convert it to a script) to reproduce the full pipeline and generate `submission.csv` under `submissions/`.

## Configuration
- Edit `configs/baseline_lgbm.yaml` to adjust paths, features, and model parameters.

## License
MIT
