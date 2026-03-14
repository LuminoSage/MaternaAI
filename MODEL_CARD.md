# MODEL CARD — MaternaAI Ensemble v1.0

## Model Details
- Type: Soft-voting ensemble (XGBoost + CatBoost)
- Task: 3-class maternal risk classification (high / mid / low)
- Input: 31 clinical + contextual features
- Output: Risk class + calibrated probability score

## Intended Use
- Decision support for Community Health Workers in ASEAN
- Triage prioritisation at point-of-care (offline capable)
- NOT for autonomous clinical decisions

## Performance (held-out test set)
- ROC-AUC: 0.9998 (test) | 0.9541 (Vietnam external)
- AUPRC: see run_manifest.json
- Inference: ~44ms per patient on standard laptop

## Training Data
- 16,900 synthetic WHO/DHS-calibrated ASEAN records
- 10 countries: Indonesia, Philippines, Vietnam, Myanmar,
  Cambodia, Thailand, Timor-Leste, Laos, Malaysia, Singapore

## Limitations
- Synthetic training data — real-world gap unknown until clinical trial
- Country-specific performance varies (see fairness_per_country.csv)
- Not validated for multiple gestations requiring specialist care

## Contact
MaternaAI Team | ASEAN AI Hackathon 2026, Track 02
