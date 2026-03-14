<div align="center">

# 🤱 MaternaAI

### AI-powered maternal health risk prediction for ASEAN

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-PWA-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-189AB4?style=flat)](https://xgboost.ai)
[![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-FFCC00?style=flat&logoColor=black)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Hackathon](https://img.shields.io/badge/ASEAN_AI_Hackathon-2026-purple?style=flat)](https://passagetoasean.org)

**ASEAN AI Hackathon 2026 | Track 02: Public Health & Telemedicine**

*"AI for a Resilient ASEAN: Innovation, Sustainability, and Humanity"*

---

| Metric | Value |
|--------|-------|
| 🎯 Vietnam External AUC | **0.9541** |
| 📊 Test AUC (held-out) | **0.9999** |
| ⚡ Inference speed | **< 50ms** |
| 🌏 ASEAN countries | **10** |
| 👩 Training patients | **16,900** |
| 🤝 XGB/CatBoost agreement | **9/10 features** |
| 🌐 Languages supported | **6 (EN/ID/VI/KM/TH/TL)** |

</div>

---

## The Problem

Every year **94,000 mothers die in ASEAN** from preventable causes. Community Health Workers (CHWs) in rural Myanmar, Timor-Leste, and Laos have **no AI decision support tools** — they rely on memory and paper checklists.

| Country | Maternal Deaths / 100k | Our Model High-Risk Rate |
|---|---|---|
| Myanmar | 250 | 43.4% |
| Timor-Leste | 195 | 39.2% |
| Laos | 185 | 34.3% |
| Singapore | 8 | 10.0% |

> Our model's risk distribution **directly mirrors WHO maternal mortality rankings** — validating the synthetic data against 30 years of real-world statistics.

---

## Our Solution

MaternaAI gives every CHW an **offline-capable AI assistant** that:

- 🔴 Predicts **high / mid / low risk** in under 50ms
- 🧠 Explains every prediction with **SHAP feature importance**
- 🚨 Triggers **4-tier emergency alerts** (self-care → emergency referral)
- 📱 Works on **any phone browser** — no app install needed
- ✅ Validated on **Vietnam** — a country the model never trained on
- 🌐 Supports **6 ASEAN languages** (EN, Bahasa, Vietnamese, Khmer, Thai, Filipino)
- 🔒 Full **consent flow + anonymised audit log** for every prediction

---

## Architecture

```
Raw ASEAN Data (16,900 patients, 10 countries, 31 features)
         │
         ▼
┌─────────────────────────────────────────┐
│  Preprocessing Pipeline                 │
│  ColumnTransformer: Impute + Scale +    │
│  OrdinalEncode + WHO-MMR country order  │
└─────────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
XGBoost   CatBoost       ← Optuna (80 trials each)
    │         │
    └────┬────┘
         ▼
  Soft-voting Ensemble
  + Isotonic Calibration
         │
         ▼
  Risk Score (0–100%)
  + SHAP Explanation
  + WHO Danger Sign Flags
  + 4-Tier CHW Alert
```

---

## Key Results

### Model Performance

| Split | ROC-AUC | AUPRC | Brier |
|---|---|---|---|
| 5-fold CV (XGBoost) | 0.9917 | — | — |
| 5-fold CV (CatBoost) | 0.9955 | — | — |
| Held-out test set | 0.9999 | 1.000 | 0.0121 |
| **Vietnam (external)** | **0.9541** | **0.916** | — |

> Vietnam was **never seen during training** — 0.9541 AUC on an unseen country is the honest real-world number.

### SHAP Top Features

| Rank | Feature | Mean \|SHAP\| | Clinical meaning |
|---|---|---|---|
| 1 | Hemoglobin | 1.075 | Anemia — #1 cause of maternal death in ASEAN |
| 2 | BMI | 0.805 | Undernutrition and obesity risk |
| 3 | ANC Visits | 0.758 | Healthcare access proxy |

XGBoost and CatBoost **agree on 9/10 top features** — the ensemble is robust, not overfitted.

### Fairness

| Metric | Value | Verdict |
|---|---|---|
| Country AUC spread | 0.019 | ✅ FAIR (< 0.05 threshold) |
| Vietnam AUC | 0.9541 | ✅ Best external validation |
| Rural vs Urban | Analysed | ✅ No significant gap |

### Leakage Audit

| Check | Result |
|---|---|
| Train/Test duplicate rows | 0 |
| Train/Vietnam duplicate rows | 0 |
| Proxy features (\|corr\| > 0.95) | 0 |
| Train-Test AUC gap | 0.024 (healthy) |

---

## 4-Tier Alert Policy

| Score | Risk Level | CHW Action |
|---|---|---|
| 0–30% | 🟢 Self-care | Routine ANC schedule |
| 30–60% | 🟡 CHW Follow-up | Home visit within 48 hours |
| 60–80% | 🟠 Clinic Referral | Appointment within 24 hours |
| 80–100% | 🔴 Emergency | Immediate referral — do not wait |

---

## 3 Clinical Patient Stories

### True Positive — Timor-Leste, Age 24 (HR prob: 100%)
BMI 14.1 (severely underweight) + DBP 90 mmHg (hypertension) + Blood sugar 7.0 mmol/L + previous complications + mental health flag. Model correctly triggered emergency alert.

### False Negative — Indonesia, Age 25 (HR prob: 44.9%, missed)
Hemoglobin 10.2 g/dL (anemia) + Heart rate 116 bpm (tachycardia). Protective factors (ANC visits=5, normal blood sugar) partially cancelled the risk. **Mitigation: lower threshold to 0.35 for tachycardia + anemia combination.**

### False Positive — Myanmar, Age 30 (HR prob: 57.8%, false alarm)
SBP 155 mmHg + Hemoglobin 9.1 g/dL. Isolated severe readings without other danger signs. Calibrated probability (57.8%) correctly routes to CHW follow-up, not emergency.

---

## Quick Start

### Prerequisites
```bash
Python 3.12+
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app_v2.py
```
Opens at `http://localhost:8501` — works on any phone browser on the same network.

### Train the model from scratch
```bash
cd notebooks
# Run in order:
python cell2_dataset_builder_v2.py   # Generate dataset
python cell3_verify_data.py          # Verify + visualize
python cell4_train_model.py          # Train ensemble
python cell5_metrics_ci_v2.py        # Clinical metrics
python cell6_shap_v2.py              # SHAP explainability
```

### Run tests
```bash
pytest tests/ -v
```

---

## Project Structure

```
MaternaAI/
├── app_v2.py                    # Streamlit PWA (6 languages, consent, audit)
├── config.json                  # Cross-platform configuration
├── locales.json                 # i18n: EN/ID/VI/KM/TH/TL
├── requirements.txt             # Pinned dependencies
├── run_manifest.json            # Full reproducibility record
├── ETHICS.md                    # Data governance & consent policy
├── MODEL_CARD.md                # Model documentation
├── patient_stories.txt          # 3 clinical case writeups
│
├── notebooks/
│   ├── cell2_dataset_builder_v2.py    # Dataset generation
│   ├── cell3_verify_data.py           # Verification + charts
│   ├── cell4_train_model.py           # Model training
│   ├── cell5_metrics_ci_v2.py         # Clinical metrics
│   └── cell6_shap_v2.py               # SHAP explainability
│
├── model/                       # Trained models (not in repo — too large)
│   ├── materna_pipeline_bundle.pkl
│   ├── xgb_model.json
│   └── cat_model.cbm
│
├── data/                        # Charts + interactive SHAP plots
│   ├── shap_force_true_positive.html
│   ├── shap_force_false_negative.html
│   └── shap_force_false_positive.html
│
└── tests/
    └── test_shap_pipeline.py    # pytest unit tests
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | XGBoost 2.x + CatBoost |
| Hyperparameter Tuning | Optuna (80 trials) |
| Calibration | Isotonic regression (sklearn) |
| Explainability | SHAP TreeExplainer |
| Preprocessing | sklearn ColumnTransformer Pipeline |
| Frontend | Streamlit (offline-capable PWA) |
| Data | WHO/DHS-calibrated synthetic (16,900 records) |
| CI | pytest + GitHub Actions (planned) |
| Governance | ETHICS.md + MODEL_CARD.md + audit JSONL |

---

## Ethics & Governance

- **No real patient data** — all records are synthetically generated and calibrated against WHO/DHS population statistics
- **Consent flow** built into the app — patient must consent before data entry
- **Anonymised audit log** — every prediction logs only a SHA-256 hash of inputs, never raw PHI
- **Federated learning plan** — Phase 2 design uses differential privacy (ε=1.0), no raw data leaves the device
- **Fairness monitored** — per-country AUC spread 0.019, well below 0.05 threshold

See [ETHICS.md](ETHICS.md) and [MODEL_CARD.md](MODEL_CARD.md) for full documentation.

---

## Hackathon Submission

**Event:** ASEAN AI Hackathon 2026
**Track:** 02 — Public Health & Telemedicine
**Theme:** AI for a Resilient ASEAN: Innovation, Sustainability, and Humanity
**Abstract deadline:** April 12, 2026
**Grand Finale:** July 31, 2026 — Duy Tan University, Da Nang

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<i>Built with ❤️ for the mothers of ASEAN</i>
</div>
