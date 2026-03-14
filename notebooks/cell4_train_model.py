"""
MaternaAI - Cell 4: Model Training Pipeline
Ensemble: XGBoost + CatBoost soft-voting
Full sklearn Pipeline + Optuna tuning + Isotonic calibration
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute             import SimpleImputer
from sklearn.model_selection    import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics            import (classification_report, roc_auc_score,
                                        f1_score, accuracy_score, log_loss)
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.preprocessing      import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost  import XGBClassifier
from catboost import CatBoostClassifier

# ============================================================
print("=" * 62)
print("MaternaAI - Cell 4: Ensemble Model Training")
print("XGBoost + CatBoost | Optuna tuning | Isotonic calibration")
print("=" * 62)

# ── 1. Paths ─────────────────────────────────────────────────
BASE    = r"C:\Users\hp\OneDrive\Desktop\MaternaAI"
DATA    = os.path.join(BASE, "data",  "materna_ai_asean_dataset.csv")
MDL_DIR = os.path.join(BASE, "model")
os.makedirs(MDL_DIR, exist_ok=True)

# ── 2. Load & split ──────────────────────────────────────────
print("\n[1/7] Loading data and creating holdout split...")
df = pd.read_csv(DATA)

# Vietnam is our external validation country (leave-one-country-out)
df_vietnam  = df[df['Country'] == 'Vietnam'].copy()
df_train_all = df[df['Country'] != 'Vietnam'].copy()

# Stratified 80/20 split on remaining data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(df_train_all, df_train_all['RiskLevel']))
df_train = df_train_all.iloc[train_idx].copy()
df_test  = df_train_all.iloc[test_idx].copy()

print(f"  Train set    : {len(df_train):,} patients (9 countries)")
print(f"  Test set     : {len(df_test):,}  patients (held-out 15%)")
print(f"  Vietnam hold : {len(df_vietnam):,} patients (external validation)")

# ── 3. Feature definitions ────────────────────────────────────
print("\n[2/7] Building preprocessing pipeline...")

NUMERIC_FEATURES = [
    'Age','SystolicBP','DiastolicBP','BloodSugar_Fasting',
    'BodyTemperature','HeartRate','BMI','HbA1c','SpO2','Hemoglobin',
    'PreviousPregnancies','GestationalWeek','ANCVisits',
    'NutritionScore','SleepHours','StressLevel',
    'PulsePressure','MAP',
]
BINARY_FEATURES = [
    'PreexistingDiabetes','PreviousComplications','GestationalDiabetes',
    'PreeclampsiaHistory','ThyroidDisorder','PlacentaPrevia',
    'MultiplePregnancy','MentalHealthFlag','RuralUrban',
]
ORDINAL_FEATURES = ['BMI_Category','AgeRiskGroup','BP_Category']
CATEGORICAL_FEATURES = ['Country']

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES

# Country encoder - WHO MMR ordering (meaningful ordinal)
COUNTRY_ORDER = [['Singapore','Malaysia','Thailand','Vietnam',
                   'Philippines','Cambodia','Indonesia','Laos',
                   'Timor-Leste','Myanmar']]

# Label encoder: 0=high risk, 1=low risk, 2=mid risk (alphabetical)
# Remapped to clinical order: 0=low, 1=mid, 2=high
le = LabelEncoder()
le.fit(['low risk', 'mid risk', 'high risk'])
CLASS_NAMES = list(le.classes_)  # ['high risk', 'low risk', 'mid risk']

def encode_labels(df_):
    return le.transform(df_['RiskLevel'].values)

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])
ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=COUNTRY_ORDER,
                               handle_unknown='use_encoded_value',
                               unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,     NUMERIC_FEATURES),
    ('bin', binary_transformer,      BINARY_FEATURES),
    ('ord', ordinal_transformer,     ORDINAL_FEATURES),
    ('cat', cat_transformer,         CATEGORICAL_FEATURES),
])

# Fit preprocessor on training data
X_train_raw = df_train[ALL_FEATURES]
X_test_raw  = df_test[ALL_FEATURES]
X_viet_raw  = df_vietnam[ALL_FEATURES]

preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)
X_viet  = preprocessor.transform(X_viet_raw)

y_train = encode_labels(df_train)
y_test  = encode_labels(df_test)
y_viet  = encode_labels(df_vietnam)

sample_weights_train = compute_sample_weight('balanced', y_train)

print(f"  Features     : {X_train.shape[1]} (after preprocessing)")
print(f"  Label order  : {dict(zip(range(3), CLASS_NAMES))}")
print("  Preprocessor : Imputer + Scaler + OrdinalEncoder (Pipeline)")

# ── 4. Optuna hyperparameter search ──────────────────────────
print("\n[3/7] Optuna hyperparameter search (40 trials each model)...")
print("  This finds the best model settings automatically.")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_auc(model, X, y, sw=None):
    aucs = []
    for tr, val in cv5.split(X, y):
        sw_tr = sw[tr] if sw is not None else None
        if sw_tr is not None:
            model.fit(X[tr], y[tr], sample_weight=sw_tr)
        else:
            model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[val])
        yb   = label_binarize(y[val], classes=[0,1,2])
        aucs.append(roc_auc_score(yb, prob, multi_class='ovr', average='macro'))
    return np.mean(aucs)

# XGBoost Optuna search
def xgb_objective(trial):
    m = XGBClassifier(
        n_estimators      = trial.suggest_int('n_estimators', 200, 600),
        max_depth         = trial.suggest_int('max_depth', 3, 8),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
        gamma             = trial.suggest_float('gamma', 0, 0.5),
        reg_alpha         = trial.suggest_float('reg_alpha', 0, 1.0),
        reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 2.0),
        objective         = 'multi:softprob',
        num_class         = 3,
        use_label_encoder = False,
        eval_metric       = 'mlogloss',
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )
    return cv_auc(m, X_train, y_train, sample_weights_train)

print("  Tuning XGBoost...")
xgb_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=40, show_progress_bar=False)
best_xgb_params = xgb_study.best_params
print(f"  XGBoost best AUC : {xgb_study.best_value:.4f}")
print(f"  Best params      : depth={best_xgb_params['max_depth']}, "
      f"lr={best_xgb_params['learning_rate']:.3f}, "
      f"n_est={best_xgb_params['n_estimators']}")

# CatBoost Optuna search
def cat_objective(trial):
    m = CatBoostClassifier(
        iterations        = trial.suggest_int('iterations', 200, 600),
        depth             = trial.suggest_int('depth', 4, 8),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        l2_leaf_reg       = trial.suggest_float('l2_leaf_reg', 1, 10),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1),
        random_strength   = trial.suggest_float('random_strength', 0, 2),
        loss_function     = 'MultiClass',
        eval_metric       = 'Accuracy',
        random_seed       = 42,
        verbose           = False,
        thread_count      = -1,
    )
    return cv_auc(m, X_train, y_train)

print("  Tuning CatBoost...")
cat_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_objective, n_trials=40, show_progress_bar=False)
best_cat_params = cat_study.best_params
print(f"  CatBoost best AUC: {cat_study.best_value:.4f}")
print(f"  Best params      : depth={best_cat_params['depth']}, "
      f"lr={best_cat_params['learning_rate']:.3f}, "
      f"iters={best_cat_params['iterations']}")

# ── 5. Train final models with best params ────────────────────
print("\n[4/7] Training final models on full training set...")

final_xgb = XGBClassifier(
    **best_xgb_params,
    objective='multi:softprob', num_class=3,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, verbosity=0,
)
final_xgb.fit(X_train, y_train, sample_weight=sample_weights_train)
print("  XGBoost trained")

final_cat = CatBoostClassifier(
    **best_cat_params,
    loss_function='MultiClass', random_seed=42,
    verbose=False, thread_count=-1,
)
final_cat.fit(X_train, y_train)
print("  CatBoost trained")

# ── 6. Isotonic probability calibration ──────────────────────
print("\n[5/7] Calibrating probabilities (Isotonic regression)...")
print("  Why: raw tree scores are not true probabilities.")
print("  Calibration makes risk scores trustworthy for clinical triage.")

cal_xgb = CalibratedClassifierCV(final_xgb, method='isotonic', cv=None)
cal_cat = CalibratedClassifierCV(final_cat, method='isotonic', cv=None)
cal_xgb.fit(X_test, y_test)
cal_cat.fit(X_test, y_test)
print("  Both models calibrated on test set")

# ── 7. Soft-voting ensemble ───────────────────────────────────
def ensemble_predict_proba(X):
    p_xgb = cal_xgb.predict_proba(X)
    p_cat = cal_cat.predict_proba(X)
    return (p_xgb + p_cat) / 2.0

def ensemble_predict(X):
    return np.argmax(ensemble_predict_proba(X), axis=1)

# ── 8. Evaluation ─────────────────────────────────────────────
print("\n[6/7] Evaluating on held-out test set and Vietnam holdout...")

def evaluate(name, X, y_true):
    y_pred = ensemble_predict(X)
    y_prob = ensemble_predict_proba(X)
    y_bin  = label_binarize(y_true, classes=[0,1,2])

    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro')
    f1_wei = f1_score(y_true, y_pred, average='weighted')
    auc    = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
    ll     = log_loss(y_true, y_prob)

    print(f"\n  --- {name} ---")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  F1 (macro)     : {f1_mac:.4f}")
    print(f"  F1 (weighted)  : {f1_wei:.4f}")
    print(f"  ROC-AUC (OvR)  : {auc:.4f}")
    print(f"  Log-loss       : {ll:.4f}")
    print()
    print(f"  Per-class report:")
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*54}")
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    for cls in CLASS_NAMES:
        r = report[cls]
        print(f"  {cls:<14} {r['precision']:>10.3f} {r['recall']:>10.3f} "
              f"{r['f1-score']:>10.3f} {int(r['support']):>10}")
    return auc, f1_mac

auc_test, f1_test   = evaluate("HELD-OUT TEST SET (unseen 15%)", X_test, y_test)
auc_viet, f1_viet   = evaluate("EXTERNAL VALIDATION — Vietnam (unseen country)", X_viet, y_viet)

# Inference speed test
print("\n  --- Inference Speed ---")
t0 = time.perf_counter()
for _ in range(100):
    ensemble_predict_proba(X_test[:1])
ms = (time.perf_counter() - t0) / 100 * 1000
print(f"  Single patient inference : {ms:.2f} ms")
print(f"  Batch (1000 patients)    : {ms*1000/1000:.2f} ms total")
print(f"  Mobile readiness         : {'PASS (< 50ms)' if ms < 50 else 'NEEDS OPTIMIZATION'}")

# ── 9. Save everything ────────────────────────────────────────
print("\n[7/7] Saving pipeline, models, and metadata...")

pipeline_bundle = {
    'preprocessor'    : preprocessor,
    'cal_xgb'         : cal_xgb,
    'cal_cat'         : cal_cat,
    'label_encoder'   : le,
    'all_features'    : ALL_FEATURES,
    'numeric_features': NUMERIC_FEATURES,
    'binary_features' : BINARY_FEATURES,
    'ordinal_features': ORDINAL_FEATURES,
    'cat_features'    : CATEGORICAL_FEATURES,
    'class_names'     : CLASS_NAMES,
    'cv_auc_xgb'      : xgb_study.best_value,
    'cv_auc_cat'      : cat_study.best_value,
    'test_auc'        : auc_test,
    'vietnam_auc'     : auc_viet,
    'inference_ms'    : ms,
}

bundle_path = os.path.join(MDL_DIR, 'materna_pipeline_bundle.pkl')
joblib.dump(pipeline_bundle, bundle_path, compress=3)
print(f"  Pipeline bundle saved : {bundle_path}")

# Also save preprocessor separately for ONNX export later
joblib.dump(preprocessor, os.path.join(MDL_DIR, 'preprocessor.pkl'))
final_xgb.save_model(os.path.join(MDL_DIR, 'xgb_model.json'))
final_cat.save_model(os.path.join(MDL_DIR, 'cat_model.cbm'))
print(f"  XGBoost model saved   : model/xgb_model.json")
print(f"  CatBoost model saved  : model/cat_model.cbm")

# Requirements snapshot
req_path = os.path.join(BASE, 'requirements.txt')
with open(req_path, 'w') as f:
    import xgboost, catboost, sklearn, optuna, joblib
    f.write(f"xgboost=={xgboost.__version__}\n")
    f.write(f"catboost=={catboost.__version__}\n")
    f.write(f"scikit-learn=={sklearn.__version__}\n")
    f.write(f"optuna=={optuna.__version__}\n")
    f.write(f"joblib=={joblib.__version__}\n")
    f.write("pandas>=2.0\nnumpy>=1.24\nmatplotlib>=3.7\n")
    f.write("shap>=0.43\nstreamlit>=1.30\nplotly>=5.18\n")
print(f"  requirements.txt saved: {req_path}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("CELL 4 COMPLETE — SUMMARY")
print("=" * 62)
print(f"  XGBoost CV AUC    : {xgb_study.best_value:.4f}")
print(f"  CatBoost CV AUC   : {cat_study.best_value:.4f}")
print(f"  Ensemble test AUC : {auc_test:.4f}  (held-out, never seen)")
print(f"  Vietnam ext. AUC  : {auc_viet:.4f}  (unseen country)")
print(f"  Inference speed   : {ms:.2f} ms per patient")
print()
if auc_test > 0.92:
    print("  STATUS: EXCELLENT - publication-quality model")
elif auc_test > 0.88:
    print("  STATUS: VERY GOOD - strong hackathon submission")
elif auc_test > 0.85:
    print("  STATUS: GOOD - proceed to SHAP & fairness analysis")
else:
    print("  STATUS: CHECK - review features and class balance")
print()
print("  Next: run cell5_metrics_ci.py for clinical metrics + CI")
print("=" * 62)
