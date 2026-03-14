"""
MaternaAI - pytest: SHAP pipeline unit tests
Run: pytest tests/test_shap_pipeline.py -v
"""
import numpy as np, joblib, os, pytest

BASE    = r"C:\Users\hp\OneDrive\Desktop\MaternaAI"
MDL_DIR = os.path.join(BASE, "model")
OUT_DIR = os.path.join(BASE, "data")

@pytest.fixture(scope="module")
def bundle():
    return joblib.load(os.path.join(MDL_DIR, "materna_pipeline_bundle.pkl"))

def test_bundle_keys(bundle):
    for key in ['preprocessor','cal_xgb','cal_cat','label_encoder',
                'all_features','class_names']:
        assert key in bundle, f"Missing key: {key}"

def test_feature_count(bundle):
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE, "data", "materna_ai_asean_dataset.csv"))
    X  = bundle['preprocessor'].transform(df[bundle['all_features']].head(5))
    try:
        names = bundle['preprocessor'].get_feature_names_out()
        assert len(names) == X.shape[1], "Feature name count mismatch"
    except AttributeError:
        assert len(bundle['all_features']) == X.shape[1]

def test_ensemble_probabilities(bundle):
    import pandas as pd
    df   = pd.read_csv(os.path.join(BASE, "data", "materna_ai_asean_dataset.csv"))
    X    = bundle['preprocessor'].transform(df[bundle['all_features']].head(10))
    prob = (bundle['cal_xgb'].predict_proba(X) +
            bundle['cal_cat'].predict_proba(X)) / 2.0
    assert prob.shape == (10, 3), f"Unexpected prob shape: {prob.shape}"
    assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-5), "Probs dont sum to 1"

def test_shap_dashboard_exists():
    path = os.path.join(OUT_DIR, "shap_dashboard_v2.png")
    assert os.path.exists(path), "shap_dashboard_v2.png not found"
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    assert w > 1000 and h > 800, f"Dashboard too small: {w}x{h}"

def test_patient_stories_exist():
    path = os.path.join(BASE, "patient_stories.txt")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    for section in ["TRUE POSITIVE", "FALSE NEGATIVE", "FALSE POSITIVE"]:
        assert section in text, f"Missing story section: {section}"
    assert "RECOMMENDED CLINICAL ACTION" in text

def test_top_features_clinical(bundle):
    """Top SHAP features should be clinically expected ones."""
    import pandas as pd, shap as sh
    df    = pd.read_csv(os.path.join(BASE, "data", "materna_ai_asean_dataset.csv"))
    X     = bundle['preprocessor'].transform(df[bundle['all_features']].head(200))
    xgb   = bundle['cal_xgb'].estimator
    exp   = sh.TreeExplainer(xgb)
    sv    = exp.shap_values(X)
    if isinstance(sv, list): sv = np.stack(sv, axis=2)
    mean_abs = np.abs(sv[:,:,0]).mean(0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    expected = {"Hemoglobin","BloodSugar_Fasting","HbA1c","MAP",
                "SystolicBP","PreeclampsiaHistory","ANCVisits"}
    all_feats = bundle['all_features']
    top3_names = {all_feats[i] for i in top3_idx if i < len(all_feats)}
    overlap = top3_names & expected
    assert len(overlap) >= 1, (
        f"Top 3 features {top3_names} have no overlap with expected clinical "
        f"features {expected}. Check for data leakage or feature issues.")
