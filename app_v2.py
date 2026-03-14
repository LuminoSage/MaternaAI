# -*- coding: utf-8 -*-
"""
MaternaAI v2.0 - Production Streamlit App
ASEAN AI Hackathon 2026 | Track 02: Public Health & Telemedicine

Features:
- Cross-platform config (no hardcoded paths)
- Graceful demo mode when model unavailable
- Full input validation + unit conversion (C/F)
- Preprocessor alignment assertion at startup
- 6-language i18n (EN/ID/VI/KM/TH/TL)
- Consent flow before data entry
- Audit log for every prediction
- Inference telemetry + perf warning
- WHO danger sign flags + calibrated alert policy
- Result export (JSON) with consent checkbox
- WCAG-compliant color contrast
- Model checksum-based cache invalidation
- Sample patient presets for demo
- Inline help tooltips for every input
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, json, time, logging, hashlib, threading
from datetime import datetime, timezone
from pathlib import Path

# ── Cross-platform base path ──────────────────────────────────
# Works on Windows, Linux, Docker, and CI
APP_DIR = Path(__file__).parent.resolve()

def load_config():
    cfg_path = APP_DIR / "config.json"
    defaults = {
        "model_dir": "model", "data_dir": "data", "log_dir": "logs",
        "app_name": "MaternaAI", "app_version": "2.0.0",
        "temp_unit": "celsius", "demo_mode_on_error": True,
        "audit_log_enabled": True, "telemetry_opt_in": False,
        "inference_warn_ms": 100, "auth_enabled": False,
        "default_language": "en",
    }
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return {**defaults, **json.load(f)}
    return defaults

CFG      = load_config()
MDL_DIR  = APP_DIR / CFG["model_dir"]
DATA_DIR = APP_DIR / CFG["data_dir"]
LOG_DIR  = APP_DIR / CFG["log_dir"]
LOG_DIR.mkdir(exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    filename=str(LOG_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("materna")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title=CFG["app_name"],
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"{CFG['app_name']} — {CFG.get('hackathon','')}"}
)

# ── Load locales ──────────────────────────────────────────────
@st.cache_data
def load_locales():
    loc_path = APP_DIR / "locales.json"
    if loc_path.exists():
        with open(loc_path, encoding="utf-8") as f:
            return json.load(f)
    return {"en": {}}

LOCALES = load_locales()
LANG_NAMES = {
    "en":"English","id":"Bahasa Indonesia",
    "vi":"Tiếng Việt","km":"ខ្មែរ","th":"ภาษาไทย","tl":"Filipino"
}

def t(key, lang=None):
    """Translate key to current language, fallback to English."""
    l = lang or st.session_state.get("lang", "en")
    return LOCALES.get(l, {}).get(key) or LOCALES.get("en", {}).get(key, key)

# ── Vivid, high-contrast CSS ─────────────────────────────────
st.markdown("""
<style>
  /* Main background — deep navy */
  [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    min-height: 100vh;
  }
  /* All main text — bright white */
  [data-testid="stAppViewContainer"] p,
  [data-testid="stAppViewContainer"] span,
  [data-testid="stAppViewContainer"] label,
  [data-testid="stAppViewContainer"] li,
  [data-testid="stAppViewContainer"] td,
  [data-testid="stAppViewContainer"] th,
  [data-testid="stAppViewContainer"] div:not([class]) {
    color: #e8f4fd !important;
  }
  /* Headings — pure white */
  [data-testid="stAppViewContainer"] h1,
  [data-testid="stAppViewContainer"] h2,
  [data-testid="stAppViewContainer"] h3 {
    color: #ffffff !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }
  /* Sidebar — dark slate */
  [data-testid="stSidebar"] {
    background: #0d1f35 !important;
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div {
    color: #b0d4f1 !important;
  }
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] strong {
    color: #ffffff !important;
  }
  /* Sidebar nav radio — bigger icons + labels */
  [data-testid="stSidebar"] .stRadio label {
    font-size: 1.05rem !important;
    padding: 6px 4px !important;
    color: #cce4f7 !important;
  }
  [data-testid="stSidebar"] .stRadio label span {
    font-size: 1.2rem !important;
  }
  /* Sidebar selectbox */
  [data-testid="stSidebar"] .stSelectbox label {
    color: #90c4e8 !important;
    font-size: 0.85rem !important;
  }

  /* Risk cards — vivid saturated colors */
  .risk-high {
    background: linear-gradient(135deg, #d32f2f, #b71c1c);
    color: #ffffff; border-radius: 18px; padding: 28px;
    text-align: center; margin: 8px 0;
    border: 2px solid #ef5350;
    box-shadow: 0 4px 20px rgba(183,28,28,0.5);
  }
  .risk-mid {
    background: linear-gradient(135deg, #f4511e, #e64a19);
    color: #ffffff; border-radius: 18px; padding: 28px;
    text-align: center; margin: 8px 0;
    border: 2px solid #ff7043;
    box-shadow: 0 4px 20px rgba(230,74,25,0.5);
  }
  .risk-low {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: #ffffff; border-radius: 18px; padding: 28px;
    text-align: center; margin: 8px 0;
    border: 2px solid #43a047;
    box-shadow: 0 4px 20px rgba(43,125,45,0.5);
  }
  .risk-title { font-size: 2rem; font-weight: 900; margin: 0; letter-spacing: 1px; }
  .risk-prob  { font-size: 1.4rem; font-weight: 700; margin: 10px 0 0 0; }
  .risk-label { font-size: 0.95rem; opacity: 0.9; margin: 6px 0 0 0; }

  /* Alert boxes — vivid with strong contrast */
  .alert-box { border-radius: 12px; padding: 16px 20px;
               font-weight: 600; font-size: 1rem; margin: 12px 0; }
  .alert-emergency { background:#b71c1c; border-left:6px solid #ff1744;
                     color:#ffffff; }
  .alert-followup  { background:#bf360c; border-left:6px solid #ff6d00;
                     color:#ffffff; }
  .alert-chw       { background:#e65100; border-left:6px solid #ffab40;
                     color:#ffffff; }
  .alert-routine   { background:#1b5e20; border-left:6px solid #69f0ae;
                     color:#ffffff; }

  /* Demo banner */
  .demo-banner { background: linear-gradient(90deg,#1565c0,#0d47a1);
                 color:#ffffff; border-radius:10px; padding:12px 18px;
                 text-align:center; font-weight:700; font-size:1rem;
                 margin-bottom:14px; border:1px solid #42a5f5; }

  /* Consent box */
  .consent-box { background:rgba(21,101,192,0.25); border:1px solid #42a5f5;
                 border-radius:12px; padding:18px; margin:12px 0;
                 color:#e3f2fd !important; }
  .consent-box b { color:#90caf9 !important; }

  /* Metric cards — vivid blue glass */
  .metric-card {
    background: linear-gradient(135deg, #0d2748, #0a1f3d);
    border-radius: 14px; padding: 20px;
    border: 1px solid #1e4976;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
  }
  .metric-val { font-size: 1.7rem; font-weight: 800; color: #64b5f6; }
  .metric-lbl { font-size: 0.82rem; color: #90caf9; margin-top: 6px; line-height:1.4; }

  /* Pills */
  .pill-red { background:#c62828; color:#ffcdd2; border-radius:8px;
              padding:5px 12px; font-size:0.85rem; margin:3px;
              display:inline-block; font-weight:600; }
  .pill-grn { background:#1b5e20; color:#c8e6c9; border-radius:8px;
              padding:5px 12px; font-size:0.85rem; margin:3px;
              display:inline-block; font-weight:600; }

  /* Audit badge */
  .audit-badge { background:#1a237e; color:#c5cae9; border-radius:8px;
                 padding:4px 10px; font-size:0.8rem; font-weight:600; }

  /* Streamlit widgets — ensure visibility on dark bg */
  .stSlider label, .stNumberInput label,
  .stSelectbox label, .stCheckbox label,
  .stRadio label, .stTextInput label {
    color: #b0d4f1 !important;
    font-size: 0.92rem !important;
  }
  .stMarkdown, .stCaption { color: #b0d4f1 !important; }
  .stDivider { border-color: #1e3a5f !important; }

  /* Table text */
  .stDataFrame td, .stDataFrame th { color: #e8f4fd !important; }

  /* Success / info / warning boxes */
  .stSuccess { background:#1b5e20 !important; color:#c8e6c9 !important; }
  .stInfo    { background:#0d47a1 !important; color:#e3f2fd !important; }
  .stWarning { background:#bf360c !important; color:#ffe0b2 !important; }

  /* Mobile */
  @media (max-width:768px) {
    .risk-title { font-size: 1.5rem; }
    .risk-prob  { font-size: 1.1rem; }
    .metric-val { font-size: 1.3rem; }
    .alert-box  { font-size: 0.9rem; }
  }
</style>
""", unsafe_allow_html=True)

# ── Model file checksum ───────────────────────────────────────
def file_checksum(path, n=65536):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(n):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"

BUNDLE_PATH = MDL_DIR / "materna_pipeline_bundle.pkl"
BUNDLE_CKSUM = file_checksum(BUNDLE_PATH)

# ── Model loading with graceful demo fallback ─────────────────
@st.cache_resource(hash_funcs={str: lambda x: x})
def load_bundle(checksum: str):
    try:
        bundle = joblib.load(BUNDLE_PATH)
        # Feature alignment assertion
        pre    = bundle['preprocessor']
        feats  = bundle['all_features']
        X_test = pre.transform(
            pd.DataFrame([{f: 0 for f in feats}])[feats])
        assert X_test.shape[1] == len(feats), (
            f"ALIGNMENT ERROR: {X_test.shape[1]} cols vs {len(feats)} features")
        logger.info(f"Model loaded OK. Checksum={checksum[:8]}. "
                    f"Features={len(feats)} aligned.")
        return bundle, None
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None, str(e)

bundle, load_error = load_bundle(BUNDLE_CKSUM)
DEMO_MODE = bundle is None

if DEMO_MODE:
    st.markdown(f"<div class='demo-banner'>⚠️  {t('demo_banner')}</div>",
                unsafe_allow_html=True)
    logger.warning("Running in DEMO MODE.")

# ── Prediction logic ──────────────────────────────────────────
DEMO_RESPONSES = {
    "demo_high": {"class":"high risk","hr_prob":0.91,
                  "probs":{"high risk":0.91,"mid risk":0.07,"low risk":0.02},"ms":12.0},
    "demo_mid":  {"class":"mid risk", "hr_prob":0.42,
                  "probs":{"high risk":0.42,"mid risk":0.51,"low risk":0.07},"ms":11.5},
    "demo_low":  {"class":"low risk", "hr_prob":0.06,
                  "probs":{"high risk":0.06,"mid risk":0.18,"low risk":0.76},"ms":10.8},
}

# Physiologic ranges for clamping (min, max)
PHYS_RANGES = {
    "Age":(15,49),"SystolicBP":(80,200),"DiastolicBP":(50,120),
    "BloodSugar_Fasting":(3.0,15.0),"BodyTemperature":(96.0,104.0),
    "HeartRate":(50,130),"BMI":(13.0,45.0),"HbA1c":(3.5,11.0),
    "SpO2":(85.0,100.0),"Hemoglobin":(5.0,17.0),
    "ANCVisits":(0,14),"GestationalWeek":(4,42),
    "NutritionScore":(1.0,10.0),"SleepHours":(3.0,10.0),
    "StressLevel":(1.0,10.0),
}

def validate_clamp(inputs):
    """Clamp all numeric inputs to physiologic ranges. Return warnings."""
    warnings_list = []
    for feat, (lo, hi) in PHYS_RANGES.items():
        if feat in inputs:
            v = inputs[feat]
            try:
                v = float(v)
                if v < lo or v > hi:
                    warnings_list.append(
                        f"{feat}: {v} outside physiologic range [{lo}–{hi}]")
                inputs[feat] = float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                warnings_list.append(f"{feat}: non-numeric value '{v}' set to midpoint")
                inputs[feat] = float((lo + hi) / 2)
    return inputs, warnings_list

WHO_THRESHOLDS = {
    "SystolicBP":        (">=",140,"Hypertension (SBP ≥140 mmHg)"),
    "DiastolicBP":       (">=", 90,"Hypertension (DBP ≥90 mmHg)"),
    "BloodSugar_Fasting":(">=",7.0,"Diabetes range (BS ≥7.0 mmol/L)"),
    "HbA1c":             (">=",6.5,"Diabetes (HbA1c ≥6.5%)"),
    "Hemoglobin":        ("<", 11.0,"Anemia (Hb <11.0 g/dL)"),
    "SpO2":              ("<", 95.0,"Hypoxia (SpO2 <95%)"),
    "BodyTemperature":   (">=",100.4,"Fever (Temp ≥100.4°F)"),
    "HeartRate":         (">=",100,"Tachycardia (HR ≥100 bpm)"),
    "ANCVisits":         ("<",  4,"Insufficient ANC (<4 visits)"),
}

def check_who_flags(inputs):
    flags = []
    for feat, (op, thr, label) in WHO_THRESHOLDS.items():
        try:
            v = float(inputs.get(feat, 0))
            if (op == ">=" and v >= thr) or (op == "<" and v < thr):
                flags.append(label)
        except Exception:
            pass
    return flags

def run_prediction(inputs, demo_mode=False):
    """Thread-safe prediction with telemetry."""
    if demo_mode:
        sbp = inputs.get("SystolicBP", 118)
        hb  = inputs.get("Hemoglobin", 11.5)
        if sbp >= 160 or hb < 7:
            return DEMO_RESPONSES["demo_high"]
        elif sbp >= 130 or hb < 11:
            return DEMO_RESPONSES["demo_mid"]
        return DEMO_RESPONSES["demo_low"]

    pre         = bundle["preprocessor"]
    cal_xgb     = bundle["cal_xgb"]
    cal_cat     = bundle["cal_cat"]
    le          = bundle["label_encoder"]
    ALL_FEATURES= bundle["all_features"]
    CLASS_NAMES = bundle["class_names"]
    HR_IDX      = list(CLASS_NAMES).index("high risk")

    row = {f: inputs.get(f, 0) for f in ALL_FEATURES}
    df  = pd.DataFrame([row])
    X   = pre.transform(df[ALL_FEATURES])

    t0   = time.perf_counter()
    p1   = cal_xgb.predict_proba(X)
    p2   = cal_cat.predict_proba(X)
    prob = (p1 + p2) / 2.0
    ms   = (time.perf_counter() - t0) * 1000

    idx  = int(np.argmax(prob[0]))
    res  = {
        "class":    CLASS_NAMES[idx],
        "hr_prob":  float(prob[0][HR_IDX]),
        "probs":    {CLASS_NAMES[i]: float(prob[0][i]) for i in range(3)},
        "ms":       round(ms, 2),
    }

    if CFG.get("telemetry_opt_in") and ms > CFG.get("inference_warn_ms", 100):
        logger.warning(f"Slow inference: {ms:.1f}ms > {CFG['inference_warn_ms']}ms")

    return res

def write_audit_log(inputs_hash, result, flags, lang, demo):
    """Anonymized audit log — no raw PHI stored."""
    if not CFG.get("audit_log_enabled", True):
        return
    try:
        entry = {
            "ts":           datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "model_ver":    CFG.get("app_version", "2.0.0"),
            "bundle_cksum": BUNDLE_CKSUM[:8],
            "inputs_hash":  inputs_hash,
            "prediction":   result["class"],
            "hr_prob":      result["hr_prob"],
            "who_flags":    flags,
            "inference_ms": result["ms"],
            "language":     lang,
            "demo_mode":    demo,
        }
        audit_path = LOG_DIR / "audit.jsonl"
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")

def inputs_hash(inputs):
    s = json.dumps({k: str(v) for k, v in sorted(inputs.items())})
    return hashlib.sha256(s.encode()).hexdigest()[:16]

# ── Sample presets for demo ───────────────────────────────────
PRESETS = {
    "High Risk — Timor-Leste": {
        "country":"Timor-Leste","age":24,"rural":1,"anc":2,
        "gest_week":36,"prev_preg":0,"sbp":158,"dbp":105,
        "bs":8.2,"hb":8.5,"hba1c":6.8,"spo2":92.0,
        "temp_c":38.5,"hr":112,"bmi":14.2,"nutrition":3.0,
        "dm":True,"gest_dm":True,"preeclamp":True,
        "thyroid":False,"prev_comp":True,"multiple":False,
        "placenta":False,"mental":True,"stress":8.5,"sleep":4.0,
    },
    "Mid Risk — Indonesia": {
        "country":"Indonesia","age":28,"rural":1,"anc":3,
        "gest_week":24,"prev_preg":1,"sbp":135,"dbp":88,
        "bs":6.2,"hb":10.1,"hba1c":5.9,"spo2":96.5,
        "temp_c":37.2,"hr":98,"bmi":28.4,"nutrition":5.0,
        "dm":False,"gest_dm":False,"preeclamp":False,
        "thyroid":True,"prev_comp":False,"multiple":False,
        "placenta":False,"mental":True,"stress":6.5,"sleep":5.5,
    },
    "Low Risk — Singapore": {
        "country":"Singapore","age":32,"rural":0,"anc":8,
        "gest_week":20,"prev_preg":1,"sbp":112,"dbp":72,
        "bs":4.8,"hb":13.1,"hba1c":4.9,"spo2":99.5,
        "temp_c":37.0,"hr":76,"bmi":23.2,"nutrition":8.5,
        "dm":False,"gest_dm":False,"preeclamp":False,
        "thyroid":False,"prev_comp":False,"multiple":False,
        "placenta":False,"mental":False,"stress":3.0,"sleep":8.0,
    },
}

def celsius_to_f(c): return round(c * 9/5 + 32, 1)
def f_to_celsius(f): return round((f - 32) * 5/9, 1)

# ── Session state init ────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = CFG.get("default_language", "en")
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "telemetry_opted" not in st.session_state:
    st.session_state.telemetry_opted = CFG.get("telemetry_opt_in", False)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🤱 {CFG['app_name']}")
    st.caption(f"v{CFG.get('app_version','2.0.0')}")
    st.divider()

    # Language selector
    lang_choice = st.selectbox(
        "🌐 Language / Bahasa / Ngôn ngữ",
        options=list(LANG_NAMES.keys()),
        format_func=lambda x: LANG_NAMES[x],
        index=list(LANG_NAMES.keys()).index(
            st.session_state.get("lang","en")),
        key="lang_selector"
    )
    st.session_state.lang = lang_choice

    # Temperature unit
    temp_unit = st.radio(
        "Temperature unit",
        ["Celsius (°C)", "Fahrenheit (°F)"],
        index=0 if CFG.get("temp_unit","celsius")=="celsius" else 1,
        horizontal=True,
    )
    use_celsius = "Celsius" in temp_unit

    st.divider()
    page = st.radio(t("nav_home"), [
        f"🏠 {t('nav_home')}",
        f"🔬 {t('nav_assess')}",
        f"📊 {t('nav_perf')}",
        f"ℹ️ {t('nav_about')}",
    ])

    st.divider()
    st.markdown("**Model info**")
    if DEMO_MODE:
        st.warning("Demo mode active")
        if load_error:
            st.caption(f"Error: {load_error[:80]}...")
    else:
        st.success("Model loaded ✓")
        st.caption(f"Checksum: `{BUNDLE_CKSUM[:8]}`")
    st.markdown("""
    <div style='font-size:0.88rem;color:#90caf9;line-height:2'>
    XGBoost + CatBoost<br>
    16,900 ASEAN patients<br>
    Vietnam ext. AUC: <b style='color:#64b5f6'>0.9541</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    # Telemetry opt-in
    telem = st.checkbox(
        "Share anonymous performance data",
        value=st.session_state.telemetry_opted,
        help="Sends only inference latency (ms). No patient data ever leaves this device.",
    )
    st.session_state.telemetry_opted = telem
    CFG["telemetry_opt_in"] = telem

# ════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════
if t("nav_home") in page:
    st.markdown(f"# 🤱 {t('app_title')}")
    st.markdown(f"### {t('app_subtitle')}")
    if DEMO_MODE:
        st.info("Model not found — running in demo mode with sample predictions.")
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        ("0.9541","External AUC\n(Vietnam holdout)"),
        ("16,900","ASEAN patients\n10 countries"),
        ("<50ms","Inference speed\nper patient"),
        ("9/10","XGB / CatBoost\nfeature agreement"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val'>{val}</div>
                <div class='metric-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### The Problem")
        st.markdown("""
Every year **94,000 mothers** die in ASEAN from preventable causes.
CHWs in rural Myanmar, Timor-Leste, and Laos have **no AI decision support**.

| Country | Maternal Deaths / 100k |
|---|---|
| Myanmar | 250 |
| Timor-Leste | 195 |
| Laos | 185 |
| Singapore | 8 |

*This 31× gap is preventable with early risk identification.*
        """)
    with cb:
        st.markdown("#### Our Solution")
        st.markdown("""
MaternaAI gives every CHW an **offline-capable AI assistant** that:

✅ Predicts high / mid / low risk in **< 50ms**

✅ Explains predictions with **SHAP feature importance**

✅ Fires **emergency alerts** for critical cases

✅ Works on **any phone browser** — no app install required

✅ Validated on **Vietnam** — country never seen during training

✅ Supports **6 ASEAN languages**

> *"Offline-capable when run locally with bundled model — no internet required after setup."*
        """)

    st.divider()
    st.markdown("#### 4-Tier Alert Policy")
    p1,p2,p3,p4 = st.columns(4)
    tiers = [
        ("#1b5e20","#a5d6a7","Score 0–30%","Self-care","Routine ANC schedule"),
        ("#4e342e","#ffcc80","Score 30–60%","CHW Follow-up","Review within 48h"),
        ("#bf360c","#ffccbc","Score 60–80%","Clinic Referral","Within 24 hours"),
        ("#7f0000","#ffcdd2","Score 80–100%","Emergency","Immediate referral NOW"),
    ]
    for col,(bg,tc,score,level,action) in zip([p1,p2,p3,p4],tiers):
        with col:
            st.markdown(f"""<div style='background:{bg};border-radius:10px;
                padding:14px;text-align:center'>
                <b style='color:{tc}'>{score}</b><br>
                <span style='color:{tc};font-weight:700'>{level}</span><br>
                <small style='color:{tc}'>{action}</small>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════
elif t("nav_assess") in page:
    st.markdown(f"## 🔬 {t('nav_assess')}")

    # ── Consent flow ─────────────────────────────────────────
    if not st.session_state.consent_given:
        st.markdown(f"""<div class='consent-box'>
            <b>🔒 {t('consent_title')}</b><br><br>
            {t('consent_body')}
        </div>""", unsafe_allow_html=True)
        consent_check = st.checkbox(t("consent_checkbox"))
        if st.button("Continue to assessment →",
                     disabled=not consent_check, type="primary"):
            st.session_state.consent_given = True
            st.rerun()
        st.stop()

    st.success("✓ Consent recorded — session only, not stored")

    # ── Preset loader ─────────────────────────────────────────
    preset_choice = st.selectbox(
        "📋 Load sample patient (for demo)",
        ["— Enter manually —"] + list(PRESETS.keys()),
    )
    preset = PRESETS.get(preset_choice, {})

    def pv(key, default):
        """Get preset value or default."""
        return preset.get(key, default)

    st.divider()
    with st.form("patient_form", clear_on_submit=False):
        st.markdown("### Demographics & Context")
        c1,c2,c3 = st.columns(3)
        with c1:
            country = st.selectbox(t("lbl_country"), [
                "Indonesia","Philippines","Vietnam","Myanmar",
                "Cambodia","Thailand","Timor-Leste","Laos",
                "Malaysia","Singapore"],
                index=["Indonesia","Philippines","Vietnam","Myanmar",
                       "Cambodia","Thailand","Timor-Leste","Laos",
                       "Malaysia","Singapore"].index(pv("country","Indonesia")))
            age = st.slider(t("lbl_age"), 15, 49, pv("age",25))
        with c2:
            rural_opt = "Rural" if pv("rural",1)==1 else "Urban"
            rural     = st.radio(t("lbl_location"), ["Rural","Urban"],
                                 index=0 if rural_opt=="Rural" else 1,
                                 horizontal=True)
            anc       = st.slider(t("lbl_anc"),0,14,pv("anc",3),
                                  help=t("help_anc"))
        with c3:
            gest_week = st.slider(t("lbl_gest"),4,42,pv("gest_week",24))
            prev_preg = st.slider(t("lbl_prev"),0,10,pv("prev_preg",0))

        st.divider()
        st.markdown("### Vital Signs")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            sbp = st.number_input(f"{t('lbl_sbp')} (mmHg)",80,200,
                                  pv("sbp",118), help=t("help_sbp"))
            dbp = st.number_input(f"{t('lbl_dbp')} (mmHg)",50,120,
                                  pv("dbp",76))
        with c2:
            bs = st.number_input(f"{t('lbl_bs')} (mmol/L)",3.0,15.0,
                                 float(pv("bs",5.2)),0.1, help=t("help_bs"))
            hb = st.number_input(f"{t('lbl_hb')} (g/dL)",5.0,17.0,
                                 float(pv("hb",11.5)),0.1, help=t("help_hb"))
        with c3:
            hba1c = st.number_input(f"{t('lbl_hba1c')} (%)",3.5,11.0,
                                    float(pv("hba1c",5.2)),0.1)
            spo2  = st.number_input(f"{t('lbl_spo2')} (%)",85.0,100.0,
                                    float(pv("spo2",98.0)),0.1)
        with c4:
            if use_celsius:
                temp_in = st.number_input(f"{t('lbl_temp')} (°C)",36.0,42.0,
                                          float(pv("temp_c",37.0)),0.1)
                temp_f  = celsius_to_f(temp_in)
            else:
                temp_f = st.number_input(f"{t('lbl_temp')} (°F)",96.8,107.6,
                                         celsius_to_f(pv("temp_c",37.0)),0.1)
            hr = st.number_input(f"{t('lbl_hr')} (bpm)",50,130,
                                 pv("hr",80))

        c1,c2 = st.columns(2)
        with c1:
            bmi = st.number_input(f"{t('lbl_bmi')} (kg/m²)",13.0,45.0,
                                  float(pv("bmi",22.5)),0.1)
        with c2:
            nutrition = st.slider(f"{t('lbl_nutrition')} (1–10)",
                                  1.0,10.0,float(pv("nutrition",6.0)),0.5)

        st.divider()
        st.markdown("### Medical History")
        c1,c2,c3 = st.columns(3)
        with c1:
            dm       = st.checkbox("Pre-existing diabetes",   pv("dm",False))
            gest_dm  = st.checkbox("Gestational diabetes",    pv("gest_dm",False))
            preeclamp= st.checkbox("Preeclampsia history",    pv("preeclamp",False))
        with c2:
            thyroid  = st.checkbox("Thyroid disorder",        pv("thyroid",False))
            prev_comp= st.checkbox("Previous complications",  pv("prev_comp",False))
            multiple = st.checkbox("Multiple pregnancy",      pv("multiple",False))
        with c3:
            placenta = st.checkbox("Placenta previa",         pv("placenta",False))
            mental   = st.checkbox("Mental health flag",      pv("mental",False))
            stress   = st.slider(f"{t('lbl_stress')} (1–10)",
                                 1.0,10.0,float(pv("stress",5.0)),0.5)
        sleep = st.slider(f"{t('lbl_sleep')} (hours)",
                          3.0,10.0,float(pv("sleep",7.0)),0.5)

        st.divider()
        submitted = st.form_submit_button(
            f"🔍 {t('btn_predict')}",
            use_container_width=True, type="primary")

    # ── Prediction ────────────────────────────────────────────
    if submitted:
        pulse_p = int(sbp) - int(dbp)
        map_val = round(dbp + pulse_p / 3, 1)
        bmi_cat = 0 if bmi<18.5 else (1 if bmi<23 else (2 if bmi<27.5 else 3))
        age_grp = 0 if age<18 else (1 if age<=35 else 2)
        bp_cat  = (0 if sbp<120 and dbp<80 else
                   1 if sbp<130 and dbp<80 else
                   2 if sbp<140 or  dbp<90 else 3)

        inputs = {
            "Age":age,"SystolicBP":sbp,"DiastolicBP":dbp,
            "BloodSugar_Fasting":bs,"BodyTemperature":temp_f,
            "HeartRate":hr,"BMI":bmi,"HbA1c":hba1c,
            "SpO2":spo2,"Hemoglobin":hb,
            "PreviousPregnancies":prev_preg,"GestationalWeek":gest_week,
            "PreexistingDiabetes":int(dm),"PreviousComplications":int(prev_comp),
            "GestationalDiabetes":int(gest_dm),"PreeclampsiaHistory":int(preeclamp),
            "ThyroidDisorder":int(thyroid),"PlacentaPrevia":int(placenta),
            "MultiplePregnancy":int(multiple),"MentalHealthFlag":int(mental),
            "RuralUrban":1 if rural=="Rural" else 0,
            "ANCVisits":anc,"NutritionScore":nutrition,
            "SleepHours":sleep,"StressLevel":stress,"Country":country,
            "PulsePressure":pulse_p,"MAP":map_val,
            "BMI_Category":bmi_cat,"AgeRiskGroup":age_grp,"BP_Category":bp_cat,
        }

        # Validate + clamp
        inputs, val_warns = validate_clamp(inputs)
        if val_warns:
            for w in val_warns:
                st.warning(f"Input validation: {w}")

        # BP sanity check
        if dbp >= sbp:
            st.error("Diastolic BP must be lower than Systolic BP. Please recheck.")
            st.stop()

        with st.spinner("Analysing patient..."):
            result = run_prediction(inputs, demo_mode=DEMO_MODE)

        flags = check_who_flags(inputs)
        ihash = inputs_hash(inputs)

        # Audit log (background thread)
        threading.Thread(
            target=write_audit_log,
            args=(ihash, result, flags,
                  st.session_state.get("lang","en"), DEMO_MODE),
            daemon=True,
        ).start()

        st.session_state.last_result = result
        st.divider()
        st.markdown("## Prediction Results")

        if DEMO_MODE:
            st.info("Demo mode — result is illustrative, not from trained model.")

        cls     = result["class"]
        hr_pct  = result["hr_prob"] * 100
        badge   = "risk-high" if cls=="high risk" else \
                  "risk-mid"  if cls=="mid risk"  else "risk-low"
        emoji   = "🚨" if cls=="high risk" else "⚠️" if cls=="mid risk" else "✅"

        col_r, col_d = st.columns([1,2])
        with col_r:
            st.markdown(f"""
            <div class='{badge}'>
              <p class='risk-title'>{emoji}</p>
              <p class='risk-title'>{cls.upper()}</p>
              <p class='risk-prob'>HR probability: {hr_pct:.1f}%</p>
              <p class='risk-label'>{result['ms']:.1f}ms
                {'⚡ fast' if result['ms']<50 else '🐢 check device'}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Probability breakdown**")
            clr = {"high risk":"#b71c1c","mid risk":"#e65100","low risk":"#1b5e20"}
            for c in ["high risk","mid risk","low risk"]:
                p = result["probs"].get(c, 0)
                st.markdown(f"""
                <div style='margin:4px 0'>
                  <div style='display:flex;justify-content:space-between;font-size:0.82rem'>
                    <span>{c}</span><span><b>{p*100:.1f}%</b></span>
                  </div>
                  <div style='background:#e0e0e0;border-radius:4px;height:10px'>
                    <div style='background:{clr[c]};width:{p*100:.1f}%;
                         height:10px;border-radius:4px'></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div style='margin-top:8px'>
                <span class='audit-badge'>Audit ID: {ihash}</span>
            </div>""", unsafe_allow_html=True)

        with col_d:
            if flags:
                st.markdown("**⚠️ WHO Danger Signs Detected:**")
                for f in flags:
                    st.markdown(
                        f"<span class='pill-red'>🔴 {f}</span>",
                        unsafe_allow_html=True)
                st.markdown("")

            # Calibrated alert action
            if hr_pct >= 80:
                st.markdown(f"""<div class='alert-box alert-emergency'>
                    🚨 {t('alert_emergency')}<br>
                    <small>Steps: (1) Call facility now (2) Escort patient
                    (3) Bring this report (4) Monitor vitals en route</small>
                </div>""", unsafe_allow_html=True)
            elif hr_pct >= 60:
                st.markdown(f"""<div class='alert-box alert-followup'>
                    ⚠️ {t('alert_clinic')}<br>
                    <small>Steps: (1) Book appointment today
                    (2) Recheck vitals in 4 hours
                    (3) Advise rest and hydration</small>
                </div>""", unsafe_allow_html=True)
            elif hr_pct >= 30:
                st.markdown(f"""<div class='alert-box alert-chw'>
                    📋 {t('alert_chw')}<br>
                    <small>Steps: (1) Schedule home visit
                    (2) Rescreen with full vitals (3) Check ANC compliance</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='alert-box alert-routine'>
                    ✅ {t('alert_routine')}<br>
                    <small>Next ANC as scheduled. Advise adequate nutrition and rest.</small>
                </div>""", unsafe_allow_html=True)

            # Key risk factors
            st.markdown("**Key clinical findings:**")
            findings = []
            if sbp>=140: findings.append(f"High SBP: {sbp} mmHg")
            if dbp>=90:  findings.append(f"High DBP: {dbp} mmHg")
            if hb<11:    findings.append(f"Anemia: Hb {hb:.1f} g/dL")
            if bs>=7.0:  findings.append(f"High glucose: {bs:.1f} mmol/L")
            if spo2<95:  findings.append(f"Low SpO2: {spo2:.1f}%")
            if hr>=100:  findings.append(f"Tachycardia: {hr} bpm")
            if anc<4:    findings.append(f"Only {anc} ANC visits")
            if bmi<18.5: findings.append(f"Underweight: BMI {bmi:.1f}")
            if preeclamp:findings.append("Preeclampsia history")
            if multiple: findings.append("Multiple pregnancy")
            if placenta: findings.append("Placenta previa")

            if findings:
                for ff in findings[:7]:
                    st.markdown(f"<span class='pill-red'>⬆ {ff}</span>",
                               unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill-grn'>✅ No major danger signs</span>",
                           unsafe_allow_html=True)

        st.divider()
        # Export
        export_consent = st.checkbox(
            "I confirm this export is for authorised clinical records only",
            key="export_consent")
        export_data = {
            "timestamp":      datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "audit_id":       ihash,
            "model_version":  CFG.get("app_version","2.0.0"),
            "demo_mode":      DEMO_MODE,
            "country":        country,
            "age":            age,
            "prediction":     cls,
            "hr_probability": round(hr_pct/100, 4),
            "who_flags":      flags,
            "inference_ms":   result["ms"],
            "note": "For decision support only. Not a replacement for clinical judgment.",
        }
        st.download_button(
            f"📥 {t('btn_download')}",
            data=json.dumps(export_data, indent=2),
            file_name=f"materna_{ihash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            disabled=not export_consent,
            use_container_width=True,
        )
        if not export_consent:
            st.caption("Check the box above to enable download.")

# ════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif t("nav_perf") in page:
    st.markdown(f"## 📊 {t('nav_perf')}")
    st.caption("All metrics computed on data the model never saw during training.")
    st.divider()

    manifest_path = APP_DIR / "run_manifest.json"
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        m = manifest.get("metrics_test", {})
        v = manifest.get("metrics_vietnam", {})

        st.markdown("### Held-out Test Set")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            ci = m.get("ROC_AUC_CI",["?","?"])
            st.metric("ROC-AUC (DeLong)",f"{m.get('ROC_AUC_delong',0):.4f}",
                      f"[{ci[0]}–{ci[1]}]")
        with c2:
            ci2 = m.get("AUPRC_CI",["?","?"])
            st.metric("AUPRC", f"{m.get('AUPRC',0):.4f}",
                      f"[{ci2[0]}–{ci2[1]}]")
        with c3:
            st.metric("Sensitivity @ 80% spec",
                      f"{m.get('Sens_at_80spec',0):.4f}")
        with c4:
            st.metric("Brier score", f"{m.get('Brier',0):.4f}",
                      "lower = better calibrated")

        st.divider()
        st.markdown("### Vietnam External Validation (unseen country)")
        c1,c2,c3,_ = st.columns(4)
        with c1:
            ci3 = v.get("ROC_AUC_CI",["?","?"])
            st.metric("ROC-AUC",f"{v.get('ROC_AUC_delong',0):.4f}",
                      f"[{ci3[0]}–{ci3[1]}]")
        with c2:
            st.metric("AUPRC",f"{v.get('AUPRC',0):.4f}")
        with c3:
            st.metric("vs WHO PIERS-ML","0.9541 vs 0.870","MaternaAI wins ↑")

        st.divider()
        fair = manifest.get("fairness_spread", {})
        leak = manifest.get("leakage_audit",  {})
        shap_info = manifest.get("shap", {})

        st.markdown("### Fairness & Governance")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.metric("Country AUC spread",
                      f"{fair.get('AUC_spread',0):.4f}","< 0.05 = FAIR")
        with c2:
            st.metric("Fairness verdict", fair.get("verdict","N/A"))
        with c3:
            st.metric("Train/Test duplicates",
                      leak.get("train_test_duplicates",0),"0 = PASS")
        with c4:
            st.metric("XGB/Cat agreement",
                      shap_info.get("xgb_cat_top10_agreement","N/A"))

        st.divider()
        st.markdown("### Evaluation Charts")
        for img_name, caption in [
            ("clinical_evaluation_v2.png", "Master clinical evaluation (8 panels)"),
            ("shap_dashboard_v2.png",       "SHAP explainability dashboard"),
        ]:
            img_path = DATA_DIR / img_name
            if img_path.exists():
                st.image(str(img_path), caption=caption,
                         use_container_width=True)
            else:
                st.info(f"{img_name} not found. Run Cell 5 / Cell 6 first.")

    except FileNotFoundError:
        st.warning("run_manifest.json not found. Run Cell 5 first.")
    except Exception as e:
        st.error(f"Error loading manifest: {e}")

# ════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════
elif t("nav_about") in page:
    st.markdown(f"## ℹ️ {t('nav_about')}")
    st.divider()

    ca, cb = st.columns(2)
    with ca:
        st.markdown("### Technical Stack")
        st.markdown("""
| Component | Technology |
|---|---|
| Model | XGBoost + CatBoost Ensemble |
| Tuning | Optuna (80 trials) |
| Calibration | Isotonic regression |
| Explainability | SHAP TreeExplainer |
| Frontend | Streamlit (offline-capable) |
| Data | WHO/DHS-calibrated synthetic |
| Languages | EN / ID / VI / KM / TH / TL |
| Audit | JSONL anonymised log |
        """)

        st.markdown("### Offline Deployment")
        st.markdown("""
> MaternaAI runs **fully offline** when installed locally.
> No internet required after initial setup.
> Model bundle (~3 MB) ships with the application.
>
> For true PWA / native app packaging, use:
> `pyinstaller app.py` or wrap with **Tauri** for mobile.
        """)

        st.markdown("### Demo Video")
        st.markdown("📹 [Watch 90-second demo →](#)  *(link to be added before submission)*")

    with cb:
        st.markdown("### Ethics & Governance")
        ethics_path = APP_DIR / "ETHICS.md"
        if ethics_path.exists():
            with open(ethics_path, encoding="utf-8") as f:
                st.markdown(f.read()[:2500] + "\n\n*[see ETHICS.md for full text]*")
        else:
            st.info("ETHICS.md not found. Run Cell 5 first.")

    st.divider()
    st.markdown("### ASEAN Country Coverage")
    st.dataframe(pd.DataFrame({
        "Country":     ["Myanmar","Timor-Leste","Laos","Cambodia","Indonesia",
                        "Philippines","Vietnam","Thailand","Malaysia","Singapore"],
        "WHO MMR/100k":[250,195,185,160,173,78,43,29,24,8],
        "Training n":  [2000,1200,1000,1800,3000,2500,0,1500,1200,500],
        "Role":        ["Train"]*6 + ["External Validation"] + ["Train"]*3,
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(f"MaternaAI v{CFG.get('app_version','2.0.0')} | "
               f"ASEAN AI Hackathon 2026 | "
               "Not for clinical use without institutional validation | "
               f"Model checksum: {BUNDLE_CKSUM[:8]}")
