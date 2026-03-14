# MaternaAI — Ethics & Data Governance

## Data Sources
- Synthetic data: WHO/DHS-calibrated distributions for 10 ASEAN countries
- Calibration references: WHO Global Health Observatory, DHS Program surveys
- No real patient records used. All records are computationally generated.

## IRB / Consent Status
- Current phase: synthetic data only — IRB not required
- Deployment phase: IRB approval required from each participating institution
- Patient consent protocol: opt-in, written consent in local language
- Data minimisation: only clinically necessary features collected

## De-identification
- Synthetic dataset: no real individuals — de-identification not applicable
- Deployment: all PHI stripped before model inference; no names/IDs stored

## Federated Learning Plan
- Phase 2: federated averaging across 3 pilot hospitals (Indonesia, Vietnam, Philippines)
- No raw patient data leaves the hospital — only gradient updates transmitted
- Differential privacy (epsilon=1.0) applied to gradient updates

## Image Privacy
- No patient images collected or stored
- All computer vision (if added) runs 100% on-device
- Zero images transmitted to any server

## Bias & Fairness
- Per-country AUC spread monitored (target: < 0.05)
- Rural/urban parity checked at each model update
- Adolescent (< 18) subgroup flagged for special review

## Limitations
- Trained on synthetic data — real-world performance may differ
- Validated externally on Vietnam holdout only — broader validation needed
- Not a replacement for clinical judgment — intended as decision support only

## Recommended Clinical Workflow
1. CHW enters vital signs → model outputs risk score
2. Score 0.8+ → immediate referral, do not wait for confirmation
3. Score 0.3–0.8 → follow-up protocol per facility guidelines
4. Score < 0.3 → standard ANC schedule maintained
5. All model outputs logged for monthly audit
