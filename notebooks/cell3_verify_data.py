import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

print("=" * 60)
print("MATERNA AI - Cell 3: Data Verification & Visualization")
print("=" * 60)

# Load the dataset we just built
df = pd.read_csv('../data/materna_ai_asean_dataset.csv')

print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ── BASIC CHECKS ──────────────────────────────────────────
print("\n" + "-" * 40)
print("BASIC HEALTH CHECKS")
print("-" * 40)

missing = df.isnull().sum().sum()
print(f"Missing values: {missing} (should be 0)")

duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates} (should be 0)")

print(f"\nFeature data types:")
print(df.dtypes.value_counts().to_string())

# ── RISK DISTRIBUTION ─────────────────────────────────────
print("\n" + "-" * 40)
print("RISK LEVEL DISTRIBUTION")
print("-" * 40)
risk_counts = df['RiskLevel'].value_counts()
risk_pct = df['RiskLevel'].value_counts(normalize=True).mul(100).round(1)
for risk in ['high risk', 'mid risk', 'low risk']:
    bar = '#' * int(risk_pct[risk] / 2)
    print(f"  {risk:12s}: {risk_counts[risk]:,} ({risk_pct[risk]}%) {bar}")

# ── COUNTRY BREAKDOWN ─────────────────────────────────────
print("\n" + "-" * 40)
print("RISK BY COUNTRY (shows ASEAN diversity)")
print("-" * 40)
country_risk = df.groupby('Country')['RiskLevel'].value_counts(
    normalize=True).mul(100).round(1).unstack()
print(country_risk.to_string())

# ── CLINICAL STATS ────────────────────────────────────────
print("\n" + "-" * 40)
print("CLINICAL FEATURE STATISTICS")
print("-" * 40)
clinical_cols = ['Age','SystolicBP','DiastolicBP','BloodSugar_Fasting',
                 'BMI','HbA1c','Hemoglobin','SpO2','HeartRate']
print(df[clinical_cols].describe().round(2).to_string())

# ── VISUALIZATIONS ────────────────────────────────────────
print("\nGenerating charts...")

fig = plt.figure(figsize=(20, 24))
fig.suptitle('MaternaAI - ASEAN Maternal Health Dataset Overview',
             fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {
    'high risk': '#e74c3c',
    'mid risk':  '#f39c12',
    'low risk':  '#2ecc71'
}
COUNTRY_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# Chart 1: Risk distribution pie
ax1 = fig.add_subplot(gs[0, 0])
sizes = [risk_counts.get(r, 0) for r in ['high risk', 'mid risk', 'low risk']]
colors = [COLORS[r] for r in ['high risk', 'mid risk', 'low risk']]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=['High Risk', 'Mid Risk', 'Low Risk'],
    colors=colors, autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 10})
for at in autotexts:
    at.set_fontweight('bold')
ax1.set_title('Overall Risk Distribution', fontweight='bold', pad=15)

# Chart 2: Patients per country bar
ax2 = fig.add_subplot(gs[0, 1])
country_counts = df['Country'].value_counts()
bars = ax2.barh(country_counts.index, country_counts.values,
                color=COUNTRY_COLORS, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, country_counts.values):
    ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=9, fontweight='bold')
ax2.set_xlabel('Number of Patients')
ax2.set_title('Patients per Country', fontweight='bold')
ax2.set_xlim(0, max(country_counts.values) * 1.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Chart 3: Risk by country stacked bar
ax3 = fig.add_subplot(gs[0, 2])
country_risk_abs = df.groupby('Country')['RiskLevel'].value_counts().unstack().fillna(0)
country_risk_pct = country_risk_abs.div(country_risk_abs.sum(axis=1), axis=0) * 100
risk_order = ['high risk', 'mid risk', 'low risk']
bottom = np.zeros(len(country_risk_pct))
for risk in risk_order:
    if risk in country_risk_pct.columns:
        vals = country_risk_pct[risk].values
        ax3.barh(country_risk_pct.index, vals, left=bottom,
                 color=COLORS[risk], label=risk.title(), alpha=0.85)
        bottom += vals
ax3.set_xlabel('Percentage (%)')
ax3.set_title('Risk Profile by Country', fontweight='bold')
ax3.legend(loc='lower right', fontsize=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Charts 4-9: Clinical feature distributions by risk
features_to_plot = [
    ('SystolicBP',         'Systolic BP (mmHg)'),
    ('DiastolicBP',        'Diastolic BP (mmHg)'),
    ('BloodSugar_Fasting', 'Fasting Blood Sugar (mmol/L)'),
    ('HbA1c',              'HbA1c (%)'),
    ('Hemoglobin',         'Hemoglobin (g/dL)'),
    ('BMI',                'BMI'),
]

positions = [(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
for (col, label), (row_i, col_i) in zip(features_to_plot, positions):
    ax = fig.add_subplot(gs[row_i, col_i])
    for risk, color in COLORS.items():
        subset = df[df['RiskLevel'] == risk][col]
        ax.hist(subset, bins=30, alpha=0.6, color=color,
                label=risk.title(), density=True)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'{label} by Risk Level', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Chart 10: Age distribution
ax10 = fig.add_subplot(gs[3, 0])
for risk, color in COLORS.items():
    subset = df[df['RiskLevel'] == risk]['Age']
    ax10.hist(subset, bins=25, alpha=0.6, color=color,
              label=risk.title(), density=True)
ax10.set_xlabel('Age (years)')
ax10.set_ylabel('Density')
ax10.set_title('Age Distribution by Risk', fontweight='bold')
ax10.legend(fontsize=8)
ax10.spines['top'].set_visible(False)
ax10.spines['right'].set_visible(False)

# Chart 11: Rural vs Urban risk
ax11 = fig.add_subplot(gs[3, 1])
rural_risk = df.groupby(['RuralUrban','RiskLevel']).size().unstack().fillna(0)
rural_risk.index = ['Urban','Rural']
rural_pct = rural_risk.div(rural_risk.sum(axis=1), axis=0) * 100
x = np.arange(2)
width = 0.25
for i, risk in enumerate(['high risk','mid risk','low risk']):
    if risk in rural_pct.columns:
        ax11.bar(x + i*width, rural_pct[risk], width,
                 label=risk.title(), color=COLORS[risk], alpha=0.85)
ax11.set_xticks(x + width)
ax11.set_xticklabels(['Urban','Rural'])
ax11.set_ylabel('Percentage (%)')
ax11.set_title('Risk: Rural vs Urban', fontweight='bold')
ax11.legend(fontsize=8)
ax11.spines['top'].set_visible(False)
ax11.spines['right'].set_visible(False)

# Chart 12: Correlation heatmap
ax12 = fig.add_subplot(gs[3, 2])
corr_cols = ['SystolicBP','DiastolicBP','BloodSugar_Fasting',
             'HbA1c','Hemoglobin','BMI','Age','SpO2']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax12, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', annot_kws={'size':7},
            linewidths=0.5, cbar_kws={'shrink':0.8})
ax12.set_title('Feature Correlation Heatmap', fontweight='bold')
ax12.tick_params(axis='x', rotation=45, labelsize=7)
ax12.tick_params(axis='y', rotation=0, labelsize=7)

# Save
os.makedirs('../data', exist_ok=True)
chart_path = '../data/dataset_overview.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

print(f"\nChart saved to: {chart_path}")
print("\n" + "=" * 60)
print("DATA VERIFICATION COMPLETE")
print("=" * 60)
print("\nKey findings:")
print(f"  Total patients     : {len(df):,}")
print(f"  Countries covered  : {df['Country'].nunique()}")
print(f"  Features           : {df.shape[1]-1}")
print(f"  Missing values     : {df.isnull().sum().sum()}")
print(f"  High risk patients : {risk_counts.get('high risk',0):,}")
print(f"  Rural patients     : {df['RuralUrban'].sum():,} ({df['RuralUrban'].mean()*100:.1f}%)")
print(f"  Anaemic patients   : {(df['Hemoglobin']<11).sum():,} ({(df['Hemoglobin']<11).mean()*100:.1f}%)")
print(f"  Diabetic markers   : {(df['HbA1c']>=6.5).sum():,} ({(df['HbA1c']>=6.5).mean()*100:.1f}%)")
print("\nDataset is healthy and ready for model training!")
