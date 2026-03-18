import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Load Data ──
df = pd.read_csv('pcos_data.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.columns.tolist())

# ── Clean Data ──
df = df.drop(columns=['Unnamed: 44', 'Sl. No', 'Patient File No.'])
df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
df['Fast food (Y/N)'] = df['Fast food (Y/N)'].fillna(0)

print(df['PCOS (Y/N)'].value_counts())
print(f"PCOS rate: {df['PCOS (Y/N)'].mean():.2%}")
print(df.isnull().sum())

# ── Plot 1: Symptom Comparison ──
symptoms = ['Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)']

pcos_symptoms = df[df['PCOS (Y/N)'] == 1][symptoms].mean()
no_pcos_symptoms = df[df['PCOS (Y/N)'] == 0][symptoms].mean()

symptom_df = pd.DataFrame({
    'PCOS': pcos_symptoms,
    'No PCOS': no_pcos_symptoms
})

symptom_df.plot(kind='bar', figsize=(10, 6))
plt.title('Symptom Prevalence: PCOS vs No PCOS')
plt.xlabel('Symptom')
plt.ylabel('Proportion of Patients')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('symptoms_comparison.png')
print("Symptoms plot saved!")

# ── Plot 2: Hormone Analysis ──
df_clean = df[
    (df['FSH(mIU/mL)'] < 50) &
    (df['LH(mIU/mL)'] < 50) &
    (df['FSH/LH'] < 10) &
    (df['AMH(ng/mL)'] < 10)
]

hormones = ['FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'AMH(ng/mL)']

plt.figure(figsize=(12, 4))
for i, hormone in enumerate(hormones):
    plt.subplot(1, 4, i+1)
    sns.boxplot(x='PCOS (Y/N)', y=hormone, data=df_clean)
    plt.xticks([0, 1], ['No PCOS', 'PCOS'])
    plt.title(hormone)

plt.tight_layout()
plt.savefig('hormone_analysis_clean.png')
print("Hormone plot saved!")

# ── Machine Learning Model ──
features = ['BMI', 'FSH(mIU/mL)', 'LH(mIU/mL)',
            'AMH(ng/mL)', 'Follicle No. (L)', 'Follicle No. (R)',
            'Weight gain(Y/N)', 'hair growth(Y/N)', 'Pimples(Y/N)']

X = df[features].dropna()
y = df.loc[X.index, 'PCOS (Y/N)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# ── Statistical Tests ──
hormones_test = ['FSH(mIU/mL)', 'LH(mIU/mL)', 'AMH(ng/mL)', 'BMI']
symptoms_test = ['Weight gain(Y/N)', 'hair growth(Y/N)',
                 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)']

print("\n=== Hormone Tests ===")
for hormone in hormones_test:
    group1 = df_clean[df_clean['PCOS (Y/N)'] == 1][hormone].dropna()
    group2 = df_clean[df_clean['PCOS (Y/N)'] == 0][hormone].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    significance = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(f"{hormone}: p-value = {p_value:.4f} → {significance}")

print("\n=== Symptom Tests ===")
for symptom in symptoms_test:
    group1 = df[df['PCOS (Y/N)'] == 1][symptom].dropna()
    group2 = df[df['PCOS (Y/N)'] == 0][symptom].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    significance = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(f"{symptom}: p-value = {p_value:.4f} → {significance}")

# ── Plot 3: P-value Visualization ──
results = []

for hormone in hormones_test:
    group1 = df_clean[df_clean['PCOS (Y/N)'] == 1][hormone].dropna()
    group2 = df_clean[df_clean['PCOS (Y/N)'] == 0][hormone].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    results.append({'Feature': hormone, 'p-value': p_value, 'Type': 'Hormone'})

for symptom in symptoms_test:
    group1 = df[df['PCOS (Y/N)'] == 1][symptom].dropna()
    group2 = df[df['PCOS (Y/N)'] == 0][symptom].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    results.append({'Feature': symptom, 'p-value': p_value, 'Type': 'Symptom'})

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
colors = ['red' if p < 0.05 else 'gray' for p in results_df['p-value']]
plt.barh(results_df['Feature'], -np.log10(results_df['p-value']), color=colors)
plt.axvline(x=-np.log10(0.05), color='black', linestyle='--', label='p = 0.05')
plt.xlabel('-log10(p-value)')
plt.title('Statistical Significance of PCOS Features')
plt.legend()
plt.tight_layout()
plt.savefig('pvalue_plot.png')
print("P-value plot saved!")

# ── Plot 4: PCA Analysis ──
pca_features = ['BMI', 'FSH(mIU/mL)', 'LH(mIU/mL)',
                'AMH(ng/mL)', 'Follicle No. (L)', 'Follicle No. (R)']

X_pca = df[pca_features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
X_pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
colors = df.loc[X_pca.index, 'PCOS (Y/N)'].map({0: 'blue', 1: 'red'})
plt.scatter(X_pca_result[:, 0], X_pca_result[:, 1], c=colors, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA of PCOS Clinical Features')
plt.legend(handles=[
    plt.scatter([], [], color='blue', label='No PCOS'),
    plt.scatter([], [], color='red', label='PCOS')
], labels=['No PCOS', 'PCOS'])
plt.tight_layout()
plt.savefig('pca_plot.png')
print(f"PCA plot saved!")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")