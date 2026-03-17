import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── Load Data ──
df = pd.read_csv('pcos_data.csv')

# Preview the first 5 rows
print(df.head())

# Check the shape
print(df.shape)

# Basic statistics
print(df.describe())

# Check all column names
print(df.columns.tolist())

# ── Clean Data ──
df = df.drop(columns=['Unnamed: 44', 'Sl. No', 'Patient File No.'])
df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
df['Fast food (Y/N)'] = df['Fast food (Y/N)'].fillna(0)

# Check PCOS distribution
print(df['PCOS (Y/N)'].value_counts())
print(f"PCOS rate: {df['PCOS (Y/N)'].mean():.2%}")

# Check for missing values
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