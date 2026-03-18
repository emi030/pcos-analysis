# PCOS Risk Factor Analysis

Exploratory data analysis, statistical testing, and machine learning prediction on a PCOS clinical dataset using Python.

## Dataset
- 541 patients, 45 features
- Source: Kaggle - PCOS Dataset
- Target variable: PCOS diagnosis (1 = Yes, 0 = No)
- 32.72% of patients diagnosed with PCOS

## Analysis

### Key Findings
- Weight gain, hair growth, and skin darkening are 3-4x more common in PCOS patients (p < 0.0001)
- AMH levels are significantly higher in PCOS patients, this is a key clinical marker (p = 0.0001)
- BMI is significantly elevated in PCOS patients (p = 0.0002)
- LH was the only non-significant hormone marker (p = 0.623), raising questions about its reliability across populations
- PCA of clinical features explains 49.1% of variance, with visible separation between PCOS and non-PCOS patients along PC1

### Visualizations
- Symptom prevalence comparison (PCOS vs No PCOS)
- Hormone levels analysis (FSH, LH, FSH/LH ratio, AMH)
- Statistical significance plot (-log10 p-values)
- PCA of clinical features

### Machine Learning Model
- Algorithm: Logistic Regression
- Features: BMI, hormone levels, follicle count, symptoms
- Train/Test split: 80/20
- Model Accuracy: 86.11%

## Tools
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn, scipy

## Author
Emi Rivera | Data Science Student | George Washington University
