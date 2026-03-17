# PCOS Risk Factor Analysis

Exploratory data analysis and machine learning prediction on a PCOS clinical dataset using Python.

## Dataset
- 541 patients, 45 features
- Source: Kaggle - PCOS Dataset
- Target variable: PCOS diagnosis (1 = Yes, 0 = No)
- 32.72% of patients diagnosed with PCOS

## Analysis

### Key Findings
- Weight gain, hair growth, and skin darkening are 3-4x more common in PCOS patients
- AMH levels are significantly higher in PCOS patients — a key clinical marker
- LH levels tend to be elevated in PCOS patients

### Visualizations
- Symptom prevalence comparison (PCOS vs No PCOS)
- Hormone levels analysis (FSH, LH, FSH/LH ratio, AMH)

### Machine Learning Model
- Algorithm: Logistic Regression
- Features: BMI, hormone levels, follicle count, symptoms
- Train/Test split: 80/20
- Model Accuracy: 86.11%

## Tools
- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## Author
Emi Rivera | Data Science Student | George Washington University
