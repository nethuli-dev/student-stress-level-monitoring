# Student Stress Level Monitoring

**Course:** IT2011 – Artificial Intelligence and Machine Learning
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)
**Group:** 2025-Y2-S1-MLB-B7G1-02
**Problem Domain:** Healthcare

## Team

| Reg No | Name |
|---|---|
| IT24101574 | Dhanapala N. N. |
| IT24103987 | Ilma M. S. F. |
| IT24101571 | Balasooriya K. S. B. |
| IT24101546 | De Silva G. H. T. D. |
| IT24101618 | Rosayro M. C. J. De |
| IT24101629 | Wijesundara W. M. B. H. |

## Overview

Student mental health is affected by academic pressure, social expectations, and personal/environmental factors. This project builds a Student Stress Level Monitoring system that classifies students into **Low**, **Moderate**, and **High** stress categories using psychological, academic, and social indicators, with the goal of supporting early detection while maintaining fairness and ethical data handling.

## Dataset

- **Source:** [Kaggle – Student Stress Monitoring dataset](https://www.kaggle.com/)
- **Type:** Tabular
- **Records:** 1,100 (793 after outlier removal)
- **Features:** 20 independent stress indicators + 1 target variable (`stress_level`)
- **Target classes:** `0` – Low, `1` – Moderate, `2` – High

## Pipeline

1. **Data Cleaning** – removed missing values and 11 duplicate rows
2. **Outlier Removal** – IQR method (1100 → 793 rows)
3. **Transformation** – Min-Max scaling; engineered features:
   - `health_index` ← depression, headache, blood_pressure, breathing_problem
   - `academic_stress` ← academic_performance, study_load, future_career_concerns
   - `social_environment` ← social_support, peer_pressure, bullying
4. **Feature Selection** – top 10 features via `SelectKBest` (ANOVA F-test)
5. **Dimensionality Reduction** – PCA (5 components)
6. **Modeling** – Logistic Regression, Random Forest, KNN, MLP, XGBoost, SVM (RBF), each with GridSearchCV hyperparameter tuning

## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | AUC (OvR, macro) |
|---|---|---|---|---|---|
| Logistic Regression | 0.97 | 0.97 | 0.96 | 0.97 | 0.99 |
| SVM (RBF) | 0.97 | 0.97 | 0.96 | 0.97 | 0.99 |
| KNN | 0.95 | 0.94 | 0.94 | 0.94 | 0.98 |
| Random Forest | 0.90 | 0.97 | 0.96 | 0.97 | 0.99 |
| XGBoost | 0.96 | 0.96 | 0.95 | 0.96 | 0.99 |
| MLP | 0.94 | 0.94 | 0.94 | 0.94 | 0.98 |

> **Note:** The Random Forest accuracy (0.90) is inconsistent with its macro precision/recall/F1 (0.96–0.97) — flagged for re-verification. See [open items](#known-issues--next-steps).

## Ethical Considerations & Bias Mitigation

- **Data privacy** – anonymized personal data; no identifying information used
- **Transparency** – model capabilities/limitations documented for end users
- **Fairness** – checked for systematic disadvantage across groups
- **Accountability** – decisions traceable and auditable
- **Avoiding harm** – model is not intended for unsupervised high-stakes decisions
- Mitigation steps: careful feature selection, class balance checks, normalization, subgroup evaluation, and preference for interpretable models

## Known Issues / Next Steps

- [ ] Verify Random Forest accuracy vs. precision/recall/F1 discrepancy
- [ ] Confirm engineered features (`health_index`, `academic_stress`, `social_environment`) were not combined with their own source columns in the final feature set, to rule out redundant/leaked signal
- [ ] Document class balance of `stress_level` before and after outlier removal
- [ ] Justify outlier removal criteria in the context of genuinely high-stress students

## Repository Structure

```
.
├── data/                # raw and processed datasets (or link to Kaggle source)
├── notebooks/           # EDA, preprocessing, modeling notebooks
├── results/              # metrics, confusion matrices, results.json
├── report/               # final report (PDF)
└── README.md
```

## How to Run

```bash
git clone https://github.com/<your-username>/student-stress-level-monitoring.git
cd student-stress-level-monitoring
pip install -r requirements.txt
jupyter notebook
```

## References

1. Kaggle – [PCA Beginner's Guide to Dimensionality Reduction](https://www.kaggle.com/code/vipulgandhi/pca-beginner-s-guideto-dimensionality-reduction)
2. Kaggle – [Organization Models](https://www.kaggle.com/models?owner-type=organization)
3. [scikit-learn: Cross-validation and model selection](https://scikitlearn.org/stable/modules/cross_validation.html#cross-validation-and-model-selection)
4. Medium – [A Comprehensive Guide to Multiclass Classification in ML](https://medium.com/@murpanironit/a-comprehensive-guide-to-multiclass-classification-in-machine-learning-c4f893e8161d)
5. GeeksforGeeks – [AI/ML Introduction](https://www.geeksforgeeks.org/artificial-intelligence/aimlintroduction/)
