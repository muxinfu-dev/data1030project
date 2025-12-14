# Predicting Individual Productivity from Digital Behavior

This repository contains the full codebase, data processing pipeline, modeling workflow, and final report for a machine learning project that predicts **actual productivity scores** from digital behavior and lifestyle features. The project emphasizes **robust missing-value handling**, **model comparison**, and **interpretability** using global and local explanation methods.

---

## Project Overview

As digital tools become increasingly embedded in daily life, understanding how digital habits relate to real productivity has become an important research question. This project frames productivity prediction as a **regression problem**, using behavioral, lifestyle, and psychological variables such as:

- Daily social media usage  
- Notification frequency  
- Sleep duration and screen time before bed  
- Work hours and break frequency  
- Job satisfaction and perceived productivity  

The primary objectives of this project are to:

1. Compare multiple **numeric missing-value strategies** under a controlled modeling setup.
2. Evaluate and select regression models using **cross-validated RMSE and R²**.
3. Interpret model behavior using **global feature importance** and **local SHAP explanations**.

---

## Repository Structure

The repository follows the required directory structure:
```text
.
├── data/        # Raw and preprocessed datasets
├── figures/     # All generated figures and visualizations
├── results/     # Saved predictions, evaluation outputs, model artifacts
├── report/      # Final PDF project report
├── src/         # Source code (Python scripts and notebooks)
├── .gitignore
├── data1030.yml # Conda environment file for reproducibility
├── LICENSE
└── README.md
```


- All raw and processed data files are stored in `/data`
- All generated plots and visualizations are stored in `/figures`
- Model outputs and saved results are stored in `/results`
- The final written report (PDF) is stored in `/report`
- All source code and notebooks are stored in `/src`

---

## Dataset

The dataset used in this project is publicly available on Kaggle:

**Social Media vs Productivity**  
https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity

The dataset consists of synthetic but realistic survey-style user records designed to simulate plausible relationships between digital behavior and productivity. All raw and processed versions of the dataset used in this project are included in the `data/` directory.

---

## Modeling Summary

- **Baseline model**: Mean predictor  
- **Models evaluated**:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - RBF-kernel Support Vector Regression (SVR)
  - XGBoost (with native missing-value handling)

- **Evaluation metrics**:
  - Root Mean Squared Error (RMSE)
  - R²

- **Final model**:
  - XGBoost with native missing-value handling
  - Early stopping with fixed `n_estimators = 10000`
  - Best performance on both validation and held-out test sets

- **Interpretability techniques**:
  - XGBoost global feature importance (gain, weight, cover)
  - SHAP global importance (mean absolute SHAP values)
  - SHAP force plots for individual-level explanations

---

## Reproducibility

### Python Version

- Python **3.11**

### Environment Setup (Recommended)

To reproduce the results locally, create the Conda environment using the provided YAML file:

```bash
conda env create -f data1030.yml
conda activate data1030
```

This environment includes all required packages and exact versions used during development.

**Key Packages**
- **numpy**
- **pandas**
- **scikit-learn**
- **xgboost**
- **shap**
- **matplotlib**
- **seaborn**

## Results

The baseline model, which predicts the mean training productivity score for all individuals, performs poorly with a test RMSE of approximately **1.90** and essentially no explanatory power (**$R^2\approx$ 0**). In contrast, the final **XGBoost model with native missing-value handling** achieves a test RMSE of **0.53** and an $R^2$ of **0.92**, representing an improvement of more than **200 standard deviations** relative to the baseline's variability.

Across all evaluated models—including Linear Regression, Ridge, Random Forest, and RBF-kernel SVR—XGBoost consistently achieves the lowest cross-validated RMSE and highest $R^2$, while maintaining a small train–validation gap. This indicates strong generalization rather than overfitting.

Global feature importance analyses (XGBoost gain, weight, cover, and SHAP) show that **perceived productivity** and **job satisfaction** are the most influential predictors of actual productivity. Behavioral variables such as sleep duration, screen time before bed, work hours, and daily social media usage contribute smaller but consistent refinements to predictions. Job-specific interaction features (e.g., social media time by occupation) reveal meaningful heterogeneity across job types.

Local SHAP force plots further demonstrate that individual predictions arise from different combinations of positive and negative influences. While psychological self-assessments set the baseline level of predicted productivity, lifestyle and digital habits fine-tune outcomes on a per-individual basis.

All quantitative results, figures, and detailed analyses are provided in the final report located in the `/report` directory.


## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software for academic or commercial purposes, provided that the original copyright notice and license text are included in all copies or substantial portions of the software.

See the `LICENSE` file in this repository for full details.



