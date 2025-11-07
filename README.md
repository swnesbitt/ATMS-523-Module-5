# 🌦️ Module 5 Project — Radar Parameters and Rainfall Prediction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This project explores how **polarimetric radar parameters** can be used to estimate **rain rate** using various regression models.  
The dataset, `radar_parameters.csv`, contains disdrometer-based radar measurements collected in **Huntsville, Alabama**.

---

## 📊 Dataset Description

**File:** `radar_parameters.csv`  
**Shape:** (18,969 rows × 8 columns)

| Feature | Description |
|----------|-------------|
| Zh (dBZ) | Radar reflectivity factor |
| Zdr (dB) | Differential reflectivity |
| Ldr (dB) | Linear depolarization ratio |
| Kdp (deg km⁻¹) | Specific differential phase |
| Ah (dBZ/km) | Specific attenuation |
| Adr (dB/km) | Differential attenuation |
| R (mm/hr) | Rain rate (target variable) |

---

## 🧠 Project Objectives

1. Split the data into **70% training** and **30% testing** subsets.  
2. Train and evaluate:
   - **Multiple Linear Regression**
   - **Polynomial Regression** (degrees 0–9, with 7-fold cross-validation)
   - **Random Forest Regressor** (with grid search hyperparameter tuning)
3. Compare models using:
   - **R² (Coefficient of Determination)**
   - **RMSE (Root Mean Square Error)**
4. Benchmark all models against the empirical **baseline**:  
   \\( Z = 200R^{1.6} \\)

---

## ✅ Results Summary

| Model | Train R² | Test R² | Train RMSE | Test RMSE |
|-------|-----------|----------|-------------|------------|
| Baseline (Z→R) | 0.28 | 0.36 | 7.14 | 7.19 |
| Linear Regression | 0.988 | 0.989 | 0.92 | 0.94 |
| Polynomial (deg=2) | 0.9996 | 0.9996 | 0.17 | 0.18 |
| Random Forest (best) | 0.997 | 0.988 | 0.43 | 0.99 |

**Conclusion:**  
All machine learning models outperform the baseline, but the **degree-2 polynomial regression** achieves the best overall performance.

---

## 💾 Repository Structure
├── homework/
│   └── radar_parameters.csv
├── Module5_Project_Radar_Regression.ipynb
├── README.md
├── best_linear_model.joblib
├── best_polynomial_model.joblib
├── best_random_forest_model.joblib
└── LICENSE

---

## 📄 License

This project is licensed under the **MIT License**.  
You can view the full license text here:  
[MIT License – Open Source Initiative](https://opensource.org/licenses/MIT)

---

## 👨‍💻 Author

Developed by **Nathan Makowski**  
Module 5 Project — Environmental Data Science and Machine Learning