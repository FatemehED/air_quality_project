# Air Quality Prediction Using Machine Learning Models on IoT Sensor Data

This repository contains the code and experiments for the paper **"Air Quality Prediction Using Machine Learning Models on IoT Sensor Data"** (Accepted at ICSPIS 2025).

The study compares machine learning models for real-time AQI prediction using IoT sensor data from India, with enhanced analysis addressing reviewer comments.

## 📂 Project Structure (Revised)
air_quality_project/
├── 01_initial_analysis/ # Original analysis for first submission
│ ├── 01_data_preprocessing.py
│ └── 02_modeling.py
│ └── 02_new_modeling.py # ✅Contains Stacking Ensemble & cross-validation
├── data/
│ ├── processed/
│ │ └── processed_AQI_US_EPA.csv
│ └── raw/
│ └── AQI.csv
├── figures/ # All generated graphs (Figures 2-6)
│ ├── figure2_error_boxplot.png
│ ├── figure3_residual_plots.png
│ ├── figure4_actual_vs_predicted.png
│ ├── figure5_mae_r2_comparison.png
│ └── figure6_scatter_plots.png
├── results/ # Performance metrics and summaries
│ ├── cv_model_summary.csv # ✅ 5-fold cross-validation results
│ ├── model_results.csv
│ ├── model_summary.csv
│ └── results_for_paper.txt



## 🚀 Quick Start (Revised Version)

### 1. Data Preprocessing (First)
```bash
python 01_initial_analysis/01_data_preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
Generates processed dataset at data/processed/processed_AQI_US_EPA.csv
2. Main Analysis with Enhanced Models
ash
python 02_revised_analysis/02_new_modeling.py
This enhanced script will:
```
✅ Train and evaluate 4 models (Linear Regression, Random Forest, XGBoost, Stacking Ensemble)

✅ Perform 5-fold cross-validation for robust evaluation

✅ Generate all performance figures (Figures 2-6) in figures/

✅ Save comprehensive results in results/ including cross-validation metrics

📊 Enhanced Models & Evaluation
Models Compared:
Linear Regression (Baseline)

Random Forest

XGBoost

Stacking Ensemble ✅ New hybrid model

Evaluation Metrics:
MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

5-Fold Cross-Validation ✅ Enhanced robustness check

🔬 Key Enhancements in Revised Version
Stacking Ensemble Model: Hybrid approach combining multiple base models

Comprehensive Cross-Validation: 5-fold CV for reliable performance estimation

Enhanced Error Analysis: Detailed residual plots and error distributions

Comparative Visualization: Updated figures including all four models

📌 Important Notes
Raw datasets are not included due to size and licensing

Place your raw data in data/raw/AQI.csv before running preprocessing

All file paths are configured for the project structure above

✍️ Authors
Fatemeh Ensafdoust (First Author)
📧 Email: ensafdoust@gmail.com
🔗 LinkedIn: Fatemeh's LinkedIn

Dr. S. N. TermehMousavi (Corresponding Author, Supervisor)
📧 Email: s.termehmousavi@iau.ac.ir

