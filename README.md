# Air Quality Prediction Using Machine Learning Models on IoT Sensor Data

This repository contains the code and experiments for the paper on **Air Quality Index (AQI) prediction using machine learning models**.  
We preprocess raw AQI data based on the official US-EPA AQI breakpoints and evaluate several models including Linear Regression, Random Forest, and XGBoost.

📂 Project Structure
air_quality_project/
├── data/
│   ├── processed/
│   │   └── processed_AQI_US_EPA.csv
│   └── raw/
│       └── AQI.csv
├── figures/
│   ├── figure2_error_boxplot.png
│   ├── figure3_residual_plots.png
│   ├── figure4_actual_vs_predicted.png
│   ├── figure5_mae_r2_comparison.png
│   └── figure6_scatter_plots.png
├── notebooks/
│   ├── 01_data_preprocessing.py
│   └── 02_modeling.py
├── results/
│   ├── cv_model_summary.csv
│   ├── model_results.csv
│   ├── model_summary.csv
│   └── results_for_paper.txt
├── .gitignore
├── README.md
└── requirements.txt

Notes:
Put your raw dataset file at data/raw/AQI.csv.
After running preprocessing, the script will produce data/processed/processed_AQI_US_EPA.csv.
Figures 2–6 are saved as PNG files in figures/.
Final CSV / text results are in results/.

⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/USERNAME/air_quality_project.git
cd air_quality_project
pip install -r requirements.txt

🚀 Usage
1.Data Preprocessing
python notebooks/01_data_preprocessing.py

This will generate the preprocessed dataset:
data/processed/processed_AQI_US_EPA.csv

2.Modeling & Evaluation
python notebooks/02_modeling.py

This will:
Train and evaluate ML models
Generate performance figures (Figure 2–6) in figures/
Save summary tables in results/

📊 Models & Metrics
The following models were evaluated:
-Linear Regression
-Random Forest
-XGBoost
-Metrics reported:
-MAE (Mean Absolute Error)
-RMSE (Root Mean Squared Error)
-R² (Coefficient of Determination)

📌 Notes
-Raw datasets are not included in the repository due to size and licensing.
-Make sure to place your raw data in data/raw/ before running preprocessing

✍️ Authors
-**Fatemeh Ensafdoust** (First Author)  
  📧 Email: ensafdoust@gmail.com 
  🔗 LinkedIn: [Fatemeh's LinkedIn](https://www.linkedin.com/in/fatemeh-ensafdoust-9535622a7/)  
- **Dr. S. N. TermehMousavi** (Corresponding Author, Supervisor)  
  📧 Email: s.termehmousavi@iau.ac.ir