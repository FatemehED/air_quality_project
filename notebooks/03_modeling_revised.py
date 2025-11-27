import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------- Load Processed Data -----------------
df = pd.read_csv(r'data/processed/processed_AQI_US_EPA.csv')

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
target = 'AQI'
features = pollutants

print(f"\n✅ Modeling for AQI (calculated using US-EPA formula)")

# ----------------- Train/Test Split -----------------
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- Models -----------------
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
}

# ----------------- Hybrid Model (Stacking) -----------------
stacking_model = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
    ],
    final_estimator=LinearRegression(),
    n_jobs=-1
)

models['Stacking'] = stacking_model

# ----------------- Training -----------------
predictions = {}
model_metrics = {}
summary_results = pd.DataFrame(columns=['Pollutant', 'Model', 'MAE', 'RMSE', 'R2'])

print("\n🚀 Training models ...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    model_metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    summary_results.loc[len(summary_results)] = [target, name, mae, rmse, r2]

# ----------------- Print Results -----------------
print("\n📊 Results for the paper's table:")
print("Model\t\t\tMAE\t\tRMSE\t\tR²")
print("-" * 60)
for model_name, metrics in model_metrics.items():
    print(f"{model_name:<20}\t{metrics['MAE']:.2f}\t\t{metrics['RMSE']:.2f}\t\t{metrics['R2']:.3f}")

# ----------------- Figures -----------------
output_dir = r'figures/'
os.makedirs(output_dir, exist_ok=True)

# Figure 4: Actual vs Predicted
plt.figure(figsize=(12, 6))
sample_size = min(50, len(y_test))
plt.plot(y_test.values[:sample_size], 'bo-', label='Actual AQI', linewidth=2, markersize=4)
plt.plot(predictions['Linear Regression'][:sample_size], 'r--', label='Linear Regression', linewidth=1.5)
plt.plot(predictions['Random Forest'][:sample_size], 'g-', label='Random Forest', linewidth=1.5)
plt.plot(predictions['XGBoost'][:sample_size], 'y-.', label='XGBoost', linewidth=1.5)
plt.plot(predictions['Stacking'][:sample_size], 'm:', label='Stacking Ensemble', linewidth=1.5)
plt.title('Figure 4: Actual vs. Predicted AQI values (sample of test data)', fontsize=14)
plt.xlabel('Sample Index')
plt.ylabel('AQI Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure4_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
plt.show()

# Figure 5: MAE & R² comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

mae_values = [model_metrics[m]['MAE'] for m in models.keys()]
bars1 = ax1.bar(models.keys(), mae_values, color=['lightcoral', 'lightgreen', 'orange', 'violet'])
ax1.set_title('MAE Comparison', fontsize=14)
ax1.set_ylabel('MAE Value')
ax1.tick_params(axis='x', rotation=45)
for bar, value in zip(bars1, mae_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.2f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

r2_values = [model_metrics[m]['R2'] for m in models.keys()]
bars2 = ax2.bar(models.keys(), r2_values, color=['lightcoral', 'lightgreen', 'orange', 'violet'])
ax2.set_title('R² Comparison', fontsize=14)
ax2.set_ylabel('R² Value')
ax2.tick_params(axis='x', rotation=45)
for bar, value in zip(bars2, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('Figure 5: Comparison of Model Performance (MAE & R²)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure5_mae_r2_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# Figure 6: Scatter plots (با همان فرمت قبلی)
plt.figure(figsize=(18, 5))
model_names = list(models.keys())
colors = ['lightcoral', 'lightgreen', 'orange', 'violet']

for i, (name, y_pred) in enumerate(predictions.items()):
    plt.subplot(1, 4, i+1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30, color=colors[i])
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'{name}\nMAE: {model_metrics[name]["MAE"]:.2f}, R²: {model_metrics[name]["R2"]:.3f}')
    plt.grid(True, alpha=0.3)

plt.suptitle('Figure 6: Scatter plots of Predicted vs. Actual AQI values', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure6_scatter_plots.png"), dpi=300, bbox_inches='tight')
plt.show()

# ----------------- Save Results -----------------
summary_results.to_csv(r'data\model_summary.csv', index=False)
print("\n✅ File model_summary.csv is saved.")

with open(r'data\results_for_paper.txt', 'w') as f:
    f.write("Results for table:\n")
    f.write("Model, MAE, RMSE, R²\n")
    for model_name, metrics in model_metrics.items():
        f.write(f"{model_name}, {metrics['MAE']:.4f}, {metrics['RMSE']:.4f}, {metrics['R2']:.4f}\n")
print("✅ Results for the paper were saved in results_for_paper.txt")

# ----------------- Cross-Validation -----------------
scoring = {
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'r2': 'r2'
}

print("\n🔄 Performing 5-Fold Cross-Validation...")
cv_results = {}
for name, model in models.items():
    print(f"  Cross-Validating {name}...")
    scores = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
    cv_results[name] = {
        'MAE': -1 * scores['test_mae'].mean(),
        'RMSE': -1 * scores['test_rmse'].mean(),
        'R2': scores['test_r2'].mean()
    }

cv_df = pd.DataFrame.from_dict(cv_results, orient='index')
cv_df.reset_index(inplace=True)
cv_df.rename(columns={'index': 'Model'}, inplace=True)

print("\n📊 Cross-Validation Results (5-Fold):")
print("----------------------------------------")
print(cv_df.round(3))

cv_df.to_csv(r'data\cv_model_summary.csv', index=False)
print("\n✅ Cross-Validation results saved to 'cv_model_summary.csv'")

# ----------------- ERROR ANALYSIS: Boxplot and Residual Plots -----------------

# Calculate absolute errors for each model
error_df = pd.DataFrame()
for name, y_pred in predictions.items():
    error_df[name] = np.abs(y_test - y_pred)

# Create the boxplot (Figure 2)
plt.figure(figsize=(10, 6))
sns.boxplot(data=error_df, palette="Set2")
plt.title('Figure 2: Distribution of Absolute Errors by Model', fontsize=16)
plt.ylabel('Absolute Error (AQI units)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "figure2_error_boxplot.png"), dpi=300, bbox_inches='tight')
plt.show()

# Create Residual Plots (Figure 3)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
models_list = list(predictions.keys())
colors = ['lightcoral', 'lightgreen', 'orange', 'violet']

for i, (name, y_pred) in enumerate(predictions.items()):
    residuals = y_test - y_pred
    axes[i].scatter(y_pred, residuals, alpha=0.5, color=colors[i], s=20)
    axes[i].axhline(y=0, color='black', linestyle='--')
    axes[i].set_xlabel('Predicted AQI')
    axes[i].set_ylabel('Residuals (Actual - Predicted)')
    axes[i].set_title(f'{name} Residuals')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Figure 3: Residual Plots for Each Model', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure3_residual_plots.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\n🎯 All figures have been updated with Stacking Ensemble model!")
print("📁 Figures saved in:", output_dir)