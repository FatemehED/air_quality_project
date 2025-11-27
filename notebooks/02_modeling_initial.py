import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import seaborn as sns


# ----------------- Load Processed Data -----------------
df = pd.read_csv(r'D:\WORK\air_quality_project\data\raw\processed_AQI_US_EPA.csv')

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
target = 'AQI'
features = pollutants  # همه آلاینده‌ها فیچر هستند

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
print("Model\t\tMAE\t\tRMSE\t\tR²")
print("-" * 50)
for model_name, metrics in model_metrics.items():
    print(f"{model_name}\t{metrics['MAE']:.2f}\t\t{metrics['RMSE']:.2f}\t\t{metrics['R2']:.3f}")

# ----------------- Figures -----------------
output_dir = r'D:\WORK\air_quality_project\figures'
os.makedirs(output_dir, exist_ok=True)

# Figure 4: Actual vs Predicted
plt.figure(figsize=(12, 6))
sample_size = min(50, len(y_test))
plt.plot(y_test.values[:sample_size], 'bo-', label='Actual AQI', linewidth=2, markersize=4)
plt.plot(predictions['Linear Regression'][:sample_size], 'r--', label='Linear Regression', linewidth=1.5)
plt.plot(predictions['Random Forest'][:sample_size], 'g-', label='Random Forest', linewidth=1.5)
plt.plot(predictions['XGBoost'][:sample_size], 'y-.', label='XGBoost', linewidth=1.5)
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
bars1 = ax1.bar(models.keys(), mae_values, color=['lightcoral', 'lightgreen', 'orange'])
ax1.set_title('MAE Comparison', fontsize=14)
ax1.set_ylabel('MAE Value')
ax1.tick_params(axis='x', rotation=45)
for bar, value in zip(bars1, mae_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

r2_values = [model_metrics[m]['R2'] for m in models.keys()]
bars2 = ax2.bar(models.keys(), r2_values, color=['lightcoral', 'lightgreen', 'orange'])
ax2.set_title('R² Comparison', fontsize=14)
ax2.set_ylabel('R² Value')
ax2.tick_params(axis='x', rotation=45)
for bar, value in zip(bars2, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Figure 5: Comparison of Model Performance (MAE & R²)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure5_mae_r2_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# Figure 6: Scatter plots
plt.figure(figsize=(16, 5))
for i, (name, y_pred) in enumerate(predictions.items()):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
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
summary_results.to_csv(r'D:\WORK\air_quality_project\data\model_summary.csv', index=False)
print("\n✅ File model_summary.csv is saved.")

with open(r'D:\WORK\air_quality_project\data\results_for_paper.txt', 'w') as f:
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

cv_df.to_csv(r'D:\WORK\air_quality_project\data\cv_model_summary.csv', index=False)
print("\n✅ Cross-Validation results saved to 'cv_model_summary.csv'")



# -----------------ERROR ANALYSIS: Boxplot and Residual Plots -----------------

# Calculate absolute errors for each model
errors_lr = np.abs(y_test - predictions['Linear Regression'])
errors_rf = np.abs(y_test - predictions['Random Forest'])
errors_xgb = np.abs(y_test - predictions['XGBoost'])

# Create a DataFrame for boxplot
error_df = pd.DataFrame({
    'Linear Regression': errors_lr,
    'Random Forest': errors_rf,
    'XGBoost': errors_xgb
})

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=error_df, palette="Set2")
plt.title('Figure 2: Distribution of Absolute Errors by Model', fontsize=16)
plt.ylabel('Absolute Error (AQI units)')
plt.yscale('log')  # Use log scale for better visualization if outliers are severe
plt.grid(True, alpha=0.3)
plt.savefig(r'D:\WORK\air_quality_project\figures\figure2_error_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Residual Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_list = list(predictions.keys())
colors = ['lightcoral', 'lightgreen', 'orange']

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
plt.savefig(r'D:\WORK\air_quality_project\figures\figure3_residual_plots.png', dpi=300, bbox_inches='tight')
plt.show()
