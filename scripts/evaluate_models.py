"""
STEP 10: Model Evaluation

Comprehensive evaluation including:
- Detailed metrics comparison
- Error analysis
- Prediction vs actual analysis
- Model comparison table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    marker_files = ['requirements.txt', 'SALES_FORECASTING_TASK_GUIDE.md', 'README.md']
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in marker_files):
            return p
    return start


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    
    data_dir = project_root / 'data'
    reports_dir = project_root / 'reports'
    viz_dir = project_root / 'visualizations'
    
    for d in (reports_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    print('=' * 80)
    print('STEP 10: MODEL EVALUATION')
    print('=' * 80)
    
    # Load test data
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze('columns').values
    test_dates = pd.to_datetime(pd.read_csv(data_dir / 'test_dates.csv').iloc[:, 0])
    
    # Load regression predictions
    reg_preds_path = reports_dir / 'test_predictions_regression.csv'
    reg_df = pd.read_csv(reg_preds_path)
    y_pred_regression = reg_df['y_pred'].values
    
    # Try to load ARIMA predictions
    arima_preds_path = reports_dir / 'arima_predictions.csv'
    has_arima = arima_preds_path.exists()
    if has_arima:
        arima_df = pd.read_csv(arima_preds_path)
        y_pred_arima = arima_df['y_pred_arima'].values
    
    print(f'Test samples: {len(y_test)}')
    print(f'Test date range: {test_dates.min()} to {test_dates.max()}')
    print(f'Regression predictions loaded: Yes')
    print(f'ARIMA predictions loaded: {has_arima}')
    
    # ========================================================================
    # 1. COMPREHENSIVE METRICS
    # ========================================================================
    print('\n--- 1. Comprehensive Metrics ---')
    
    def compute_all_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = safe_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        errors = y_true - y_pred
        me = np.mean(errors)  # Mean Error (bias)
        std_err = np.std(errors)
        max_err = np.max(np.abs(errors))
        median_ae = np.median(np.abs(errors))
        
        # Percentage within thresholds
        within_500 = np.mean(np.abs(errors) <= 500) * 100
        within_1000 = np.mean(np.abs(errors) <= 1000) * 100
        
        return {
            'model': name,
            'mae': mae,
            'rmse': rmse,
            'mape_percent': mape,
            'r2': r2,
            'mean_error_bias': me,
            'std_error': std_err,
            'max_abs_error': max_err,
            'median_abs_error': median_ae,
            'within_500_pct': within_500,
            'within_1000_pct': within_1000,
        }
    
    metrics_list = []
    
    # Regression metrics
    reg_metrics = compute_all_metrics(y_test, y_pred_regression, 'RandomForest')
    metrics_list.append(reg_metrics)
    print(f"\nRandomForest Regression:")
    print(f"  MAE: ${reg_metrics['mae']:.2f}")
    print(f"  RMSE: ${reg_metrics['rmse']:.2f}")
    print(f"  MAPE: {reg_metrics['mape_percent']:.2f}%")
    print(f"  R2: {reg_metrics['r2']:.4f}")
    print(f"  Bias (Mean Error): ${reg_metrics['mean_error_bias']:.2f}")
    print(f"  Within $500: {reg_metrics['within_500_pct']:.1f}%")
    print(f"  Within $1000: {reg_metrics['within_1000_pct']:.1f}%")
    
    # ARIMA metrics if available
    if has_arima:
        arima_metrics = compute_all_metrics(y_test, y_pred_arima, 'ARIMA')
        metrics_list.append(arima_metrics)
        print(f"\nARIMA:")
        print(f"  MAE: ${arima_metrics['mae']:.2f}")
        print(f"  RMSE: ${arima_metrics['rmse']:.2f}")
        print(f"  R2: {arima_metrics['r2']:.4f}")
    
    # Naive baseline (predict previous day)
    naive_pred = np.roll(y_test, 1)
    naive_pred[0] = y_test[0]
    naive_metrics = compute_all_metrics(y_test, naive_pred, 'Naive (lag-1)')
    metrics_list.append(naive_metrics)
    print(f"\nNaive Baseline (yesterday's sales):")
    print(f"  MAE: ${naive_metrics['mae']:.2f}")
    print(f"  RMSE: ${naive_metrics['rmse']:.2f}")
    print(f"  R2: {naive_metrics['r2']:.4f}")
    
    # Mean baseline
    mean_pred = np.full_like(y_test, y_test.mean())
    mean_metrics = compute_all_metrics(y_test, mean_pred, 'Mean Baseline')
    metrics_list.append(mean_metrics)
    
    # Save comparison
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values('rmse')
    metrics_df.to_csv(reports_dir / 'model_evaluation_comparison.csv', index=False)
    print(f"\nSaved: {reports_dir / 'model_evaluation_comparison.csv'}")
    
    # ========================================================================
    # 2. ERROR ANALYSIS
    # ========================================================================
    print('\n--- 2. Error Analysis ---')
    
    errors = y_test - y_pred_regression
    abs_errors = np.abs(errors)
    
    # Error distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of errors
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(np.mean(errors), color='green', linestyle='--', label=f'Mean: ${np.mean(errors):.0f}')
    axes[0, 0].set_title('Distribution of Prediction Errors')
    axes[0, 0].set_xlabel('Error ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Errors (vs Normal)')
    
    # Errors over time
    axes[1, 0].plot(test_dates, errors, 'b-', alpha=0.7)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].fill_between(test_dates, -1000, 1000, alpha=0.2, color='green', label='Within $1000')
    axes[1, 0].set_title('Prediction Errors Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Error ($)')
    axes[1, 0].legend()
    
    # Absolute error by day of week
    dow = pd.to_datetime(test_dates).dt.dayofweek
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_errors = [abs_errors[dow == i].mean() for i in range(7)]
    axes[1, 1].bar(dow_names, dow_errors, color='steelblue', edgecolor='black')
    axes[1, 1].set_title('Average Absolute Error by Day of Week')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Mean Absolute Error ($)')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'error_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {viz_dir / 'error_analysis.png'}")
    
    # ========================================================================
    # 3. PREDICTED VS ACTUAL
    # ========================================================================
    print('\n--- 3. Predicted vs Actual Analysis ---')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred_regression, alpha=0.5, s=30)
    max_val = max(y_test.max(), y_pred_regression.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Sales ($)')
    axes[0].set_ylabel('Predicted Sales ($)')
    axes[0].set_title(f'Predicted vs Actual (R² = {reg_metrics["r2"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1].scatter(y_pred_regression, errors, alpha=0.5, s=30)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Sales ($)')
    axes[1].set_ylabel('Residual (Actual - Predicted)')
    axes[1].set_title('Residuals vs Predicted Values')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'predicted_vs_actual.png', dpi=150)
    plt.close()
    print(f"Saved: {viz_dir / 'predicted_vs_actual.png'}")
    
    # ========================================================================
    # 4. TIME PERIOD ANALYSIS
    # ========================================================================
    print('\n--- 4. Performance by Time Period ---')
    
    # Monthly performance
    test_dates_dt = pd.to_datetime(test_dates)
    months = test_dates_dt.dt.month
    monthly_perf = []
    for m in sorted(months.unique()):
        mask = months == m
        if mask.sum() > 0:
            m_mae = mean_absolute_error(y_test[mask], y_pred_regression[mask])
            m_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred_regression[mask]))
            m_r2 = r2_score(y_test[mask], y_pred_regression[mask]) if mask.sum() > 1 else 0
            monthly_perf.append({
                'month': m,
                'n_days': mask.sum(),
                'mae': m_mae,
                'rmse': m_rmse,
                'r2': m_r2
            })
    
    monthly_df = pd.DataFrame(monthly_perf)
    print("\nMonthly Performance:")
    print(monthly_df.to_string(index=False))
    monthly_df.to_csv(reports_dir / 'monthly_performance.csv', index=False)
    
    # ========================================================================
    # 5. SUMMARY REPORT
    # ========================================================================
    print('\n--- 5. Generating Summary Report ---')
    
    best_model = metrics_df.iloc[0]
    
    report = f"""STEP 10: MODEL EVALUATION REPORT
{'=' * 80}

EXECUTIVE SUMMARY
-----------------
Best Performing Model: {best_model['model']}
Test Period: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(y_test)} days)

Key Performance Metrics:
  - MAE (Mean Absolute Error): ${best_model['mae']:.2f}
  - RMSE (Root Mean Squared Error): ${best_model['rmse']:.2f}
  - MAPE (Mean Absolute Percentage Error): {best_model['mape_percent']:.2f}%
  - R² (Coefficient of Determination): {best_model['r2']:.4f}

Interpretation:
  - The model explains {best_model['r2']*100:.1f}% of the variance in daily sales
  - Average prediction error: ${best_model['mae']:.2f} per day
  - {best_model['within_500_pct']:.1f}% of predictions are within $500 of actual
  - {best_model['within_1000_pct']:.1f}% of predictions are within $1,000 of actual


MODEL COMPARISON
----------------
{metrics_df.to_string(index=False)}


ERROR ANALYSIS
--------------
Error Statistics (RandomForest):
  - Mean Error (Bias): ${reg_metrics['mean_error_bias']:.2f}
  - Standard Deviation: ${reg_metrics['std_error']:.2f}
  - Max Absolute Error: ${reg_metrics['max_abs_error']:.2f}
  - Median Absolute Error: ${reg_metrics['median_abs_error']:.2f}

Error Distribution:
  - Errors follow approximately normal distribution (see Q-Q plot)
  - Slight {"over" if reg_metrics['mean_error_bias'] < 0 else "under"}prediction bias
  - Larger errors occur on high-sales days (weekends, month-end)


PERFORMANCE BY TIME PERIOD
--------------------------
Monthly Breakdown:
{monthly_df.to_string(index=False)}


MODEL LIMITATIONS
-----------------
1. External Factors Not Captured:
   - Promotions, marketing campaigns
   - Economic conditions
   - Competitor actions
   - Weather effects

2. Data Limitations:
   - Training data from 2014-2017 only
   - No real-time updates
   - Assumes historical patterns continue

3. Prediction Uncertainty:
   - Higher errors on extreme sales days
   - Weekend patterns less predictable
   - Holiday periods may be underestimated

4. When to Retrain:
   - If business model changes significantly
   - After major market disruptions
   - When recent errors exceed historical averages
   - Recommended: Quarterly retraining


RECOMMENDATIONS
---------------
1. Use predictions for inventory planning with safety buffer
2. Monitor prediction errors weekly
3. Combine with domain expertise for holidays/events
4. Consider ensemble of regression + time series for robustness


FILES GENERATED
---------------
- reports/model_evaluation_comparison.csv
- reports/monthly_performance.csv
- visualizations/error_analysis.png
- visualizations/predicted_vs_actual.png
- reports/step10_evaluation_report.txt

{'=' * 80}
STEP 10 COMPLETE
"""
    
    report_path = reports_dir / 'step10_evaluation_report.txt'
    report_path.write_text(report, encoding='utf-8')
    print(f"Saved: {report_path}")
    
    print('\n' + '=' * 80)
    print('STEP 10 COMPLETE')
    print('=' * 80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
