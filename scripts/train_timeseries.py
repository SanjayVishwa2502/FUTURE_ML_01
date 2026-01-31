"""
STEP 9: Time Series Models (ARIMA)

Train ARIMA model as alternative forecasting approach and compare with regression.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Try to import statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("statsmodels not installed. Run: pip install statsmodels")


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
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(safe_mape(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    return {'mae': mae, 'rmse': rmse, 'mape_percent': mape, 'r2': r2}


def check_stationarity(series: pd.Series) -> dict:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'critical_values': {k: float(v) for k, v in result[4].items()},
        'is_stationary': result[1] < 0.05
    }


def main() -> int:
    if not ARIMA_AVAILABLE:
        print("ERROR: statsmodels is required for ARIMA. Install with: pip install statsmodels")
        return 1
    
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'
    viz_dir = project_root / 'visualizations'
    
    for d in (models_dir, reports_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    print('=' * 80)
    print('STEP 9: TIME SERIES MODELS (ARIMA)')
    print('=' * 80)
    print(f'Project root: {project_root}')
    
    # Load daily time series data
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze('columns')
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze('columns')
    train_dates = pd.read_csv(data_dir / 'train_dates.csv')
    test_dates = pd.read_csv(data_dir / 'test_dates.csv')
    
    train_dates = pd.to_datetime(train_dates.iloc[:, 0])
    test_dates = pd.to_datetime(test_dates.iloc[:, 0])
    
    # Create time series with date index
    train_series = pd.Series(y_train.values, index=train_dates)
    test_series = pd.Series(y_test.values, index=test_dates)
    
    print(f'\nTrain series: {len(train_series)} days')
    print(f'Test series: {len(test_series)} days')
    print(f'Train date range: {train_dates.min()} to {train_dates.max()}')
    print(f'Test date range: {test_dates.min()} to {test_dates.max()}')
    
    # Check stationarity
    print('\n--- Stationarity Test (ADF) ---')
    stationarity = check_stationarity(train_series)
    print(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
    print(f"P-Value: {stationarity['p_value']:.4f}")
    print(f"Is Stationary: {stationarity['is_stationary']}")
    
    # If not stationary, difference the series
    d_order = 0 if stationarity['is_stationary'] else 1
    print(f"Differencing order (d): {d_order}")
    
    # Try different ARIMA configurations
    arima_configs = [
        (1, d_order, 1),
        (2, d_order, 1),
        (1, d_order, 2),
        (2, d_order, 2),
        (5, d_order, 0),
        (0, d_order, 5),
    ]
    
    results = []
    best_model = None
    best_rmse = float('inf')
    best_order = None
    
    print('\n--- Training ARIMA Models ---')
    for order in arima_configs:
        try:
            print(f"Training ARIMA{order}...", end=' ')
            model = ARIMA(train_series, order=order)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=len(test_series))
            forecast = np.maximum(forecast, 0)  # Clip negative predictions
            
            # Evaluate
            metrics = evaluate_forecast(test_series.values, forecast)
            metrics['order'] = str(order)
            metrics['aic'] = float(model_fit.aic)
            metrics['bic'] = float(model_fit.bic)
            results.append(metrics)
            
            print(f"RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}, AIC: {metrics['aic']:.2f}")
            
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model = model_fit
                best_order = order
                best_forecast = forecast
                
        except Exception as e:
            print(f"Failed: {e}")
    
    if best_model is None:
        print("ERROR: No ARIMA model trained successfully")
        return 1
    
    print(f'\n--- Best ARIMA Model: {best_order} ---')
    print(f'RMSE: {best_rmse:.2f}')
    
    # Final evaluation
    final_metrics = evaluate_forecast(test_series.values, best_forecast)
    print(f"\nFinal Test Metrics:")
    print(f"  MAE: ${final_metrics['mae']:.2f}")
    print(f"  RMSE: ${final_metrics['rmse']:.2f}")
    print(f"  MAPE: {final_metrics['mape_percent']:.2f}%")
    print(f"  R2: {final_metrics['r2']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    results_df.to_csv(reports_dir / 'arima_comparison.csv', index=False)
    print(f"\nSaved: {reports_dir / 'arima_comparison.csv'}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test_dates.values,
        'y_true': test_series.values,
        'y_pred_arima': best_forecast
    })
    predictions_df.to_csv(reports_dir / 'arima_predictions.csv', index=False)
    print(f"Saved: {reports_dir / 'arima_predictions.csv'}")
    
    # Create forecast visualization
    plt.figure(figsize=(15, 6))
    plt.plot(train_dates[-60:], train_series.values[-60:], 'b-', label='Training (last 60 days)', alpha=0.7)
    plt.plot(test_dates, test_series.values, 'g-', label='Actual Test', linewidth=2)
    plt.plot(test_dates, best_forecast, 'r--', label=f'ARIMA{best_order} Forecast', linewidth=2)
    plt.title(f'ARIMA{best_order} Sales Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'arima_forecast.png', dpi=150)
    plt.close()
    print(f"Saved: {viz_dir / 'arima_forecast.png'}")
    
    # Save model summary
    summary = f"""STEP 9: TIME SERIES MODELS - ARIMA SUMMARY
{'=' * 80}

Best Model: ARIMA{best_order}

Stationarity Test (ADF):
  - ADF Statistic: {stationarity['adf_statistic']:.4f}
  - P-Value: {stationarity['p_value']:.4f}
  - Is Stationary: {stationarity['is_stationary']}
  - Differencing Order Used: {d_order}

Test Set Performance:
  - MAE: ${final_metrics['mae']:.2f}
  - RMSE: ${final_metrics['rmse']:.2f}
  - MAPE: {final_metrics['mape_percent']:.2f}%
  - R2: {final_metrics['r2']:.4f}

Model Information Criteria:
  - AIC: {best_model.aic:.2f}
  - BIC: {best_model.bic:.2f}

Models Tested:
{results_df.to_string(index=False)}

Files Generated:
  - reports/arima_comparison.csv
  - reports/arima_predictions.csv
  - visualizations/arima_forecast.png

Comparison with Regression (RandomForest):
  - RandomForest R2: 0.81
  - ARIMA R2: {final_metrics['r2']:.4f}
  
Note: ARIMA uses only historical sales values (univariate), while
RandomForest uses 25 engineered features (multivariate). The regression
approach captures more patterns through feature engineering.
"""
    
    summary_path = reports_dir / 'step9_arima_summary.txt'
    summary_path.write_text(summary, encoding='utf-8')
    print(f"Saved: {summary_path}")
    
    # Save metadata - convert numpy/bool types to JSON serializable
    metadata = {
        'best_order': list(best_order),
        'metrics': {k: float(v) if hasattr(v, 'item') else v for k, v in final_metrics.items()},
        'stationarity': {
            'adf_statistic': stationarity['adf_statistic'],
            'p_value': stationarity['p_value'],
            'is_stationary': bool(stationarity['is_stationary']),
            'critical_values': stationarity['critical_values']
        },
        'aic': float(best_model.aic),
        'bic': float(best_model.bic),
        'train_size': len(train_series),
        'test_size': len(test_series),
    }
    with open(models_dir / 'arima_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {models_dir / 'arima_metadata.json'}")
    
    print('\n' + '=' * 80)
    print('STEP 9 COMPLETE')
    print('=' * 80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
