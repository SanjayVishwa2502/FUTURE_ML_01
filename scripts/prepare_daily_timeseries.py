"""
Step 7 Rebuild: Daily Time Series Artifacts

Aggregates row-level sales to daily totals, builds proper time-series features,
and saves train/test splits in Step 7 format for Step 8 modeling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    marker_files = ['requirements.txt', 'SALES_FORECASTING_TASK_GUIDE.md', 'README.md']
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in marker_files):
            return p
    return start


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    
    data_dir = project_root / 'data'
    reports_dir = project_root / 'reports'
    
    for d in (data_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    print('=' * 80)
    print('DAILY TIME SERIES PREPARATION')
    print('=' * 80)
    print(f'Project root: {project_root}')
    
    # Load featured data
    featured_path = data_dir / 'featured_superstore.csv'
    if not featured_path.exists():
        raise FileNotFoundError(f'Missing {featured_path}')
    
    df = pd.read_csv(featured_path)
    print(f'\nLoaded {len(df):,} rows from {featured_path.name}')
    
    # Parse order date
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    
    # Aggregate to daily sales
    daily = df.groupby('Order Date').agg({
        'Sales': 'sum',
        'Quantity': 'sum',
        'Profit': 'sum',
        'Discount': 'mean',
    }).reset_index()
    
    daily = daily.rename(columns={'Order Date': 'date', 'Sales': 'sales'})
    daily = daily.sort_values('date').reset_index(drop=True)
    
    print(f'\nAggregated to {len(daily):,} daily observations')
    print(f'Date range: {daily["date"].min()} to {daily["date"].max()}')
    print(f'Sales total: ${daily["sales"].sum():,.2f}')
    
    # Fill missing dates with zero sales
    full_date_range = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
    daily_full = pd.DataFrame({'date': full_date_range})
    daily_full = daily_full.merge(daily, on='date', how='left')
    
    for col in ['sales', 'Quantity', 'Profit', 'Discount']:
        daily_full[col] = daily_full[col].fillna(0)
    
    print(f'Filled missing dates: {len(daily_full):,} total days')
    
    # Time features
    daily_full['year'] = daily_full['date'].dt.year
    daily_full['month'] = daily_full['date'].dt.month
    daily_full['day'] = daily_full['date'].dt.day
    daily_full['day_of_week'] = daily_full['date'].dt.dayofweek
    daily_full['quarter'] = daily_full['date'].dt.quarter
    daily_full['week_of_year'] = daily_full['date'].dt.isocalendar().week
    daily_full['is_weekend'] = daily_full['day_of_week'].isin([5, 6]).astype(int)
    daily_full['is_month_start'] = daily_full['date'].dt.is_month_start.astype(int)
    daily_full['is_month_end'] = daily_full['date'].dt.is_month_end.astype(int)
    
    # Lag features (previous day sales)
    for lag in [1, 7, 14, 30]:
        daily_full[f'sales_lag_{lag}'] = daily_full['sales'].shift(lag)
    
    # Rolling statistics (moving averages and std)
    for window in [7, 14, 30]:
        daily_full[f'sales_rolling_mean_{window}'] = daily_full['sales'].rolling(window=window, min_periods=1).mean()
        daily_full[f'sales_rolling_std_{window}'] = daily_full['sales'].rolling(window=window, min_periods=1).std()
    
    # Exponential weighted moving average
    daily_full['sales_ewm_7'] = daily_full['sales'].ewm(span=7, adjust=False).mean()
    
    print('\nFeature engineering complete')
    print(f'Total features: {len(daily_full.columns)}')
    
    # Drop initial rows with NaN from lag features (keep at least 30 days of data for rolling)
    daily_full = daily_full.dropna().reset_index(drop=True)
    print(f'After dropping NaN: {len(daily_full):,} rows')
    
    # Time-based train/test split (80/20)
    split_idx = int(len(daily_full) * 0.8)
    train = daily_full.iloc[:split_idx].copy()
    test = daily_full.iloc[split_idx:].copy()
    
    print(f'\nTrain size: {len(train):,} days ({train["date"].min()} to {train["date"].max()})')
    print(f'Test size: {len(test):,} days ({test["date"].min()} to {test["date"].max()})')
    
    # Separate features, target, and dates
    feature_cols = [c for c in daily_full.columns if c not in ['date', 'sales']]
    
    X_train = train[feature_cols]
    y_train = train[['sales']]
    train_dates = train[['date']]
    
    X_test = test[feature_cols]
    y_test = test[['sales']]
    test_dates = test[['date']]
    
    # Save in Step 7 format
    X_train.to_csv(data_dir / 'X_train.csv', index=False)
    X_test.to_csv(data_dir / 'X_test.csv', index=False)
    y_train.to_csv(data_dir / 'y_train.csv', index=False)
    y_test.to_csv(data_dir / 'y_test.csv', index=False)
    train_dates.to_csv(data_dir / 'train_dates.csv', index=False)
    test_dates.to_csv(data_dir / 'test_dates.csv', index=False)
    
    print(f'\nSaved artifacts to {data_dir}:')
    print('  - X_train.csv, X_test.csv')
    print('  - y_train.csv, y_test.csv')
    print('  - train_dates.csv, test_dates.csv')
    
    # Summary report
    summary = f"""Daily Time Series Preparation Summary
{'=' * 80}

Data Source: {featured_path.name}
Date Range: {daily_full['date'].min()} to {daily_full['date'].max()}
Total Days: {len(daily_full):,}

Train Set:
  - Size: {len(train):,} days
  - Date range: {train['date'].min()} to {train['date'].max()}
  - Sales total: ${train['sales'].sum():,.2f}
  - Sales mean: ${train['sales'].mean():,.2f}
  - Sales std: ${train['sales'].std():,.2f}

Test Set:
  - Size: {len(test):,} days
  - Date range: {test['date'].min()} to {test['date'].max()}
  - Sales total: ${test['sales'].sum():,.2f}
  - Sales mean: ${test['sales'].mean():,.2f}
  - Sales std: ${test['sales'].std():,.2f}

Features ({len(feature_cols)}):
{', '.join(feature_cols)}

Time-based features:
  - year, month, day, day_of_week, quarter, week_of_year
  - is_weekend, is_month_start, is_month_end

Lag features (past sales):
  - sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_30

Rolling statistics:
  - sales_rolling_mean_7/14/30
  - sales_rolling_std_7/14/30
  - sales_ewm_7 (exponential weighted moving average)

Additional features from aggregation:
  - Quantity (total items sold per day)
  - Profit (total profit per day)
  - Discount (average discount per day)

Notes:
- Row-level sales aggregated to daily totals
- Missing dates filled with zero sales
- Features built on proper time-series structure
- Train/test split preserves temporal order (no leakage)
- Ready for Step 8 regression modeling
"""
    
    summary_path = reports_dir / 'daily_timeseries_preparation_summary.txt'
    summary_path.write_text(summary, encoding='utf-8')
    print(f'\nSummary saved to {summary_path}')
    
    # Feature info JSON
    feature_info = {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'train_size': len(train),
        'test_size': len(test),
        'date_range': {
            'start': str(daily_full['date'].min()),
            'end': str(daily_full['date'].max()),
        },
        'train_date_range': {
            'start': str(train['date'].min()),
            'end': str(train['date'].max()),
        },
        'test_date_range': {
            'start': str(test['date'].min()),
            'end': str(test['date'].max()),
        },
    }
    
    feature_info_path = data_dir / 'feature_info.json'
    with open(feature_info_path, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f'Feature info saved to {feature_info_path}')
    print('\n' + '=' * 80)
    print('PREPARATION COMPLETE - Ready for Step 8 modeling')
    print('=' * 80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
