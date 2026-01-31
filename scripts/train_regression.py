from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE_DEFAULT = 42


def find_project_root(start: Path) -> Path:
    start = start.resolve()

    # Prefer a file that only exists at the repo root
    marker_files = [
        'requirements.txt',
        'SALES_FORECASTING_TASK_GUIDE.md',
        'README.md',
        '.gitignore',
    ]

    for p in [start, *start.parents]:
        if any((p / m).exists() for m in marker_files) and (p / 'data').exists():
            return p

    # Fallback: nearest ancestor with expected data artifacts
    required = [
        Path('data') / 'X_train.csv',
        Path('data') / 'X_test.csv',
        Path('data') / 'y_train.csv',
        Path('data') / 'y_test.csv',
        Path('data') / 'train_dates.csv',
        Path('data') / 'test_dates.csv',
    ]
    for p in [start, *start.parents]:
        if all((p / r).exists() for r in required):
            return p

    return start


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(safe_mape(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        'mae': mae,
        'rmse': rmse,
        'mape_percent': mape,
        'r2': r2,
    }


def clip_negative(y_pred: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(y_pred, dtype=float), 0.0)


def infer_feature_types(df_features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocess(X_train: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_feature_types(X_train)

    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def assert_schema_compatible(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    missing_in_test = [c for c in train_df.columns if c not in test_df.columns]
    extra_in_test = [c for c in test_df.columns if c not in train_df.columns]
    if missing_in_test or extra_in_test:
        raise ValueError(
            'Schema mismatch between train and test. '
            f'missing_in_test={missing_in_test}, extra_in_test={extra_in_test}'
        )


def maybe_wrap_target_transform(estimator: object, use_log_target: bool) -> object:
    if not use_log_target:
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


@dataclass
class ModelResult:
    name: str
    metrics: dict
    train_seconds: float
    estimator: object
    notes: str = ''


def fit_evaluate_holdout(
    name: str,
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    clip_to_zero: bool,
    notes: str = '',
) -> ModelResult:
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_seconds = float(time.perf_counter() - start)

    y_pred = estimator.predict(X_test)
    if clip_to_zero:
        y_pred = clip_negative(y_pred)

    metrics = evaluate_regression(y_test, y_pred)
    return ModelResult(
        name=name,
        metrics=metrics,
        train_seconds=train_seconds,
        estimator=estimator,
        notes=notes,
    )


def make_pipeline(
    X_train: pd.DataFrame,
    model: object,
    scale_numeric: bool,
    use_log_target: bool,
) -> object:
    preprocess = make_preprocess(X_train=X_train, scale_numeric=scale_numeric)
    pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
    return maybe_wrap_target_transform(pipe, use_log_target=use_log_target)


def load_step7_artifacts(project_root: Path) -> dict:
    data_dir = project_root / 'data'

    paths = {
        'X_train': data_dir / 'X_train.csv',
        'X_test': data_dir / 'X_test.csv',
        'y_train': data_dir / 'y_train.csv',
        'y_test': data_dir / 'y_test.csv',
        'train_dates': data_dir / 'train_dates.csv',
        'test_dates': data_dir / 'test_dates.csv',
    }

    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing required Step 7 artifacts: ' + ', '.join(str(p) for p in missing)
        )

    X_train = pd.read_csv(paths['X_train'])
    X_test = pd.read_csv(paths['X_test'])

    y_train = pd.read_csv(paths['y_train']).squeeze('columns')
    y_test = pd.read_csv(paths['y_test']).squeeze('columns')

    train_dates_df = pd.read_csv(paths['train_dates'])
    test_dates_df = pd.read_csv(paths['test_dates'])

    train_date_col = train_dates_df.columns[0]
    test_date_col = test_dates_df.columns[0]

    train_dates = pd.to_datetime(train_dates_df[train_date_col])
    test_dates = pd.to_datetime(test_dates_df[test_date_col])

    # Ensure time order for TimeSeriesSplit and consistent reporting
    train_order = np.argsort(train_dates.values)
    test_order = np.argsort(test_dates.values)

    X_train = X_train.iloc[train_order].reset_index(drop=True)
    y_train = y_train.iloc[train_order].reset_index(drop=True)
    train_dates = train_dates.iloc[train_order].reset_index(drop=True)

    X_test = X_test.iloc[test_order].reset_index(drop=True)
    y_test = y_test.iloc[test_order].reset_index(drop=True)
    test_dates = test_dates.iloc[test_order].reset_index(drop=True)

    assert_schema_compatible(X_train, X_test)
    if not (train_dates.max() < test_dates.min()):
        raise ValueError('Time split violation: train overlaps test')

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_dates': train_dates,
        'test_dates': test_dates,
    }


def build_candidate_models(random_state: int) -> list[tuple[str, object, bool, str]]:
    # Returns list of (name, model, scale_numeric, notes)
    return [
        ('LinearRegression', LinearRegression(), True, 'baseline'),
        ('Ridge', Ridge(random_state=random_state), True, 'regularized linear'),
        ('ElasticNet', ElasticNet(random_state=random_state, max_iter=5000), True, 'regularized linear'),
        ('DecisionTree', DecisionTreeRegressor(random_state=random_state), False, 'tree'),
        ('RandomForest', RandomForestRegressor(random_state=random_state, n_estimators=300, n_jobs=-1), False, 'ensemble tree'),
        ('GradientBoosting', GradientBoostingRegressor(random_state=random_state), False, 'boosting'),
        ('HistGradientBoosting', HistGradientBoostingRegressor(random_state=random_state), False, 'boosting'),
    ]


def tune_best_candidate(
    project_root: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_log_target: bool,
    clip_to_zero: bool,
    tscv_splits: int,
    random_state: int,
    n_iter: int,
) -> tuple[str, object, dict]:
    # Two efficient candidates to tune: RandomForest and HistGradientBoosting
    tscv = TimeSeriesSplit(n_splits=tscv_splits)

    rf = make_pipeline(
        X_train=X_train,
        model=RandomForestRegressor(random_state=random_state, n_jobs=-1),
        scale_numeric=False,
        use_log_target=use_log_target,
    )

    hgb = make_pipeline(
        X_train=X_train,
        model=HistGradientBoostingRegressor(random_state=random_state),
        scale_numeric=False,
        use_log_target=use_log_target,
    )

    # If we use TransformedTargetRegressor, the underlying Pipeline params are nested
    # under the "regressor" attribute.
    prefix = 'regressor__' if use_log_target else ''

    # Note: parameters must be prefixed with the pipeline step names
    rf_param_dist = {
        f'{prefix}model__n_estimators': [200, 400, 700],
        f'{prefix}model__max_depth': [None, 10, 20, 30],
        f'{prefix}model__min_samples_split': [2, 5, 10],
        f'{prefix}model__min_samples_leaf': [1, 2, 4],
        f'{prefix}model__max_features': ['sqrt', 'log2', None],
    }

    hgb_param_dist = {
        f'{prefix}model__learning_rate': [0.03, 0.05, 0.1],
        f'{prefix}model__max_depth': [None, 6, 10],
        f'{prefix}model__max_leaf_nodes': [31, 63, 127],
        f'{prefix}model__min_samples_leaf': [10, 20, 50],
        f'{prefix}model__l2_regularization': [0.0, 0.1, 1.0],
    }

    searches: list[tuple[str, RandomizedSearchCV]] = []

    searches.append(
        (
            'RandomForest_tuned',
            RandomizedSearchCV(
                estimator=rf,
                param_distributions=rf_param_dist,
                n_iter=n_iter,
                scoring='neg_mean_absolute_error',
                cv=tscv,
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            ),
        )
    )

    searches.append(
        (
            'HistGradientBoosting_tuned',
            RandomizedSearchCV(
                estimator=hgb,
                param_distributions=hgb_param_dist,
                n_iter=n_iter,
                scoring='neg_mean_absolute_error',
                cv=tscv,
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            ),
        )
    )

    best_name = ''
    best_est = None
    best_score = -np.inf
    best_params: dict = {}

    for name, search in searches:
        print('Tuning:', name)
        search.fit(X_train, y_train)
        score = float(search.best_score_)
        print('Best CV score (neg MAE):', score)
        if score > best_score:
            best_score = score
            best_name = name
            best_est = search.best_estimator_
            best_params = dict(search.best_params_)

        # Persist each search best params immediately
        out = {
            'name': name,
            'best_score_neg_mae': score,
            'best_params': dict(search.best_params_),
            'tscv_splits': tscv_splits,
        }
        (project_root / 'reports').mkdir(parents=True, exist_ok=True)
        (project_root / 'reports' / f'{name}_best_params.json').write_text(
            json.dumps(out, indent=2), encoding='utf-8'
        )

    if best_est is None:
        raise RuntimeError('Hyperparameter search did not produce a best estimator')

    meta = {
        'best_name': best_name,
        'best_score_neg_mae': best_score,
        'best_params': best_params,
    }
    return best_name, best_est, meta


def main() -> int:
    parser = argparse.ArgumentParser(description='Step 8: Train and tune regression models, save best pipeline.')
    parser.add_argument('--use-log-target', action='store_true', default=True)
    parser.add_argument('--no-log-target', dest='use_log_target', action='store_false')
    parser.add_argument('--clip-negative', action='store_true', default=True)
    parser.add_argument('--no-clip-negative', dest='clip_negative', action='store_false')
    parser.add_argument('--tscv-splits', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE_DEFAULT)
    args = parser.parse_args()

    np.random.seed(args.random_state)

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print('Project root:', project_root)
    print('Data dir:', data_dir)

    artifacts = load_step7_artifacts(project_root)
    X_train = artifacts['X_train']
    X_test = artifacts['X_test']
    y_train = artifacts['y_train']
    y_test = artifacts['y_test']
    train_dates = artifacts['train_dates']
    test_dates = artifacts['test_dates']

    print('Train rows:', len(X_train), 'Test rows:', len(X_test))
    print('Train date range:', train_dates.min(), 'to', train_dates.max())
    print('Test date range:', test_dates.min(), 'to', test_dates.max())

    results: list[ModelResult] = []

    # Quick baseline and strong defaults
    for name, model, scale_numeric, notes in build_candidate_models(args.random_state):
        est = make_pipeline(
            X_train=X_train,
            model=model,
            scale_numeric=scale_numeric,
            use_log_target=args.use_log_target,
        )
        res = fit_evaluate_holdout(
            name=name,
            estimator=est,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            clip_to_zero=args.clip_negative,
            notes=notes,
        )
        results.append(res)
        print('Model:', name, 'Metrics:', res.metrics, 'Train_s:', round(res.train_seconds, 3))

    # Tune best candidates with time-series CV
    tuned_name, tuned_estimator, tuned_meta = tune_best_candidate(
        project_root=project_root,
        X_train=X_train,
        y_train=y_train,
        use_log_target=args.use_log_target,
        clip_to_zero=args.clip_negative,
        tscv_splits=args.tscv_splits,
        random_state=args.random_state,
        n_iter=args.n_iter,
    )

    tuned_result = fit_evaluate_holdout(
        name=tuned_name,
        estimator=tuned_estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        clip_to_zero=args.clip_negative,
        notes='tuned with TimeSeriesSplit',
    )
    results.append(tuned_result)

    # Build comparison table
    rows = []
    for r in results:
        row = {
            'model': r.name,
            'mae': r.metrics['mae'],
            'rmse': r.metrics['rmse'],
            'mape_percent': r.metrics['mape_percent'],
            'r2': r.metrics['r2'],
            'train_seconds': r.train_seconds,
            'notes': r.notes,
        }
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(by=['rmse', 'mae'], ascending=[True, True]).reset_index(drop=True)

    comparison_path = reports_dir / 'model_comparison_regression.csv'
    comparison.to_csv(comparison_path, index=False)
    print('Wrote:', comparison_path)

    best_row = comparison.iloc[0]
    best_model_name = str(best_row['model'])

    best_estimator = next(r.estimator for r in results if r.name == best_model_name)

    # Final fit on full training data (already fitted in holdout evaluation, but refit for clarity)
    best_estimator.fit(X_train, y_train)

    # Save model
    model_path = models_dir / 'best_regression_model.pkl'
    joblib.dump(best_estimator, model_path)
    print('Saved best model to:', model_path)

    # Save predictions for dashboards
    y_pred_test = best_estimator.predict(X_test)
    if args.clip_negative:
        y_pred_test = clip_negative(y_pred_test)

    pred_df = pd.DataFrame(
        {
            'date': test_dates,
            'y_true': np.asarray(y_test, dtype=float),
            'y_pred': np.asarray(y_pred_test, dtype=float),
        }
    )
    pred_path = reports_dir / 'test_predictions_regression.csv'
    pred_df.to_csv(pred_path, index=False)
    print('Wrote:', pred_path)

    # Save metadata
    metadata = {
        'project_root': str(project_root),
        'best_model_name': best_model_name,
        'best_metrics': evaluate_regression(y_test, y_pred_test),
        'use_log_target': bool(args.use_log_target),
        'clip_negative': bool(args.clip_negative),
        'tscv_splits': int(args.tscv_splits),
        'n_iter': int(args.n_iter),
        'random_state': int(args.random_state),
        'train_date_min': str(train_dates.min()),
        'train_date_max': str(train_dates.max()),
        'test_date_min': str(test_dates.min()),
        'test_date_max': str(test_dates.max()),
        'tuning': tuned_meta,
    }

    meta_path = models_dir / 'best_regression_model_metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Wrote:', meta_path)

    # Write short summary (ASCII-only)
    lines = []
    lines.append('STEP 8 MODEL DEVELOPMENT SUMMARY (REGRESSION)')
    lines.append('')
    lines.append('Best model: ' + best_model_name)
    lines.append('Metrics (holdout test):')
    for k, v in metadata['best_metrics'].items():
        lines.append(f'  - {k}: {v:.6f}')
    lines.append('')
    lines.append('Artifacts:')
    lines.append('  - model: models/best_regression_model.pkl')
    lines.append('  - metadata: models/best_regression_model_metadata.json')
    lines.append('  - comparison: reports/model_comparison_regression.csv')
    lines.append('  - test preds: reports/test_predictions_regression.csv')
    summary_path = reports_dir / 'model_development_regression_summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('Wrote:', summary_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
