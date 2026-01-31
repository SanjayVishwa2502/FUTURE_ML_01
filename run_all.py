"""
Run All Steps - Complete Pipeline Automation

This script runs the entire forecasting pipeline from data preparation to model training.
Perfect for reproducing results after cloning the repository.

Usage:
    python run_all.py

Requirements:
    - Sample - Superstore.csv must exist in data/
    - featured_superstore.csv must exist (run notebooks 02-03 first if missing)
"""
import sys
import subprocess
from pathlib import Path


def print_banner(text: str) -> None:
    """Print a formatted section banner."""
    print('\n' + '=' * 80)
    print(text.center(80))
    print('=' * 80 + '\n')


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status."""
    print_banner(f'Running: {description}')
    print(f'Script: {script_path}')
    print(f'Command: python {script_path}\n')
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f'\n✅ {description} completed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'\n❌ {description} failed with exit code {e.returncode}')
        print(f'Error: {e}')
        return False
    except Exception as e:
        print(f'\n❌ Unexpected error running {description}')
        print(f'Error: {e}')
        return False


def check_prerequisites() -> bool:
    """Check if required files exist."""
    print_banner('Checking Prerequisites')
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == 'scripts' else script_dir
    
    required_files = [
        project_root / 'data' / 'Sample - Superstore.csv',
        project_root / 'data' / 'featured_superstore.csv',
        project_root / 'scripts' / 'prepare_daily_timeseries.py',
        project_root / 'scripts' / 'train_regression.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            print(f'✅ Found: {file_path.name}')
        else:
            print(f'❌ Missing: {file_path}')
            all_exist = False
    
    if not all_exist:
        print('\n⚠️  Missing required files!')
        print('Please ensure:')
        print('  1. Sample - Superstore.csv is in data/')
        print('  2. featured_superstore.csv is in data/ (run notebooks 02-03 if missing)')
        return False
    
    print('\n✅ All prerequisites met')
    return True


def main() -> int:
    """Run the complete pipeline."""
    print_banner('SALES FORECASTING - COMPLETE PIPELINE')
    print('This will run all steps from data preparation to model training.')
    print('Estimated time: 3-5 minutes\n')
    
    # Determine project root
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == 'scripts':
        project_root = script_dir.parent
        scripts_dir = script_dir
    else:
        project_root = script_dir
        scripts_dir = project_root / 'scripts'
    
    print(f'Project root: {project_root}')
    print(f'Scripts directory: {scripts_dir}\n')
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Pipeline steps
    steps = [
        (
            scripts_dir / 'prepare_daily_timeseries.py',
            'Step 7: Daily Time Series Preparation'
        ),
        (
            scripts_dir / 'train_regression.py',
            'Step 8: Regression Model Training'
        ),
        (
            scripts_dir / 'train_timeseries.py',
            'Step 9: ARIMA Time Series Models'
        ),
        (
            scripts_dir / 'evaluate_models.py',
            'Step 10: Comprehensive Model Evaluation'
        ),
        (
            scripts_dir / 'create_visualizations.py',
            'Step 11: Visualization & Business Insights'
        ),
    ]
    
    # Run each step
    for script_path, description in steps:
        if not script_path.exists():
            print(f'\n❌ Script not found: {script_path}')
            return 1
        
        success = run_script(script_path, description)
        if not success:
            print(f'\n❌ Pipeline failed at: {description}')
            return 1
    
    # Success summary
    print_banner('PIPELINE COMPLETE')
    print('✅ All steps completed successfully!\n')
    print('Generated artifacts:')
    print('  📁 data/X_train.csv, X_test.csv, y_train.csv, y_test.csv')
    print('  📁 data/train_dates.csv, test_dates.csv, feature_info.json')
    print('  📁 models/best_regression_model.pkl')
    print('  📁 models/arima_metadata.json')
    print('  📁 reports/model_comparison_regression.csv')
    print('  📁 reports/arima_predictions.csv')
    print('  📁 reports/model_evaluation_comparison.csv')
    print('  📁 reports/BUSINESS_REPORT.md')
    print('  📁 visualizations/forecast_chart.png')
    print('  📁 visualizations/business_dashboard.png')
    print('  📁 visualizations/trend_analysis.png')
    print('\nNext steps:')
    print('  - Review reports/BUSINESS_REPORT.md for executive summary')
    print('  - Review reports/PRESENTATION_GUIDE.md for demo preparation')
    print('  - Review reports/LINKEDIN_POST_DRAFT.md for social sharing')
    print('  - Open visualizations/ for charts and dashboards')
    print('\n' + '=' * 80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
