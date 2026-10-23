"""Reproduce all published metrics with: python train.py."""
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from model import ROOT, FEATURES, load_data, candidates

def main():
    df, audit = load_data()
    # Exact duplicates are removed before splitting to avoid identical rows crossing partitions.
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.smoker)
    X, y = train[FEATURES], train.charges
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = candidates()
    rows = []
    # No test data participates in model choice, preprocessing, or CV.
    for name, model in models.items():
        errors = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        rows.append({'model': name, 'cv_mae': float(errors.mean()), 'cv_mae_std': float(errors.std())})
    ranking = pd.DataFrame(rows).sort_values('cv_mae')
    winner = ranking.iloc[0]['model']
    model = models[winner].fit(X, y)
    pred = model.predict(test[FEATURES])
    metrics = {'selected_model': winner, 'test_mae': mean_absolute_error(test.charges, pred),
               'test_rmse': float(np.sqrt(mean_squared_error(test.charges, pred))),
               'test_r2': r2_score(test.charges, pred)}
    baseline = models['Median baseline'].fit(X, y)
    linear = models['Ridge regression'].fit(X, y)
    metrics['baseline_test_mae'] = mean_absolute_error(test.charges, baseline.predict(test[FEATURES]))
    metrics['ridge_test_mae'] = mean_absolute_error(test.charges, linear.predict(test[FEATURES]))
    metrics['mae_reduction_vs_baseline_percent'] = 100 * (1 - metrics['test_mae']/metrics['baseline_test_mae'])
    metrics['mae_reduction_vs_ridge_percent'] = 100 * (1 - metrics['test_mae']/metrics['ridge_test_mae'])
    reports = ROOT / 'reports'; reports.mkdir(exist_ok=True)
    ranking.to_csv(reports / 'cross_validation.csv', index=False)
    diagnostics = test.reset_index(drop=True).copy()
    diagnostics['predicted_charges'] = pred
    diagnostics['absolute_error'] = abs(diagnostics.charges - pred)
    # Publish aggregate diagnostics only, not individual test records.
    groups = []
    for feature in ['smoker', 'gender', 'region']:
        for group, part in diagnostics.groupby(feature):
            groups.append({'feature': feature, 'group': group, 'n': len(part), 'mae': part.absolute_error.mean()})
    pd.DataFrame(groups).to_csv(reports / 'subgroup_errors.csv', index=False)
    importance = permutation_importance(model, test[FEATURES], test.charges, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
    pd.DataFrame({'feature': FEATURES, 'importance_mae': importance.importances_mean,
                  'importance_std': importance.importances_std}).sort_values('importance_mae', ascending=False).to_csv(reports / 'feature_importance.csv', index=False)
    metadata = {'audit': audit, 'train_rows': len(train), 'test_rows': len(test),
                'split': '80/20, random_state=42, stratified by smoker', 'cv': '5-fold shuffled KFold on training partition only',
                'data_sha256': hashlib.sha256((ROOT/'data/medical.xlsx').read_bytes()).hexdigest(),
                'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
                'metrics': metrics}
    (reports / 'metrics.json').write_text(json.dumps(metadata, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    axes[0].scatter(test.charges, pred, s=18, alpha=.6, color='#2563eb')
    bound = max(test.charges.max(), pred.max())
    axes[0].plot([0,bound],[0,bound], '--', color='#64748b')
    axes[0].set(xlabel='Observed annual charges (dataset units)', ylabel='Predicted annual charges', title='Held-out predictions')
    axes[1].scatter(pred, test.charges.to_numpy()-pred, s=18, alpha=.6, color='#0d9488')
    axes[1].axhline(0, linestyle='--', color='#64748b')
    axes[1].set(xlabel='Predicted annual charges', ylabel='Observed - predicted', title='Residuals reveal remaining error')
    fig.savefig(reports / 'diagnostics.png', dpi=160); plt.close(fig)
    # The demo uses this exact evaluated training-only fit. No silent all-data refit.
    joblib.dump({'model': model, 'metadata': metadata}, ROOT / 'model.joblib')
    print(ranking.to_string(index=False)); print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
