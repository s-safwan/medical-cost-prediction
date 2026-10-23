"""Validated data loading and leakage-safe model definitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parent
NUMERIC = ['age', 'bmi', 'children']
CATEGORICAL = ['gender', 'smoker', 'region']
FEATURES = NUMERIC + CATEGORICAL
CATEGORIES = {'gender': ['female', 'male'], 'smoker': ['no', 'yes'],
              'region': ['northeast', 'northwest', 'southeast', 'southwest']}

def load_data(path=ROOT / 'data' / 'medical.xlsx'):
    df = pd.read_excel(path)
    missing = set(FEATURES + ['charges']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = df[FEATURES + ['charges']].copy()
    initial = len(df)
    for c in NUMERIC + ['charges']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CATEGORICAL:
        df[c] = df[c].str.strip().str.lower()
        if not df[c].dropna().isin(CATEGORIES[c]).all():
            raise ValueError(f'Unexpected category in {c}')
    invalid_target = df['charges'].isna() | ~np.isfinite(df['charges']) | (df['charges'] < 0)
    df = df.loc[~invalid_target].copy()
    for c, low, high in [('age', 18, 100), ('bmi', 10, 80), ('children', 0, 20)]:
        valid = df[c].dropna()
        if not valid.between(low, high).all():
            raise ValueError(f'Out-of-range {c}')
        if c != 'bmi' and not (valid % 1 == 0).all():
            raise ValueError(f'{c} must be integral')
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    audit = {'raw_rows': initial, 'invalid_target_rows_removed': int(invalid_target.sum()),
             'exact_duplicates_removed': duplicates, 'clean_rows': len(df)}
    return df, audit

def make_pipeline(regressor):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('preprocess', ColumnTransformer([('numeric', numeric, NUMERIC),
                                                      ('categorical', categorical, CATEGORICAL)])),
                     ('regressor', regressor)])

def candidates():
    return {
        'Median baseline': make_pipeline(DummyRegressor(strategy='median')),
        'Ridge regression': make_pipeline(Ridge(alpha=1.0)),
        'Random forest': make_pipeline(RandomForestRegressor(n_estimators=250, min_samples_leaf=5, random_state=42, n_jobs=1)),
        'Gradient boosting': make_pipeline(GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, loss='huber', random_state=42))
    }

def validate_record(record):
    for c in FEATURES:
        if c not in record:
            raise ValueError(f'Missing {c}')
    for c, low, high in [('age', 18, 100), ('bmi', 10, 80), ('children', 0, 20)]:
        v = float(record[c])
        if not np.isfinite(v) or not low <= v <= high or (c != 'bmi' and v % 1):
            raise ValueError(f'Invalid {c}')
    for c in CATEGORICAL:
        if record[c] not in CATEGORIES[c]:
            raise ValueError(f'Invalid {c}')
    return pd.DataFrame([record], columns=FEATURES)
