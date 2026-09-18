"""Train the classifier (is_claim) and regressor (new_claim_value) pipelines.

Each model is a single sklearn Pipeline of [preprocessing, estimator]. This is the
structural fix for the leakage bug in the original notebook: .fit() is only ever
called on the training split, and .predict() on the test split calls .transform()
under the hood (never .fit_transform()) because the preprocessing step is already
fitted.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from xgboost import XGBClassifier

from data import load_and_clean
from features import build_preprocess_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Default hyperparameters -- these mirror the tuned values found in the original
# notebook's grid search. Running src/tune.py will search for (and can overwrite)
# better values, including L1/L2 regularization strength.
DEFAULT_CLF_PARAMS = dict(
    n_estimators=290, max_depth=7, learning_rate=0.04, subsample=0.5,
    colsample_bytree=0.55, gamma=0.1, min_child_weight=18,
    reg_alpha=1, reg_lambda=1, random_state=RANDOM_STATE, eval_metric='error',
)
DEFAULT_REG_PARAMS = dict(
    penalty='l2', alpha=0.004, learning_rate='invscaling', eta0=0.003,
    max_iter=200, tol=1e-5, random_state=RANDOM_STATE,
)


def make_claim_value_bins(df: pd.DataFrame) -> pd.Series:
    bins = [0.0, 5000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000, np.inf]
    labels = np.arange(1, 12)
    return pd.cut(df['new_claim_value'], bins=bins, labels=labels, include_lowest=True)


def build_classification_split(df: pd.DataFrame):
    df = df.copy()
    df['claim_value_cat'] = make_claim_value_bins(df)
    X = df.drop(columns=['new_claim_value', 'is_claim'])
    y = df['is_claim']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=X['claim_value_cat']
    )
    X_train = X_train.drop(columns=['claim_value_cat'])
    X_test = X_test.drop(columns=['claim_value_cat'])
    return X_train, X_test, y_train, y_test


def build_regression_split(df: pd.DataFrame):
    claim_df = df[df['new_claim_value'] > 0].copy()
    X = claim_df.drop(columns=['new_claim_value', 'is_claim'])
    y = claim_df['new_claim_value']
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_classifier(X_train, y_train, params: dict = None) -> Pipeline:
    params = params or DEFAULT_CLF_PARAMS
    pipe = Pipeline([
        ('preprocess', build_preprocess_pipeline()),
        ('model', XGBClassifier(**params)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_regressor(X_train, y_train, params: dict = None) -> Pipeline:
    params = params or DEFAULT_REG_PARAMS
    pipe = Pipeline([
        ('preprocess', build_preprocess_pipeline()),
        ('model', SGDRegressor(**params)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def main(data_path: str, models_dir: str, clf_params_path: str = None, reg_params_path: str = None):
    df = load_and_clean(data_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    clf_params = json.loads(Path(clf_params_path).read_text()) if clf_params_path else None
    reg_params = json.loads(Path(reg_params_path).read_text()) if reg_params_path else None

    X_train, X_test, y_train, y_test = build_classification_split(df)
    clf_pipeline = train_classifier(X_train, y_train, clf_params)
    joblib.dump(clf_pipeline, models_dir / 'classifier_pipeline.joblib')
    print(f'Saved classifier pipeline to {models_dir / "classifier_pipeline.joblib"}')

    Xr_train, Xr_test, yr_train, yr_test = build_regression_split(df)
    reg_pipeline = train_regressor(Xr_train, yr_train, reg_params)
    joblib.dump(reg_pipeline, models_dir / 'regressor_pipeline.joblib')
    print(f'Saved regressor pipeline to {models_dir / "regressor_pipeline.joblib"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/car_insurance_claim.csv')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--clf-params', default=None, help='Path to JSON file of tuned classifier params')
    parser.add_argument('--reg-params', default=None, help='Path to JSON file of tuned regressor params')
    args = parser.parse_args()
    main(args.data_path, args.models_dir, args.clf_params, args.reg_params)
