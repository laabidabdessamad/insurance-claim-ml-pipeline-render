"""Hyperparameter tuning with Optuna, tracked in MLflow.

Both search spaces include L1/L2 regularization strength (reg_alpha/reg_lambda for
XGBoost, alpha/l1_ratio/penalty for SGDRegressor) rather than leaving them at
defaults. Every trial is logged to MLflow; the best trial's params are saved to
JSON for train.py to consume, and the best pipeline is registered in the MLflow
Model Registry.
"""
import argparse
import json
from pathlib import Path

import mlflow
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import SGDRegressor
from xgboost import XGBClassifier

from data import load_and_clean
from features import build_preprocess_pipeline
from train import build_classification_split, build_regression_split, RANDOM_STATE
from sklearn.pipeline import Pipeline

N_TRIALS_DEFAULT = 30


def _make_clf_pipeline(params: dict) -> Pipeline:
    return Pipeline([
        ('preprocess', build_preprocess_pipeline()),
        ('model', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', **params)),
    ])


def _make_reg_pipeline(params: dict) -> Pipeline:
    return Pipeline([
        ('preprocess', build_preprocess_pipeline()),
        ('model', SGDRegressor(random_state=RANDOM_STATE, **params)),
    ])


def tune_classifier(X_train, y_train, n_trials: int) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 400),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.4, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
            gamma=trial.suggest_float('gamma', 0.0, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),   # L1
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),  # L2
        )
        pipe = _make_clf_pipeline(params)
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='average_precision', n_jobs=-1)
        mean_score = scores.mean()
        with mlflow.start_run(nested=True, run_name=f'clf_trial_{trial.number}'):
            mlflow.log_params(params)
            mlflow.log_metric('cv_pr_auc', mean_score)
        return mean_score

    study = optuna.create_study(direction='maximize', study_name='classifier_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_regressor(X_train, y_train, n_trials: int) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
        params = dict(
            penalty=penalty,
            alpha=trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            learning_rate=trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'optimal']),
            eta0=trial.suggest_float('eta0', 1e-4, 0.1, log=True),
            max_iter=trial.suggest_int('max_iter', 200, 2000),
            tol=trial.suggest_float('tol', 1e-6, 1e-3, log=True),
        )
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        pipe = _make_reg_pipeline(params)
        scores = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        mean_rmse = scores.mean()
        with mlflow.start_run(nested=True, run_name=f'reg_trial_{trial.number}'):
            mlflow.log_params(params)
            mlflow.log_metric('cv_rmse', mean_rmse)
        return mean_rmse

    study = optuna.create_study(direction='minimize', study_name='regressor_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main(data_path: str, n_trials: int, output_dir: str, mlflow_uri: str):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment('insurance_claim_portfolio')

    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = build_classification_split(df)
    Xr_train, Xr_test, yr_train, yr_test = build_regression_split(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name='classifier_optuna_search'):
        best_clf_params, best_clf_score = tune_classifier(X_train, y_train, n_trials)
        mlflow.log_params({f'best_{k}': v for k, v in best_clf_params.items()})
        mlflow.log_metric('best_cv_pr_auc', best_clf_score)
        best_clf_pipeline = _make_clf_pipeline(best_clf_params)
        best_clf_pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(
            best_clf_pipeline, name='classifier_pipeline', registered_model_name='claim_classifier',
            serialization_format='pickle',
        )
        print(f'Best classifier CV PR-AUC: {best_clf_score:.4f}')
        print(f'Best classifier params: {best_clf_params}')

    with mlflow.start_run(run_name='regressor_optuna_search'):
        best_reg_params, best_reg_score = tune_regressor(Xr_train, yr_train, n_trials)
        mlflow.log_params({f'best_{k}': v for k, v in best_reg_params.items()})
        mlflow.log_metric('best_cv_rmse', best_reg_score)
        best_reg_pipeline = _make_reg_pipeline(best_reg_params)
        best_reg_pipeline.fit(Xr_train, yr_train)
        mlflow.sklearn.log_model(
            best_reg_pipeline, name='regressor_pipeline', registered_model_name='claim_regressor',
            serialization_format='pickle',
        )
        print(f'Best regressor CV RMSE: {best_reg_score:.2f}')
        print(f'Best regressor params: {best_reg_params}')

    (output_dir / 'best_clf_params.json').write_text(json.dumps(best_clf_params, indent=2))
    (output_dir / 'best_reg_params.json').write_text(json.dumps(best_reg_params, indent=2))
    print(f'Saved tuned params to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/car_insurance_claim.csv')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument('--output-dir', default='models/tuned_params')
    parser.add_argument('--mlflow-uri', default='sqlite:///mlflow.db')
    args = parser.parse_args()
    main(args.data_path, args.n_trials, args.output_dir, args.mlflow_uri)
