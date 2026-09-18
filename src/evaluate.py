"""Evaluate the saved pipelines on the held-out test split.

Uses .predict()/.predict_proba() directly on the fitted Pipeline -- this calls
.transform() on the preprocessing step (fitted on train only), never .fit_transform().
That's what makes these numbers a true held-out estimate, unlike the original
notebook's evaluation cells.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_recall_curve, roc_auc_score,
)

from data import load_and_clean
from train import build_classification_split, build_regression_split


def evaluate_classifier(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Precision/recall at the default 0.5 threshold, for reference alongside PR-AUC
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    return {
        'f1_weighted': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc_average_precision': float(pr_auc),
        'positive_class_rate_test': float(y_test.mean()),
    }


def evaluate_regressor(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Naive baseline: always predict the training mean claim value
    baseline_pred = np.full_like(y_test, fill_value=y_test.mean(), dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))

    return {
        'rmse': rmse,
        'mae': mae,
        'baseline_rmse': baseline_rmse,
        'baseline_mae': baseline_mae,
        'rmse_improvement_pct': float((baseline_rmse - rmse) / baseline_rmse * 100),
    }


def shap_feature_importance(pipeline, X_sample, top_n: int = 10, plot_path: str = None) -> list:
    """Computes mean absolute SHAP values for the model step of a fitted Pipeline.

    X_sample should be raw (untransformed) rows; we run them through the fitted
    preprocessing step first so SHAP sees the same feature space the model was
    trained on.
    """
    preprocess = pipeline.named_steps['preprocess']
    model = pipeline.named_steps['model']
    X_transformed = preprocess.transform(X_sample)

    explainer = shap.TreeExplainer(model) if hasattr(model, 'get_booster') else shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    values = shap_values.values
    if values.ndim == 3:  # multi-class / binary classifier with two output columns
        values = values[:, :, -1]
    mean_abs = np.abs(values).mean(axis=0)
    feature_names = list(X_transformed.columns)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True)[:top_n]

    if plot_path:
        names = [r[0] for r in ranked][::-1]
        vals = [r[1] for r in ranked][::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(names, vals)
        plt.xlabel('Mean |SHAP value|')
        plt.title('Top feature importances')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()

    return [{'feature': name, 'mean_abs_shap': float(val)} for name, val in ranked]


def main(data_path: str, models_dir: str, output_path: str = None):
    df = load_and_clean(data_path)
    models_dir = Path(models_dir)

    clf_pipeline = joblib.load(models_dir / 'classifier_pipeline.joblib')
    reg_pipeline = joblib.load(models_dir / 'regressor_pipeline.joblib')

    _, X_test, _, y_test = build_classification_split(df)
    clf_metrics = evaluate_classifier(clf_pipeline, X_test, y_test)

    _, Xr_test, _, yr_test = build_regression_split(df)
    reg_metrics = evaluate_regressor(reg_pipeline, Xr_test, yr_test)

    clf_importance = shap_feature_importance(
        clf_pipeline, X_test.sample(min(300, len(X_test)), random_state=42),
        plot_path=str(models_dir / 'classifier_shap_importance.png'),
    )
    reg_importance = shap_feature_importance(
        reg_pipeline, Xr_test.sample(min(300, len(Xr_test)), random_state=42),
        plot_path=str(models_dir / 'regressor_shap_importance.png'),
    )
    clf_metrics['top_shap_features'] = clf_importance
    reg_metrics['top_shap_features'] = reg_importance

    results = {'classifier': clf_metrics, 'regressor': reg_metrics}
    print(json.dumps(results, indent=2))

    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    matplotlib.use('Agg')  # no display available when run as a CLI script
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/car_insurance_claim.csv')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--output', default='models/evaluation_results.json')
    args = parser.parse_args()
    main(args.data_path, args.models_dir, args.output)
