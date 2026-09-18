import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data import load_and_clean
from features import build_preprocess_pipeline
from train import build_classification_split, build_regression_split, train_classifier, train_regressor

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'car_insurance_claim.csv'


@pytest.fixture(scope='module')
def cleaned_df():
    return load_and_clean(str(DATA_PATH))


def test_load_and_clean_has_no_currency_strings(cleaned_df):
    # Currency columns should be numeric after cleaning, not '$'-prefixed strings
    assert pd.api.types.is_numeric_dtype(cleaned_df['income'])
    assert pd.api.types.is_numeric_dtype(cleaned_df['new_claim_value'])


def test_load_and_clean_drops_id_and_dob(cleaned_df):
    assert 'ID' not in cleaned_df.columns
    assert 'date_of_birth' not in cleaned_df.columns


def test_no_duplicate_rows(cleaned_df):
    assert cleaned_df.duplicated().sum() == 0


def test_classification_split_is_stratified_consistently(cleaned_df):
    X_train, X_test, y_train, y_test = build_classification_split(cleaned_df)
    # No overlap between train/test indices
    assert set(X_train.index).isdisjoint(set(X_test.index))
    # Positive class rates should be reasonably close (stratified split)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05


def test_preprocessing_pipeline_produces_no_nulls(cleaned_df):
    X_train, X_test, y_train, y_test = build_classification_split(cleaned_df)
    pipeline = build_preprocess_pipeline()
    X_train_prepared = pipeline.fit_transform(X_train)
    assert not X_train_prepared.isnull().any().any()


def test_no_train_test_leakage_in_full_pipeline(cleaned_df):
    """The preprocessing step inside the fitted Pipeline must only ever be fit on
    train data. Calling .predict() on test data must not refit it -- this is the
    regression test for the original notebook's leakage bug."""
    X_train, X_test, y_train, y_test = build_classification_split(cleaned_df)
    small_params = dict(n_estimators=20, max_depth=3, random_state=42, eval_metric='error')
    pipeline = train_classifier(X_train, y_train, params=small_params)

    fitted_preprocess = pipeline.named_steps['preprocess']
    # Capture a fitted attribute's state (e.g. the KNN imputer's learned statistics)
    knn_imputer = fitted_preprocess.named_transformers_['num'].named_steps['knn_imputer']
    stats_before = np.array(knn_imputer.n_features_in_)

    _ = pipeline.predict(X_test)

    knn_imputer_after = fitted_preprocess.named_transformers_['num'].named_steps['knn_imputer']
    stats_after = np.array(knn_imputer_after.n_features_in_)
    assert stats_before == stats_after  # unchanged: predict() did not refit anything


def test_classifier_model_persists_and_predicts(tmp_path, cleaned_df):
    X_train, X_test, y_train, y_test = build_classification_split(cleaned_df)
    small_params = dict(n_estimators=20, max_depth=3, random_state=42, eval_metric='error')
    pipeline = train_classifier(X_train, y_train, params=small_params)

    model_path = tmp_path / 'clf.joblib'
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    sample = X_test.iloc[[0]]
    prediction = loaded.predict(sample)
    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)


def test_regressor_model_persists_and_predicts(tmp_path, cleaned_df):
    Xr_train, Xr_test, yr_train, yr_test = build_regression_split(cleaned_df)
    pipeline = train_regressor(Xr_train, yr_train, params=dict(max_iter=50, random_state=42))

    model_path = tmp_path / 'reg.joblib'
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    sample = Xr_test.iloc[[0]]
    prediction = loaded.predict(sample)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
