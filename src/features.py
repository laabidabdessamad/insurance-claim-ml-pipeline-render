"""Preprocessing pipeline: custom transformers + a ColumnTransformer builder.

This is used inside a single sklearn Pipeline alongside the model (see train.py),
so preprocessing is only ever *fit* on training data. Calling .predict() on held-out
data automatically calls .transform() (never .fit_transform()) on the already-fitted
pipeline -- this is the structural fix for the test-set leakage bug in the original
notebook.
"""
import numpy as np
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Transformers output pandas DataFrames (not numpy arrays) so downstream steps can
# reference columns by name -- matches the original notebook's global config.
set_config(transform_output='pandas')

NUMERICAL_COLS = [
    'num_young_drivers', 'age', 'num_of_children', 'years_job_held_for',
    'income', 'value_of_home', 'commute_dist', 'vehicle_value',
    'policy_tenure', '5_year_total_claims_value', '5_year_num_of_claims',
    'license_points', 'vehicle_age',
]
CAT_COLS_ORD = ['highest_education']
CAT_COLS_BIN = ['single_parent', 'married', 'gender', 'type_of_use', 'licence_revoked', 'address_type']
CAT_COLS_ONE_HOT = ['occupation', 'vehicle_type']
COLS_TO_DROP = ['red_vehicle']
SKEWED_FEATURES = ['income', 'value_of_home', 'commute_dist', 'vehicle_value', 'policy_tenure', 'license_points']
EDUCATION_RANK = [['<High School', 'High School', 'Bachelors', 'Masters', 'PhD']]


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops a fixed set of columns. Renamed usage stays honest: this is a drop, not a transform."""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        self.fitted_ = True  # marks this transformer as fitted for sklearn's check_is_fitted
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

    def get_feature_names_out(self, input_features=None):
        return None


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """Applies a square-root transform to reduce right-skew in numerical features.

    Named for what it does (sqrt), unlike the original notebook's `log_of_feature`
    function, which was misleadingly named despite applying np.sqrt.
    """

    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns_to_transform] = np.sqrt(X[self.columns_to_transform])
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_preprocess_pipeline() -> ColumnTransformer:
    """Builds the full preprocessing ColumnTransformer.

    Fitting this (as part of a parent Pipeline with a model) only ever happens on
    training data -- callers should never call .fit_transform() on held-out data.
    """
    cols_to_drop_pipeline = Pipeline([('col_dropper', ColumnDropper(COLS_TO_DROP))])

    num_pipeline = Pipeline([
        ('knn_imputer', KNNImputer(n_neighbors=2)),
        ('sqrt', SqrtTransformer(SKEWED_FEATURES)),
        ('scaler', StandardScaler()),
    ])

    cat_ord_pipeline = Pipeline([
        ('simple_imputer', SimpleImputer(strategy='most_frequent')),
        ('ord_encoder', OrdinalEncoder(categories=EDUCATION_RANK)),
    ])

    cat_bin_pipeline = Pipeline([
        ('simple_imputer', SimpleImputer(strategy='most_frequent')),
        ('binary_encoder', OrdinalEncoder()),
    ])

    cat_one_hot_pipeline = Pipeline([
        ('cat_simple_imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
    ])

    return ColumnTransformer([
        ('drop_features', cols_to_drop_pipeline, COLS_TO_DROP),
        ('num', num_pipeline, NUMERICAL_COLS),
        ('cat_ord', cat_ord_pipeline, CAT_COLS_ORD),
        ('cat_bin', cat_bin_pipeline, CAT_COLS_BIN),
        ('cat_one_hot', cat_one_hot_pipeline, CAT_COLS_ONE_HOT),
    ])
