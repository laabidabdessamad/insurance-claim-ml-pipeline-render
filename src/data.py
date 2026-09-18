"""Load and clean the raw car insurance claim dataset."""
import pandas as pd

COLUMN_RENAME_MAP = {
    'KIDSDRIV': 'num_young_drivers',
    'BIRTH': 'date_of_birth',
    'AGE': 'age',
    'HOMEKIDS': 'num_of_children',
    'YOJ': 'years_job_held_for',
    'INCOME': 'income',
    'PARENT1': 'single_parent',
    'HOME_VAL': 'value_of_home',
    'MSTATUS': 'married',
    'GENDER': 'gender',
    'EDUCATION': 'highest_education',
    'OCCUPATION': 'occupation',
    'TRAVTIME': 'commute_dist',
    'CAR_USE': 'type_of_use',
    'BLUEBOOK': 'vehicle_value',
    'TIF': 'policy_tenure',
    'CAR_TYPE': 'vehicle_type',
    'RED_CAR': 'red_vehicle',
    'OLDCLAIM': '5_year_total_claims_value',
    'CLM_FREQ': '5_year_num_of_claims',
    'REVOKED': 'licence_revoked',
    'MVR_PTS': 'license_points',
    'CLM_AMT': 'new_claim_value',
    'CAR_AGE': 'vehicle_age',
    'CLAIM_FLAG': 'is_claim',
    'URBANICITY': 'address_type',
}

CURRENCY_COLS = ['income', 'value_of_home', 'vehicle_value', '5_year_total_claims_value', 'new_claim_value']
Z_PREFIX_COLS = ['married', 'gender', 'highest_education', 'occupation', 'vehicle_type', 'address_type']
DROP_COLS = ['ID', 'date_of_birth']


def _format_currency_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].replace(r'[\$,]', '', regex=True).astype('Int64')
    return df


def _remove_z_prefix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].replace('z_', '', regex=True)
    return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning steps as the original notebook:
    rename columns, drop duplicates, strip currency formatting and 'z_' prefixes,
    and drop identifier/redundant columns.
    """
    raw = pd.read_csv(csv_path)
    df = raw.copy()
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    df.drop_duplicates(inplace=True)
    df = _format_currency_cols(df, CURRENCY_COLS)
    df = _remove_z_prefix(df, Z_PREFIX_COLS)
    df.drop(columns=DROP_COLS, inplace=True)
    return df
