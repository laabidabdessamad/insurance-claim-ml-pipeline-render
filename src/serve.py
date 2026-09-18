"""FastAPI inference service.

Loads the two saved pipelines (classifier + regressor) once at startup and serves
predictions over HTTP. This is what the Streamlit app calls in production, instead
of loading models locally -- the same code runs behind Fargate.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'

app = FastAPI(title='Insurance Claim Inference API')

_classifier = None
_regressor = None


def _get_models():
    global _classifier, _regressor
    if _classifier is None:
        _classifier = joblib.load(MODELS_DIR / 'classifier_pipeline.joblib')
    if _regressor is None:
        _regressor = joblib.load(MODELS_DIR / 'regressor_pipeline.joblib')
    return _classifier, _regressor


class ClaimRequest(BaseModel):
    num_young_drivers: int
    age: float
    num_of_children: int
    years_job_held_for: float
    income: float
    single_parent: str = Field(description="'Yes' or 'No'")
    value_of_home: float
    married: str = Field(description="'Yes' or 'No'")
    gender: str = Field(description="'M' or 'F'")
    highest_education: str
    occupation: str
    commute_dist: float
    type_of_use: str
    vehicle_value: float
    policy_tenure: float
    vehicle_type: str
    red_vehicle: str
    five_year_total_claims_value: float = Field(alias='5_year_total_claims_value')
    five_year_num_of_claims: int = Field(alias='5_year_num_of_claims')
    licence_revoked: str
    license_points: int
    vehicle_age: float
    address_type: str

    class Config:
        populate_by_name = True


class ClaimResponse(BaseModel):
    claim_probability: float
    expected_claim_value_if_claim: float


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=ClaimResponse)
def predict(request: ClaimRequest):
    classifier, regressor = _get_models()
    row = pd.DataFrame([request.dict(by_alias=True)])

    claim_probability = float(classifier.predict_proba(row)[:, 1][0])
    expected_claim_value = float(regressor.predict(row)[0])

    return ClaimResponse(
        claim_probability=claim_probability,
        expected_claim_value_if_claim=expected_claim_value,
    )
