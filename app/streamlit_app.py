"""Streamlit client for the insurance claim inference service.

This app does NOT load any model locally -- it only calls the FastAPI service's
/predict endpoint over HTTP. In production that's the Fargate-hosted API behind
the ALB; for local development it defaults to a locally-running `uvicorn` instance.

Configure the API URL via the API_URL environment variable, or edit DEFAULT_API_URL
below. Once deployed, set API_URL to the ALB's DNS name (or a custom domain in
front of it), e.g. http://<load-balancer-dns>.
"""
import os

import requests
import streamlit as st

DEFAULT_API_URL = "http://insura-infer-oeuqdmjyqi37-283485911.eu-west-3.elb.amazonaws.com/"
API_URL = os.environ.get("API_URL", DEFAULT_API_URL)

st.set_page_config(page_title="Car Insurance Claim Risk", page_icon="🚗", layout="centered")

st.title("🚗 Car insurance claim risk")
st.caption(
    "Estimates the probability a policyholder files a claim, and the expected "
    "claim value if one occurs. Calls a live FastAPI inference service -- no "
    "model runs in this browser tab."
)

with st.sidebar:
    st.subheader("API connection")
    st.code(API_URL, language=None)
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.ok:
            st.success("API reachable")
        else:
            st.error(f"API responded with status {health.status_code}")
    except requests.RequestException as exc:
        st.error(f"Can't reach the API: {exc}")
    st.caption("Set the API_URL environment variable to point at a different endpoint.")

st.subheader("Policyholder details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=16, max_value=100, value=45)
    gender = st.selectbox("Gender", ["M", "F"])
    married = st.selectbox("Married", ["Yes", "No"])
    single_parent = st.selectbox("Single parent", ["No", "Yes"])
    num_of_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
    num_young_drivers = st.number_input("Young drivers in household", min_value=0, max_value=5, value=0)
    highest_education = st.selectbox(
        "Highest education", ["<High School", "High School", "Bachelors", "Masters", "PhD"], index=2,
    )
    occupation = st.selectbox(
        "Occupation",
        ["Blue Collar", "Clerical", "Doctor", "Home Maker", "Lawyer", "Manager", "Professional", "Student"],
        index=6,
    )
    income = st.number_input("Annual income ($)", min_value=0, value=60000, step=1000)
    years_job_held_for = st.number_input("Years in current job", min_value=0, max_value=60, value=10)

with col2:
    value_of_home = st.number_input("Home value ($)", min_value=0, value=200000, step=1000)
    address_type = st.selectbox("Area type", ["Highly Urban/ Urban", "Highly Rural/ Rural"])
    commute_dist = st.number_input("Commute distance (miles)", min_value=0, value=20)
    type_of_use = st.selectbox("Vehicle use", ["Private", "Commercial"])
    vehicle_type = st.selectbox(
        "Vehicle type", ["Minivan", "Panel Truck", "Pickup", "SUV", "Sports Car", "Van"],
    )
    vehicle_value = st.number_input("Vehicle value ($)", min_value=0, value=15000, step=500)
    vehicle_age = st.number_input("Vehicle age (years)", min_value=0, max_value=30, value=5)
    red_vehicle = st.selectbox("Red vehicle", ["no", "yes"])
    policy_tenure = st.number_input("Years with this insurer (policy tenure)", min_value=0, max_value=30, value=5)

st.subheader("Driving and claims history")
col3, col4 = st.columns(2)
with col3:
    licence_revoked = st.selectbox("Licence ever revoked", ["No", "Yes"])
    license_points = st.number_input("License points", min_value=0, max_value=20, value=0)
with col4:
    claims_5yr = st.number_input("Claims in the last 5 years", min_value=0, max_value=20, value=0)
    claims_5yr_value = st.number_input(
        "Total value of claims in the last 5 years ($)", min_value=0, value=0, step=100,
    )

if st.button("Estimate claim risk", type="primary"):
    payload = {
        "num_young_drivers": int(num_young_drivers),
        "age": float(age),
        "num_of_children": int(num_of_children),
        "years_job_held_for": float(years_job_held_for),
        "income": float(income),
        "single_parent": single_parent,
        "value_of_home": float(value_of_home),
        "married": married,
        "gender": gender,
        "highest_education": highest_education,
        "occupation": occupation,
        "commute_dist": float(commute_dist),
        "type_of_use": type_of_use,
        "vehicle_value": float(vehicle_value),
        "policy_tenure": float(policy_tenure),
        "vehicle_type": vehicle_type,
        "red_vehicle": red_vehicle,
        "5_year_total_claims_value": float(claims_5yr_value),
        "5_year_num_of_claims": int(claims_5yr),
        "licence_revoked": licence_revoked,
        "license_points": int(license_points),
        "vehicle_age": float(vehicle_age),
        "address_type": address_type,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Claim probability", f"{result['claim_probability']:.1%}")
        with col_b:
            st.metric(
                "Expected claim value (if a claim occurs)",
                f"${result['expected_claim_value_if_claim']:,.0f}",
            )
        st.caption(
            "The claim value estimate applies only if a claim occurs -- it is not "
            "weighted by the probability above."
        )
    except requests.HTTPError as exc:
        st.error(f"API returned an error: {exc.response.status_code} — {exc.response.text}")
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the API: {exc}")
