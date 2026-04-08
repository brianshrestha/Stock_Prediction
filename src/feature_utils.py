import json
import numpy as np
import pandas as pd
import boto3
import streamlit as st


def load_data_from_s3():
    session = boto3.Session(
        aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"],
        region_name="us-east-1"
    )

    s3 = session.client("s3")

    bucket = "brian-shrestha-s3-bucket"
    key = "SP500Data.csv"

    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"], index_col=0)
    return df


def convert_input_pca_regression(request_body, request_content_type):
    """
    Convert a small JSON payload from Streamlit into a full feature row
    matching the KPCA regression model input schema.
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    user_data = json.loads(request_body)

    # Load dataset from S3 instead of local file
    dataset = load_data_from_s3()

    # Match notebook setup
    target = "NVDA"
    return_period = 5

    # Rebuild X exactly like training
    X = np.log(dataset.drop([target], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + "_CR_Cum" for name in X.columns]
    X = X.dropna()

    # User-controlled features from Streamlit
    feature_1 = "AOS_CR_Cum"
    feature_2 = "AFL_CR_Cum"

    if feature_1 not in user_data or feature_2 not in user_data:
        raise ValueError(f"Payload must include '{feature_1}' and '{feature_2}'")

    val_1 = float(user_data[feature_1])
    val_2 = float(user_data[feature_2])

    # Find nearest historical row so the remaining features stay realistic
    distances = np.sqrt(
        (X[feature_1] - val_1) ** 2 +
        (X[feature_2] - val_2) ** 2
    )

    closest_index = distances.idxmin()
    closest_row = X.loc[[closest_index]].copy()

    # Overwrite the two user-provided values
    closest_row[feature_1] = val_1
    closest_row[feature_2] = val_2

    return closest_row
