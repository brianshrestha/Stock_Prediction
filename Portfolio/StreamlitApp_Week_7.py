import os
import sys
import warnings
import tempfile
import tarfile
import posixpath

import boto3
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sagemaker
import shap
import streamlit as st
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sklearn.pipeline import Pipeline

warnings.simplefilter("ignore")

# -----------------------------
# Path setup
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair  # noqa: E402

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Pair Trading Model Deployment", layout="wide")
st.title("👨‍💻 Pair Trading Model Deployment")

# -----------------------------
# Secrets / AWS config
# -----------------------------
aws_creds = st.secrets["aws_credentials"]
aws_id = aws_creds["AWS_ACCESS_KEY_ID"]
aws_secret = aws_creds["AWS_SECRET_ACCESS_KEY"]
aws_token = aws_creds.get("AWS_SESSION_TOKEN", None)
aws_bucket = aws_creds["AWS_BUCKET"]
aws_endpoint = aws_creds["AWS_ENDPOINT"]
aws_region = aws_creds.get("AWS_REGION", "us-east-1")

# -----------------------------
# Model config
# Final trained pair: GOOG + EMR
# Keep column order consistent everywhere
# -----------------------------
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline": "finalized_pair_model.tar.gz",
    "pipeline_s3_prefix": "sklearn-pipeline-deployment",
    "explainer_s3_key": "explainer/explainer_pair.shap",
    "keys": ["EMR", "GOOG"],  # match final training input columns
    "inputs": [
        {"name": "EMR", "type": "number", "min": 0.0, "default": 110.0, "step": 1.0},
        {"name": "GOOG", "type": "number", "min": 0.0, "default": 180.0, "step": 1.0},
    ],
}

# -----------------------------
# AWS session
# -----------------------------
@st.cache_resource
def get_session(access_key: str, secret_key: str, session_token: str | None, region: str):
    kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region,
    }
    if session_token:
        kwargs["aws_session_token"] = session_token
    return boto3.Session(**kwargs)


session = get_session(aws_id, aws_secret, aws_token, aws_region)
sm_session = sagemaker.Session(boto_session=session)

# -----------------------------
# Historical base data
# This should return the raw two-column price history used
# to append a new observation before scoring.
# -----------------------------
@st.cache_data
def load_base_features() -> pd.DataFrame:
    df = extract_features_pair().copy()
    missing_cols = [c for c in MODEL_INFO["keys"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"extract_features_pair() is missing expected columns: {missing_cols}")
    return df[MODEL_INFO["keys"]].copy()


df_features = load_base_features()

# -----------------------------
# Helpers to load artifacts from S3
# -----------------------------
@st.cache_resource
def load_pipeline(_session, bucket: str, s3_prefix: str, tar_filename: str):
    s3_client = _session.client("s3")
    local_tar_path = os.path.join(tempfile.gettempdir(), tar_filename)

    # download tar.gz
    s3_client.download_file(
        Bucket=bucket,
        Key=f"{s3_prefix}/{os.path.basename(tar_filename)}",
        Filename=local_tar_path,
    )

    # extract tar.gz to temp folder
    extract_dir = os.path.join(tempfile.gettempdir(), "pair_model_extract")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_files = [f for f in tar.getnames() if f.endswith(".joblib")]

    if not joblib_files:
        raise FileNotFoundError("No .joblib file found inside model tar.gz")

    joblib_path = os.path.join(extract_dir, os.path.basename(joblib_files[0]))
    return joblib.load(joblib_path)


@st.cache_resource
def load_shap_explainer(_session, bucket: str, s3_key: str, local_path: str):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# -----------------------------
# Endpoint prediction
# -----------------------------
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )

    try:
        # Send only values; endpoint expects tabular rows
        raw_pred = predictor.predict(input_df.values.tolist())

        # Handle common response formats
        if isinstance(raw_pred, list):
            pred_val = raw_pred[-1] if len(raw_pred) > 0 else None
        elif isinstance(raw_pred, dict):
            # fallback for JSON object response
            pred_val = raw_pred.get("predictions", [None])[-1]
        else:
            pred_val = raw_pred

        if pred_val is None:
            return "Error: empty prediction returned by endpoint", 500

        pred_val = int(pred_val)
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(pred_val, pred_val), 200

    except Exception as e:
        return f"Error: {str(e)}", 500


# -----------------------------
# Local explainability (SHAP)
# -----------------------------
def display_explanation(input_df: pd.DataFrame, _session, bucket: str):
    explainer = load_shap_explainer(
        _session=_session,
        bucket=bucket,
        s3_key=MODEL_INFO["explainer_s3_key"],
        local_path=os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"]),
    )

    full_pipeline = load_pipeline(
        _session=_session,
        bucket=bucket,
        s3_prefix=MODEL_INFO["pipeline_s3_prefix"],
        tar_filename=MODEL_INFO["pipeline"],
    )

    # Exclude sampler and model; keep preprocessing/feature engineering only
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])

    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names = full_pipeline.named_steps["feature_selection"].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")

    # Multiclass-safe indexing
    sample_shap = shap_values[0, :, 0] if len(shap_values.shape) == 3 else shap_values[0]

    shap.plots.waterfall(sample_shap, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    top_feature = pd.Series(sample_shap.values, index=sample_shap.feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# -----------------------------
# UI
# -----------------------------
st.subheader("Inputs")

with st.form("pred_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                label=inp["name"],
                min_value=inp["min"],
                value=inp["default"],
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")


# -----------------------------
# Inference flow
# -----------------------------
if submitted:
    try:
        # Base historical data used by PairFeatureEngineer(window=60)
        base_df = df_features[MODEL_INFO["keys"]].copy()

        # Append the latest user-provided row
        new_row = pd.DataFrame(
            [[user_inputs[k] for k in MODEL_INFO["keys"]]],
            columns=MODEL_INFO["keys"],
        )

        input_df = pd.concat([base_df, new_row], ignore_index=True)

        st.write("### Input Snapshot")
        st.dataframe(input_df.tail(5))

        prediction, status = call_model_api(input_df)

        if status == 200:
            st.metric("Prediction Result", prediction)
            display_explanation(input_df, session, aws_bucket)
        else:
            st.error(prediction)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
