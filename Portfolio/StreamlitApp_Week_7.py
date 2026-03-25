import os
import sys
import warnings
import tarfile
import tempfile

import joblib
import boto3
import shap
import sagemaker
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.pipeline import Pipeline

# -----------------------------
# Setup & Path Configuration
# -----------------------------
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Pair Trading Model Deployment", layout="wide")
st.title("👨‍💻 Pair Trading Model Deployment")

# -----------------------------
# Access secrets
# -----------------------------
aws_creds = st.secrets["aws_credentials"]
aws_id = aws_creds["AWS_ACCESS_KEY_ID"]
aws_secret = aws_creds["AWS_SECRET_ACCESS_KEY"]
aws_token = aws_creds.get("AWS_SESSION_TOKEN", None)
aws_bucket = aws_creds["AWS_BUCKET"]
aws_endpoint = aws_creds["AWS_ENDPOINT"]
aws_region = aws_creds.get("AWS_REGION", "us-east-1")

# -----------------------------
# AWS Session Management
# -----------------------------
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token=None):
    kwargs = {
        "aws_access_key_id": aws_id,
        "aws_secret_access_key": aws_secret,
        "region_name": aws_region,
    }
    if aws_token:
        kwargs["aws_session_token"] = aws_token
    return boto3.Session(**kwargs)


session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# -----------------------------
# Data & Model Configuration
# -----------------------------
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline": "finalized_pair_model.tar.gz",
    "pipeline_prefix": "sklearn-pipeline-deployment",
    "explainer_key": "explainer/explainer_pair.shap",
    "keys": ["EMR", "GOOG"],
    "inputs": [
        {"name": "EMR", "type": "number", "min": 0.0, "default": 116.0, "step": 1.0},
        {"name": "GOOG", "type": "number", "min": 0.0, "default": 180.0, "step": 1.0},
    ],
}

# -----------------------------
# Base historical data
# Uses extract_features_pair() as requested
# -----------------------------
@st.cache_data
def load_base_features():
    return extract_features_pair()


df_features = load_base_features()

# -----------------------------
# Load pipeline from S3
# -----------------------------
@st.cache_resource
def load_pipeline(_session, bucket, prefix):
    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]
    local_tar_path = os.path.join(tempfile.gettempdir(), filename)

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{prefix}/{os.path.basename(filename)}",
        Filename=local_tar_path,
    )

    extract_dir = os.path.join(tempfile.gettempdir(), "pair_model_extract")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_files = [f for f in tar.getnames() if f.endswith(".joblib")]

    if not joblib_files:
        raise FileNotFoundError("No .joblib file found inside the model tar.gz")

    joblib_path = os.path.join(extract_dir, os.path.basename(joblib_files[0]))
    return joblib.load(joblib_path)

# -----------------------------
# Load SHAP explainer from S3
# -----------------------------
@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# -----------------------------
# Prediction Logic
# -----------------------------
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )

    try:
        raw_pred = predictor.predict(input_df.values.tolist())

        if isinstance(raw_pred, list):
            pred_val = raw_pred[-1] if raw_pred else None
        elif isinstance(raw_pred, dict):
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
# Local Explainability
# -----------------------------
def display_explanation(input_df, _session, bucket):
    explainer = load_shap_explainer(
        _session,
        bucket,
        MODEL_INFO["explainer_key"],
        os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"]),
    )

    full_pipeline = load_pipeline(
        _session,
        bucket,
        MODEL_INFO["pipeline_prefix"],
    )

    # exclude sampler and model
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    feature_names = full_pipeline.named_steps["feature_selection"].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")

    sample_shap = shap_values[0, :, 0] if len(shap_values.shape) == 3 else shap_values[0]

    shap.plots.waterfall(sample_shap, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    top_feature = pd.Series(sample_shap.values, index=sample_shap.feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# -----------------------------
# Streamlit UI
# -----------------------------
with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"],
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
        data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

        base_df = df_features[MODEL_INFO["keys"]].copy()
        new_row = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])
        input_df = pd.concat([base_df, new_row], ignore_index=True)

        st.subheader("Input Snapshot")
        st.dataframe(input_df.tail(5))

        res, status = call_model_api(input_df)

        if status == 200:
            st.metric("Prediction Result", res)
            display_explanation(input_df, session, aws_bucket)
        else:
            st.error(res)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
