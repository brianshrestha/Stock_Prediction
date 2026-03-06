import os
import tarfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import boto3
import shap
import joblib
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Bitcoin Signal Predictor", layout="wide")
st.title("Bitcoin Buy / Hold / Sell Signal Predictor")

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]

@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()

MODEL_FILE = "finalized_bitcoin_model.tar.gz"
EXPLAINER_FILE = "explainer_bitcoin.shap"
MODEL_KEY = f"sklearn-pipeline-deployment/{MODEL_FILE}"
EXPLAINER_KEY = f"explainer/{EXPLAINER_FILE}"

@st.cache_resource
def load_pipeline():
    s3 = session.client("s3")
    local_tar = os.path.join(tempfile.gettempdir(), MODEL_FILE)
    s3.download_file(aws_bucket, MODEL_KEY, local_tar)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_file))

@st.cache_resource
def load_explainer():
    s3 = session.client("s3")
    local_explainer = os.path.join(tempfile.gettempdir(), EXPLAINER_FILE)
    if not os.path.exists(local_explainer):
        s3.download_file(aws_bucket, EXPLAINER_KEY, local_explainer)

    with open(local_explainer, "rb") as f:
        return shap.Explainer.load(f)

model = load_pipeline()
explainer = load_explainer()

st.subheader("Enter Technical Indicators")

momentum_7 = st.number_input("Momentum_7", value=0.0)
sma_7 = st.number_input("SMA_7", value=0.0)
sma_21 = st.number_input("SMA_21", value=0.0)
volatility_7 = st.number_input("Volatility_7", value=0.0)
roc_7 = st.number_input("ROC_7", value=0.0)

input_df = pd.DataFrame([{
    "Momentum_7": momentum_7,
    "SMA_7": sma_7,
    "SMA_21": sma_21,
    "Volatility_7": volatility_7,
    "ROC_7": roc_7
}])

if st.button("Predict Signal"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Prediction: BUY")
    elif prediction == -1:
        st.error("Prediction: SELL")
    else:
        st.warning("Prediction: HOLD")

    preprocessing_pipeline = Pipeline(steps=model.steps[:-1])
    transformed = preprocessing_pipeline.transform(input_df)

    selected_mask = model.named_steps["feature_selection"].get_support()
    feature_names = input_df.columns[selected_mask]

    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    shap_values = explainer(transformed_df)

    exp = shap.Explanation(
        values=shap_values.values[0, :, 2],
        base_values=shap_values.base_values[0, 2],
        data=transformed_df.iloc[0],
        feature_names=feature_names
    )

    st.subheader("SHAP Explanation")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(exp, show=False)
    st.pyplot(fig)
