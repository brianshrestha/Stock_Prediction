from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path

import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [ROOT, ROOT / "data"]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if (p / "test_transaction.csv").exists()), ROOT)

MODEL_PATH_CANDIDATES = [
    ROOT / "final_models" / "finalized_fraud_model.joblib",
    ROOT / "model_skl12.joblib",
    ROOT / "model.joblib",
]
MODEL_PATH = next((p for p in MODEL_PATH_CANDIDATES if p.exists()), MODEL_PATH_CANDIDATES[0])

SUMMARY_PATH = ROOT / "final_outputs" / "final_summary.json"
TOP_FEATURES_PATH = ROOT / "final_outputs" / "final_top_features.csv"
PREDICTIONS_PATH = ROOT / "final_outputs" / "professor_test_predictions.csv"


def _secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _secret2(group: str, key: str, default=None):
    try:
        if group in st.secrets and key in st.secrets[group]:
            return st.secrets[group][key]
    except Exception:
        pass
    return _secret(key, default)


USE_SAGEMAKER = bool(_secret("SAGEMAKER_ENDPOINT") or _secret2("aws_credentials", "AWS_ENDPOINT"))


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_runtime_client():
    endpoint = _secret("SAGEMAKER_ENDPOINT") or _secret2("aws_credentials", "AWS_ENDPOINT")
    if not endpoint:
        return None, None

    region = _secret("AWS_DEFAULT_REGION", "us-east-1") or _secret2("aws_credentials", "AWS_DEFAULT_REGION", "us-east-1")
    aws_id = _secret("AWS_ACCESS_KEY_ID") or _secret2("aws_credentials", "AWS_ACCESS_KEY_ID")
    aws_secret = _secret("AWS_SECRET_ACCESS_KEY") or _secret2("aws_credentials", "AWS_SECRET_ACCESS_KEY")
    aws_token = _secret("AWS_SESSION_TOKEN") or _secret2("aws_credentials", "AWS_SESSION_TOKEN")

    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
    )
    return runtime, endpoint


@st.cache_resource
def get_boto_session():
    aws_id = _secret2("aws_credentials", "AWS_ACCESS_KEY_ID")
    aws_secret = _secret2("aws_credentials", "AWS_SECRET_ACCESS_KEY")
    aws_token = _secret2("aws_credentials", "AWS_SESSION_TOKEN")
    region = _secret2("aws_credentials", "AWS_DEFAULT_REGION", _secret("AWS_DEFAULT_REGION", "us-east-1"))
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name=region,
    )


@st.cache_resource
def load_pipeline_from_s3():
    sess = get_boto_session()
    s3 = sess.client("s3")
    bucket = _secret2("aws_credentials", "AWS_BUCKET")
    key = _secret2("aws_credentials", "PIPELINE_KEY", "fraud-pipeline-deployment/model.tar.gz")

    if not bucket:
        raise RuntimeError("Missing AWS_BUCKET in secrets.")
    if not key:
        raise RuntimeError("Missing PIPELINE_KEY in secrets.")

    local_tar = os.path.join(tempfile.gettempdir(), "fraud_pipeline.tar.gz")
    s3.download_file(bucket, key, local_tar)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_name = [n for n in tar.getnames() if n.endswith(".joblib")][0]

    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_name))


@st.cache_resource
def load_explainer_from_s3():
    sess = get_boto_session()
    s3 = sess.client("s3")
    bucket = _secret2("aws_credentials", "AWS_BUCKET")
    key = _secret2("aws_credentials", "EXPLAINER_KEY", "")

    if not bucket:
        raise RuntimeError("Missing AWS_BUCKET in secrets.")
    if not key:
        raise RuntimeError("EXPLAINER_KEY is empty. Configure it in secrets or skip SHAP.")

    local_path = os.path.join(tempfile.gettempdir(), "fraud_explainer.shap")
    s3.download_file(bucket, key, local_path)
    return joblib.load(local_path)


@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    return {"holdout_metrics": {"roc_auc": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0}}


@st.cache_data
def load_top_features():
    if TOP_FEATURES_PATH.exists():
        return pd.read_csv(TOP_FEATURES_PATH).head(15)
    return pd.DataFrame({"feature": [], "importance_mean": []})


@st.cache_data
def load_professor_sample():
    p = DATA_DIR / "test_transaction.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p).sample(250, random_state=42).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("Unnamed")], errors="ignore")

    if "TransactionAmt" in out.columns:
        if "TransactionAmt_clip" not in out.columns:
            if len(out) > 1:
                lo, hi = out["TransactionAmt"].quantile([0.01, 0.99])
            else:
                lo = hi = out["TransactionAmt"].iloc[0]
            out["TransactionAmt_clip"] = out["TransactionAmt"].clip(lo, hi)
        out["amt_log"] = np.log1p(out["TransactionAmt"])
        out["amt_log_clip"] = np.log1p(out["TransactionAmt_clip"])
        out["amt_is_round"] = (out["TransactionAmt"] % 1 == 0).astype(int)
        out["amt_cents"] = ((out["TransactionAmt"] * 100) % 100).round(0)

    if "TransactionDT" in out.columns:
        out["txn_hour"] = (out["TransactionDT"] // 3600) % 24
        out["txn_day"] = (out["TransactionDT"] // 86400) % 7
        out["txn_week"] = (out["TransactionDT"] // (86400 * 7)).astype(int)
        out["txn_month"] = (out["TransactionDT"] // (86400 * 30)).astype(int)

    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in out.columns:
            out[col] = out[col].fillna("missing").astype(str).str.lower()
            out[f"{col}_tld"] = out[col].str.split(".").str[-1].fillna("missing")
            out[f"{col}_provider"] = out[col].str.split(".").str[0].fillna("missing")

    if {"P_emaildomain", "R_emaildomain"}.issubset(out.columns):
        out["email_match"] = (out["P_emaildomain"] == out["R_emaildomain"]).astype(int)
    if {"card1", "card2"}.issubset(out.columns):
        out["card1_card2"] = out["card1"].astype(str) + "_" + out["card2"].astype(str)
    if {"addr1", "addr2"}.issubset(out.columns):
        out["addr1_addr2"] = out["addr1"].astype(str) + "_" + out["addr2"].astype(str)

    if "DeviceInfo" in out.columns:
        device = out["DeviceInfo"].fillna("missing").astype(str).str.lower()
        out["device_is_missing"] = (device == "missing").astype(int)
        out["device_brand"] = device.str.split(r"[\s/]", regex=True).str[0]

    out["missing_count"] = out.isna().sum(axis=1)
    out["missing_rate"] = out.isna().mean(axis=1)
    return out


def _parse_endpoint_scores(raw: str) -> list[float]:
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "predictions" in parsed:
        preds = parsed["predictions"]
        if preds and isinstance(preds[0], dict):
            return [float(p.get("fraud_score", p.get("score", p.get("probability", 0.0)))) for p in preds]
        return [float(x) for x in preds]
    if isinstance(parsed, list):
        out = []
        for x in parsed:
            if isinstance(x, dict):
                out.append(float(x.get("fraud_score", x.get("score", x.get("probability", 0.0)))))
            else:
                out.append(float(x))
        return out
    raise ValueError("Unexpected endpoint response format")


def endpoint_score(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    runtime, endpoint = load_runtime_client()
    if runtime is None:
        raise RuntimeError("SageMaker endpoint config missing in Streamlit secrets.")

    prepared = add_features(frame)
    csv_body = prepared.to_csv(index=False)

    resp = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="text/csv",
        Accept="application/json",
        Body=csv_body,
    )

    raw = resp["Body"].read().decode("utf-8").strip()
    scores = _parse_endpoint_scores(raw)

    if len(scores) != len(frame):
        raise ValueError(f"Prediction length mismatch: got {len(scores)} scores for {len(frame)} rows.")

    scored = frame.copy()
    scored["fraud_score"] = scores
    scored["flag_for_review"] = scored["fraud_score"] >= threshold
    return scored


def local_score(frame: pd.DataFrame, model_bundle, threshold: float) -> pd.DataFrame:
    enriched = add_features(frame)

    if isinstance(model_bundle, dict) and "model" in model_bundle:
        feature_columns = model_bundle.get("feature_columns")
        model = model_bundle["model"]
        X = enriched.reindex(columns=feature_columns) if feature_columns else enriched
        local_threshold = float(model_bundle.get("threshold", threshold))
    else:
        model = model_bundle
        X = enriched
        local_threshold = threshold

    scores = model.predict_proba(X)[:, 1]
    scored = frame.copy()
    scored["fraud_score"] = scores
    scored["flag_for_review"] = scores >= local_threshold
    return scored


st.set_page_config(page_title="IEEE Fraud Detection", layout="wide")

st.write("bucket:", _secret2("aws_credentials", "AWS_BUCKET"))
st.write("explainer:", _secret2("aws_credentials", "EXPLAINER_KEY"))
st.write("pipeline:", _secret2("aws_credentials", "PIPELINE_KEY"))

summary = load_summary()
top_features = load_top_features()

summary = load_summary()
top_features = load_top_features()

bundle = None
if not USE_SAGEMAKER:
    bundle = load_model_bundle()

if USE_SAGEMAKER:
    st.success("Using SageMaker endpoint scoring.")
elif bundle is not None:
    st.info(f"Using local model scoring: {MODEL_PATH.name}")
else:
    st.error("No scoring backend available. Add SageMaker secrets or include local model file.")
    st.stop()

st.title("IEEE-CIS Fraud Detection")
st.caption("Final project Streamlit demo using professor-provided transaction files.")

metric_cols = st.columns(4)
metric_cols[0].metric("Held-Out ROC-AUC", f"{summary['holdout_metrics']['roc_auc']:.3f}")
metric_cols[1].metric("Held-Out PR-AUC", f"{summary['holdout_metrics']['pr_auc']:.3f}")
metric_cols[2].metric("Precision", f"{summary['holdout_metrics']['precision']:.1%}")
metric_cols[3].metric("Recall", f"{summary['holdout_metrics']['recall']:.1%}")

default_threshold = 0.5
if bundle is not None and isinstance(bundle, dict) and "threshold" in bundle:
    default_threshold = float(bundle["threshold"])

threshold = st.sidebar.slider("Fraud alert threshold", 0.05, 0.90, default_threshold, 0.05)

uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
if uploaded is not None:
    input_df = pd.read_csv(uploaded)
    label = "Uploaded transactions"
else:
    input_df = load_professor_sample()
    label = "Sample from professor test transactions"

if input_df.empty:
    st.warning("No input rows found. Upload a CSV or add test_transaction.csv to app data.")
    st.stop()

try:
    if USE_SAGEMAKER:
        scored = endpoint_score(input_df, threshold)
        backend_used = "SageMaker endpoint"
    else:
        scored = local_score(input_df, bundle, threshold)
        backend_used = f"Local model ({MODEL_PATH.name})"
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

st.info(f"Scoring backend: {backend_used}")

left, right = st.columns([2, 1])
with left:
    st.subheader(label)
    st.dataframe(
        scored.sort_values("fraud_score", ascending=False).head(30),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Review Queue")
    st.metric("Transactions scored", f"{len(scored):,}")
    st.metric("Flagged", f"{int(scored['flag_for_review'].sum()):,}")
    st.metric("Alert rate", f"{scored['flag_for_review'].mean():.1%}")

st.subheader("Decision Transparency (SHAP)")
row_idx = st.number_input("Row index to explain", min_value=0, max_value=len(input_df)-1, value=0, step=1)
run_shap = st.checkbox("Generate SHAP explanation (slow)", value=False)

if run_shap:
    explainer_key = _secret2("aws_credentials", "EXPLAINER_KEY", "")
    if not explainer_key:
        st.info("SHAP explainer not configured. Add EXPLAINER_KEY in Streamlit secrets.")
    else:
        try:
            
            bundle_or_pipe = load_pipeline_from_s3()
            explainer = load_explainer_from_s3()
            
            pipe = bundle_or_pipe["model"] if isinstance(bundle_or_pipe, dict) and "model" in bundle_or_pipe else bundle_or_pipe
            
            x_row = add_features(input_df.iloc[[row_idx]].copy())
            
            pre = pipe.named_steps.get("preprocess") if hasattr(pipe, "named_steps") else None
            if pre is None:
                st.info(f"Pipeline preprocessor step not found for SHAP. Loaded type: {type(pipe)}")
            else:
                req = getattr(pre, "feature_names_in_", None)
                if req is not None:
                    for c in req:
                        if c not in x_row.columns:
                            x_row[c] = np.nan
                    x_row = x_row.loc[:, req]

                x_trans = pre.transform(x_row)
                x_trans = x_trans.toarray() if hasattr(x_trans, "toarray") else x_trans

                shap_values = explainer(x_trans)

                fig = plt.figure(figsize=(11, 5))
                shap.plots.waterfall(shap_values[0], max_display=12, show=False)
                st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.info(f"SHAP unavailable: {e}")
else:
    st.info("Enable SHAP checkbox to compute row-level explanation.")

st.subheader("Top Model Drivers")
st.dataframe(top_features, use_container_width=True, hide_index=True)

st.subheader("Batch Predictions File")
st.write(f"Saved professor test predictions: `{PREDICTIONS_PATH}`")

# Debug
st.write("USE_SAGEMAKER:", USE_SAGEMAKER)
st.write("endpoint:", _secret("SAGEMAKER_ENDPOINT"), _secret2("aws_credentials", "AWS_ENDPOINT"))
