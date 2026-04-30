from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [ROOT, ROOT / "data"]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if (p / "test_transaction.csv").exists()), ROOT)

MODEL_PATH = ROOT / "final_models" / "finalized_fraud_model.joblib"
SUMMARY_PATH = ROOT / "final_outputs" / "final_summary.json"
TOP_FEATURES_PATH = ROOT / "final_outputs" / "final_top_features.csv"
PREDICTIONS_PATH = ROOT / "final_outputs" / "professor_test_predictions.csv"
SHAP_BAR_PATH = ROOT / "final_figures" / "shap_bar.png"
SHAP_WATERFALL_PATH = ROOT / "final_figures" / "shap_waterfall.png"


def _secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


USE_SAGEMAKER = bool(_secret("SAGEMAKER_ENDPOINT"))


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_runtime_client():
    endpoint = _secret("SAGEMAKER_ENDPOINT")
    if not endpoint:
        return None, None
    region = _secret("AWS_DEFAULT_REGION", "us-east-1")
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        aws_access_key_id=_secret("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_secret("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=_secret("AWS_SESSION_TOKEN"),
    )
    return runtime, endpoint


@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    return {
        "holdout_metrics": {"roc_auc": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0}
    }


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


def endpoint_score(frame: pd.DataFrame, threshold: float):
    runtime, endpoint = load_runtime_client()
    if runtime is None:
        raise RuntimeError("SageMaker endpoint config missing in Streamlit secrets.")
    payload = frame.to_dict(orient="records")
    resp = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(resp["Body"].read().decode("utf-8"))
    scores = np.array(body if isinstance(body, list) else body.get("scores", []), dtype=float)

    scored = frame.copy()
    scored["fraud_score"] = scores
    scored["flag_for_review"] = scores >= threshold
    return scored


def local_score(frame: pd.DataFrame, model_bundle: dict, threshold: float):
    feature_columns = model_bundle["feature_columns"]
    model = model_bundle["model"]
    enriched = add_features(frame)
    X = enriched.reindex(columns=feature_columns)
    scores = model.predict_proba(X)[:, 1]
    scored = frame.copy()
    scored["fraud_score"] = scores
    scored["flag_for_review"] = scores >= threshold
    return scored


st.set_page_config(page_title="IEEE Fraud Detection", layout="wide")

summary = load_summary()
top_features = load_top_features()

bundle = load_model_bundle()
if USE_SAGEMAKER:
    st.success("Using SageMaker endpoint scoring.")
elif bundle is not None:
    st.info("Using local model scoring.")
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

default_threshold = float(bundle["threshold"]) if bundle is not None and "threshold" in bundle else 0.5
threshold = st.sidebar.slider("Fraud alert threshold", 0.05, 0.90, default_threshold, 0.05)

uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
if uploaded is not None:
    input_df = pd.read_csv(uploaded)
    label = "Uploaded transactions"
else:
    input_df = load_professor_sample()
    label = "Sample from professor test transactions"

if input_df.empty:
    st.warning("No input rows found. Upload a CSV or add `test_transaction.csv` to app data.")
    st.stop()

if USE_SAGEMAKER:
    scored = endpoint_score(input_df, threshold)
else:
    scored = local_score(input_df, bundle, threshold)

left, right = st.columns([2, 1])
with left:
    st.subheader(label)
    st.dataframe(scored.sort_values("fraud_score", ascending=False).head(30), use_container_width=True, hide_index=True)

with right:
    st.subheader("Review Queue")
    st.metric("Transactions scored", f"{len(scored):,}")
    st.metric("Flagged", f"{int(scored['flag_for_review'].sum()):,}")
    st.metric("Alert rate", f"{scored['flag_for_review'].mean():.1%}")

st.subheader("Top Model Drivers")
st.dataframe(top_features, use_container_width=True, hide_index=True)

st.subheader("SHAP Explanation")
if SHAP_WATERFALL_PATH.exists():
    st.image(str(SHAP_WATERFALL_PATH), caption="Local SHAP waterfall plot")
if SHAP_BAR_PATH.exists():
    st.image(str(SHAP_BAR_PATH), caption="Global SHAP feature importance")
if not SHAP_WATERFALL_PATH.exists() and not SHAP_BAR_PATH.exists():
    st.info("SHAP images not found; showing fallback plot.")
    if not top_features.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_data = top_features.head(10).sort_values("importance_mean")
        ax.barh(plot_data["feature"], plot_data["importance_mean"], color="#2f6f73")
        ax.set_xlabel("Permutation importance")
        ax.set_title("Fallback explanation: top features")
        st.pyplot(fig)

st.subheader("Batch Predictions File")
st.write(f"Saved professor test predictions: `{PREDICTIONS_PATH}`")
