import os
import json
import numpy as np
import pandas as pd


def convert_input_pca_regression(request_body, request_content_type):
    """
    Convert a small JSON payload from Streamlit into a full feature row
    matching the KPCA regression model's expected input schema.

    Expected JSON input example:
    {
        "AOS_CR_Cum": 10.0,
        "AFL_CR_Cum": -5.0
    }

    Returns:
        pandas.DataFrame with one row and all model feature columns
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    user_data = json.loads(request_body)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    file_path = os.path.join(project_root, "Portfolio", "SP500Data.csv")

    dataset = pd.read_csv(file_path, index_col=0)

    # Match your notebook target and feature engineering
    target = "NVDA"
    return_period = 5

    # Rebuild X exactly like training
    X = np.log(dataset.drop([target], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + "_CR_Cum" for name in X.columns]
    X = X.dropna()

    # Streamlit user-controlled features
    feature_1 = "AOS_CR_Cum"
    feature_2 = "AFL_CR_Cum"

    if feature_1 not in user_data or feature_2 not in user_data:
        raise ValueError(f"JSON payload must include '{feature_1}' and '{feature_2}'")

    val_1 = float(user_data[feature_1])
    val_2 = float(user_data[feature_2])

    # Find nearest historical row so the full feature vector is realistic
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
