import pandas as pd

def prepare_input(input_dict, selected_features, scaler):
    """
    input_dict: dict of raw user inputs
    selected_features: list from features.pkl
    scaler: fitted StandardScaler
    """
    df = pd.DataFrame([input_dict])

    # Ensure all expected columns exist
    for feat in selected_features:
        if feat not in df.columns:
            df[feat] = 0.0

    # Reorder
    df = df[selected_features]

    # Scale
    X = scaler.transform(df.values)
    return X
