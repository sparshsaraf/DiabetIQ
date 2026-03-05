import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_export():

    # Load dataset
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    # Features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Save feature names for inference
    feature_names = X.columns.tolist()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Export artifacts
    joblib.dump(model, "saved_models/model.pkl")
    joblib.dump(scaler, "saved_models/scaler.pkl")
    joblib.dump(feature_names, "saved_models/features.pkl")

    print("✔ Model, scaler & features saved successfully.")

if __name__ == "__main__":
    train_and_export()
