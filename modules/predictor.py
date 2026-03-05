def predict_from_array(model):
    def _predict(X):
        label = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        return label, proba
    return _predict
