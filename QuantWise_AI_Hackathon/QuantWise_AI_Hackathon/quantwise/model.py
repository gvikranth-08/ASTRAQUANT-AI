import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .features import FEATURE_COLUMNS

def train_and_predict(data: pd.DataFrame, confidence_threshold: float):
    """Chronological split with a small purge gap to reduce label leakage."""
    split = int(len(data) * 0.75)
    purge = 3
    train = data.iloc[:max(0, split-purge)].copy()
    test = data.iloc[split:].copy()
    if len(train) < 50 or len(test) < 20:
        raise ValueError("Not enough rows for a reliable chronological train/test split.")

    X_train, y_train = train[FEATURE_COLUMNS], train["Target"]
    X_test, y_test = test[FEATURE_COLUMNS], test["Target"]
    model = RandomForestClassifier(
        n_estimators=250, max_depth=7, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    test["Raw_Signal"] = predictions.astype(int)
    test["Confidence"] = confidence
    test["Filtered_Signal"] = test["Raw_Signal"].where(test["Confidence"] >= confidence_threshold, 0)
    accuracy = accuracy_score(y_test, predictions)
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })
    return model, test, accuracy, importance
