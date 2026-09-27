import os
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

class EntityMatcher:
    def __init__(self, model_type="lightgbm", calibrate=False, **kwargs):
        self.model_type = model_type
        self.calibrate = calibrate
        
        if model_type == "lightgbm":
            base_model = LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 300),
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_depth=kwargs.get("max_depth", 8),
                num_leaves=kwargs.get("num_leaves", 31),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
            )
        elif model_type == "xgboost":
            base_model = XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 300),
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_depth=kwargs.get("max_depth", 6),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
                eval_metric="logloss",
            )
        elif model_type == "hist_gb":
            base_model = HistGradientBoostingClassifier(
                max_iter=kwargs.get("n_estimators", 200),
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_depth=kwargs.get("max_depth", 8),
                random_state=kwargs.get("random_state", 42),
            )
        elif model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 150),
                max_depth=kwargs.get("max_depth", 15),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
            )
        elif model_type == "logistic":
            base_model = LogisticRegression(
                max_iter=1000,
                C=kwargs.get("C", 1.0),
                random_state=kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if calibrate:
            self.model = CalibratedClassifierCV(estimator=base_model, cv=3, method="sigmoid")
        else:
            self.model = base_model

    def fit(self, X, y):
        calib_str = " (with calibration)" if self.calibrate else ""
        print(f"Training {self.model_type}{calib_str} on {X.shape[0]:,} samples ({X.shape[1]} features)...", flush=True)
        self.model.fit(X, y)
        print("Model training complete.", flush=True)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"model_type": self.model_type, "calibrate": self.calibrate, "model": self.model}, f)
        print(f"Model saved successfully to {filepath}", flush=True)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(model_type=data["model_type"], calibrate=data.get("calibrate", False))
        obj.model = data["model"]
        print(f"Model loaded successfully from {filepath}", flush=True)
        return obj
