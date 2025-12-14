"""Evaluation metrics for imbalanced classification."""

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    """Model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "threshold": self.threshold,
        }


class Evaluator:
    """Evaluates models with metrics suited for imbalanced data."""

    def __init__(self, model):
        self.model = model

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
        log_to_mlflow: bool = True,
    ) -> Metrics:
        """Compute metrics. PR-AUC is preferred over ROC-AUC for imbalanced data."""
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = Metrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_proba),
            pr_auc=average_precision_score(y_test, y_proba),
            threshold=threshold,
        )

        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metrics(metrics.to_dict())

        return metrics

    def find_best_threshold(
        self, X_val: pd.DataFrame, y_val: pd.Series, optimize_for: str = "f1"
    ) -> float:
        """Find threshold that maximizes chosen metric."""
        y_proba = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        if optimize_for == "f1":
            scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(scores)
        elif optimize_for == "recall":
            valid = precisions >= 0.5
            best_idx = np.where(valid)[0][-1] if valid.any() else 0
        else:
            raise ValueError(f"Unknown metric: {optimize_for}")

        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    def print_report(
        self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
    ):
        """Print classification report and confusion matrix."""
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        print(f"Threshold: {threshold:.3f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
