"""Model training with MLflow tracking."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import mlflow

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    validation_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Try to import cuML for GPU-accelerated sklearn models
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


class ModelType(str, Enum):
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NET = "neural_net"


@dataclass
class TrainConfig:
    """Training configuration."""

    experiment_name: str = "fraud-detection"
    model_type: ModelType = ModelType.XGBOOST
    use_class_weight: bool = True
    random_state: int = 42
    cv_folds: int = 5
    use_gpu: bool = True  # Use GPU for XGBoost if available


class Trainer:
    """Handles model training with MLflow tracking."""

    PARAM_GRIDS = {
        ModelType.LOGISTIC: {
            "C": [0.01, 0.1, 1, 10],  # Inverse regularization strength
            "penalty": ["l1", "l2"],
            "solver": ["saga"],  # Supports both L1 and L2
            "class_weight": ["balanced", None],
        },
        ModelType.RANDOM_FOREST: {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],  # Regularization
            "min_samples_leaf": [1, 2, 4],    # Regularization
            "class_weight": ["balanced", None],
        },
        ModelType.XGBOOST: {
            "n_estimators": [100, 200],
            "max_depth": [5, 7],
            "learning_rate": [0.1, 0.2],
            "scale_pos_weight": [10, 50],
            # Regularization
            "reg_alpha": [0, 0.1, 1],       # L1
            "reg_lambda": [1, 2, 5],         # L2
            "min_child_weight": [1, 3, 5],
        },
        ModelType.NEURAL_NET: {
            "classifier__hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
            "classifier__alpha": [0.0001, 0.001, 0.01],  # L2 regularization
            "classifier__learning_rate_init": [0.001, 0.01],
        },
    }

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.model = None
        self.run_id: str | None = None

    def get_model(self, **kwargs) -> Any:
        """Create model instance. Tries GPU first, falls back to CPU on failure."""
        class_weight = "balanced" if self.config.use_class_weight else None

        if self.config.model_type == ModelType.LOGISTIC:
            if self.config.use_gpu and CUML_AVAILABLE:
                try:
                    return cuLR(max_iter=1000, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU LogisticRegression failed: {e}. Falling back to CPU.")
            return LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=self.config.random_state,
                **kwargs,
            )

        elif self.config.model_type == ModelType.RANDOM_FOREST:
            if self.config.use_gpu and CUML_AVAILABLE:
                try:
                    return cuRF(n_estimators=100, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU RandomForest failed: {e}. Falling back to CPU.")
            return RandomForestClassifier(
                class_weight=class_weight,
                n_estimators=100,
                random_state=self.config.random_state,
                **kwargs,
            )

        elif self.config.model_type == ModelType.XGBOOST:
            scale_pos_weight = kwargs.pop("scale_pos_weight", 50)
            if self.config.use_gpu:
                try:
                    return XGBClassifier(
                        n_estimators=100,
                        random_state=self.config.random_state,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        device="cuda",
                        tree_method="hist",
                        **kwargs,
                    )
                except Exception as e:
                    logger.warning(f"GPU XGBoost failed: {e}. Falling back to CPU.")
            return XGBClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                **kwargs,
            )

        elif self.config.model_type == ModelType.NEURAL_NET:
            # sklearn MLP doesn't support GPU, would need PyTorch for GPU
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    random_state=self.config.random_state,
                    **kwargs,
                )),
            ])

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, scoring: str = "f1"
    ) -> dict:
        """Run stratified k-fold cross-validation."""
        model = self.get_model()
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        return {"mean": scores.mean(), "std": scores.std(), "scores": scores.tolist()}

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict | None = None,
        scoring: str = "f1",
    ) -> dict:
        """Grid search for best hyperparameters."""
        param_grid = param_grid or self.PARAM_GRIDS.get(self.config.model_type, {})
        model = self.get_model()

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(X, y)

        self.model = search.best_estimator_
        return {"best_params": search.best_params_, "best_score": search.best_score_}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        run_name: str | None = None,
        **model_kwargs,
    ) -> str:
        """Train model and log to MLflow."""
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=run_name):
            if self.model is None:
                self.model = self.get_model(**model_kwargs)

            # Log params (handle pipeline case)
            params = (
                self.model.get_params()
                if not isinstance(self.model, Pipeline)
                else self.model.named_steps["classifier"].get_params()
            )
            mlflow.log_params({
                "model_type": self.config.model_type.value,
                "use_class_weight": self.config.use_class_weight,
                **{k: v for k, v in params.items() if not callable(v)},
            })

            self.model.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.model, "model")

            self.run_id = mlflow.active_run().info.run_id
            return self.run_id

    @staticmethod
    def compare_models(
        X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, scoring: str = "f1"
    ) -> pd.DataFrame:
        """Compare all models using cross-validation."""
        results = []

        for model_type in ModelType:
            config = TrainConfig(model_type=model_type, cv_folds=cv_folds)
            trainer = Trainer(config)
            cv_result = trainer.cross_validate(X, y, scoring=scoring)

            results.append({
                "model": model_type.value,
                "mean_score": cv_result["mean"],
                "std_score": cv_result["std"],
            })

        return pd.DataFrame(results).sort_values("mean_score", ascending=False)

    def get_learning_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_sizes: np.ndarray | None = None,
        scoring: str = "f1",
    ) -> dict:
        """Generate learning curve data for overfitting/underfitting analysis.

        Returns train/validation scores at different training set sizes.
        - Underfitting: both scores low
        - Overfitting: train high, validation low (big gap)
        - Good fit: both scores high and close together
        """
        model = self.get_model()
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_mean": train_scores.mean(axis=1).tolist(),
            "train_std": train_scores.std(axis=1).tolist(),
            "val_mean": val_scores.mean(axis=1).tolist(),
            "val_std": val_scores.std(axis=1).tolist(),
        }

    def get_validation_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_name: str,
        param_range: list,
        scoring: str = "f1",
    ) -> dict:
        """Generate validation curve for a specific hyperparameter.

        Shows how train/validation score changes with hyperparameter value.
        Helps find optimal value before overfitting.
        """
        model = self.get_model()
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        return {
            "param_range": param_range,
            "train_mean": train_scores.mean(axis=1).tolist(),
            "train_std": train_scores.std(axis=1).tolist(),
            "val_mean": val_scores.mean(axis=1).tolist(),
            "val_std": val_scores.std(axis=1).tolist(),
        }
