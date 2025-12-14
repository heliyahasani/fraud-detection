"""Data loading and preprocessing utilities."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


@dataclass
class DataLoader:
    """Handles data loading, splitting, and resampling for imbalanced data."""

    target: str = "is_fraud"
    test_size: float = 0.2
    random_state: int = 42
    datetime_cols: list[str] = field(default_factory=lambda: ["trans_date_trans_time", "dob"])

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load CSV and parse datetime columns."""
        df = pd.read_csv(path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        for col in self.datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        return df

    def load_parquet(self, path: str | Path) -> pd.DataFrame:
        """Load parquet file."""
        return pd.read_parquet(path)

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split into train/test with stratification for imbalanced data."""
        X = df.drop(columns=[self.target])
        y = df[self.target]

        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def apply_smote(
        self, X: pd.DataFrame, y: pd.Series, sampling_strategy: float = 0.5
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to oversample minority class.

        Args:
            X: Features
            y: Target
            sampling_strategy: Ratio of minority to majority after resampling.
                0.5 means minority will be 50% of majority count.

        Returns:
            Resampled X, y
        """
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    def get_class_weights(self, y: pd.Series) -> dict[int, float]:
        """Calculate class weights for imbalanced data.

        Returns weights inversely proportional to class frequency.
        """
        counts = y.value_counts()
        total = len(y)

        return {
            cls: total / (len(counts) * count)
            for cls, count in counts.items()
        }
