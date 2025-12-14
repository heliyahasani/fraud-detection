"""Feature engineering pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    night_start: int = 22
    night_end: int = 6
    n_risk_quantiles: int = 3
    target: str = "is_fraud"


class FeatureEngineer:
    """Transforms raw transactions into model features."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.risk_maps: dict[str, dict] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit risk mappings and transform data."""
        df = self._add_time_features(df)
        df = self._add_amount_features(df)
        df = self._add_category_features(df)
        df = self._add_customer_features(df)
        df = self._add_geo_features(df)

        for col in ["category", "state", "merchant"]:
            df = self._fit_risk_labels(df, col)

        return self._encode(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted risk mappings."""
        if not self.risk_maps:
            raise ValueError("Call fit_transform first")

        df = self._add_time_features(df)
        df = self._add_amount_features(df)
        df = self._add_category_features(df)
        df = self._add_customer_features(df)
        df = self._add_geo_features(df)

        for col, risk_map in self.risk_maps.items():
            df[f"{col}_risk"] = df[col].map(risk_map).fillna("medium")

        return self._encode(df)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["month"] = df["trans_date_trans_time"].dt.month
        df["is_night"] = (
            (df["hour"] >= self.config.night_start)
            | (df["hour"] <= self.config.night_end)
        ).astype(int)
        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_amt"] = np.log1p(df["amt"])
        return df

    def _add_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_online"] = df["category"].str.endswith("_net").astype(int)
        return df

    def _add_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
        return df

    def _add_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
        )
        return df

    def _fit_risk_labels(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Fit risk labels. Low-count entities get 'medium'."""
        df = df.copy()

        stats = df.groupby(column).agg(
            fraud_rate=(self.config.target, "mean"),
            count=(self.config.target, "count"),
        )

        median_count = stats["count"].median()
        low_count = stats["count"] < median_count

        reliable = stats[~low_count]
        labels = pd.qcut(
            reliable["fraud_rate"],
            q=self.config.n_risk_quantiles,
            labels=["low", "medium", "high"],
        )

        risk_map = labels.to_dict()
        for entity in stats[low_count].index:
            risk_map[entity] = "medium"

        self.risk_maps[column] = risk_map
        df[f"{column}_risk"] = df[column].map(risk_map)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(
            df,
            columns=["category_risk", "state_risk", "merchant_risk", "gender"],
            prefix=["cat_risk", "state_risk", "merch_risk", "gender"],
            drop_first=True,
        )
