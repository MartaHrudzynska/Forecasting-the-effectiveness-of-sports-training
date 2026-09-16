"""
pipeline.py — Головний оркестратор системи
Реалізує повний ланцюжок A1 → A2 → A3 → A4
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.a1_data_intake.loader  import load_csv, save_processed, load_processed
from src.a2_metrics.metrics     import compute_all_metrics
from src.a3_analysis.features   import (
    build_feature_matrix, get_regression_data,
    get_classification_data, get_clustering_data,
)
from src.a3_analysis.regression     import train_regression_models, predict_next_steps
from src.a3_analysis.classification import train_classification_models, predict_readiness
from src.a3_analysis.clustering     import train_clustering, predict_cluster
from src.a4_recommendations.recommender import generate_recommendation
from src.database.mongo_db           import get_db

MODELS_DIR = Path(__file__).parent / "models"


class FitnessPipeline:
    """
    Головний клас системи.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.regression_results:     Optional[Dict] = None
        self.classification_results: Optional[Dict] = None
        self.clustering_result:      Optional[Dict] = None
        self.is_trained: bool = False
        self.db = get_db()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Завантаження, валідація, очищення."""
        self.df = load_csv(filepath)
        if self.db.connected:
            saved = self.db.insert_daily_records(self.df)
            print(f"[Pipeline] MongoDB: збережено {saved} документів.")
        save_processed(self.df)
        return self.df

    def load_processed_data(self) -> pd.DataFrame:
        """Завантаження вже обробленого датасету."""
        self.df = load_processed()
        return self.df


    def compute_metrics(self) -> pd.DataFrame:
        """Розрахунок CTL, ATL, TSB, Readiness."""
        assert self.df is not None, "Спочатку завантажте дані через load_data()."
        self.df = compute_all_metrics(self.df)
        if self.db.connected:
            self.db.insert_daily_records(self.df)
        return self.df


    def build_features(self) -> pd.DataFrame:
        """Конструювання часових ознак, лагів, цільових змінних."""
        assert self.df is not None
        self.df = build_feature_matrix(self.df)
        return self.df

    def train_all_models(self) -> Dict:
        """Навчання всіх трьох типів моделей."""
        assert self.df is not None

        print("\n" + "="*60)
        print("Навчання регресійних моделей (Steps Forecast)")
        print("="*60)
        X_reg, y_reg = get_regression_data(self.df)
        self.regression_results = train_regression_models(X_reg, y_reg)

        print("\n" + "="*60)
        print("Навчання класифікаційних моделей (Readiness)")
        print("="*60)
        X_clf, y_clf = get_classification_data(self.df)
        self.classification_results = train_classification_models(X_clf, y_clf)

        print("\n" + "="*60)
        print("Кластеризація тренувальної поведінки")
        print("="*60)
        profile_df = get_clustering_data(self.df)
        self.clustering_result = train_clustering(profile_df)

        cluster_map = self.clustering_result["result_df"].set_index("user_id")["cluster_name"].to_dict()
        self.df["cluster_name"] = self.df["user_id"].map(cluster_map).fillna("Невідомо")
        if self.db.connected:
            self.db.insert_daily_records(self.df)
        self.is_trained = True
        print("\nНавчання всіх моделей завершено.")
        return {
            "regression":     self.regression_results,
            "classification": self.classification_results,
            "clustering":     self.clustering_result,
        }


    def predict_for_user(self, user_id) -> Dict:
        """
        Повний цикл прогнозування та рекомендацій для одного користувача.
        """
        assert self.df is not None and self.is_trained, \
            "Спочатку виконайте run_full_pipeline()."

        user_df = self.df[self.df["user_id"] == user_id].sort_values("date")
        if user_df.empty:
            return {"error": f"Користувача {user_id} не знайдено."}

        last_row = user_df.iloc[-1]

        reg_pred = predict_next_steps(
            last_row,
            self.regression_results["features"],
            self.regression_results["best_model"],
            self.regression_results["scaler"],
        )

        clf_pred = predict_readiness(
            last_row,
            self.classification_results["features"],
            self.classification_results["best_model"],
            self.classification_results["scaler"],
        )

        cluster_name = last_row.get("cluster_name", "Невідомо")

        rec = generate_recommendation(
            tsb=float(last_row.get("tsb", 0)),
            readiness_level=clf_pred["readiness_level"],
            predicted_steps=reg_pred["predicted_steps"],
            cluster_name=cluster_name,
        )
        if self.db.connected:
            self.db.insert_prediction(user_id, reg_pred, clf_pred, cluster_name)
            self.db.insert_recommendation(rec, user_id)

        return {
            "user_id":        user_id,
            "last_date":      str(last_row["date"]),
            "regression":     reg_pred,
            "classification": clf_pred,
            "cluster":        cluster_name,
            "recommendation": rec,
            "current_tsb":    float(last_row.get("tsb", 0)),
            "current_ctl":    float(last_row.get("ctl", 0)),
            "current_atl":    float(last_row.get("atl", 0)),
            "readiness_level": last_row.get("readiness_level", "N/A"),
        }
    def run_full_pipeline(self, filepath: str) -> Dict:
        print("\nЗапуск повного пайплайну системи\n")
        if self.db.connected:
            print(f"[Pipeline] MongoDB активна: {self.db.db_name}")
        else:
            print("[Pipeline] MongoDB недоступна — файловий режим (CSV).")
        self.load_data(filepath)
        self.compute_metrics()
        self.build_features()
        results = self.train_all_models()
        print("\nПайплайн завершено успішно.")
        return results
    def get_user_list(self):
        if self.df is None:
            return []
        return sorted(self.df["user_id"].unique().tolist())
    def get_user_dataframe(self, user_id) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        return self.df[self.df["user_id"] == user_id].sort_values("date")
    def get_db_stats(self) -> Dict:
        """Статистика колекцій MongoDB."""
        return self.db.get_collection_stats()
