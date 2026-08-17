# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Anomali Tespiti ve Fiyat Modelleme
Bu modül, iki ML modeli tanımlar:
  1. PricePredictor  : Araç özelliklerine göre beklenen piyasa fiyatını tahmin eder.
  2. AnomalyDetector : İlanların olağandışılık skorunu hesaplar (IsolationForest).
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from src.data_cleaning import PRICE_FEATURE_COLUMNS, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Dizin Sabitleri
# ---------------------------------------------------------------------------

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(_BASE_DIR, "models")

PRICE_MODEL_PATH    = os.path.join(MODELS_DIR, "price_predictor.joblib")
ANOMALY_MODEL_PATH  = os.path.join(MODELS_DIR, "anomaly_detector.joblib")


# ---------------------------------------------------------------------------
# Yardımcı
# ---------------------------------------------------------------------------

def _ensure_models_dir():
    """models/ dizini yoksa oluşturur."""
    os.makedirs(MODELS_DIR, exist_ok=True)


# ===========================================================================
# 1. Fiyat Tahmin Modeli
# ===========================================================================

def build_price_predictor() -> Pipeline:
    """
    Fiyat tahmini için sklearn Pipeline oluşturur.

    Pipeline adımları
    -----------------
    imputer  : Eksik değerleri medyan ile doldurur.
    regressor: HistGradientBoostingRegressor (NaN toleranslı, hızlı).
    """
    pipeline = Pipeline([
        ("imputer",   SimpleImputer(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])
    return pipeline


def train_price_predictor(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Fiyat tahmincisini eğitir ve döndürür.

    Parametreler
    ------------
    X_train : pd.DataFrame  — PRICE_FEATURE_COLUMNS özellik matrisi
    y_train : pd.Series     — Gerçek fiyat değerleri (log1p dönüştürülmüş)
    """
    model = build_price_predictor()
    model.fit(X_train[PRICE_FEATURE_COLUMNS], y_train)
    return model


def save_price_predictor(model: Pipeline):
    """Eğitilmiş fiyat tahmincisini diske kaydeder."""
    _ensure_models_dir()
    joblib.dump(model, PRICE_MODEL_PATH)
    print(f"[PricePredictor] Kaydedildi -> {PRICE_MODEL_PATH}")


def load_price_predictor() -> Pipeline:
    """Daha önce kaydedilmiş fiyat tahmincisini yükler."""
    if not os.path.exists(PRICE_MODEL_PATH):
        raise FileNotFoundError(
            f"Fiyat tahmin modeli bulunamadı: {PRICE_MODEL_PATH}\n"
            "Lütfen önce 'python src/train_models.py' çalıştırın."
        )
    return joblib.load(PRICE_MODEL_PATH)


def predict_price(model: Pipeline, features: dict) -> float:
    """
    Tek bir ilan için beklenen piyasa fiyatını tahmin eder.

    Dönen Değer : float — Tahmin edilen USD fiyatı
    """
    X = pd.DataFrame([features])[PRICE_FEATURE_COLUMNS]
    log_price = model.predict(X)[0]
    # Eğitimde log1p uygulandığından tersini alıyoruz
    return float(np.expm1(log_price))


# ===========================================================================
# 2. Anomali Tespit Modeli
# ===========================================================================

def build_anomaly_detector(contamination: float = 0.05) -> Pipeline:
    """
    Anomali tespiti için sklearn Pipeline oluşturur.

    Pipeline adımları
    -----------------
    imputer  : Eksik değerleri medyan ile doldurur.
    scaler   : StandardScaler ile normalize eder.
    detector : IsolationForest — negatif skorlar daha anormaldir.

    Parametreler
    ------------
    contamination : float
        Veri setinde beklenen anormal ilan oranı (varsayılan %5).
    """
    pipeline = Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("scaler",   StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    return pipeline


def train_anomaly_detector(X_train: pd.DataFrame,
                           contamination: float = 0.05) -> Pipeline:
    """
    Anomali dedektörünü eğitir ve döndürür.

    Parametreler
    ------------
    X_train       : pd.DataFrame — FEATURE_COLUMNS özellik matrisi
    contamination : float        — Beklenen anormallik oranı
    """
    model = build_anomaly_detector(contamination=contamination)
    model.fit(X_train[FEATURE_COLUMNS])
    return model


def save_anomaly_detector(model: Pipeline):
    """Eğitilmiş anomali dedektörünü diske kaydeder."""
    _ensure_models_dir()
    joblib.dump(model, ANOMALY_MODEL_PATH)
    print(f"[AnomalyDetector] Kaydedildi -> {ANOMALY_MODEL_PATH}")


def load_anomaly_detector() -> Pipeline:
    """Daha önce kaydedilmiş anomali dedektörünü yükler."""
    if not os.path.exists(ANOMALY_MODEL_PATH):
        raise FileNotFoundError(
            f"Anomali tespit modeli bulunamadı: {ANOMALY_MODEL_PATH}\n"
            "Lütfen önce 'python src/train_models.py' çalıştırın."
        )
    return joblib.load(ANOMALY_MODEL_PATH)


def get_anomaly_score(model: Pipeline, features: dict) -> float:
    """
    Tek bir ilan için anomali skoru döndürür.

    Dönen Değer
    -----------
    float
        IsolationForest'in ham karar skoru (score_samples).
        Daha negatif değer → daha anormal.
        Normalize edilmiş [0, 1] aralığına çevrilir:
          0 = tamamen normal, 1 = tamamen anormal.
    """
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    raw_score = model.named_steps["detector"].score_samples(
        model.named_steps["scaler"].transform(
            model.named_steps["imputer"].transform(X)
        )
    )[0]
    # IsolationForest skoru genellikle [-0.5, 0.5] aralığındadır.
    # [0, 1] → yüksek = daha anormal şeklinde normalize ediyoruz.
    normalized = float(np.clip((-raw_score - 0.0) / 0.5, 0.0, 1.0))
    return round(normalized, 4)
