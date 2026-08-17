# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - ML Çıkarım Modülü

Bu modül; eğitilmiş modelleri yükleyerek tek bir ilan sözlüğü için
ML tabanlı risk analizi yapar ve kural tabanlı modelle uyumlu bir
çıktı formatında sonuç döndürür.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from src.data_cleaning import (
    preprocess_listing,
    FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
)
from src.anomaly_detection import (
    load_price_predictor,
    load_anomaly_detector,
    MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Güven Skoru Regresyon Modeli Yolu
# ---------------------------------------------------------------------------

TRUST_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, "trust_score_regressor.joblib")

# ---------------------------------------------------------------------------
# Model Önbellekleme (her çağrıda disk I/O'dan kaçın)
# ---------------------------------------------------------------------------

_price_model  = None
_anomaly_model = None
_trust_model  = None


def _get_models():
    """Modelleri gerektiğinde yükler ve önbelleğe alır."""
    global _price_model, _anomaly_model, _trust_model

    if _price_model is None:
        _price_model = load_price_predictor()

    if _anomaly_model is None:
        _anomaly_model = load_anomaly_detector()

    if _trust_model is None:
        if not os.path.exists(TRUST_SCORE_MODEL_PATH):
            raise FileNotFoundError(
                f"Güven skoru modeli bulunamadı: {TRUST_SCORE_MODEL_PATH}\n"
                "Lütfen önce 'python src/train_models.py' çalıştırın."
            )
        _trust_model = joblib.load(TRUST_SCORE_MODEL_PATH)

    return _price_model, _anomaly_model, _trust_model


def models_available() -> bool:
    """
    Tüm model dosyalarının mevcut olup olmadığını kontrol eder.
    Streamlit sidebar'daki ML toggle'ını devre dışı bırakmak için kullanılır.
    """
    paths = [
        os.path.join(MODELS_DIR, "price_predictor.joblib"),
        os.path.join(MODELS_DIR, "anomaly_detector.joblib"),
        TRUST_SCORE_MODEL_PATH,
    ]
    return all(os.path.exists(p) for p in paths)


# ---------------------------------------------------------------------------
# Risk Seviyesi Eşleme
# ---------------------------------------------------------------------------

def _score_to_level(score: float) -> str:
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Medium Risk"
    elif score >= 40:
        return "High Risk"
    return "Very High Risk"


# ---------------------------------------------------------------------------
# Ana ML Analiz Fonksiyonu
# ---------------------------------------------------------------------------

def analyze_listing_ml(listing: dict) -> dict:
    """
    Tek bir ilan sözlüğünü ML modelleri ile analiz eder.

    Parametreler
    ------------
    listing : dict
        Ham ilan verisi (main.py veya app.py'den gelen sözlük).

    Dönen Değer
    -----------
    dict
        {
          "trust_score"     : int       — 0-100 arası güven skoru,
          "risk_level"      : str       — Low / Medium / High / Very High Risk,
          "risk_reasons"    : list[str] — ML tarafından üretilen risk nedenleri,
          "recommendations" : list[str] — Alıcı tavsiyeleri,
          "predicted_price" : float     — ML'in tahmin ettiği piyasa değeri ($),
          "price_deviation" : float     — (gerçek - tahmin) / tahmin oranı,
          "anomaly_score"   : float     — [0,1] normalize edilmiş anomali skoru,
        }
    """
    price_model, anomaly_model, trust_model = _get_models()

    # 1. Özellik mühendisliği
    features = preprocess_listing(listing)

    # 2. Fiyat tahmini
    X_price = pd.DataFrame([features])[PRICE_FEATURE_COLUMNS]
    log_pred   = price_model.predict(X_price)[0]
    pred_price = float(np.expm1(log_pred))

    actual_price = features["price"]
    if pred_price > 0:
        price_dev = (actual_price - pred_price) / pred_price
    else:
        price_dev = 0.0
    price_dev = float(np.clip(price_dev, -2.0, 2.0))

    # 3. Anomali skoru
    X_all = pd.DataFrame([features])[FEATURE_COLUMNS]
    raw_score = anomaly_model.named_steps["detector"].score_samples(
        anomaly_model.named_steps["scaler"].transform(
            anomaly_model.named_steps["imputer"].transform(X_all)
        )
    )[0]
    anomaly_score = float(np.clip((-raw_score) / 0.5, 0.0, 1.0))

    # 4. Güven skoru tahmini
    X_ext = pd.DataFrame([{
        **features,
        "price_deviation": price_dev,
        "anomaly_score":   anomaly_score,
    }])
    EXT_FEATURES = FEATURE_COLUMNS + ["price_deviation", "anomaly_score"]
    raw_trust = trust_model.predict(X_ext[EXT_FEATURES])[0]
    trust_score = int(np.clip(round(raw_trust), 0, 100))

    # 5. Risk nedenleri (ML tarafından yorumlanan)
    risk_reasons   = []
    recommendations = []

    # --- Fiyat sapması ---
    if price_dev < -0.40:
        pct = abs(price_dev) * 100
        risk_reasons.append(
            f"ML Fiyat Sapması: İlan fiyatı (${actual_price:,.0f}), "
            f"beklenen piyasa değerinin (${pred_price:,.0f}) "
            f"%{pct:.0f} altında."
        )
        recommendations.append(
            "Bu fiyat, ML modeline göre piyasa değerinin çok altındadır. "
            "Kaparo dolandırıcılığı veya gizli hasar riski taşıyabilir. "
            "Aracı fiziksel olarak incelemeden kesinlikle ödeme yapmayın."
        )
    elif price_dev > 0.40:
        pct = price_dev * 100
        risk_reasons.append(
            f"ML Fiyat Sapması: İlan fiyatı (${actual_price:,.0f}), "
            f"beklenen piyasa değerinin (${pred_price:,.0f}) "
            f"%{pct:.0f} üzerinde."
        )
        recommendations.append(
            "Araç piyasa değerinin üzerinde fiyatlandırılmış. "
            "Özel donanım veya nadirlik gibi gerekçeleri satıcıdan açıklamasını isteyin."
        )

    # --- Anomali tespiti ---
    if anomaly_score > 0.65:
        risk_reasons.append(
            f"ML Anomali Tespiti: Araç özellikleri olağandışı bir kombinasyon "
            f"sergilmektedir (Anomali Skoru: {anomaly_score:.2f}/1.00)."
        )
        recommendations.append(
            "ML modeli bu ilanı veri setindeki benzerlerinden belirgin şekilde "
            "farklı buluyor. Tüm belgeleri dikkatlice kontrol edin."
        )
    elif anomaly_score > 0.45:
        risk_reasons.append(
            f"ML Anomali Tespiti: Araç bazı sıradışı özellikler içeriyor "
            f"(Anomali Skoru: {anomaly_score:.2f}/1.00)."
        )
        recommendations.append(
            "İlanın detaylarını benzer araçlarla karşılaştırın; "
            "tutarsızlık varsa satıcıyı bilgilendirin."
        )

    # --- Düşük güven skoru ---
    if trust_score < 40 and not risk_reasons:
        risk_reasons.append(
            "ML modeli bu ilanı genel olarak düşük güvenilirlikli buluyor."
        )
        recommendations.append(
            "Lütfen araç belgelerini ve satıcı bilgilerini titizlikle inceleyin."
        )

    risk_level = _score_to_level(trust_score)

    return {
        "trust_score":     trust_score,
        "risk_level":      risk_level,
        "risk_reasons":    risk_reasons,
        "recommendations": recommendations,
        "predicted_price": round(pred_price, 2),
        "price_deviation": round(price_dev, 4),
        "anomaly_score":   round(anomaly_score, 4),
    }


# ---------------------------------------------------------------------------
# Test Bloğu
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not models_available():
        print("Modeller henüz eğitilmedi.")
        print("Lütfen önce: python src/train_models.py")
        sys.exit(1)

    test_cases = [
        {
            "name": "Temiz Araç",
            "listing": {
                "VIN": "1FTFW1EF5FAXXXXXX", "title_status": "clean",
                "price": 18500, "odometer": 85000, "year": 2018,
                "description": "2018 model kazasız bakımlı aile arabası.",
                "image_url": "http://example.com/car.jpg",
                "condition": "excellent", "fuel": "gas",
                "transmission": "automatic", "drive": "4wd",
            },
        },
        {
            "name": "Şüpheli Ucuz Araç",
            "listing": {
                "VIN": None, "title_status": "salvage",
                "price": 300, "odometer": 350000, "year": 2005,
                "description": "Satılık.", "image_url": "",
                "condition": "", "fuel": "gas",
                "transmission": "automatic", "drive": "fwd",
            },
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*55}")
        print(f"  {tc['name']}")
        print("=" * 55)
        result = analyze_listing_ml(tc["listing"])
        for k, v in result.items():
            print(f"  {k:20s}: {v}")
