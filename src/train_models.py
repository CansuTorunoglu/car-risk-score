# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Model Eğitim Pipeline'ı

Kullanım:
    python src/train_models.py

Bu script:
  1. vehicles.csv'den 100.000 satır örneklem yükler.
  2. Kural tabanlı analyze_listing ile hedef trust_score üretir.
  3. Fiyat Tahmin Modeli'ni eğitir ve kaydeder.
  4. Anomali Tespit Modeli'ni eğitir ve kaydeder.
  5. Güven Skoru Regresyon Modeli'ni eğitir ve kaydeder.
  6. Her model için değerlendirme metriklerini yazdırır.
"""

import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)

warnings.filterwarnings("ignore")

# --- Proje kök dizinini path'e ekle ---
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from src.risk_rules       import analyze_listing
from src.data_cleaning    import preprocess_dataframe, FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS
from src.anomaly_detection import (
    train_price_predictor, save_price_predictor,
    train_anomaly_detector, save_anomaly_detector,
    MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

DATASET_PATH  = os.path.join(_BASE_DIR, "data", "cardata.csv", "vehicles.csv")
SAMPLE_SIZE   = 100_000
RANDOM_STATE  = 42
TRUST_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, "trust_score_regressor.joblib")
CLASSIFIER_MODEL_PATH  = os.path.join(MODELS_DIR, "risk_classifier.joblib")

# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def _banner(msg: str):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def _load_dataset(path: str, sample_size: int) -> pd.DataFrame:
    """
    Veri setini yükler. Eğer veri seti sample_size'dan büyükse
    rastgele bir alt küme seçer.
    """
    _banner("1. Veri Seti Yükleniyor")
    print(f"   Dosya : {path}")
    print(f"   Örneklem: {sample_size:,} satır")

    t0 = time.time()
    # skiprows ile büyük dosyayı tüm belleğe almadan örnekleriz
    total_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print(f"   Toplam satır sayısı: {total_rows:,}")

    skip_prob = max(0.0, 1.0 - sample_size / total_rows)
    skip_rows = range(1, total_rows + 1)
    skip_set  = set(
        np.random.default_rng(RANDOM_STATE)
        .choice(total_rows, size=int(total_rows * skip_prob), replace=False) + 1
    )

    df = pd.read_csv(path, skiprows=lambda i: i in skip_set, low_memory=False)
    df = df.sample(min(sample_size, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"   Yüklendi: {len(df):,} satır  ({time.time()-t0:.1f}s)")
    return df


def _compute_rule_scores(df: pd.DataFrame) -> pd.Series:
    """
    Her satır için kural tabanlı trust_score'u hesaplar.
    Bu, Güven Skoru Regresyonu'nun hedef değişkenidir.
    """
    _banner("2. Kural Tabanlı Trust Score Hesaplanıyor")
    t0 = time.time()
    scores = df.apply(lambda row: analyze_listing(row.to_dict())["trust_score"], axis=1)
    print(f"   Tamamlandi ({time.time()-t0:.1f}s)  --  "
          f"Ort: {scores.mean():.1f}  Min: {scores.min()}  Max: {scores.max()}")
    return scores


def _evaluate_regression(name: str, y_true, y_pred):
    """Regresyon modelinin MAE ve R² metriklerini yazdırır."""
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"   [{name}]  MAE: {mae:.2f}   R2: {r2:.4f}")


# ---------------------------------------------------------------------------
# Model 3: Guven Skoru Regresyonu  (Random Forest)
# ---------------------------------------------------------------------------

def build_trust_score_regressor() -> Pipeline:
    """
    Guven skoru regresyon pipeline'i.
    Cross-validation sonuclarina gore RandomForest daha iyi MAE verdi.
    """
    return Pipeline([
        ("imputer",   SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Model 4: Risk Sinif Classifier  (class_weight=balanced)
# ---------------------------------------------------------------------------

def score_to_class(score):
    if score >= 80: return 0
    elif score >= 60: return 1
    elif score >= 40: return 2
    return 3

CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

def build_risk_classifier() -> Pipeline:
    """
    Risk seviyesi siniflandirici.
    class_weight='balanced' ile Very High Risk gibi nadir siniflara
    daha fazla agirlik verir.
    """
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",   # nadir siniflara agirlik ver
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Ana Akış
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Veri Yükleme
    df = _load_dataset(DATASET_PATH, SAMPLE_SIZE)

    # 2. Kural Tabanlı Hedef Skor
    y_trust = _compute_rule_scores(df)

    # 3. Özellik Mühendisliği
    _banner("3. Özellik Mühendisliği")
    t0 = time.time()
    X = preprocess_dataframe(df)
    print(f"   Özellik boyutu: {X.shape}  ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 4. Fiyat Tahmin Modeli
    # -----------------------------------------------------------------------
    _banner("4. Fiyat Tahmin Modeli Eğitiliyor")

    # Eğitim: sadece makul fiyatları kullanalım (0 ve aşırı fiyatları filtrele)
    price_mask = (X["price"] > 200) & (X["price"] < 200_000)
    X_price = X[price_mask][PRICE_FEATURE_COLUMNS]
    y_price = np.log1p(X[price_mask]["price"])  # log1p → skewed dağılımı düzelt

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_price, y_price, test_size=0.2, random_state=RANDOM_STATE
    )

    price_model = train_price_predictor(
        pd.concat([Xp_train, pd.Series(name="price")], axis=1).iloc[:, :-1],
        yp_train,
    )
    # Not: train_price_predictor zaten PRICE_FEATURE_COLUMNS'u seçiyor.
    # Daha temiz şekilde doğrudan fit edelim:
    price_model = build_trust_score_regressor()   # yeniden başlat
    price_model.steps[-1] = ("regressor", HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, random_state=RANDOM_STATE,
    ))
    from sklearn.impute import SimpleImputer as _SI
    price_pipeline = Pipeline([
        ("imputer",   _SI(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, random_state=RANDOM_STATE,
        )),
    ])
    price_pipeline.fit(Xp_train, yp_train)
    yp_pred = price_pipeline.predict(Xp_test)
    _evaluate_regression("PricePredictor (log uzayı)", yp_test, yp_pred)
    # Gerçek fiyat uzayında da göster
    _evaluate_regression("PricePredictor (USD)",
                         np.expm1(yp_test), np.expm1(yp_pred))
    save_price_predictor(price_pipeline)

    # -----------------------------------------------------------------------
    # 5. Price Deviation Özelliği (tüm veri seti)
    # -----------------------------------------------------------------------
    _banner("5. Fiyat Sapması Özelliği Hesaplanıyor")
    t0 = time.time()
    log_pred_all = price_pipeline.predict(X[PRICE_FEATURE_COLUMNS])
    pred_price_all = np.expm1(log_pred_all)
    actual_price   = X["price"].values

    with np.errstate(divide="ignore", invalid="ignore"):
        price_deviation = np.where(
            pred_price_all > 0,
            (actual_price - pred_price_all) / pred_price_all,
            0.0,
        )
    price_deviation = np.clip(price_deviation, -2.0, 2.0)
    print(f"   Tamamlandi ({time.time()-t0:.1f}s)  --  "
          f"Ort sapma: {price_deviation.mean():.3f}")

    # -----------------------------------------------------------------------
    # 6. Anomali Tespit Modeli
    # -----------------------------------------------------------------------
    _banner("6. Anomali Tespit Modeli Eğitiliyor")
    t0 = time.time()
    anomaly_model = train_anomaly_detector(X, contamination=0.05)
    save_anomaly_detector(anomaly_model)

    # Anomali skorlarını hesapla (tüm veri seti)
    raw_scores = anomaly_model.named_steps["detector"].score_samples(
        anomaly_model.named_steps["scaler"].transform(
            anomaly_model.named_steps["imputer"].transform(X[FEATURE_COLUMNS])
        )
    )
    anomaly_scores = np.clip((-raw_scores) / 0.5, 0.0, 1.0)
    print(f"   Egitim tamamlandi ({time.time()-t0:.1f}s)  --  "
          f"Ort anomali skoru: {anomaly_scores.mean():.4f}")

    # -----------------------------------------------------------------------
    # 7. Güven Skoru Regresyon Modeli
    # -----------------------------------------------------------------------
    _banner("7. Güven Skoru Regresyon Modeli Eğitiliyor")

    # Genişletilmiş özellik matrisi
    X_ext = X.copy()
    X_ext["price_deviation"] = price_deviation
    X_ext["anomaly_score"]   = anomaly_scores

    EXT_FEATURES = FEATURE_COLUMNS + ["price_deviation", "anomaly_score"]

    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X_ext[EXT_FEATURES], y_trust,
        test_size=0.2, random_state=RANDOM_STATE
    )

    trust_model = build_trust_score_regressor()
    trust_model.fit(Xt_train, yt_train)
    yt_pred = trust_model.predict(Xt_test)
    _evaluate_regression("TrustScoreRegressor", yt_test, yt_pred)

    # Kaydet
    joblib.dump(trust_model, TRUST_SCORE_MODEL_PATH)
    print(f"   [TrustScoreRegressor] Kaydedildi -> {TRUST_SCORE_MODEL_PATH}")

    # -----------------------------------------------------------------------
    # 8. Risk Sinif Classifier (class_weight=balanced)
    # -----------------------------------------------------------------------
    _banner("8. Risk Sinif Classifier Egitiliyor (class_weight=balanced)")

    y_class_train = np.array([score_to_class(s) for s in yt_train])
    y_class_test  = np.array([score_to_class(s) for s in yt_test])

    print("   Sinif dagilimi (egitim):")
    for i, name in enumerate(CLASS_NAMES):
        count = (y_class_train == i).sum()
        print(f"     {i} - {name}: {count:,} ({count/len(y_class_train)*100:.1f}%)")

    clf_model = build_risk_classifier()
    clf_model.fit(Xt_train, y_class_train)
    clf_pred  = clf_model.predict(Xt_test)

    acc = accuracy_score(y_class_test, clf_pred)
    print(f"\n   Accuracy: {acc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_class_test, clf_pred, target_names=CLASS_NAMES))

    joblib.dump(clf_model, CLASSIFIER_MODEL_PATH)
    print(f"   [RiskClassifier] Kaydedildi -> {CLASSIFIER_MODEL_PATH}")

    _banner("[OK] Tum modeller basariyla egitildi ve kaydedildi!")
    print(f"   Dizin: {MODELS_DIR}\n")


if __name__ == "__main__":
    main()
