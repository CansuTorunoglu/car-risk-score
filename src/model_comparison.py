# -*- coding: utf-8 -*-
"""
Model Karsilastirma Scripti
- Farkli regresyon modellerini karsilastirir
- Trust score'u 4 sinifa cevirip classification yapar
- Confusion matrix ve diger metrikleri hesaplar
- Sonuclari ekrana yazdirir
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    accuracy_score
)
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from src.risk_rules    import analyze_listing
from src.data_cleaning import preprocess_dataframe, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(_BASE_DIR, "data", "cardata.csv", "vehicles.csv")
SAMPLE_SIZE  = 30_000   # Karsilastirma icin daha kucuk orneklem (hiz icin)
RANDOM_STATE = 42

def score_to_class(score):
    if score >= 80: return 0   # Low Risk
    elif score >= 60: return 1  # Medium Risk
    elif score >= 40: return 2  # High Risk
    else: return 3              # Very High Risk

CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

def banner(msg):
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Veri yukle
# ---------------------------------------------------------------------------
banner("1. Veri Yukleniyor")
total_rows = sum(1 for _ in open(DATASET_PATH, encoding="utf-8")) - 1
skip_set = set(
    np.random.default_rng(RANDOM_STATE)
    .choice(total_rows, size=int(total_rows * (1 - SAMPLE_SIZE/total_rows)), replace=False) + 1
)
df = pd.read_csv(DATASET_PATH, skiprows=lambda i: i in skip_set, low_memory=False)
df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"  Yuklendi: {len(df):,} satir")

# 2. Hedef degisken
banner("2. Trust Score Hesaplaniyor")
t0 = time.time()
y_score = df.apply(lambda row: analyze_listing(row.to_dict())["trust_score"], axis=1).values
y_class = np.array([score_to_class(s) for s in y_score])
print(f"  Tamamlandi ({time.time()-t0:.1f}s)")
print(f"  Sinif dagilimi:")
for i, name in enumerate(CLASS_NAMES):
    count = (y_class == i).sum()
    print(f"    {i} - {name}: {count:,} ({count/len(y_class)*100:.1f}%)")

# 3. Ozellik muh.
banner("3. Ozellik Muhendisligi")
X = preprocess_dataframe(df)
print(f"  Ozellik boyutu: {X.shape}")

# Train/test bolme
X_train, X_test, ys_train, ys_test, yc_train, yc_test = train_test_split(
    X, y_score, y_class, test_size=0.2, random_state=RANDOM_STATE
)

# Imputer (NaN doldurucu)
imp = SimpleImputer(strategy="median")
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)

# ---------------------------------------------------------------------------
# 4. REGRESYON KARSILASTIRMASI
# ---------------------------------------------------------------------------
banner("4. Regresyon Modelleri Karsilastirmasi")

regressors = {
    "Linear Regression"         : LinearRegression(),
    "Ridge Regression"          : Ridge(alpha=1.0),
    "Decision Tree"             : DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "K-Nearest Neighbors"       : KNeighborsRegressor(n_neighbors=10),
    "Random Forest"             : RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting"         : GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "HistGradientBoosting [BIZ]": HistGradientBoostingRegressor(max_iter=200, random_state=RANDOM_STATE),
}

print(f"\n  {'Model':<30} {'MAE':>8} {'R2':>8} {'Sure(s)':>8}")
print("  " + "-" * 56)
reg_results = {}
for name, model in regressors.items():
    t0 = time.time()
    model.fit(X_train_imp, ys_train)
    pred = model.predict(X_test_imp)
    mae  = mean_absolute_error(ys_test, pred)
    r2   = r2_score(ys_test, pred)
    dur  = time.time() - t0
    reg_results[name] = {"mae": mae, "r2": r2, "pred": pred}
    marker = " <-- KULLANDIK" if "BIZ" in name else ""
    print(f"  {name:<30} {mae:>8.2f} {r2:>8.4f} {dur:>7.1f}s{marker}")

# ---------------------------------------------------------------------------
# 5. CLASSIFICATION KARSILASTIRMASI
# ---------------------------------------------------------------------------
banner("5. Classification Modelleri Karsilastirmasi")

classifiers = {
    "Random Forest Classifier"       : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "HistGradientBoosting Classifier": __import__("sklearn.ensemble", fromlist=["HistGradientBoostingClassifier"]).HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE),
}

print(f"\n  {'Model':<35} {'Accuracy':>10} {'Sure(s)':>8}")
print("  " + "-" * 56)
clf_results = {}
for name, model in classifiers.items():
    t0 = time.time()
    model.fit(X_train_imp, yc_train)
    pred_c = model.predict(X_test_imp)
    acc    = accuracy_score(yc_test, pred_c)
    dur    = time.time() - t0
    clf_results[name] = {"pred": pred_c, "acc": acc}
    print(f"  {name:<35} {acc:>10.4f} {dur:>7.1f}s")

# ---------------------------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------------------------
banner("6. Confusion Matrix -- Random Forest Classifier")

# Regresyon tahminini sinifa donustur
best_reg_pred_class = np.array([score_to_class(s) for s in reg_results["HistGradientBoosting [BIZ]"]["pred"]])

print("\n  [Regression -> Class donusum] HistGradientBoosting:")
cm_reg = confusion_matrix(yc_test, best_reg_pred_class)
_header = f"  {'':>18}" + "".join(f" {n[:8]:>10}" for n in CLASS_NAMES)
print(_header)
for i, row in enumerate(cm_reg):
    print(f"  {CLASS_NAMES[i][:16]:>18}" + "".join(f" {v:>10}" for v in row))

print(f"\n  Dogruluk (Accuracy): {accuracy_score(yc_test, best_reg_pred_class):.4f}")

print("\n\n  [Random Forest Classifier dogrudan]:")
if clf_results:
    rf_pred = clf_results["Random Forest Classifier"]["pred"]
    cm_clf  = confusion_matrix(yc_test, rf_pred)
    print(_header)
    for i, row in enumerate(cm_clf):
        print(f"  {CLASS_NAMES[i][:16]:>18}" + "".join(f" {v:>10}" for v in row))
    print(f"\n  Dogruluk (Accuracy): {clf_results['Random Forest Classifier']['acc']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(yc_test, rf_pred, target_names=CLASS_NAMES))

# ---------------------------------------------------------------------------
# 7. CROSS-VALIDATION (Genelleme Gucu)
# ---------------------------------------------------------------------------
banner("7. 5-Fold Cross Validation -- En Iyi 3 Model")

top3 = ["HistGradientBoosting [BIZ]", "Random Forest", "Gradient Boosting"]
print(f"\n  {'Model':<30} {'CV MAE Ort':>12} {'CV MAE Std':>12}")
print("  " + "-" * 55)
for name in top3:
    model = regressors[name]
    cv_scores = cross_val_score(model, X_train_imp, ys_train,
                                cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    mean_mae = -cv_scores.mean()
    std_mae  = cv_scores.std()
    print(f"  {name:<30} {mean_mae:>12.4f} {std_mae:>12.4f}")

banner("Karsilastirma Tamamlandi!")
