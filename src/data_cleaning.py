# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Veri Temizleme ve Özellik Mühendisliği
Bu modül; ham ilan sözlüğünü veya DataFrame'ini ML modellerine hazır
sayısal/kategorik özelliklere dönüştürür.
"""

import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

CURRENT_YEAR: int = datetime.datetime.now().year

# Kategorik sütunlar için bilinen geçerli değerler (eğitim sırasında kullanılır)
TITLE_STATUS_VALUES = ["clean", "rebuilt", "salvage", "lien", "missing", "parts only", "unknown"]
CONDITION_VALUES    = ["new", "like new", "excellent", "good", "fair", "salvage", "unknown"]
FUEL_VALUES         = ["gas", "diesel", "electric", "hybrid", "other", "unknown"]
TRANSMISSION_VALUES = ["automatic", "manual", "other", "unknown"]
DRIVE_VALUES        = ["4wd", "fwd", "rwd", "unknown"]

# En yaygin 20 uretici; geri kalanlari 'other' olarak kodla
MANUFACTURER_VALUES = [
    "ford", "chevrolet", "toyota", "honda", "nissan",
    "jeep", "ram", "gmc", "dodge", "bmw",
    "mercedes-benz", "subaru", "hyundai", "volkswagen", "kia",
    "audi", "lexus", "buick", "cadillac", "chrysler",
    "other",
]

# En yaygin ABD eyaletleri (frekansa gore sirali)
STATE_VALUES = [
    "ca", "fl", "tx", "ny", "oh", "pa", "mi", "wa", "nc", "ga",
    "il", "co", "az", "or", "va", "mn", "wi", "mo", "sc", "in",
    "other",
]

CYLINDERS_VALUES = ["4 cylinders", "6 cylinders", "8 cylinders", "10 cylinders", "12 cylinders", "5 cylinders", "3 cylinders", "other", "unknown"]
TYPE_VALUES = ["sedan", "SUV", "pickup", "truck", "other", "coupe", "hatchback", "wagon", "van", "mini-van", "convertible", "bus", "offroad", "unknown"]


# ML modeli icin kullanilacak ozellik sutunlari
FEATURE_COLUMNS = [
    "price",
    "car_age",
    "odometer",
    "has_vin",
    "has_image",
    "description_len",
    "title_status_encoded",
    "condition_encoded",
    "fuel_encoded",
    "transmission_encoded",
    "drive_encoded",
    "manufacturer_encoded",   # YENİ
    "state_encoded",          # YENİ
    "cylinders_encoded",      # YENİ
    "type_encoded",           # YENİ
    "model_hash",             # YENİ
]

# Fiyat tahmincisi icin kullanilacak ozellik sutunlari (price haric)
PRICE_FEATURE_COLUMNS = [
    "car_age",
    "odometer",
    "title_status_encoded",
    "condition_encoded",
    "fuel_encoded",
    "transmission_encoded",
    "drive_encoded",
    "manufacturer_encoded",   # YENİ
    "state_encoded",          # YENİ
    "cylinders_encoded",      # YENİ
    "type_encoded",           # YENİ
    "model_hash",             # YENİ
]


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def _is_empty(val) -> bool:
    """Değerin boş, None, NaN veya tanımsız olup olmadığını kontrol eder."""
    if val is None:
        return True
    val_str = str(val).strip().lower()
    return val_str in ("", "nan", "none", "nat", "<na>")


def _safe_float(val):
    """Girdiyi güvenli biçimde float'a dönüştürür; başarısızsa None döner."""
    if _is_empty(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """Girdiyi güvenli biçimde int'e dönüştürür; başarısızsa None döner."""
    if _is_empty(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _encode_category(val, known_values: list) -> int:
    """
    Kategorik değeri sıralı tam sayıya (ordinal) dönüştürür.
    Bilinmeyen değerler 'unknown' olarak eşlenir (listenin son elemanı).
    """
    if _is_empty(val):
        return len(known_values) - 1          # 'unknown' indeksi
    val_str = str(val).strip().lower()
    if val_str in known_values:
        return known_values.index(val_str)
    return len(known_values) - 1              # Bilinmeyen → 'unknown'


# ---------------------------------------------------------------------------
# Ana Fonksiyon: Tekil İlan Sözlüğü → Özellik Satırı
# ---------------------------------------------------------------------------

def preprocess_listing(listing: dict) -> dict:
    """
    Ham ilan sözlüğünü ML modeli için hazır bir özellik sözlüğüne dönüştürür.

    Parametreler
    ------------
    listing : dict
        Kullanıcıdan veya veri setinden gelen ham ilan verisi.

    Dönen Değer
    -----------
    dict
        FEATURE_COLUMNS'daki her sütun için sayısal bir değer içeren sözlük.
        Eksik değerler güvenli varsayılanlarla doldurulur.
    """
    # --- Sayısal alanlar ---
    price      = _safe_float(listing.get("price"))   or 0.0
    odometer   = _safe_float(listing.get("odometer")) or 0.0
    year       = _safe_int(listing.get("year"))
    car_age    = (CURRENT_YEAR - year) if year is not None else 10   # varsayılan: 10 yıl

    # --- İkili göstergeler ---
    vin       = listing.get("VIN", "")
    image_url = listing.get("image_url", "")
    has_vin   = int(not _is_empty(vin))
    has_image = int(not _is_empty(image_url))

    # --- Metin uzunluğu ---
    desc = listing.get("description", "")
    description_len = len(str(desc).strip()) if not _is_empty(desc) else 0

    # --- Kategorik kodlamalar ---
    title_status_encoded  = _encode_category(listing.get("title_status"),  TITLE_STATUS_VALUES)
    condition_encoded     = _encode_category(listing.get("condition"),     CONDITION_VALUES)
    fuel_encoded          = _encode_category(listing.get("fuel"),          FUEL_VALUES)
    transmission_encoded  = _encode_category(listing.get("transmission"),  TRANSMISSION_VALUES)
    drive_encoded         = _encode_category(listing.get("drive"),         DRIVE_VALUES)
    manufacturer_encoded  = _encode_category(listing.get("manufacturer"),  MANUFACTURER_VALUES)
    state_encoded         = _encode_category(listing.get("state"),         STATE_VALUES)
    cylinders_encoded     = _encode_category(listing.get("cylinders"),     CYLINDERS_VALUES)
    type_encoded          = _encode_category(listing.get("type"),          TYPE_VALUES)
    
    # Model ismi binlerce farkli deger alabilir, bunlari hash fonksiyonuyla 100 kategoriye ayiriyoruz
    model_str = str(listing.get("model", "")).strip().lower()
    if not model_str or model_str == "nan":
        model_hash = -1
    else:
        import zlib
        model_hash = zlib.crc32(model_str.encode("utf-8")) % 100

    return {
        "price":                price,
        "car_age":              car_age,
        "odometer":             odometer,
        "has_vin":              has_vin,
        "has_image":            has_image,
        "description_len":      description_len,
        "title_status_encoded": title_status_encoded,
        "condition_encoded":    condition_encoded,
        "fuel_encoded":         fuel_encoded,
        "transmission_encoded": transmission_encoded,
        "drive_encoded":        drive_encoded,
        "manufacturer_encoded": manufacturer_encoded,
        "state_encoded":        state_encoded,
        "cylinders_encoded":    cylinders_encoded,
        "type_encoded":         type_encoded,
        "model_hash":           model_hash,
    }


# ---------------------------------------------------------------------------
# Toplu İşleme: DataFrame → ML-Ready DataFrame
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setindeki tüm satırları `preprocess_listing` ile işler ve
    ML özellik DataFrame'ini döndürür.

    Parametreler
    ------------
    df : pd.DataFrame
        Ham Craigslist veri seti (veya bir alt kümesi).

    Dönen Değer
    -----------
    pd.DataFrame
        FEATURE_COLUMNS sütunlarını içeren temiz özellik tablosu.
    """
    rows = df.apply(lambda row: preprocess_listing(row.to_dict()), axis=1)
    feature_df = pd.DataFrame(list(rows))
    return feature_df[FEATURE_COLUMNS]
