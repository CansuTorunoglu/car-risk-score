# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Terminal Arayüzü (main.py)
Kullanıcı terminal üzerinden ilan bilgilerini girer ve
Kural Tabanlı veya Makine Öğrenmesi modunu seçerek risk analizi alır.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk_rules import analyze_listing
from src.risk_ml    import analyze_listing_ml, models_available


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_input(prompt: str, default: str = "") -> str:
    val = input(prompt).strip()
    return val if val else default


# ---------------------------------------------------------------------------
# Giriş Formu
# ---------------------------------------------------------------------------

def collect_listing() -> dict:
    """Kullanıcıdan araç ilan bilgilerini toplar."""
    clear_screen()
    print("=" * 60)
    print("                 YENİ İLAN VERİ GİRİŞİ                  ")
    print(" (Bilgi yoksa boş bırakıp Enter'a basabilirsiniz)         ")
    print("=" * 60)

    vin          = get_input("1. Şasi Numarası (VIN)       : ")
    print("\n[Seçenekler: clean, salvage, rebuilt, lien, parts only, missing]")
    title_status = get_input("2. Ruhsat Durumu             : ")
    price        = get_input("3. Satış Fiyatı ($)          : ")
    odometer     = get_input("4. Kilometre (Mil)           : ")
    year         = get_input("5. Model Yılı                : ")
    description  = get_input("6. İlan Açıklaması           : ")
    image_url    = get_input("7. Görsel URL                : ")
    print("\n[Seçenekler: new, like new, excellent, good, fair, salvage]")
    condition    = get_input("8. Araç Kondisyonu           : ")
    print("\n[Seçenekler: gas, diesel, electric, hybrid, other]")
    fuel         = get_input("9. Yakıt Tipi                : ")
    print("\n[Seçenekler: automatic, manual, other]")
    transmission = get_input("10. Vites Tipi               : ")
    print("\n[Seçenekler: 4wd, fwd, rwd]")
    drive        = get_input("11. Çekiş Tipi               : ")

    return {
        "VIN":          vin,
        "title_status": title_status,
        "price":        price,
        "odometer":     odometer,
        "year":         year,
        "description":  description,
        "image_url":    image_url,
        "condition":    condition,
        "fuel":         fuel,
        "transmission": transmission,
        "drive":        drive,
    }


# ---------------------------------------------------------------------------
# Sonuç Yazdırma
# ---------------------------------------------------------------------------

def print_result(result: dict, use_ml: bool):
    """Analiz sonuçlarını terminal üzerinde formatlı biçimde gösterir."""
    score  = result["trust_score"]
    level  = result["risk_level"]
    risk_labels = {
        "Low Risk":       "Düşük Risk     ✅",
        "Medium Risk":    "Orta Risk      ⚠️",
        "High Risk":      "Yüksek Risk    🚨",
        "Very High Risk": "Çok Yüksek Risk🔴",
    }

    print("\n" + "=" * 60)
    mode_tag = "[ML]" if use_ml else "[KURAL]"
    print(f"          ANALİZ SONUÇLARI  {mode_tag}")
    print("=" * 60)
    print(f" Güven Skoru     : {score} / 100")
    print(f" Risk Seviyesi   : {risk_labels.get(level, level)}")

    if use_ml:
        pred = result.get("predicted_price", 0)
        dev  = result.get("price_deviation", 0) * 100
        anom = result.get("anomaly_score", 0)
        sign = "▼" if dev < 0 else ("▲" if dev > 0 else "~")
        print(f" Tahmin Fiyatı   : ${pred:,.0f}   ({sign} %{abs(dev):.1f} sapma)")
        print(f" Anomali Skoru   : {anom:.2f} / 1.00  (0=Normal, 1=Anormal)")

    print("-" * 60)
    print(" Risk Nedenleri:")
    reasons = result["risk_reasons"]
    if not reasons:
        print("  - Herhangi bir risk sinyali tespit edilmedi.")
    else:
        for i, r in enumerate(reasons, 1):
            print(f"  {i}. {r}")

    print("-" * 60)
    print(" Alıcı Tavsiyeleri:")
    recs = result["recommendations"]
    if not recs:
        print("  - Özel bir tavsiye yok; genel alım kontrollerini yapın.")
    else:
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Analiz Akışı
# ---------------------------------------------------------------------------

def analyze_flow(use_ml: bool):
    listing = collect_listing()

    print("\n" + "-" * 60)
    print("Analiz ediliyor...")
    print("-" * 60)

    if use_ml:
        result = analyze_listing_ml(listing)
    else:
        result = analyze_listing(listing)

    print_result(result, use_ml)
    input("\nAna menüye dönmek için Enter'a basın...")


# ---------------------------------------------------------------------------
# Ana Menü
# ---------------------------------------------------------------------------

def menu():
    ml_ready = models_available()

    while True:
        clear_screen()
        print("=" * 60)
        print("     İKİNCİ EL ARAÇ İLAN RİSK DEĞERLENDİRME SİSTEMİ     ")
        print("=" * 60)
        print(" 1. Yeni İlan Analiz Et  →  Kural Tabanlı Mod")
        if ml_ready:
            print(" 2. Yeni İlan Analiz Et  →  Makine Öğrenmesi Modu")
        else:
            print(" 2. [ML Modu] Modeller Henüz Eğitilmedi — Devre Dışı")
            print("    (python src/train_models.py ile eğitebilirsiniz)")
        print(" 3. Sistemden Çıkış")
        print("=" * 60)

        choice = input("Seçiminiz (1-3): ").strip()

        if choice == "1":
            analyze_flow(use_ml=False)
        elif choice == "2":
            if ml_ready:
                analyze_flow(use_ml=True)
            else:
                input("\nML modelleri henüz eğitilmedi! Enter'a basın...")
        elif choice == "3":
            print("\nSistemden çıkış yapılıyor. Güvenli sürüşler! 🚗")
            break
        else:
            input("\nGeçersiz seçim! Enter'a basın...")


if __name__ == "__main__":
    menu()
