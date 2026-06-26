# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Terminal Arayüzü (main.py)
Bu dosya, kullanıcının terminal üzerinden manuel araç ilan verisi girmesini
ve bu ilanın risk analiz sonuçlarını görüntülemesini sağlar.
"""

import os
import sys

# Modüllerin bulunabilmesi için çalışma dizinini yolumuza ekliyoruz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Risk kuralları fonksiyonumuzu import ediyoruz
from src.risk_rules import analyze_listing

def clear_screen():
    """
    Terminal ekranını temizleyerek temiz bir arayüz sunar.
    Windows ve Unix sistemlerle uyumludur.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def get_input(prompt: str, default: str = "") -> str:
    """
    Kullanıcıdan girdi alır. Boş girişlerde varsayılan değeri döner.
    """
    user_val = input(prompt).strip()
    if not user_val:
        return default
    return user_val

def menu():
    """
    Terminal arayüzü ana menü döngüsü.
    """
    while True:
        clear_screen()
        print("=" * 60)
        print("     İKİNCİ EL ARAÇ İLAN RİSK DEĞERLENDİRME SİSTEMİ     ")
        print("=" * 60)
        print(" 1. Yeni Bir İlanı Analiz Et")
        print(" 2. Sistemden Çıkış")
        print("=" * 60)
        
        choice = input("Lütfen yapmak istediğiniz işlemi seçin (1-2): ").strip()
        
        if choice == "1":
            analyze_flow()
        elif choice == "2":
            print("\nSistemden çıkış yapılıyor. Güvenli sürüşler dileriz!")
            break
        else:
            input("\nGeçersiz seçim! Ana menüye dönmek için Enter'a basın...")

def analyze_flow():
    """
    Kullanıcıdan tek tek alanları alan ve analizi çalıştıran akış.
    """
    clear_screen()
    print("=" * 60)
    print("                 YENİ İLAN VERİ GİRİŞİ                  ")
    print(" (Bilgi yoksa doğrudan boş bırakıp Enter'a basabilirsiniz) ")
    print("=" * 60)
    
    # Kullanıcıdan alanların toplanması
    vin = get_input("1. Şasi Numarası (VIN) [Örn: 1FTFW1EF5FAXXXXXX]: ")
    
    print("\n[Ruhsat Seçenekleri: clean, salvage, rebuilt, lien, parts only, missing]")
    title_status = get_input("2. Ruhsat Durumu (title_status) [Örn: clean]: ")
    
    price = get_input("3. Satış Fiyatı ($) [Örn: 12500]: ")
    odometer = get_input("4. Kilometre (Mil Cinsinden) [Örn: 85000]: ")
    year = get_input("5. Model Yılı (Year) [Örn: 2018]: ")
    description = get_input("6. İlan Açıklaması (Description): ")
    image_url = get_input("7. İlan Görsel URL'si (image_url): ")
    
    print("\n[Kondisyon Seçenekleri: new, like new, excellent, good, fair, salvage]")
    condition = get_input("8. Araç Kondisyonu (condition) [Örn: good]: ")
    
    # Girişleri bir sözlüğe dönüştürüyoruz
    listing = {
        "VIN": vin,
        "title_status": title_status,
        "price": price,
        "odometer": odometer,
        "year": year,
        "description": description,
        "image_url": image_url,
        "condition": condition
    }
    
    print("\n" + "-" * 60)
    print("İlan verileri analiz ediliyor...")
    print("-" * 60)
    
    # analyze_listing fonksiyonunun çağrılması
    result = analyze_listing(listing)
    
    # Sonuçların yazdırılması
    print("\n" + "=" * 60)
    print("                 ANALİZ SONUÇLARI                       ")
    print("=" * 60)
    
    score = result["trust_score"]
    level = result["risk_level"]
    
    # Risk Seviyelerinin Türkçe karşılıkları
    risk_levels_tr = {
        "Low Risk": "Düşük Risk (Güvenilir İlan)",
        "Medium Risk": "Orta Risk (Dikkat Edilmeli)",
        "High Risk": "Yüksek Risk (Şüpheli İlan)",
        "Very High Risk": "Çok Yüksek Risk (Tehlikeli / Hasarlı İlan)"
    }
    level_tr = risk_levels_tr.get(level, level)
    
    print(f" Güven Skoru (Trust Score) : {score} / 100")
    print(f" Risk Seviyesi (Risk Level): {level} -> {level_tr}")
    print("-" * 60)
    
    # Risk Nedenleri
    print(" Tespit Edilen Risk Nedenleri (Risk Reasons):")
    reasons = result["risk_reasons"]
    if not reasons:
        print("  - Herhangi bir risk sinyali tespit edilmedi.")
    else:
        for idx, reason in enumerate(reasons, 1):
            print(f"  {idx}. {reason}")
            
    print("-" * 60)
    
    # Alıcı Tavsiyeleri
    print(" Alıcıya Tavsiyeler (Recommendations):")
    recs = result["recommendations"]
    if not recs:
        print("  - Özel bir tavsiye bulunmuyor, genel kontrolleri yapabilirsiniz.")
    else:
        for idx, rec in enumerate(recs, 1):
            print(f"  {idx}. {rec}")
            
    print("=" * 60)
    input("\nAna menüye dönmek için Enter'a basın...")

if __name__ == "__main__":
    menu()
