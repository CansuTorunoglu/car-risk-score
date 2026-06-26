# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Core Risk Scoring Rules
Bu dosya, kullanılmış araç ilanlarının çeşitli risk sinyallerini analiz eden
ve güven skoru hesaplayan kuralları içerir.
"""

import datetime
import math

def analyze_listing(listing: dict) -> dict:
    """
    Bir araç ilanını analiz eder ve güven skoru, risk seviyesi,
    risk nedenleri ve alıcı tavsiyelerini içeren bir rapor döner.
    
    Parametreler:
        listing (dict): İlan verilerini içeren sözlük.
        
    Dönen Değer (dict):
        {
            "trust_score": int,
            "risk_level": str,
            "risk_reasons": list,
            "recommendations": list
        }
    """
    # Girdi doğrulaması: eğer ilan dict değilse varsayılan en yüksek riskli yapıyı döndür
    if not isinstance(listing, dict):
        return {
            "trust_score": 0,
            "risk_level": "Very High Risk",
            "risk_reasons": ["Geçersiz ilan formatı (Sözlük olmalıdır)"],
            "recommendations": ["İlan verisi okunamadı. Lütfen teknik ekiple görüşün."]
        }

    # Başlangıç güven skoru 100 üzerinden hesaplanır
    trust_score = 100
    risk_reasons = []
    recommendations = []

    # --- Yardımcı Yardımcı Kontrol Fonksiyonları (Turkish-friendly) ---
    
    def is_empty(val) -> bool:
        """
        Değerin boş, None, NaN veya tanımsız olup olmadığını kontrol eder.
        Pandas / Numpy / Python boşluk tiplerini kapsar.
        """
        if val is None:
            return True
        val_str = str(val).strip().lower()
        if val_str in ["", "nan", "none", "nat", "<na>"]:
            return True
        return False

    def safe_float(val):
        """
        Girdiyi güvenli bir şekilde float tipine dönüştürür. Hata durumunda None döner.
        """
        if is_empty(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def safe_int(val):
        """
        Girdiyi güvenli bir şekilde int tipine dönüştürür. Hata durumunda None döner.
        """
        if is_empty(val):
            return None
        try:
            # Önce float'a çevirip sonra int yapmak "2015.0" gibi stringleri kurtarır
            return int(float(val))
        except (ValueError, TypeError):
            return None

    # --- KURAL 1: Şasi Numarası (VIN) Kontrolü (-20 Puan) ---
    vin = listing.get("VIN")
    if is_empty(vin):
        trust_score -= 20
        risk_reasons.append("Şasi numarası (VIN) eksik.")
        recommendations.append("Satıcıdan şasi numarasını (VIN) talep edin ve geçmiş raporunu (Tramer/Carfax) sorgulayın.")

    # --- KURAL 2: Ruhsat Durumu (Title Status) Kontrolü ---
    title_status = listing.get("title_status")
    if is_empty(title_status):
        trust_score -= 10
        risk_reasons.append("Ruhsat durumu (title status) bilgisi eksik.")
        recommendations.append("Satışa engel bir durum olmadığını doğrulamak için satıcıdan ruhsat detaylarını isteyin.")
    else:
        status_str = str(title_status).strip().lower()
        if status_str == "salvage":
            trust_score -= 30
            risk_reasons.append("Araç salvage (hurda belgeli / ağır hasarlı) ruhsat durumuna sahip.")
            recommendations.append("Ağır hasarlı araçların can güvenliği riski taşıyabileceğini unutmayın; profesyonel ekspertiz yaptırmadan almayın.")
        elif status_str == "rebuilt":
            trust_score -= 20
            risk_reasons.append("Araç rebuilt (yeniden toplanmış) ruhsat durumuna sahip.")
            recommendations.append("Yeniden toplanmış araçların şasi ve airbag kontrollerini uzman bir serviste detaylıca yaptırın.")
        elif status_str == "parts only":
            trust_score -= 30
            risk_reasons.append("Araç sadece parça (parts only) amaçlı satılık.")
            recommendations.append("Bu araç yasal olarak trafiğe çıkamaz, sadece yedek parça olarak değerlendirilebilir.")
        elif status_str == "lien":
            trust_score -= 15
            risk_reasons.append("Araç üzerinde lien (rehin / haciz / hak mahrumiyeti) kaydı olabilir.")
            recommendations.append("Noter satışı öncesinde aracın üzerindeki tüm hak mahrumiyetlerinin temizlendiğinden emin olun.")
        elif status_str in ["missing", "unknown"]:
            trust_score -= 10
            risk_reasons.append("Ruhsat durumu (title status) bilinmiyor veya eksik.")
            recommendations.append("Ruhsatın aslını görmeyi talep edin ve yasal durumunu e-Devlet/ilgili kurumlardan sorgulayın.")

    # --- KURAL 3: Fiyat Kontrolü ---
    price = safe_float(listing.get("price"))
    if price is None:
        # Fiyat belirtilmemişse genel bir kesinti yap
        trust_score -= 10
        risk_reasons.append("İlan fiyat bilgisi bulunmuyor.")
        recommendations.append("Fiyatı girilmemiş ilanların detaylarını satıcı ile görüşerek netleştirin.")
    else:
        if price == 0:
            trust_score -= 25
            risk_reasons.append("Fiyat 0 olarak girilmiş (yanıltıcı ilan).")
            recommendations.append("Gerçek satış fiyatını netleştirmek için satıcıyla iletişime geçin, peşinat tuzaklarına dikkat edin.")
        elif 0 < price <= 500:
            trust_score -= 20
            risk_reasons.append(f"Fiyat şüpheli derecede çok düşük (${price:,.2f}).")
            recommendations.append("Bu fiyat gerçekçi değildir. Kaparo dolandırıcılığı veya yanıltıcı ilan (peşinat tutarı) olma ihtimali çok yüksektir.")
        elif price > 150000:
            trust_score -= 10
            risk_reasons.append(f"Fiyat piyasa ortalamasının çok üzerinde (${price:,.2f}).")
            recommendations.append("Fiyatın yüksek olmasının nedenini (özel donanım, nadirlik vb.) sorgulayın ve piyasa değerini karşılaştırın.")

    # --- KURAL 4: Kilometre (Odometer) Kontrolü ---
    odometer = safe_float(listing.get("odometer"))
    if odometer is not None:
        if odometer > 300000:
            trust_score -= 15
            risk_reasons.append(f"Kilometre çok yüksek ({odometer:,.0f} mil).")
            recommendations.append("Yüksek kilometreli araçların motor ve şanzıman durumunu kompresyon testi dahil detaylı inceletin.")

    # --- KURAL 5: Yaş ve Düşük Kilometre İlişkisi (Odometer Rollback) ---
    year = safe_int(listing.get("year"))
    if year is not None:
        current_year = datetime.datetime.now().year
        car_age = current_year - year
        
        # Yaş 15'ten büyük ve km 0 ile 10000 arasındaysa (geri çekilme fraudu şüphesi)
        if odometer is not None and odometer > 0 and odometer < 10000 and car_age > 15:
            trust_score -= 25
            risk_reasons.append(f"Çok eski araç ({car_age} yaşında) olmasına rağmen kilometresi şüpheli derecede düşük ({odometer:,.0f} mil).")
            recommendations.append("Çok eski araçlarda aşırı düşük kilometre, sayaç sıfırlanması veya geri çekilmesi (odometer rollback) belirtisi olabilir. Servis/muayene geçmişini inceleyin.")

    # --- KURAL 6: Açıklama Uzunluğu Kontrolü ---
    description = listing.get("description")
    if is_empty(description):
        trust_score -= 10
        risk_reasons.append("İlan açıklama metni bulunmuyor.")
        recommendations.append("Detaylı bilgi içermeyen boş ilanlardan uzak durun, satıcıdan araç durumu hakkında detay isteyin.")
    else:
        desc_len = len(str(description).strip())
        if desc_len < 30:
            trust_score -= 10
            risk_reasons.append("Açıklama metni çok kısa (30 karakterden az).")
            recommendations.append("Very short description: Yetersiz bilgi içeren ilanlar güvensiz olabilir; satıcıdan araç geçmişi hakkında detaylı yazılı bilgi isteyin.")

    # --- KURAL 7: Görsel (Image) Kontrolü ---
    image_url = listing.get("image_url")
    if is_empty(image_url):
        trust_score -= 5
        risk_reasons.append("İlanda araç görseli bulunmuyor.")
        recommendations.append("Fotoğrafı olmayan ilanlar yanıltıcı olabilir. Satıcıdan güncel araç fotoğraflarını göndermesini isteyin.")

    # --- KURAL 8: Kondisyon Bilgisi Kontrolü ---
    condition = listing.get("condition")
    if is_empty(condition):
        trust_score -= 5
        risk_reasons.append("Araç kondisyon (durum) bilgisi eksik.")
        recommendations.append("Aracın genel mekanik ve kozmetik kondisyonunu öğrenmek için satıcıya durumunu sorun.")

    # Güven skorunu 0 ile 100 arasında sınırla
    trust_score = max(0, min(100, trust_score))

    # --- RİSK SEVİYESİ TANIMLAMA ---
    if trust_score >= 80:
        risk_level = "Low Risk"
    elif trust_score >= 60:
        risk_level = "Medium Risk"
    elif trust_score >= 40:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"

    return {
        "trust_score": int(trust_score),
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "recommendations": recommendations
    }

# --- TEST BLOKU ---
if __name__ == "__main__":
    print("--- Risk Rules Test Başlatılıyor ---")
    
    # Test 1: Mükemmel, Düşük Riskli İlan
    clean_car = {
        "VIN": "1FTFW1EF5FAXXXXXX",
        "title_status": "clean",
        "price": 18500,
        "odometer": 85000,
        "year": 2018,
        "description": "2018 model temiz kazasız bakımlı aile arabası. İstediğiniz servise gösterebilirsiniz.",
        "image_url": "http://example.com/car.jpg",
        "condition": "excellent"
    }
    
    # Test 2: Çok Yüksek Riskli İlan (Birçok kuralı ihlal eden ilan)
    risky_car = {
        "VIN": None,                     # -20
        "title_status": "salvage",       # -30
        "price": 0,                      # -25
        "odometer": 350000,              # -15
        "year": 2005,                    # car_age = 21, ama odometer > 10000 olduğu için rollback tetiklenmez
        "description": "Satılık.",        # -10 (kısa açıklama)
        "image_url": "",                 # -5
        "condition": ""                  # -5
    }

    # Test 3: Odometer Rollback Şüpheli İlan
    rollback_car = {
        "VIN": "1FTFW1EF5FAXXXXXX",
        "title_status": "clean",
        "price": 12000,
        "odometer": 2500,                # Düşük km
        "year": 2002,                    # Yaşı 24 (rollback tetiklenmeli: -25)
        "description": "Garaj arabası, neredeyse hiç kullanılmadı. Orjinal kilometredir.",
        "image_url": "http://example.com/car.jpg",
        "condition": "like new"
    }

    print("\n[TEST 1] Temiz Araç Raporu:")
    report_clean = analyze_listing(clean_car)
    for k, v in report_clean.items():
        print(f"  {k}: {v}")

    print("\n[TEST 2] Şüpheli/Hasarlı Araç Raporu:")
    report_risky = analyze_listing(risky_car)
    for k, v in report_risky.items():
        print(f"  {k}: {v}")

    print("\n[TEST 3] Sayaç Dolandırıcılığı (Rollback) Şüpheli Araç Raporu:")
    report_rollback = analyze_listing(rollback_car)
    for k, v in report_rollback.items():
        print(f"  {k}: {v}")
