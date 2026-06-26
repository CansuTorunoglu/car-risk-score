# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Streamlit Web Uygulaması (app.py)
Bu dosya, kullanıcının web arayüzü üzerinden araç ilanı verilerini girerek
güven ve risk durumunu görsel olarak analiz etmesini sağlar.
"""

import streamlit as st
import datetime
import os
import sys

# Modüllerin bulunabilmesi için çalışma dizinini yolumuza ekliyoruz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Risk kuralları fonksiyonunu import ediyoruz
from src.risk_rules import analyze_listing

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Used Car Listing Risk Score System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Stil Tanımlamaları (CSS)
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #2E5BFF, #FF4B4B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #6C757D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-title {
        font-weight: 700;
        color: #1E293B;
        border-left: 5px solid #2E5BFF;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: rgba(46, 91, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2E5BFF;
        margin-bottom: 10px;
    }
    .card-error {
        background-color: rgba(255, 75, 75, 0.05);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #FF4B4B;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SİDEBAR (YAN MENÜ) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=90)
    st.markdown("### **Car Risk Score**")
    st.info("**Proje Adı:**\nUsed Car Listing Risk Score System")
    st.markdown("**Veri Seti:**\nCraigslist Cars/Trucks Kaggle Dataset")
    st.markdown("**Ürün Amacı:**\nİkinci el araç ilanlarındaki şüpheli ve yanıltıcı durumları tespit ederek alıcıların karar verme süreçlerini destekler.")
    st.divider()
    st.caption("⚠️ **Not:**\nBu sistem bir karar destek aracıdır. Kesin bir dolandırıcılık tespiti yapmaz, sadece ilan özniteliklerine dayalı risk analiz skorlaması üretir.")

# --- ANA SAYFA BAŞLIĞI ---
st.markdown('<div class="main-title">Used Car Listing Risk Score System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bu uygulama, ikinci el araç ilanlarını güvenilirlik ve risk açısından analiz eder.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Araç İlan Bilgilerini Giriniz</div>', unsafe_allow_html=True)

# Form yapısı
with st.form("listing_form", clear_on_submit=False):
    # Kolon yerleşimi ile formu daha estetik hale getiriyoruz
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input(
            "Araç Model Yılı (Year)", 
            min_value=1900, 
            max_value=datetime.datetime.now().year + 1, 
            value=2018, 
            step=1,
            help="Aracın üretim yılı"
        )
        
        price = st.number_input(
            "Satış Fiyatı ($)", 
            min_value=0, 
            value=15000, 
            step=500,
            help="İlanda yazan satış fiyatı. 0 peşinat tuzaklarını test etmek için kullanılabilir."
        )
        
        odometer = st.number_input(
            "Kilometre (Odometer - Mil)", 
            min_value=0, 
            value=65000, 
            step=1000,
            help="Aracın katettiği mesafe (Mil cinsinden)"
        )
        
        condition = st.selectbox(
            "Araç Kondisyonu (Condition)",
            options=["excellent", "good", "fair", "like new", "new", "salvage", "unknown", "empty"],
            index=0,
            help="İlanda belirtilen araç durumu"
        )

    with col2:
        vin = st.text_input(
            "Şasi Numarası (VIN)", 
            value="1FTFW1EF5FAXXXXXX", 
            placeholder="17 haneli şasi numarası",
            help="Boş bırakılması güven skorunu düşürecektir."
        )
        
        title_status = st.selectbox(
            "Ruhsat Durumu (Title Status)",
            options=["clean", "salvage", "rebuilt", "lien", "missing", "parts only", "unknown"],
            index=0,
            help="Aracın yasal mülkiyet ve hasar belgesi durumu"
        )
        
        image_url = st.text_input(
            "İlan Görsel URL'si (Image URL)", 
            value="http://images.craigslist.org/example.jpg", 
            placeholder="http://...",
            help="İlanda fotoğraf bulunup bulunmadığı kontrol edilir."
        )
        
        description = st.text_area(
            "İlan Açıklaması (Description)", 
            value="Araç temiz kullanılmış, tüm bakımları zamanında yapılmıştır. Kazasızdır.",
            height=100,
            help="Açıklama uzunluğu analizi için metin giriniz."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    # Form gönderme butonu
    submit_button = st.form_submit_button("İlanı Analiz Et", type="primary")

# --- ANALİZ SONUÇLARININ GÖSTERİLMESİ ---
if submit_button:
    # Verileri dict yapısında topluyoruz
    listing_data = {
        "year": year,
        "price": price,
        "odometer": odometer,
        "condition": "" if condition == "empty" else condition,
        "VIN": vin,
        "title_status": title_status,
        "image_url": image_url,
        "description": description
    }
    
    # Analizi çalıştır
    result = analyze_listing(listing_data)
    
    # Değişkenleri al
    score = result["trust_score"]
    level = result["risk_level"]
    reasons = result["risk_reasons"]
    recs = result["recommendations"]
    
    st.markdown('<div class="section-title">Risk Analiz Raporu</div>', unsafe_allow_html=True)
    
    # Skoru ve Risk Seviyesini Gösterme
    score_col, level_col = st.columns(2)
    
    with score_col:
        st.metric(label="Güven Skoru (Trust Score)", value=f"{score} / 100")
        st.progress(score / 100.0)
        
    with level_col:
        # Risk seviyesine göre renklendirme ve Türkçe karşılık
        if level == "Low Risk":
            st.success(f"**Risk Seviyesi: Düşük Risk (Low Risk)**")
            st.info("Bu ilan genel olarak güvenli kriterlere uymaktadır. Ancak yine de fiziki kontrolleri aksatmayın.")
        elif level == "Medium Risk":
            st.warning(f"**Risk Seviyesi: Orta Risk (Medium Risk)**")
            st.markdown("İlanda bazı şüpheli durumlar tespit edilmiştir. İnceleme önerilir.")
        else:
            st.error(f"**Risk Seviyesi: {level}**")
            st.markdown("İlan yüksek derecede riskli unsurlar içermektedir. Dikkatli olun!")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Risk Nedenleri ve Tavsiyeler
    reasons_col, recs_col = st.columns(2)
    
    with reasons_col:
        st.subheader("Risk Nedenleri (Risk Reasons)")
        if not reasons:
            st.write("🟢 **Herhangi bir önemli risk sinyali tespit edilmedi.**")
        else:
            # HTML ile kırmızı kenarlıklı kart yapısında listeleme
            for reason in reasons:
                st.markdown(f'<div class="card-error">🚨 {reason}</div>', unsafe_allow_html=True)
                
    with recs_col:
        st.subheader("Alıcı Tavsiyeleri (Recommendations)")
        if not recs:
            st.write("🟢 **Özel bir aksiyon önerilmemektedir. Klasik alım prosedürlerini izleyin.**")
        else:
            # HTML ile mavi kenarlıklı kart yapısında listeleme
            for rec in recs:
                st.markdown(f'<div class="card">💡 {rec}</div>', unsafe_allow_html=True)
