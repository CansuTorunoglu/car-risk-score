# -*- coding: utf-8 -*-
"""
Used Car Listing Risk Score System - Streamlit Web Uygulaması (app.py)
Kullanıcı web arayüzü üzerinden araç ilan bilgilerini girer ve
iki farklı moddan birini seçerek risk analizi yapar:
  - Kural Tabanlı (Rule-Based)
  - Makine Öğrenmesi (ML-Based)
"""

import datetime
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk_rules import analyze_listing
from src.risk_ml    import analyze_listing_ml, models_available

# ---------------------------------------------------------------------------
# Sayfa Konfigürasyonu
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Listing Risk Score System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Outfit:wght@700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-title {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(90deg, #2E5BFF, #9B51E0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.6rem;
    margin-bottom: 0.3rem;
  }
  .subtitle {
    color: #64748B;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
  }
  .section-title {
    font-weight: 700;
    color: #1E293B;
    border-left: 5px solid #2E5BFF;
    padding-left: 10px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-size: 1.15rem;
  }
  .card {
    background: rgba(46, 91, 255, 0.06);
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 4px solid #2E5BFF;
    margin-bottom: 10px;
    font-size: 0.95rem;
  }
  .card-error {
    background: rgba(239, 68, 68, 0.07);
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 4px solid #EF4444;
    margin-bottom: 10px;
    font-size: 0.95rem;
  }
  .ml-badge {
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }
  .rule-badge {
    background: linear-gradient(90deg, #0EA5E9, #2E5BFF);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }
  .metric-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
  }
  .metric-label { color: #64748B; font-size: 0.82rem; font-weight: 600; margin-bottom: 4px; }
  .metric-value { font-size: 1.6rem; font-weight: 800; color: #1E293B; }
  .deviation-neg  { color: #EF4444; }
  .deviation-warn { color: #F59E0B; }
  .deviation-ok   { color: #10B981; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
    st.markdown("### **Car Risk Score**")

    st.divider()

    ml_ready = models_available()
    mode_options = ["🔵 Kural Tabanlı", "🟣 Makine Öğrenmesi"]

    if not ml_ready:
        st.warning(
            "**ML Modelleri Henüz Eğitilmedi**\n\n"
            "Terminal'de şunu çalıştırın:\n"
            "```\npython src/train_models.py\n```"
        )
        mode = mode_options[0]
        st.radio("Risk Modeli", mode_options, index=0, disabled=True)
    else:
        mode = st.radio("Risk Modeli", mode_options, index=0)

    st.divider()

    st.info(
        "**Proje:** Used Car Listing Risk Score System\n\n"
        "**Veri:** Craigslist Cars/Trucks (Kaggle)"
    )
    st.caption(
        "⚠️ Bu sistem bir karar destek aracıdır. "
        "Kesin dolandırıcılık tespiti yapmaz."
    )

use_ml = "Makine Öğrenmesi" in mode

# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Used Car Listing Risk Score System</div>', unsafe_allow_html=True)

badge_html = (
    '<span class="ml-badge">🤖 Makine Öğrenmesi Modu</span>'
    if use_ml else
    '<span class="rule-badge">📋 Kural Tabanlı Mod</span>'
)
st.markdown(
    f'<div class="subtitle">İkinci el araç ilanlarını güvenilirlik açısından analiz edin. &nbsp; {badge_html}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Giriş Formu
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Araç İlan Bilgilerini Giriniz</div>', unsafe_allow_html=True)

with st.form("listing_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Araç Model Yılı",
            min_value=1900, max_value=datetime.datetime.now().year + 1,
            value=2018, step=1, help="Aracın üretim yılı",
        )
        price = st.number_input(
            "Satış Fiyatı ($)",
            min_value=0, value=15000, step=500,
            help="İlanda yazan satış fiyatı",
        )
        odometer = st.number_input(
            "Kilometre (Mil)",
            min_value=0, value=65000, step=1000,
            help="Aracın katettiği mesafe (mil cinsinden)",
        )
        condition = st.selectbox(
            "Araç Kondisyonu",
            options=["excellent", "good", "fair", "like new", "new", "salvage", "unknown", "empty"],
            index=0,
        )
        fuel = st.selectbox(
            "Yakıt Tipi",
            options=["gas", "diesel", "electric", "hybrid", "other", "unknown"],
            index=0,
        )

    with col2:
        vin = st.text_input(
            "Şasi Numarası (VIN)",
            value="1FTFW1EF5FAXXXXXX",
            placeholder="17 haneli şasi numarası",
            help="Boş bırakmak güven skorunu düşürür.",
        )
        title_status = st.selectbox(
            "Ruhsat Durumu",
            options=["clean", "salvage", "rebuilt", "lien", "missing", "parts only", "unknown"],
            index=0,
        )
        transmission = st.selectbox(
            "Vites Tipi",
            options=["automatic", "manual", "other", "unknown"],
            index=0,
        )
        drive = st.selectbox(
            "Çekiş Tipi",
            options=["4wd", "fwd", "rwd", "unknown"],
            index=0,
        )
        image_url = st.text_input(
            "Görsel URL",
            value="http://images.craigslist.org/example.jpg",
            placeholder="http://...",
        )
        description = st.text_area(
            "İlan Açıklaması",
            value="Araç temiz kullanılmış, tüm bakımları zamanında yapılmıştır. Kazasızdır.",
            height=95,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("🔍 İlanı Analiz Et", type="primary")

# ---------------------------------------------------------------------------
# Analiz ve Sonuçlar
# ---------------------------------------------------------------------------
if submit:
    listing_data = {
        "year":         year,
        "price":        price,
        "odometer":     odometer,
        "condition":    "" if condition == "empty" else condition,
        "VIN":          vin,
        "title_status": title_status,
        "image_url":    image_url,
        "description":  description,
        "fuel":         fuel,
        "transmission": transmission,
        "drive":        drive,
    }

    with st.spinner("Analiz yapılıyor..."):
        if use_ml:
            result = analyze_listing_ml(listing_data)
        else:
            result = analyze_listing(listing_data)

    score   = result["trust_score"]
    level   = result["risk_level"]
    reasons = result["risk_reasons"]
    recs    = result["recommendations"]

    st.markdown('<div class="section-title">Risk Analiz Raporu</div>', unsafe_allow_html=True)

    # --- Üst Metrik Satırı ---
    if use_ml:
        m1, m2, m3, m4 = st.columns(4)
    else:
        m1, m2 = st.columns(2)

    with m1:
        st.metric("Güven Skoru", f"{score} / 100")
        st.progress(score / 100.0)

    with m2:
        risk_colors = {
            "Low Risk":       st.success,
            "Medium Risk":    st.warning,
            "High Risk":      st.error,
            "Very High Risk": st.error,
        }
        risk_labels = {
            "Low Risk":       "✅ Düşük Risk",
            "Medium Risk":    "⚠️ Orta Risk",
            "High Risk":      "🚨 Yüksek Risk",
            "Very High Risk": "🔴 Çok Yüksek Risk",
        }
        risk_colors.get(level, st.error)(f"**Risk Seviyesi: {risk_labels.get(level, level)}**")

    if use_ml:
        pred_price = result.get("predicted_price", 0)
        price_dev  = result.get("price_deviation", 0)
        anom_score = result.get("anomaly_score", 0)

        with m3:
            dev_pct = price_dev * 100
            if price_dev < -0.30:
                cls = "deviation-neg"
                arrow = "▼"
            elif price_dev > 0.30:
                cls = "deviation-warn"
                arrow = "▲"
            else:
                cls = "deviation-ok"
                arrow = "~"
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">ML Tahmini Piyasa Değeri</div>'
                f'<div class="metric-value">${pred_price:,.0f}</div>'
                f'<div class="{cls}" style="font-size:0.9rem;font-weight:600">'
                f'{arrow} %{abs(dev_pct):.1f} sapma</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with m4:
            anom_pct = anom_score * 100
            anom_cls = "deviation-neg" if anom_score > 0.65 else (
                       "deviation-warn" if anom_score > 0.45 else "deviation-ok")
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Anomali Skoru</div>'
                f'<div class="metric-value {anom_cls}">{anom_score:.2f}</div>'
                f'<div style="font-size:0.82rem;color:#64748B">0 = Normal &nbsp;|&nbsp; 1 = Anormal</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Risk Nedenleri & Tavsiyeler ---
    rc, tc_ = st.columns(2)

    with rc:
        st.subheader("🚨 Risk Nedenleri")
        if not reasons:
            st.markdown("🟢 **Herhangi bir önemli risk sinyali tespit edilmedi.**")
        else:
            for r in reasons:
                st.markdown(f'<div class="card-error">🚨 {r}</div>', unsafe_allow_html=True)

    with tc_:
        st.subheader("💡 Alıcı Tavsiyeleri")
        if not recs:
            st.markdown("🟢 **Özel bir aksiyon önerilmemektedir.**")
        else:
            for r in recs:
                st.markdown(f'<div class="card">💡 {r}</div>', unsafe_allow_html=True)
