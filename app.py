import streamlit as st
import pandas as pd
import joblib
import lightgbm

st.set_page_config(
    page_title="OVARIS",
    page_icon="🌸",
    layout="wide"
)

package = joblib.load(
    "pcos_streamlit_package.pkl"
)

model = package["model"]
features = package["features"]

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:72px;
    font-weight:800;
    color:#C75D7C;
}

.sub-title{
    text-align:center;
    font-size:24px;
    color:gray;
    margin-bottom:30px;
}

.stButton button{
    width:100%;
    height:60px;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🌸 OVARIS
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Ovarian Risk Assessment and Screening System
</div>
""", unsafe_allow_html=True)

st.markdown("### 👤 Identitas Pasien")

nama = st.text_input(
    "Nama Pasien",
    placeholder="Masukkan nama pasien"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Dasar",
    "Menstruasi",
    "Laboratorium",
    "Gejala",
    "USG"
])

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Usia (tahun)",
            min_value=10,
            max_value=100
        )

        weight = st.number_input(
            "Berat Badan (kg)",
            min_value=20.0
        )

        pulse = st.number_input(
            "Detak Jantung (bpm)",
            min_value=40
        )

        rr = st.number_input(
            "Laju Pernapasan (napas/menit)",
            min_value=10
        )

    with col2:

        height = st.number_input(
            "Tinggi Badan (cm)",
            min_value=100.0
        )

        blood_group = st.selectbox(
            "Golongan Darah",
            [11,12,13,14,15,16,17,18]
        )

        hb = st.number_input(
            "Hemoglobin (g/dL)",
            min_value=0.0
        )

    bmi = weight / ((height/100)**2)

    st.info(
        f"BMI Otomatis: {bmi:.2f}"
    )
    
with tab2:

    cycle = st.selectbox(
        "Siklus Menstruasi Tidak Teratur",
        ["Tidak","Ya"]
    )

    cycle_length = st.number_input(
        "Panjang Siklus (hari)",
        min_value=1
    )

    marriage = st.number_input(
        "Lama Pernikahan (tahun)",
        min_value=0
    )

    pregnant = st.selectbox(
        "Pernah Hamil",
        ["Tidak","Ya"]
    )

    abortions = st.number_input(
        "Jumlah Keguguran",
        min_value=0
    )
    
with tab3:

    col1, col2 = st.columns(2)

    with col1:

        beta1 = st.number_input("Beta HCG I")

        beta2 = st.number_input("Beta HCG II")

        fsh = st.number_input("FSH")

        lh = st.number_input("LH")

        tsh = st.number_input("TSH")

        amh = st.number_input("AMH")

    with col2:

        prl = st.number_input("Prolaktin")

        vitd = st.number_input("Vitamin D3")

        prg = st.number_input("Progesteron")

        rbs = st.number_input("Gula Darah Sewaktu")

        waist = st.number_input(
            "Lingkar Pinggang (inch)"
        )

        hip = st.number_input(
            "Lingkar Pinggul (inch)"
        )

    ratio = waist / hip if hip > 0 else 0

    fsh_lh = fsh / lh if lh > 0 else 0
    
with tab4:

    weight_gain = st.selectbox(
        "Kenaikan Berat Badan",
        ["Tidak","Ya"]
    )

    hair_growth = st.selectbox(
        "Pertumbuhan Rambut Berlebih",
        ["Tidak","Ya"]
    )

    skin_darkening = st.selectbox(
        "Penggelapan Kulit",
        ["Tidak","Ya"]
    )

    hair_loss = st.selectbox(
        "Kerontokan Rambut",
        ["Tidak","Ya"]
    )

    pimples = st.selectbox(
        "Jerawat",
        ["Tidak","Ya"]
    )

    fast_food = st.selectbox(
        "Konsumsi Fast Food",
        ["Tidak","Ya"]
    )

    exercise = st.selectbox(
        "Olahraga Rutin",
        ["Tidak","Ya"]
    )
    
with tab5:

    systolic = st.number_input(
        "Tekanan Darah Sistolik"
    )

    diastolic = st.number_input(
        "Tekanan Darah Diastolik"
    )

    follicle_l = st.number_input(
        "Jumlah Folikel Ovarium Kiri"
    )

    follicle_r = st.number_input(
        "Jumlah Folikel Ovarium Kanan"
    )

    avg_l = st.number_input(
        "Rata-rata Ukuran Folikel Kiri (mm)"
    )

    avg_r = st.number_input(
        "Rata-rata Ukuran Folikel Kanan (mm)"
    )

    endometrium = st.number_input(
        "Ketebalan Endometrium (mm)"
    )
    
# =====================================
# ANALISIS RISIKO
# =====================================

st.write("")
st.write("")

if st.button("🔍 Analisis Risiko PCOS"):

    # DataFrame sesuai urutan model training
    df = pd.DataFrame([{
        "_Age_yrs_": "Usia",
    "Weight_Kg_": "Berat Badan",
    "Height_Cm_": "Tinggi Badan",
    "BMI": "Indeks Massa Tubuh (BMI)",
    "Blood_Group": "Golongan Darah",

    "Pulse_rate_bpm_": "Detak Jantung",
    "RR_breaths_min_": "Laju Pernapasan",

    "Hb_g_dl_": "Hemoglobin",

    "Cycle_R_I_": "Siklus Menstruasi Tidak Teratur",
    "Cycle_length_days_": "Panjang Siklus Menstruasi",
    "Marraige_Status_Yrs_": "Lama Pernikahan",
    "Pregnant_Y_N_": "Riwayat Kehamilan",
    "No_of_aborptions": "Jumlah Keguguran",

    "_I_beta_HCG_mIU_mL_": "Beta HCG I",
    "II_beta_HCG_mIU_mL_": "Beta HCG II",

    "FSH_mIU_mL_": "Kadar FSH",
    "LH_mIU_mL_": "Kadar LH",
    "FSH_LH": "Rasio FSH/LH",

    "Hip_inch_": "Lingkar Pinggul",
    "Waist_inch_": "Lingkar Pinggang",
    "Waist_Hip_Ratio": "Rasio Pinggang-Pinggul",

    "TSH_mIU_L_": "Kadar TSH",
    "AMH_ng_mL_": "Kadar AMH",
    "PRL_ng_mL_": "Kadar Prolaktin",
    "Vit_D3_ng_mL_": "Kadar Vitamin D3",
    "PRG_ng_mL_": "Kadar Progesteron",
    "RBS_mg_dl_": "Gula Darah Sewaktu",

    "Weight_gain_Y_N_": "Kenaikan Berat Badan",
    "hair_growth_Y_N_": "Pertumbuhan Rambut Berlebih",
    "Skin_darkening_Y_N_": "Penggelapan Kulit",
    "Hair_loss_Y_N_": "Kerontokan Rambut",
    "Pimples_Y_N_": "Jerawat",
    "Fast_food_Y_N_": "Konsumsi Fast Food",
    "Reg_Exercise_Y_N_": "Olahraga Rutin",

    "BP__Systolic_mmHg_": "Tekanan Darah Sistolik",
    "BP__Diastolic_mmHg_": "Tekanan Darah Diastolik",

    "Follicle_No_L_": "Jumlah Folikel Ovarium Kiri",
    "Follicle_No_R_": "Jumlah Folikel Ovarium Kanan",

    "Avg_F_size_L_mm_": "Ukuran Folikel Kiri",
    "Avg_F_size_R_mm_": "Ukuran Folikel Kanan",

    "Endometrium_mm_": "Ketebalan Endometrium"
    }])

    # Samakan urutan fitur dengan model
    df = df[features]

    # Prediksi
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.markdown("---")

    st.markdown("## 📊 Hasil Analisis Risiko PCOS")

    st.info(
        f"👤 Nama Pasien: {nama}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Skor Risiko",
            f"{prob*100:.2f}%"
        )

        st.progress(float(prob))

    with col2:

        if pred == 1:

            st.error(
                "⚠ Risiko PCOS Tinggi"
            )

        else:

            st.success(
                "✅ Risiko PCOS Rendah"
            )

    # Faktor utama model
    st.markdown("### 📌 Faktor Utama")

    try:

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "feature": features,
            "importance": importance
        })

        top_features = (
            imp_df
            .sort_values(
                "importance",
                ascending=False
            )
            .head(5)
        )

        feature_names = {
            "AMH_ng_mL_": "Kadar AMH",
            "Follicle_No_R_": "Jumlah Folikel Ovarium Kanan",
            "Follicle_No_L_": "Jumlah Folikel Ovarium Kiri",
            "Cycle_R_I_": "Siklus Menstruasi Tidak Teratur",
            "Weight_gain_Y_N_": "Kenaikan Berat Badan",
            "hair_growth_Y_N_": "Pertumbuhan Rambut Berlebih",
            "Skin_darkening_Y_N_": "Penggelapan Kulit",
            "LH_mIU_mL_": "Kadar LH",
            "TSH_mIU_L_": "Kadar TSH",
            "RBS_mg_dl_": "Gula Darah Sewaktu"
        }

        for _, row in top_features.iterrows():

            nama = feature_names.get(
                row["feature"],
                row["feature"]
            )

            st.write(f"• {nama}")

    except:
        st.info("Faktor utama belum tersedia.")

    # Rekomendasi
    st.markdown("### 📋 Rekomendasi")

    if pred == 1:

        st.warning("""
        • Konsultasikan hasil skrining kepada dokter spesialis kandungan.

        • Pertimbangkan pemeriksaan hormon reproduksi lanjutan.

        • Lakukan pemantauan siklus menstruasi secara rutin.

        • Terapkan pola makan sehat dan aktivitas fisik teratur.

        • Hasil ini merupakan skrining awal dan bukan diagnosis medis.
        """)

    else:

        st.success("""
        • Risiko PCOS tergolong rendah berdasarkan data yang dimasukkan.

        • Tetap pertahankan pola hidup sehat.

        • Lakukan pemeriksaan berkala jika muncul gejala PCOS.

        • Hasil ini merupakan skrining awal dan bukan diagnosis medis.
        """)