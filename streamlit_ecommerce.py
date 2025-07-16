import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# ------------------------------------------------------------------------------------------------ TITLE 
st.markdown("""
    <h1 style='text-align: center; color: #996515;;'>
        💰 Prediksi Churn 💰
    </h1>
    <h1 style='text-align: center; color: #996515;;'>
            Ecommerce
    </h1>
    <p style='text-align: center; font-size: 16px;'>
        Isi form di sidebar atau upload file CSV untuk memprediksi apakah customer akan churn.
    </p>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------ REQUIRED COLUMNS
required_columns = [
    "Tenure", "PreferredLoginDevice", "CityTier", "WarehouseToHome",
    "PreferredPaymentMode", "Gender", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "PreferedOrderCat", "SatisfactionScore", "MaritalStatus", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount"
]

# ------------------------------------------------------------------------------------------------ SIDEBAR - FORM MANUAL
st.sidebar.header("📝 Input Data Customer (Manual)")
with st.sidebar.form("input_form"):
    Tenure = st.number_input("Tenure", min_value=0, max_value=31, value=10)
    PreferredLoginDevice = st.selectbox("PreferredLoginDevice", ['Mobile Phone', 'Computer'])
    CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=2)
    WarehouseToHome = st.number_input("WarehouseToHome", min_value=5, max_value=127, value=16)
    PreferredPaymentMode = st.selectbox("PreferredPaymentMode", ['Debit Card', 'UPI', 'Cash on Delivery', 'E wallet', 'Credit Card'])
    Gender = st.selectbox("Gender", ['Female', 'Male'])
    HourSpendOnApp = st.number_input("HourSpendOnApp", min_value=0, max_value=5, value=1)
    NumberOfDeviceRegistered = st.number_input("NumberOfDeviceRegistered", min_value=1, max_value=6, value=3)
    PreferedOrderCat = st.selectbox("PreferedOrderCat", ['Fashion', 'Grocery', 'Laptop & Accessory', 'Mobile', 'Others'])
    SatisfactionScore = st.number_input("SatisfactionScore", min_value=1, max_value=5, value=3)
    MaritalStatus = st.selectbox("MaritalStatus", ['Divorced', 'Married', 'Single'])
    NumberOfAddress = st.number_input("NumberOfAddress", min_value=1, max_value=22, value=3)
    Complain = st.radio("Complain", ["yes", "no"])
    OrderAmountHikeFromlastYear = st.number_input("OrderAmountHikeFromlastYear", min_value=11, max_value=26, value=15)
    CouponUsed = st.number_input("CouponUsed", min_value=0, max_value=16, value=2)
    OrderCount = st.number_input("OrderCount", min_value=1, max_value=16, value=2)
    DaySinceLastOrder = st.number_input("DaySinceLastOrder", min_value=0, max_value=46, value=1)
    CashbackAmount = st.number_input("CashbackAmount", min_value=0, max_value=325, value=178)
    submitted = st.form_submit_button("📈 Prediksi Manual")

# ------------------------------------------------------------------------------------------------ SIDEBAR - UPLOAD FILE
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload File CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV untuk batch prediksi", type=["csv"])
predict_file = st.sidebar.button("📊 Prediksi dari File")

# ------------------------------------------------------------------------------------------------ SEGMENTATION FUNCTION

def combine_lrfm_scores(row):
    return (
        str(int(row['l_score'])) +
        str(int(row['r_score'])) +
        str(int(row['f_score'])) +
        str(int(row['m_score']))
    )

seg_map = {
    r'[1-2][1-4][1-4][1-2]': 'New Cust Low Value',
    r'[1-2][1-4][1-4][3-4]': 'New Cust High Value',
    r'[3-4][1-2][1-2][1-2]': 'Old Cust Inactive',
    r'[3-4][1-2][3-4][3-4]': 'Need Attention',
    r'[3-4][1-2][1-4][1-4]': 'At Risk',
    r'[3-4][3-4][3-4][3-4]': 'Loyal Cust',
    r'[3-4][3-4][3-4][1-2]': 'Potential Loyal Cust',
    r'[3-4][3-4][1-2][1-4]': 'Reactivated Old Cust'
}

def get_segment_combined(score_combined):
    for pattern, segment_name in seg_map.items():
        if pd.Series([score_combined]).str.match(pattern).iloc[0]:
            return segment_name
    return 'Uncategorized'

def segment_lrfm(df):
    df_lrfm = df.copy()[['Tenure', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount']]
    df_lrfm.rename(columns={'Tenure': 'length',
                            'DaySinceLastOrder': 'recency',
                            'OrderCount': 'frequency',
                            'CashbackAmount': 'monetary'}, inplace=True)
    df_lrfm['l_score'] = np.where(df_lrfm['length'] > 15, 4,
                                  np.where(df_lrfm['length'] > 8, 3,
                                           np.where(df_lrfm['length'] > 2, 2, 1)))
    df_lrfm['r_score'] = np.where(df_lrfm['recency'] > 7, 4,
                                  np.where(df_lrfm['recency'] > 4, 3,
                                           np.where(df_lrfm['recency'] > 2, 2, 1)))
    df_lrfm['f_score'] = np.where(df_lrfm['frequency'] > 4, 4,
                                  np.where(df_lrfm['frequency'] > 2, 3,
                                           np.where(df_lrfm['frequency'] > 1, 2, 1)))
    df_lrfm['m_score'] = np.where(df_lrfm['monetary'] > 197.95, 4,
                                  np.where(df_lrfm['monetary'] > 163.87, 3,
                                           np.where(df_lrfm['monetary'] > 145.91, 2, 1)))
    df_lrfm['lrfm_score_combined'] = df_lrfm.apply(combine_lrfm_scores, axis=1)
    df['Segment'] = df_lrfm['lrfm_score_combined'].apply(get_segment_combined)
    return df

# ------------------------------------------------------------------------------------------------ LOAD MODEL
model_path = r"I:\My Drive\Final Project Data\IPYNB\IPYNB\READY_TO_PUSH_\model_ecommerce_churn.sav"

# ------------------------------------------------------------------------------------------------ PREDIKSI FILE CSV
if uploaded_file is not None and predict_file:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ File CSV tidak valid. Kolom berikut hilang atau salah nama: {missing_cols}")
    else:
        df['Complain'] = np.where(df['Complain'] == 'yes', 1, 0)
        df = segment_lrfm(df)

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                model_loaded = pickle.load(file)

            y_proba = model_loaded.predict_proba(df)[:, 1]
            pred_label = np.where(y_proba > 0.35, 1, 0)
            df['ChurnPrediction'] = np.where(pred_label == 1, "Churn", "Tidak Churn")
            df['ChurnProbability'] = y_proba

            st.subheader("📄 Hasil Prediksi Batch")
            st.dataframe(df[['PreferredLoginDevice', 'CityTier', 'CashbackAmount',
                             'Segment', 'ChurnPrediction', 'ChurnProbability']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil", csv, "hasil_prediksi.csv", "text/csv")
        else:
            st.error(f"⚠️ File model tidak ditemukan di: `{model_path}`")

# ------------------------------------------------------------------------------------------------ PREDIKSI MANUAL
elif submitted:
    df = pd.DataFrame({
        "Tenure": [Tenure],
        "PreferredLoginDevice": [PreferredLoginDevice],
        "CityTier": [CityTier],
        "WarehouseToHome": [WarehouseToHome],
        "PreferredPaymentMode": [PreferredPaymentMode],
        "Gender": [Gender],
        "HourSpendOnApp": [HourSpendOnApp],
        "NumberOfDeviceRegistered": [NumberOfDeviceRegistered],
        "PreferedOrderCat": [PreferedOrderCat],
        "SatisfactionScore": [SatisfactionScore],
        "MaritalStatus": [MaritalStatus],
        "NumberOfAddress": [NumberOfAddress],
        "Complain": [1 if Complain == 'yes' else 0],
        "OrderAmountHikeFromlastYear": [OrderAmountHikeFromlastYear],
        "CouponUsed": [CouponUsed],
        "OrderCount": [OrderCount],
        "DaySinceLastOrder": [DaySinceLastOrder],
        "CashbackAmount": [CashbackAmount]
    })

    df = segment_lrfm(df)
    st.subheader("📊 Data yang Dimasukkan")
    st.write(df)

    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            model_loaded = pickle.load(file)

        y_proba = model_loaded.predict_proba(df)[:, 1]
        kelas = np.where(y_proba > 0.50, 1, 0)

        st.subheader("📈 Hasil Prediksi")
        st.subheader(f"Customer termasuk kategori:\n {df['Segment'].loc[0]}")
        
        if kelas == 1:
            st.success("✅ **Customer DIPREDIKSI AKAN churn.**")
        else:
            st.error("❌ **Customer DIPREDIKSI TIDAK AKAN churn.**")

        st.markdown(f"**🎯 Probabilitas Churn:** `{y_proba[0]:.2%}`")
    else:
        st.error(f"⚠️ File model tidak ditemukan di: `{model_path}`")
