import streamlit as st

st.set_page_config(
    page_title="Dashboard EDA & Prediksi",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
from pathlib import Path
import uuid
import warnings

warnings.filterwarnings('ignore')

try:
    from eda_module import EDAGenerator
    from prediction_lstm import LSTMPredictor
except Exception as e:
    st.error(f"Kesalahan saat mengimpor modul: {e}")
    st.stop()

@st.cache_resource
def load_predictor():
    try:
        return LSTMPredictor()
    except Exception as e:
        st.error(f"Kesalahan saat memuat model: {e}")
        return None

predictor = load_predictor()

if predictor is None:
    st.error("Gagal memuat model prediksi. Silakan periksa file model.")
    st.stop()

st.title("📊 Dashboard Analisis Data Eksploratif & Prediksi LSTM")
st.markdown("Unggah dataset Anda untuk analisis data eksploratif (EDA) otomatis dan prediksi 5 tahun menggunakan model LSTM")

tab1, tab2, tab3 = st.tabs(["📈 Analisis Data Eksploratif", "🎯 Buat Prediksi", "ℹ️ Tentang"])

with tab1:
    st.header("Upload Dataset for EDA")
    
    eda_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        key='eda_uploader'
    )
    
    if eda_file is not None:
        try:
            if eda_file.name.endswith('.csv'):
                df = pd.read_csv(eda_file)
            else:
                df = pd.read_excel(eda_file)
            
            with st.spinner('Generating EDA...'):
                session_id = str(uuid.uuid4())
                eda_gen = EDAGenerator(df, session_id, "plots")
                
                dataset_info = eda_gen.get_dataset_info()
                summary_stats = eda_gen.get_summary_statistics()
                missing_values = eda_gen.get_missing_values()
                numerical_cols, categorical_cols = eda_gen.get_column_types()
                plots = eda_gen.generate_all_plots()
            
            st.success("✅ EDA Generated Successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Dataset Overview")
                st.metric("Rows", f"{dataset_info['n_rows']:,}")
                st.metric("Columns", dataset_info['n_columns'])
                st.metric("Memory Usage", dataset_info['memory_usage'])
                
                with st.expander("View All Columns"):
                    st.write(", ".join(dataset_info['columns']))
                
                st.subheader("📊 Column Types")
                st.write(f"**Numerical:** {len(numerical_cols)} columns")
                st.write(f"**Categorical:** {len(categorical_cols)} columns")
            
            with col2:
                st.subheader("❌ Missing Values")
                if missing_values:
                    missing_df = pd.DataFrame(missing_values)
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✓ No missing values found!")
            
            st.subheader("📊 Summary Statistics")
            if summary_stats:
                stats_df = pd.DataFrame(summary_stats)
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No numerical columns for statistics")
            
            st.subheader("📈 Visualizations")
            
            plot_titles = {
                'correlation_heatmap': 'Correlation Heatmap',
                'distributions': 'Distribution Plots',
                'boxplots': 'Box Plots (Outlier Detection)',
                'categorical_plots': 'Categorical Variables'
            }
            
            cols = st.columns(2)
            col_idx = 0
            
            for key, path in plots.items():
                if Path(path).exists():
                    with cols[col_idx % 2]:
                        st.image(path, caption=plot_titles.get(key, key), use_container_width=True)
                    col_idx += 1
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab2:
    st.header("Input Data untuk Prediksi 5 Tahun")
    
    with st.expander("ℹ️ Tentang Model LSTM", expanded=False):
        st.info("""
        **Model LSTM Multi-Output untuk Prediksi Time Series**
        
        Model ini memprediksi **intensity_nca** untuk **5 tahun ke depan** berdasarkan data **5 tahun sebelumnya**.
        
        **Kolom yang Diperlukan (7 fitur):**
        - `nca` - Nilai Natural Capital Accounting
        - `gdp` - Gross Domestic Product (PDB)
        - `growth_gdp` - Tingkat pertumbuhan GDP
        - `growth_nca` - Tingkat pertumbuhan NCA
        - `loggrowth_gdp` - Log pertumbuhan GDP
        - `loggrowth_nca` - Log pertumbuhan NCA
        - `intensity_nca` - Rasio NCA/GDP (dihitung otomatis jika tidak ada)
        
        **Kolom Opsional:**
        - `year` - Untuk konteks temporal
        - `code` atau `countryname` - Untuk identifikasi entitas
        
        **Cara Kerja:**
        - **Lookback**: Menggunakan 5 tahun berturut-turut sebagai input
        - **Horizon**: Memprediksi 5 tahun ke depan
        - **Sequences**: Dari N baris, menghasilkan N-4 urutan prediksi
        - **Contoh**: 20 baris → 16 urutan, masing-masing memprediksi 5 tahun ke depan
        
        **Persyaratan minimum**: Setidaknya 5 baris data diperlukan
        """)
        
        if st.button("Tampilkan Detail Model"):
            model_info = predictor.get_model_info()
            st.json(model_info)
    
    st.markdown("### 📝 Masukkan Data untuk 5 Tahun Terakhir")
    st.markdown("Isi indikator ekonomi untuk 5 tahun berturut-turut untuk memprediksi 5 tahun ke depan.")
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("#### Informasi Negara/Entitas (Opsional)")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            country_code = st.text_input("Kode Negara", value="USA", help="contoh: USA, IDN, CHN")
        with col_info2:
            country_name = st.text_input("Nama Negara", value="United States", help="Nama negara lengkap")
        
        st.markdown("#### Masukkan Data untuk 5 Tahun")
        
        data_rows = []
        current_year = 2024
        
        for i in range(5):
            year = current_year - 4 + i
            st.markdown(f"**Tahun {year}**")
            
            cols = st.columns(7)
            
            with cols[0]:
                year_val = st.number_input(f"Tahun", value=year, key=f"year_{i}", disabled=True)
            with cols[1]:
                nca = st.number_input(f"NCA", value=None, format="%.2f", key=f"nca_{i}", 
                                     help="Natural Capital Accounting", placeholder="Masukkan nilai NCA")
            with cols[2]:
                gdp = st.number_input(f"GDP", value=None, format="%.2f", key=f"gdp_{i}",
                                     help="Produk Domestik Bruto", placeholder="Masukkan nilai GDP")
            with cols[3]:
                growth_gdp = st.number_input(f"Pertumbuhan GDP", value=None, format="%.4f", key=f"growth_gdp_{i}",
                                            help="Tingkat Pertumbuhan GDP", placeholder="Masukkan pertumbuhan GDP")
            with cols[4]:
                growth_nca = st.number_input(f"Pertumbuhan NCA", value=None, format="%.4f", key=f"growth_nca_{i}",
                                            help="Tingkat Pertumbuhan NCA", placeholder="Masukkan pertumbuhan NCA")
            with cols[5]:
                loggrowth_gdp = st.number_input(f"Log Pertumbuhan GDP", value=None, format="%.4f", key=f"loggrowth_gdp_{i}",
                                               help="Log dari Pertumbuhan GDP", placeholder="Masukkan log pertumbuhan GDP")
            with cols[6]:
                loggrowth_nca = st.number_input(f"Log Pertumbuhan NCA", value=None, format="%.4f", key=f"loggrowth_nca_{i}",
                                               help="Log dari Pertumbuhan NCA", placeholder="Masukkan log pertumbuhan NCA")
            
            data_rows.append({
                'year': year_val,
                'code': country_code,
                'countryname': country_name,
                'nca': nca,
                'gdp': gdp,
                'growth_gdp': growth_gdp,
                'growth_nca': growth_nca,
                'loggrowth_gdp': loggrowth_gdp,
                'loggrowth_nca': loggrowth_nca
            })
        
        submitted = st.form_submit_button("🔮 Buat Prediksi 5 Tahun", type="primary", use_container_width=True)
    
    if submitted:
        try:
            df = pd.DataFrame(data_rows)
            
            st.info(f"📋 Data disiapkan: {len(df)} tahun data historis")
            
            with st.spinner('Membuat Prediksi 5 tahun...'):
                predictions, start_years, model_info = predictor.predict(df)
                
                predictions_df = predictor.format_predictions_as_dataframe(
                    predictions, start_years, df
                )
                
                session_id = str(uuid.uuid4())
                output_path = f"results/{session_id}_predictions.csv"
                predictor.save_predictions(predictions_df, str(output_path))
            
            st.success("✅ Prediksi 5 Tahun Berhasil Dibuat!")
            
            st.info(f"""
            📊 Berhasil membuat **{len(predictions)} urutan Prediksi**
            - Meramalkan **{model_info['horizon']} tahun** ke depan
            - Berdasarkan **{model_info['lookback']} tahun** data historis
            - Variabel target: **{model_info['target']}**
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Informasi Model")
                st.write(f"**Model:** {model_info['model_name']}")
                st.write(f"**Tipe:** {model_info['model_type']}")
                st.write(f"**Lookback:** {model_info['lookback']} tahun")
                st.write(f"**Horizon:** {model_info['horizon']} tahun")
                st.write(f"**Target:** {model_info['target']}")
                
                st.subheader("📈 Tahun Prediksi")
                if len(predictions) > 0:
                    base_year = data_rows[-1]['year']
                    forecast_years = [base_year + i for i in range(1, 6)]
                    st.write(f"**Tahun Dasar:** {base_year}")
                    st.write(f"**Tahun Prediksi:** {', '.join(map(str, forecast_years))}")
            
            with col2:
                st.subheader("🎯 Hasil Prediksi")
                
                if len(predictions) > 0:
                    forecast_values = predictions[0]
                    
                    st.write(f"**Prediksi {model_info['target'].upper()}:**")
                    for i, val in enumerate(forecast_values, 1):
                        year = data_rows[-1]['year'] + i
                        st.metric(f"Tahun {year}", f"{val:.6f}", delta=None)
                    
                    st.write("")
                    st.write(f"**Min:** {predictions.min():.6f}")
                    st.write(f"**Maks:** {predictions.max():.6f}")
                    st.write(f"**Rata-rata:** {predictions.mean():.6f}")
            
            st.subheader("📋 Tabel Prediksi Detail")
            st.dataframe(predictions_df, use_container_width=True)
            
            # Visualization
            if len(predictions) > 0:
                st.subheader("📈 Visualisasi Prediksi")
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Historical data
                historical_years = [row['year'] for row in data_rows]
                historical_intensity = [row['nca'] / row['gdp'] for row in data_rows]
                
                # Forecast data (dimulai dari tahun terakhir historis untuk menghubungkan garis)
                last_year = data_rows[-1]['year']
                last_intensity = historical_intensity[-1]
                
                forecast_years = [data_rows[-1]['year'] + i for i in range(1, 6)]
                forecast_values = predictions[0].tolist()
                
                # Gabungkan data untuk garis yang terhubung
                # Garis historis
                ax.plot(historical_years, historical_intensity, 'o-', label='Data Historis', 
                       linewidth=2.5, markersize=8, color='#2E86AB', zorder=3)
                
                # Garis prediksi yang terhubung dari titik terakhir historis
                connected_years = [last_year] + forecast_years
                connected_values = [last_intensity] + forecast_values
                ax.plot(connected_years, connected_values, 's-', label='Prediksi', 
                       linewidth=2.5, markersize=8, color='#E63946', linestyle='--', zorder=3)
                
                # Garis vertikal pembatas
                ax.axvline(x=last_year, color='gray', linestyle=':', linewidth=1.5, 
                          alpha=0.7, label='Batas Prediksi', zorder=2)
                
                # Tambahkan area shading untuk prediksi
                ax.axvspan(last_year, forecast_years[-1], alpha=0.1, color='red', zorder=1)
                
                ax.set_xlabel('Tahun', fontsize=13, fontweight='bold')
                ax.set_ylabel('Intensitas NCA (NCA/GDP)', fontsize=13, fontweight='bold')
                ax.set_title(f'{country_name} - Prediksi Intensitas NCA 5 Tahun', 
                           fontsize=15, fontweight='bold', pad=20)
                ax.legend(loc='best', fontsize=11, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)",
                    data=f,
                    file_name=f"forecast_{country_code}_{session_id}.csv",
                    mime="text/csv",
                    type="primary"
                )
                    
        except Exception as e:
            st.error(f"Kesalahan saat membuat prediksi: {str(e)}")

with tab3:
    st.header("Tentang Aplikasi Ini")
    
    st.markdown("""
    Aplikasi ini menyediakan dua fungsi utama:
    
    ### 1. Analisis Data Eksploratif (EDA)
    Secara otomatis menghasilkan analisis komprehensif meliputi:
    - Ringkasan dataset dan statistik
    - Analisis nilai yang hilang
    - Heatmap korelasi
    - Plot distribusi untuk fitur numerik
    - Box plot untuk deteksi outlier
    - Analisis variabel kategorikal
    
    ### 2. Prediksi Time Series LSTM
    Menggunakan model LSTM Multi-Output yang telah dilatih untuk Prediksi 5 tahun:
    - **Tipe Model:** LSTM (Long Short-Term Memory)
    - **Lookback:** 5 tahun data historis
    - **Horizon:** Memprediksi 5 tahun ke depan
    - **Target:** intensity_nca (rasio NCA/GDP)
    - **Fitur:** 7 indikator ekonomi
    - **Metode Input:** Form entri data manual
    
    ### Arsitektur Model
    - **Bentuk Input:** (5 timesteps, 7 fitur)
    - **Bentuk Output:** 5 prediksi masa depan
    - **File Model:** multi_output_lstm_h5_lookback5.keras
    
    ### Fitur yang Diperlukan
    1. NCA (Natural Capital Accounting)
    2. GDP (Produk Domestik Bruto)
    3. Tingkat Pertumbuhan GDP
    4. Tingkat Pertumbuhan NCA
    5. Log Pertumbuhan GDP
    6. Log Pertumbuhan NCA
    7. Intensitas NCA (dihitung otomatis jika tidak disediakan)
    
    ### Cara Menggunakan
    1. Masukkan informasi negara/entitas
    2. Isi data ekonomi untuk 5 tahun berturut-turut
    3. Klik "Buat Prediksi 5 Tahun"
    4. Lihat prediksi dan unduh hasilnya
    """)
