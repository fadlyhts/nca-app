import streamlit as st

st.set_page_config(
    page_title="EDA & Prediction Dashboard",
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
    st.error(f"Error importing modules: {e}")
    st.stop()

@st.cache_resource
def load_predictor():
    try:
        return LSTMPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = load_predictor()

if predictor is None:
    st.error("Failed to load predictor. Please check model files.")
    st.stop()

st.title("📊 EDA & LSTM Prediction Dashboard")
st.markdown("Upload your dataset for automatic exploratory data analysis and 5-year forecasting using LSTM model")

tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "🎯 Make Predictions", "ℹ️ About"])

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
    st.header("Input Data for 5-Year Forecasting")
    
    with st.expander("ℹ️ About LSTM Model", expanded=False):
        st.info("""
        **Multi-Output LSTM Model for Time Series Forecasting**
        
        This model predicts **intensity_nca** for the **next 5 years** based on the **previous 5 years** of data.
        
        **Required Columns (7 features):**
        - `nca` - Natural Capital Accounting value
        - `gdp` - Gross Domestic Product  
        - `growth_gdp` - GDP growth rate
        - `growth_nca` - NCA growth rate
        - `loggrowth_gdp` - Log of GDP growth
        - `loggrowth_nca` - Log of NCA growth
        - `intensity_nca` - NCA/GDP ratio (auto-calculated if missing)
        
        **Optional Columns:**
        - `year` - For temporal context
        - `code` or `countryname` - For entity identification
        
        **How it works:**
        - **Lookback**: Uses 5 consecutive years as input
        - **Horizon**: Predicts next 5 years  
        - **Sequences**: From N rows, generates N-4 prediction sequences
        - **Example**: 20 rows → 16 sequences, each predicting 5 future years
        
        **Minimum data requirement**: At least 5 rows needed
        """)
        
        if st.button("Show Model Details"):
            model_info = predictor.get_model_info()
            st.json(model_info)
    
    st.markdown("### 📝 Enter Data for the Last 5 Years")
    st.markdown("Fill in the economic indicators for 5 consecutive years to predict the next 5 years.")
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("#### Country/Entity Information (Optional)")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            country_code = st.text_input("Country Code", value="USA", help="e.g., USA, IDN, CHN")
        with col_info2:
            country_name = st.text_input("Country Name", value="United States", help="Full country name")
        
        st.markdown("#### Enter Data for 5 Years")
        
        data_rows = []
        current_year = 2024
        
        for i in range(5):
            year = current_year - 4 + i
            st.markdown(f"**Year {year}**")
            
            cols = st.columns(7)
            
            with cols[0]:
                year_val = st.number_input(f"Year", value=year, key=f"year_{i}", disabled=True)
            with cols[1]:
                nca = st.number_input(f"NCA", value=100.0, format="%.2f", key=f"nca_{i}", 
                                     help="Natural Capital Accounting")
            with cols[2]:
                gdp = st.number_input(f"GDP", value=20000.0, format="%.2f", key=f"gdp_{i}",
                                     help="Gross Domestic Product")
            with cols[3]:
                growth_gdp = st.number_input(f"GDP Growth", value=0.03, format="%.4f", key=f"growth_gdp_{i}",
                                            help="GDP Growth Rate")
            with cols[4]:
                growth_nca = st.number_input(f"NCA Growth", value=0.02, format="%.4f", key=f"growth_nca_{i}",
                                            help="NCA Growth Rate")
            with cols[5]:
                loggrowth_gdp = st.number_input(f"Log GDP Growth", value=0.029, format="%.4f", key=f"loggrowth_gdp_{i}",
                                               help="Log of GDP Growth")
            with cols[6]:
                loggrowth_nca = st.number_input(f"Log NCA Growth", value=0.019, format="%.4f", key=f"loggrowth_nca_{i}",
                                               help="Log of NCA Growth")
            
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
        
        submitted = st.form_submit_button("🔮 Generate 5-Year Forecast", type="primary", use_container_width=True)
    
    if submitted:
        try:
            df = pd.DataFrame(data_rows)
            
            st.info(f"📋 Data prepared: {len(df)} years of historical data")
            
            with st.spinner('Generating 5-year forecasts...'):
                predictions, start_years, model_info = predictor.predict(df)
                
                predictions_df = predictor.format_predictions_as_dataframe(
                    predictions, start_years, df
                )
                
                session_id = str(uuid.uuid4())
                output_path = f"results/{session_id}_predictions.csv"
                predictor.save_predictions(predictions_df, str(output_path))
            
            st.success("✅ 5-Year Forecast Generated Successfully!")
            
            st.info(f"""
            📊 Generated **{len(predictions)} forecast sequence(s)**
            - Forecasting **{model_info['horizon']} years** into the future
            - Based on **{model_info['lookback']} years** of historical data
            - Target variable: **{model_info['target']}**
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Model Information")
                st.write(f"**Model:** {model_info['model_name']}")
                st.write(f"**Type:** {model_info['model_type']}")
                st.write(f"**Lookback:** {model_info['lookback']} years")
                st.write(f"**Horizon:** {model_info['horizon']} years")
                st.write(f"**Target:** {model_info['target']}")
                
                st.subheader("📈 Forecast Years")
                if len(predictions) > 0:
                    base_year = data_rows[-1]['year']
                    forecast_years = [base_year + i for i in range(1, 6)]
                    st.write(f"**Base Year:** {base_year}")
                    st.write(f"**Forecast Years:** {', '.join(map(str, forecast_years))}")
            
            with col2:
                st.subheader("🎯 Forecast Results")
                
                if len(predictions) > 0:
                    forecast_values = predictions[0]
                    
                    st.write(f"**{model_info['target'].upper()} Predictions:**")
                    for i, val in enumerate(forecast_values, 1):
                        year = data_rows[-1]['year'] + i
                        st.metric(f"Year {year}", f"{val:.6f}", delta=None)
                    
                    st.write("")
                    st.write(f"**Min:** {predictions.min():.6f}")
                    st.write(f"**Max:** {predictions.max():.6f}")
                    st.write(f"**Mean:** {predictions.mean():.6f}")
            
            st.subheader("📋 Detailed Forecast Table")
            st.dataframe(predictions_df, use_container_width=True)
            
            # Visualization
            if len(predictions) > 0:
                st.subheader("📈 Forecast Visualization")
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Historical data
                historical_years = [row['year'] for row in data_rows]
                historical_intensity = [row['nca'] / row['gdp'] for row in data_rows]
                
                # Forecast data
                forecast_years = [data_rows[-1]['year'] + i for i in range(1, 6)]
                forecast_values = predictions[0]
                
                ax.plot(historical_years, historical_intensity, 'o-', label='Historical', linewidth=2, markersize=8)
                ax.plot(forecast_years, forecast_values, 's--', label='Forecast', linewidth=2, markersize=8, color='red')
                ax.axvline(x=data_rows[-1]['year'], color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
                
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('Intensity NCA', fontsize=12)
                ax.set_title(f'{country_name} - NCA Intensity Forecast', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Forecast Results (CSV)",
                    data=f,
                    file_name=f"forecast_{country_code}_{session_id}.csv",
                    mime="text/csv",
                    type="primary"
                )
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    This application provides two main functionalities:
    
    ### 1. Exploratory Data Analysis (EDA)
    Automatically generates comprehensive analysis including:
    - Dataset overview and statistics
    - Missing values analysis
    - Correlation heatmap
    - Distribution plots for numerical features
    - Box plots for outlier detection
    - Categorical variable analysis
    
    ### 2. LSTM Time Series Forecasting
    Uses a pre-trained Multi-Output LSTM model for 5-year forecasting:
    - **Model Type:** LSTM (Long Short-Term Memory)
    - **Lookback:** 5 years of historical data
    - **Horizon:** Predicts next 5 years
    - **Target:** intensity_nca (NCA/GDP ratio)
    - **Features:** 7 economic indicators
    - **Input Method:** Manual data entry form
    
    ### Model Architecture
    - **Input Shape:** (5 timesteps, 7 features)
    - **Output Shape:** 5 future predictions
    - **Model File:** multi_output_lstm_h5_lookback5.keras
    
    ### Required Features
    1. NCA (Natural Capital Accounting)
    2. GDP (Gross Domestic Product)
    3. GDP Growth Rate
    4. NCA Growth Rate
    5. Log GDP Growth
    6. Log NCA Growth
    7. Intensity NCA (auto-calculated if not provided)
    
    ### How to Use
    1. Enter country/entity information
    2. Fill in economic data for 5 consecutive years
    3. Click "Generate 5-Year Forecast"
    4. View predictions and download results
    """)
