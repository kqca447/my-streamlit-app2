import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（使用系统自带，避免乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # 如果乱码后续再改
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="农业环境决策系统", layout="wide")
st.title("🌾 农业环境智能决策系统")
st.markdown("基于历史日均数据，预测未来多天AQI")

@st.cache_data
def load_data():
    # 只加载一个文件（优先2025）
    files = [f for f in os.listdir('.') if f.endswith('.csv') and 'pollutants' in f.lower()]
    if not files:
        return None, "未找到CSV文件"
    # 优先选择2025年的文件
    file_2025 = [f for f in files if '2025' in f]
    file = file_2025[0] if file_2025 else files[0]
    df = pd.read_csv(file, encoding='gb18030')
    # 重命名列
    df.rename(columns={'date': 'date', 'hour': 'hour', 'station': 'station'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
    df = df.sort_values(['station', 'datetime']).reset_index(drop=True)
    return df, f"已加载 {file}"

df, msg = load_data()
if df is None:
    st.error(msg)
    st.stop()
st.sidebar.success(msg)
st.sidebar.info(f"共 {len(df)} 条记录，{df['station'].nunique()} 个站点")

stations = sorted(df['station'].unique())
selected_station = st.sidebar.selectbox("站点", stations)
start_date = st.sidebar.date_input("开始日期", df['datetime'].min().date())
end_date = st.sidebar.date_input("结束日期", df['datetime'].max().date())
pred_days = st.sidebar.slider("预测天数", 1, 7, 3)

# 过滤数据
df_station = df[df['station'] == selected_station].copy()
mask = (df_station['datetime'].dt.date >= start_date) & (df_station['datetime'].dt.date <= end_date)
df_filtered = df_station[mask]
if df_filtered.empty:
    st.warning("无数据，请调整日期")
    st.stop()

# 转日均
df_hour = df_filtered.set_index('datetime')
numeric_cols = df_hour.select_dtypes(include=[np.number]).columns
df_daily = df_hour[numeric_cols].resample('D').mean().reset_index().dropna(subset=['AQI'])

# 简单预测（移动平均）
st.subheader("历史AQI趋势")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df_daily['datetime'], df_daily['AQI'], color='orange')
ax.set_xlabel("日期")
ax.set_ylabel("AQI")
ax.set_title(f"{selected_station} AQI趋势")
st.pyplot(fig)

st.subheader(f"未来{pred_days}天预测（移动平均）")
last_7 = df_daily['AQI'].iloc[-7:].mean()
future_dates = [(df_daily['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]
future_preds = [last_7] * pred_days
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(future_dates, future_preds, marker='o', color='red')
ax2.set_xlabel("日期")
ax2.set_ylabel("AQI")
ax2.set_title("预测值")
st.pyplot(fig2)
st.table(pd.DataFrame({"日期": future_dates, "预测AQI": future_preds}))
