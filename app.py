import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

# ==================== 自动查找中文字体（解决 Linux 乱码） ====================
def get_chinese_font():
    # 常见中文字体名称（按优先级）
    font_names = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC',
                  'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei']
    for name in font_names:
        for font in fm.fontManager.ttflist:
            if name.lower() in font.name.lower():
                return font.name
    # 如果都没找到，尝试模糊匹配
    for font in fm.fontManager.ttflist:
        if any(key in font.name.lower() for key in ['cjk', 'hei', 'zen', 'droidsansfallback']):
            return font.name
    return 'sans-serif'  # 最终回退

chinese_font = get_chinese_font()
plt.rcParams['font.sans-serif'] = [chinese_font]
plt.rcParams['axes.unicode_minus'] = False

# 可选：在日志中显示使用的字体（调试用）
print(f"Using font: {chinese_font}")

st.set_page_config(page_title="农业环境决策系统", layout="wide")
st.title("🌾 农业环境智能决策系统")
st.markdown("基于历史日均数据，预测未来多天AQI")

# ==================== 数据加载（无 widget 调用） ====================
@st.cache_data
def load_data():
    all_csv = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
    if not all_csv:
        return None, [], "当前目录下没有找到任何 CSV 文件。"
    
    pattern = re.compile(r'.*pollutants.*hourly.*\.csv', re.IGNORECASE)
    matched_files = [f for f in all_csv if pattern.match(f)]
    if not matched_files:
        return None, all_csv, f"未找到匹配的数据文件。需要文件名包含 'pollutants' 和 'hourly'。找到的 CSV: {all_csv}"
    
    df_list = []
    loaded_info = []
    for file in matched_files:
        success = False
        for enc in ['gb18030', 'utf-8', 'gbk', 'latin1']:
            try:
                df_temp = pd.read_csv(file, encoding=enc)
                df_list.append(df_temp)
                loaded_info.append(f"{file} (编码: {enc})")
                success = True
                break
            except:
                continue
        if not success:
            loaded_info.append(f"{file} (解码失败)")
    
    if not df_list:
        return None, matched_files, "未能成功读取任何数据文件。"
    
    df = pd.concat(df_list, ignore_index=True)
    
    # 必要列的中英文映射
    required_original = {'date': ['date', '日期'], 'hour': ['hour', '小时'], 'station': ['station', '站点']}
    for std_name, possible_names in required_original.items():
        found = None
        for pn in possible_names:
            if pn in df.columns:
                found = pn
                break
        if found is None:
            return None, matched_files, f"缺少必要列: {std_name} (尝试过 {possible_names})。实际列名: {list(df.columns)}"
        if found != std_name:
            df.rename(columns={found: std_name}, inplace=True)
    
    # 污染物列名映射（不区分大小写）
    pollutant_map = {'aqi': 'AQI', 'pm2.5': 'PM2.5', 'o3': 'O3'}
    for old_name in df.columns:
        lower = old_name.lower()
        if lower in pollutant_map:
            df.rename(columns={old_name: pollutant_map[lower]}, inplace=True)
    
    # 检查是否有 AQI, PM2.5, O3 列（如果没有，尝试找近似列）
    if 'AQI' not in df.columns:
        # 尝试找任何包含 'aqi' 的列
        aqi_cols = [c for c in df.columns if 'aqi' in c.lower()]
        if aqi_cols:
            df.rename(columns={aqi_cols[0]: 'AQI'}, inplace=True)
        else:
            return None, matched_files, f"找不到 AQI 列。实际列名: {list(df.columns)}"
    if 'PM2.5' not in df.columns:
        pm_cols = [c for c in df.columns if 'pm2.5' in c.lower() or 'pm25' in c.lower()]
        if pm_cols:
            df.rename(columns={pm_cols[0]: 'PM2.5'}, inplace=True)
        else:
            st.info("未找到 PM2.5 列，部分图表将跳过。")
    if 'O3' not in df.columns:
        o3_cols = [c for c in df.columns if 'o3' in c.lower()]
        if o3_cols:
            df.rename(columns={o3_cols[0]: 'O3'}, inplace=True)
        else:
            st.info("未找到 O₃ 列，部分图表将跳过。")
    
    # 构造 datetime
    try:
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
    except Exception as e:
        return None, matched_files, f"时间列转换失败: {e}"
    
    df = df.sort_values(['station', 'datetime']).reset_index(drop=True)
    return df, loaded_info, None

# 加载数据
df, loaded_info, error_msg = load_data()

if error_msg:
    st.error(f"❌ {error_msg}")
    if df is None and 'loaded_info' in locals():
        with st.expander("调试信息：当前目录下的 CSV 文件"):
            st.write(loaded_info if loaded_info else [])
    st.stop()
else:
    # 显示成功加载的文件（可选，去掉也不影响）
    for info in loaded_info:
        if "解码失败" in info:
            st.sidebar.error(f"❌ {info}")
        else:
            st.sidebar.success(f"✅ {info}")
    st.sidebar.info(f"📊 总计加载 {len(df)} 条记录，{df['station'].nunique()} 个站点")

# 检查站点列
if 'station' not in df.columns:
    st.error("❌ 数据中缺少 'station' 列，无法继续。")
    st.stop()
stations = sorted(df['station'].unique())

# ==================== 侧边栏交互 ====================
st.sidebar.header("🔧 交互设置")
selected_station = st.sidebar.selectbox("📌 选择监测站点", stations)
start_date = st.sidebar.date_input("开始日期", df['datetime'].min().date(),
                                   min_value=df['datetime'].min().date(),
                                   max_value=df['datetime'].max().date())
end_date = st.sidebar.date_input("结束日期", df['datetime'].max().date(),
                                 min_value=df['datetime'].min().date(),
                                 max_value=df['datetime'].max().date())
pred_days = st.sidebar.slider("🔮 预测未来天数", 1, 7, 3)

# 调试开关
show_debug = st.sidebar.checkbox("显示调试信息")
if show_debug:
    st.sidebar.write("当前工作目录:", os.getcwd())
    st.sidebar.write("所有 CSV 文件:", [f for f in os.listdir('.') if f.endswith('.csv')])
    st.sidebar.write("数据列名:", df.columns.tolist())
    st.sidebar.write("前5行数据:", df.head())

# 过滤数据
df_station = df[df['station'] == selected_station].copy()
mask = (df_station['datetime'].dt.date >= start_date) & (df_station['datetime'].dt.date <= end_date)
df_filtered = df_station[mask].copy()
if df_filtered.empty:
    st.warning("所选时间范围内无数据，请调整日期。")
    st.stop()

# 转为日均值
df_hour = df_filtered.set_index('datetime')
numeric_cols = df_hour.select_dtypes(include=[np.number]).columns
df_daily = df_hour[numeric_cols].resample('D').mean().reset_index()
df_daily = df_daily.dropna(subset=['AQI'])

# 显示数据统计
st.sidebar.subheader("📊 当前数据统计（日均值）")
st.sidebar.write(f"AQI 范围: {df_daily['AQI'].min():.1f} - {df_daily['AQI'].max():.1f}")
if 'PM2.5' in df_daily.columns:
    st.sidebar.write(f"PM2.5 范围: {df_daily['PM2.5'].min():.1f} - {df_daily['PM2.5'].max():.1f}")
if 'O3' in df_daily.columns:
    st.sidebar.write(f"O₃ 范围: {df_daily['O3'].min():.1f} - {df_daily['O3'].max():.1f}")

# ==================== 1. 历史 AQI 折线图 ====================
st.subheader("📈 历史 AQI 变化趋势")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_daily['datetime'], df_daily['AQI'], color='orange', linewidth=1.5)
ax1.set_xlabel("日期")
ax1.set_ylabel("AQI 指数")
ax1.set_title(f"{selected_station} 站点 AQI 日均值变化")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ==================== 2. PM2.5 和 O₃ 变化 ====================
if 'PM2.5' in df_daily.columns and 'O3' in df_daily.columns:
    st.subheader("🌫️ 主要污染物变化（PM2.5 与 O₃）")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df_daily['datetime'], df_daily['PM2.5'], label='PM2.5 (µg/m³)', color='red', alpha=0.7)
    ax2.plot(df_daily['datetime'], df_daily['O3'], label=r'$O_3$ (µg/m³)', color='blue', alpha=0.7)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("浓度 (µg/m³)")
    ax2.set_title(f"{selected_station} 站点 PM2.5 与 $O_3$ 日均值变化")
    ax2.legend()
    max_val = max(df_daily['PM2.5'].max(), df_daily['O3'].max())
    ax2.set_ylim(0, max_val * 1.1 if max_val > 0 else 100)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# ==================== 3. 空气质量等级分布饼图 ====================
st.subheader("📊 空气质量等级分布")
def aqi_level(aqi):
    if aqi <= 50: return "优"
    elif aqi <= 100: return "良"
    elif aqi <= 150: return "轻度污染"
    elif aqi <= 200: return "中度污染"
    elif aqi <= 300: return "重度污染"
    else: return "严重污染"

df_daily['等级'] = df_daily['AQI'].apply(aqi_level)
level_counts = df_daily['等级'].value_counts()
total = level_counts.sum()
level_counts = level_counts[level_counts / total >= 0.05]
if level_counts.sum() < total:
    level_counts['其他'] = total - level_counts.sum()

fig3, ax3 = plt.subplots(figsize=(9, 9))
wedges, texts, autotexts = ax3.pie(level_counts, labels=None, autopct='%1.1f%%',
                                   startangle=90, pctdistance=0.8,
                                   textprops={'fontsize': 12})
ax3.legend(wedges, level_counts.index, title="空气质量等级", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax3.set_title(f"{selected_station} 站点空气质量等级分布", fontsize=14)
ax3.axis('equal')
plt.tight_layout()
st.pyplot(fig3)

# ==================== 4. 预测未来 AQI ====================
st.subheader(f"🔮 未来 {pred_days} 天 AQI 趋势预测（集成模型）")

def prepare_features_multioutput(df_daily, target='AQI', n_lags=7, forecast_horizons=[1,2,3]):
    df = df_daily.copy()
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df['rolling_mean_3'] = df[target].rolling(3).mean()
    df['rolling_mean_7'] = df[target].rolling(7).mean()
    if 'PM2.5' in df.columns:
        df['pm25_lag1'] = df['PM2.5'].shift(1)
    if 'O3' in df.columns:
        df['o3_lag1'] = df['O3'].shift(1)
    df = df.select_dtypes(include=[np.number]).dropna()
    for h in forecast_horizons:
        df[f'target_{h}'] = df[target].shift(-h)
    df = df.dropna()
    feature_cols = [c for c in df.columns if not c.startswith('target_') and c != target]
    X = df[feature_cols]
    y = df[[f'target_{h}' for h in forecast_horizons]]
    return X, y, feature_cols

if len(df_daily) < 30:
    st.warning("历史数据不足，无法训练模型，使用简单移动平均预测。")
    last_7 = df_daily['AQI'].iloc[-7:].mean()
    future_dates = [(df_daily['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]
    future_preds = [last_7] * pred_days
else:
    horizons = list(range(1, pred_days+1))
    X, y, feature_cols = prepare_features_multioutput(df_daily, forecast_horizons=horizons)
    if len(X) < 20:
        st.warning("有效样本不足，使用移动平均。")
        last_7 = df_daily['AQI'].iloc[-7:].mean()
        future_dates = [(df_daily['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]
        future_preds = [last_7] * pred_days
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        rf_base = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf_multi = MultiOutputRegressor(rf_base, n_jobs=-1)
        rf_multi.fit(X, y)
        try:
            import xgboost as xgb
            xgb_base = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            xgb_multi = MultiOutputRegressor(xgb_base, n_jobs=-1)
            xgb_multi.fit(X, y)
            use_xgb = True
        except ImportError:
            use_xgb = False
            st.info("未安装 xgboost，仅使用随机森林进行预测。")
        latest_features = X.iloc[-1:].copy()
        preds_rf = rf_multi.predict(latest_features)[0]
        if use_xgb:
            preds_xgb = xgb_multi.predict(latest_features)[0]
            preds = 0.5 * preds_rf + 0.5 * preds_xgb
        else:
            preds = preds_rf
        future_preds = preds.tolist()
        future_dates = [(df_daily['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(future_dates, future_preds, marker='o', linestyle='-', color='red', linewidth=2, label='集成模型预测值')
ax4.set_xlabel("日期")
ax4.set_ylabel("AQI")
ax4.set_title(f"{selected_station} 站点未来{pred_days}天AQI预测")
ax4.grid(True, alpha=0.3)
ax4.legend()
st.pyplot(fig4)

pred_df = pd.DataFrame({"日期": future_dates, "预测AQI": [f"{v:.1f}" for v in future_preds]})
st.table(pred_df)

# ==================== 5. 农田作物风险等级 ====================
st.subheader("🌱 农田作物臭氧与污染风险等级")
last_day = df_daily.iloc[-1]
o3_now = last_day.get('O3', 0)
aqi_now = last_day['AQI']
date_now = last_day['datetime'].strftime('%Y-%m-%d')
if o3_now > 120 or aqi_now > 150:
    risk, color = "高风险", "red"
elif o3_now > 80 or aqi_now > 100:
    risk, color = "中等风险", "orange"
else:
    risk, color = "低风险", "green"
st.markdown(f"**{date_now}**  O₃: {o3_now:.1f} µg/m³, AQI: {aqi_now:.0f}")
st.markdown(f"<span style='color:{color}; font-size:24px; font-weight:bold'>风险等级：{risk}</span>", unsafe_allow_html=True)

# ==================== 6. 农事操作建议 ====================
st.subheader("📋 农事操作建议")
advice = []
if o3_now > 100:
    advice.append("⚠️ 臭氧浓度较高，敏感作物建议喷施保护剂。")
elif o3_now > 70:
    advice.append("🔔 臭氧接近阈值，注意观测作物叶片变化。")
if aqi_now > 150:
    advice.append("😷 严重污染，建议暂停露天农事操作，佩戴防护口罩。")
elif aqi_now > 100:
    advice.append("🍃 轻度污染，减少户外作业时间。")
if o3_now <= 70 and aqi_now <= 100:
    advice.append("✅ 天气条件良好，适宜施肥、打药、收割等农事活动。")
for a in advice:
    st.write(f"- {a}")

with st.expander("📄 查看当前站点数据预览"):
    preview_cols = ['datetime', 'AQI']
    if 'PM2.5' in df_hour.columns:
        preview_cols.append('PM2.5')
    if 'O3' in df_hour.columns:
        preview_cols.append('O3')
    st.dataframe(df_hour.reset_index()[preview_cols].head(20))

st.caption("数据来源：北京市空气质量监测站点 | 预测模型：随机森林 + XGBoost（可选）加权集成 | 风险标准参考《环境空气质量标准》")
