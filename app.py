import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import re
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    st.warning("LightGBM 未安装，将仅使用随机森林进行预测。请运行 pip install lightgbm")

# ==================== 中文字体加载 ====================
font_path = r"D:\wqy-microhei.ttc"   # 使用原始字符串避免转义问题
if os.path.exists(font_path):
    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        st.sidebar.success("✅ 中文字体加载成功")
    except Exception as e_mon:
        plt.rcParams['font.sans-serif'] = ['sans-serif']
        st.sidebar.error(f"字体加载异常: {e_mon}")
else:
    plt.rcParams['font.sans-serif'] = ['sans-serif']
    st.sidebar.warning("字体文件 D:\\wqy-microhei.ttc 未找到，中文可能显示为方框")


st.set_page_config(page_title="农业环境决策系统", layout="wide")
st.title("🌾 农业环境智能决策系统")
st.markdown("基于历史日均数据，预测未来多天AQI | 支持多站点对比、参数调节、置信区间")

# ==================== 数据加载 ====================
@st.cache_data
def load_data():
    all_csv = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
    if not all_csv:
        return None, [], "没有找到CSV文件"
    pattern = re.compile(r'.*pollutants.*hourly.*\.csv', re.IGNORECASE)
    matched = [f for f in all_csv if pattern.match(f)]
    if not matched:
        return None, all_csv, f"未找到 pollutants hourly 文件: {all_csv}"
    # 优先2025
    file = next((f for f in matched if '2025' in f), matched[0])
    for enc in ['gb18030', 'utf-8', 'gbk', 'latin1']:
        try:
            df = pd.read_csv(file, encoding=enc)
            used_enc = enc
            break
        except:
            continue
    else:
        return None, [file], f"无法解码 {file}"
    
    # 列名标准化（统一转为小写后再将关键列映射回大写）
    rename = {'日期':'date','小时':'hour','站点':'station','AQI(日)':'AQI'}
    df.rename(columns=rename, inplace=True)
    # 将所有列名转为小写
    df.columns = [c.lower() for c in df.columns]
    
    # 将关键列名映射回标准大写（保证后续代码可用）
    if 'aqi' in df.columns:
        df.rename(columns={'aqi':'AQI'}, inplace=True)
    if 'pm2.5' in df.columns:
        df.rename(columns={'pm2.5':'PM2.5'}, inplace=True)
    if 'o3' in df.columns:
        df.rename(columns={'o3':'O3'}, inplace=True)
    
    # 必要列检查（现在使用大写 AQI，其他列保持小写）
    for col in ['date','hour','station','AQI']:
        if col not in df.columns:
            return None, [file], f"缺少列 {col}, 现有列: {list(df.columns)}"
    
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
    df = df.sort_values(['station','datetime']).reset_index(drop=True)
    
    # 优化内存（列名已大写）
    for col in ['AQI','PM2.5','O3']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df, [f"{file} (编码:{used_enc})"], None

df, info, err = load_data()
if err:
    st.error(err)
    st.stop()
for i in info:
    st.sidebar.success(i)
st.sidebar.info(f"📊 总记录 {len(df)}，站点 {df['station'].nunique()}")

stations = sorted(df['station'].unique())

# ==================== 侧边栏交互组件 ====================
st.sidebar.header("🔧 交互设置")
# 功能2: 多站点选择
selected_stations = st.sidebar.multiselect("📌 选择对比站点", stations, default=[stations[0]])
# 功能3: 滑动时间窗口
date_min = df['datetime'].min().date()
date_max = df['datetime'].max().date()
date_range = st.sidebar.slider("📅 时间范围", min_value=date_min, max_value=date_max,
                                value=(date_min, date_max), format="YYYY-MM-DD")
start_date, end_date = date_range
# 预测天数
pred_days = st.sidebar.slider("🔮 预测未来天数", 1, 7, 3)

# 功能4: 模型选择
model_option = st.sidebar.selectbox(
    "🤖 预测模型",
    ["加权融合 (RF+LGB)", "仅随机森林", "仅LightGBM"] if LGB_AVAILABLE else ["仅随机森林"]
)

# 功能6: 超参数调节
st.sidebar.subheader("模型参数 (调整后自动重训练)")
rf_n_estimators = st.sidebar.slider("随机森林 树数量", 10, 100, 50, 10)
rf_max_depth = st.sidebar.slider("随机森林 最大深度", 3, 15, 6, 1)
if LGB_AVAILABLE:
    lgb_n_estimators = st.sidebar.slider("LightGBM 树数量", 20, 150, 80, 10)
    lgb_max_depth = st.sidebar.slider("LightGBM 最大深度", 3, 12, 5, 1)
    lgb_lr = st.sidebar.number_input("LightGBM 学习率", min_value=0.01, max_value=0.3, value=0.05, step=0.01)

# 功能1: 相关性热图开关
show_corr = st.sidebar.checkbox("📊 显示污染物相关性热图")

# 调试信息开关
show_debug = st.sidebar.checkbox("🐞 调试信息")
if show_debug:
    st.sidebar.write("当前目录:", os.getcwd())
    st.sidebar.write("CSV文件:", [f for f in os.listdir('.') if f.endswith('.csv')])

# ==================== 数据过滤与日均值 ====================
# 接收多站点数据，但后续历史曲线和预测按单个站点？为了简化，预测仍使用第一个选中站点，但对比曲线展示所有选中站点
primary_station = selected_stations[0] if selected_stations else stations[0]
# 过滤时间
df_filtered = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)].copy()
if df_filtered.empty:
    st.warning("所选时间范围无数据")
    st.stop()

# 计算每日均值（对于所有站点，分站点处理）
df_hour = df_filtered.set_index('datetime')
# 为了对比曲线，需要按站点分别resample
daily_list = []
for stn in selected_stations:
    df_stn = df_filtered[df_filtered['station']==stn].copy()
    if df_stn.empty:
        continue
    df_stn = df_stn.set_index('datetime')
    daily = df_stn[['AQI']].resample('D').mean().dropna().reset_index()
    daily['station'] = stn
    daily_list.append(daily)
if not daily_list:
    st.warning("所选站点在时间范围内无数据")
    st.stop()
df_daily_all = pd.concat(daily_list, ignore_index=True)

# 对于预测，使用主站点的日均数据
df_primary = df_filtered[df_filtered['station']==primary_station].copy()
if df_primary.empty:
    st.warning(f"主站点 {primary_station} 无数据")
    st.stop()
df_primary = df_primary.set_index('datetime')
df_daily_primary = df_primary[['AQI','PM2.5','O3']].resample('D').mean().dropna().reset_index()

# ==================== 1. 历史AQI对比曲线（功能2） ====================
st.subheader("📈 历史 AQI 趋势 - 多站点对比")
fig_hist, ax_hist = plt.subplots(figsize=(12,4))
for stn in selected_stations:
    d = df_daily_all[df_daily_all['station']==stn]
    if not d.empty:
        ax_hist.plot(d['datetime'], d['AQI'], label=stn, alpha=0.8)
ax_hist.set_xlabel("日期")
ax_hist.set_ylabel("AQI")
ax_hist.set_title(f"多站点 AQI 日均值对比")
ax_hist.legend()
ax_hist.grid(True, alpha=0.3)
st.pyplot(fig_hist)

# ==================== 2. 污染物相关性热图（功能1） ====================
if show_corr:
    st.subheader("📊 污染物相关性矩阵")
    # 使用主站点的日均数据包含更多污染物
    corr_data = df_daily_primary[['AQI','PM2.5','O3']].dropna()
    if len(corr_data) > 1:
        corr = corr_data.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(6,5))
        im = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(corr.columns)))
        ax_corr.set_yticks(range(len(corr.columns)))
        ax_corr.set_xticklabels(corr.columns)
        ax_corr.set_yticklabels(corr.columns)
        plt.colorbar(im, ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("数据点不足，无法计算相关性")

# ==================== 3. 预测功能（模型选择 + 参数调节 + 置信区间） ====================
st.subheader(f"🔮 未来 {pred_days} 天 AQI 预测（{model_option}）")

def prepare_features(df_daily, target='AQI', n_lags=5, forecast_horizons=None):
    df = df_daily.copy()
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df['rolling_mean_7'] = df[target].rolling(7, min_periods=1).mean()
    if 'PM2.5' in df.columns:
        df['pm25_lag1'] = df['PM2.5'].shift(1)
    if 'O3' in df.columns:
        df['o3_lag1'] = df['O3'].shift(1)
    df = df.select_dtypes(include=[np.number]).dropna()
    if forecast_horizons is None:
        forecast_horizons = list(range(1, 4))
    for h in forecast_horizons:
        df[f'target_{h}'] = df[target].shift(-h)
    df = df.dropna()
    feature_cols = [c for c in df.columns if not c.startswith('target_') and c != target]
    X = df[feature_cols].astype(np.float32)
    y = df[[f'target_{h}' for h in forecast_horizons]].astype(np.float32)
    return X, y, feature_cols

if len(df_daily_primary) < 20:
    st.warning("历史数据不足，使用移动平均预测")
    last_7 = df_daily_primary['AQI'].iloc[-min(7,len(df_daily_primary)):].mean()
    future_dates = [(df_daily_primary['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]
    future_preds = [last_7] * pred_days
    pred_std = [0]*pred_days
else:
    horizons = list(range(1, pred_days+1))
    X, y, _ = prepare_features(df_daily_primary, n_lags=5, forecast_horizons=horizons)
    if len(X) < 20:
        st.warning("有效样本不足，使用移动平均")
        last_7 = df_daily_primary['AQI'].iloc[-7:].mean()
        future_dates = [(df_daily_primary['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]
        future_preds = [last_7] * pred_days
        pred_std = [0]*pred_days
    else:
        # 训练随机森林
        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=-1)
        rf_multi = MultiOutputRegressor(rf, n_jobs=-1)
        rf_multi.fit(X, y)
        latest_X = X.iloc[-1:].copy()
        # 预测并计算置信区间（基于树的预测标准差）
        # 对于多输出，需要分别处理每个输出，我们简化：只对第一个输出（第1天）计算置信区间，或对所有输出分别计算（但图例复杂）
        # 这里为了演示，只对第1天的预测值计算标准差，并以阴影表示
        # 实现：获取每个树的预测结果
        rf_estimators = rf_multi.estimators_  # list of estimators for each output
        # 对于第0个目标（未来第1天）
        tree_preds = np.array([tree.predict(latest_X) for tree in rf_estimators])
        pred_day1_mean = tree_preds.mean()
        pred_day1_std = tree_preds.std()
        # 对于多天，我们使用所有目标的标准差（简化：用第一天代表不确定性）
        # 完整做法：对每个 horizon 计算标准差
        all_preds = rf_multi.predict(latest_X)[0]  # (horizons,)
        if LGB_AVAILABLE and model_option != "仅随机森林":
            lgb_model = lgb.LGBMRegressor(
                n_estimators=lgb_n_estimators, max_depth=lgb_max_depth,
                learning_rate=lgb_lr, random_state=42, verbose=-1
            )
            lgb_multi = MultiOutputRegressor(lgb_model, n_jobs=-1)
            lgb_multi.fit(X, y)
            lgb_preds = lgb_multi.predict(latest_X)[0]
            if model_option == "加权融合 (RF+LGB)":
                final_preds = 0.5 * all_preds + 0.5 * lgb_preds
                use_fusion = True
            elif model_option == "仅LightGBM":
                final_preds = lgb_preds
                use_fusion = False
        else:
            final_preds = all_preds
            use_fusion = False
        
        future_preds = final_preds.tolist()
        # 置信区间（基于随机森林的标准差，扩展到所有天数）
        # 简单做法：用第一天的标准差作为参考量
        pred_std = [pred_day1_std] * pred_days
        future_dates = [(df_daily_primary['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]

# ==================== 绘制预测图（含置信区间） ====================
fig_pred, ax_pred = plt.subplots(figsize=(10,5))
ax_pred.plot(future_dates, future_preds, marker='o', color='red', linewidth=2, label='预测值')
# 置信区间阴影
upper = [future_preds[i] + 1.96 * pred_std[i] for i in range(pred_days)]
lower = [future_preds[i] - 1.96 * pred_std[i] for i in range(pred_days)]
ax_pred.fill_between(future_dates, lower, upper, color='red', alpha=0.2, label='95% 置信区间')
ax_pred.set_xlabel("日期")
ax_pred.set_ylabel("AQI")
ax_pred.set_title(f"{primary_station} 站点未来{pred_days}天AQI预测")
ax_pred.grid(True, alpha=0.3)
ax_pred.legend()
st.pyplot(fig_pred)

# 预测表格
pred_df = pd.DataFrame({"日期": future_dates, "预测AQI": [f"{v:.1f}" for v in future_preds]})
st.table(pred_df)

# ==================== 功能9: 动态文字说明 ====================
st.subheader("📋 智能农事建议")
# 比较预测均值与历史均值
if len(df_daily_primary) > 0:
    hist_mean = df_daily_primary['AQI'].mean()
    pred_mean = np.mean(future_preds)
    if pred_mean < hist_mean * 0.9:
        trend = "显著转好"
        advice_text = "👍 未来空气质量预计改善，适宜加强通风、进行施肥等农事操作。"
    elif pred_mean > hist_mean * 1.1:
        trend = "可能转差"
        advice_text = "⚠️ 未来空气质量可能恶化，建议减少户外作业，对敏感作物采取保护措施。"
    else:
        trend = "平稳"
        advice_text = "🟢 未来空气质量保持稳定，可正常安排农事活动。"
    st.info(f"**趋势分析**: 相比历史均值 ({hist_mean:.1f})，未来{pred_days}天预测均值 ({pred_mean:.1f}) 呈现{trend}趋势。\n\n{advice_text}")
else:
    st.info("历史数据不足，无法生成趋势建议。")

# 原有的风险等级和农事建议保留
st.subheader("🌱 农田作物臭氧与污染风险等级")
if len(df_daily_primary) > 0:
    last_day = df_daily_primary.iloc[-1]
    # 注意：列名已是大写 'O3'
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
    
    advice_list = []
    if o3_now > 100:
        advice_list.append("⚠️ 臭氧浓度较高，敏感作物建议喷施保护剂。")
    elif o3_now > 70:
        advice_list.append("🔔 臭氧接近阈值，注意观测作物叶片变化。")
    if aqi_now > 150:
        advice_list.append("😷 严重污染，建议暂停露天农事操作，佩戴防护口罩。")
    elif aqi_now > 100:
        advice_list.append("🍃 轻度污染，减少户外作业时间。")
    if o3_now <= 70 and aqi_now <= 100:
        advice_list.append("✅ 天气条件良好，适宜施肥、打药、收割等农事活动。")
    for a in advice_list:
        st.write(f"- {a}")

with st.expander("📄 查看当前数据预览"):
    st.dataframe(df_daily_primary[['datetime','AQI','PM2.5','O3']].tail(20))

st.caption("数据来源：北京市空气质量监测站点 | 模型支持随机森林与LightGBM融合 | 置信区间基于随机森林树间标准差")
