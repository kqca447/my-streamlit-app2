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

# ==================== 中文字体加载（使用相对路径，需上传字体文件到仓库根目录） ====================
font_path = 'wqy-microhei.ttc'   # 相对路径，请确保该文件已上传到 GitHub 仓库根目录
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
    st.sidebar.warning("字体文件 wqy-microhei.ttc 未找到，请上传该文件到仓库根目录，否则中文可能显示为方框。")

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
    for enc in ['gb18030', 'utf-8-sig', 'utf-8', 'gbk', 'latin1']:  # 加入 utf-8-sig 处理 BOM
        try:
            df = pd.read_csv(file, encoding=enc)
            used_enc = enc
            break
        except:
            continue
    else:
        return None, [file], f"无法解码 {file}"
    
    # 列名标准化：去除首尾空格、统一小写
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 定义标准列名映射（可能的原始列名 -> 标准名）
    # 日期、小时、站点
    rename_map = {}
    # 日期列
    date_candidates = ['date', '日期', '时间']
    for c in date_candidates:
        if c in df.columns:
            rename_map[c] = 'date'
            break
    # 小时列
    hour_candidates = ['hour', '小时']
    for c in hour_candidates:
        if c in df.columns:
            rename_map[c] = 'hour'
            break
    # 站点列
    station_candidates = ['station', '站点']
    for c in station_candidates:
        if c in df.columns:
            rename_map[c] = 'station'
            break
    # AQI列
    aqi_candidates = ['aqi', 'aqi(日)', '空气质量指数']
    for c in aqi_candidates:
        if c in df.columns:
            rename_map[c] = 'AQI'   # 最终大写
            break
    # PM2.5列：模糊匹配
    pm25_candidates = ['pm2.5', 'pm25', 'pm2_5']
    found_pm25 = None
    for c in df.columns:
        if any(p in c for p in pm25_candidates):
            found_pm25 = c
            break
    if found_pm25:
        rename_map[found_pm25] = 'PM2.5'
    # O3列
    o3_candidates = ['o3', '臭氧']
    found_o3 = None
    for c in df.columns:
        if any(p in c for p in o3_candidates):
            found_o3 = c
            break
    if found_o3:
        rename_map[found_o3] = 'O3'
    
    # 执行重命名
    df.rename(columns=rename_map, inplace=True)
    
    # 检查必要列是否存在
    required_cols = ['date', 'hour', 'station', 'AQI']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, [file], f"缺少必要列 {missing}，现有列: {list(df.columns)}"
    
    # 如果 PM2.5 或 O3 仍然不存在，创建空列（填充 NaN）
    if 'PM2.5' not in df.columns:
        df['PM2.5'] = np.nan
    if 'O3' not in df.columns:
        df['O3'] = np.nan
    
    # 构造 datetime
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
    df = df.sort_values(['station', 'datetime']).reset_index(drop=True)
    
    # 优化内存
    for col in ['AQI', 'PM2.5', 'O3']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
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
    # 显示数据框的列名（用于诊断）
    st.sidebar.write("DataFrame 列名:", list(df.columns))

# ==================== 数据过滤与日均值 ====================
primary_station = selected_stations[0] if selected_stations else stations[0]
# 过滤时间
df_filtered = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)].copy()
if df_filtered.empty:
    st.warning("所选时间范围无数据")
    st.stop()

# 计算每日均值（所有站点，仅 AQI 用于对比曲线）
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

# 对于预测，使用主站点的日均数据（动态选择存在的列）
df_primary = df_filtered[df_filtered['station']==primary_station].copy()
if df_primary.empty:
    st.warning(f"主站点 {primary_station} 无数据")
    st.stop()
df_primary = df_primary.set_index('datetime')

# 确定可用的列（AQI必须存在，PM2.5和O3可选）
available_cols = ['AQI']
if 'PM2.5' in df_primary.columns:
    available_cols.append('PM2.5')
if 'O3' in df_primary.columns:
    available_cols.append('O3')
df_daily_primary = df_primary[available_cols].resample('D').mean().dropna().reset_index()

# 如果 PM2.5 或 O3 列缺失但后续代码需要，已经在 df_daily_primary 中不存在，不会出错
# 但为了后面热图和农事建议，可以手动补全（如果缺失则填充 NaN）
if 'PM2.5' not in df_daily_primary.columns:
    df_daily_primary['PM2.5'] = np.nan
if 'O3' not in df_daily_primary.columns:
    df_daily_primary['O3'] = np.nan

# ==================== 1. 历史AQI对比曲线 ====================
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

# ==================== 新增：PM2.5 多站点对比曲线 ====================
if 'PM2.5' in df.columns and not df_filtered['PM2.5'].isna().all():
    st.subheader("📉 历史 PM2.5 趋势 - 多站点对比")
    daily_pm25_list = []
    for stn in selected_stations:
        df_stn = df_filtered[df_filtered['station']==stn].copy()
        if df_stn.empty:
            continue
        df_stn = df_stn.set_index('datetime')
        # 如果该站点 PM2.5 全部为 NaN，则跳过
        if df_stn['PM2.5'].isna().all():
            continue
        daily = df_stn[['PM2.5']].resample('D').mean().dropna().reset_index()
        daily['station'] = stn
        daily_pm25_list.append(daily)
    if daily_pm25_list:
        df_daily_pm25 = pd.concat(daily_pm25_list, ignore_index=True)
        fig_pm25, ax_pm25 = plt.subplots(figsize=(12,4))
        for stn in selected_stations:
            d = df_daily_pm25[df_daily_pm25['station']==stn]
            if not d.empty:
                ax_pm25.plot(d['datetime'], d['PM2.5'], label=stn, alpha=0.8)
        ax_pm25.set_xlabel("日期")
        ax_pm25.set_ylabel("PM2.5 (µg/m³)")
        ax_pm25.set_title("多站点 PM2.5 日均值对比")
        ax_pm25.legend()
        ax_pm25.grid(True, alpha=0.3)
        st.pyplot(fig_pm25)
    else:
        st.info("所选站点在时间范围内无 PM2.5 有效数据")
else:
    st.info("数据中未找到 PM2.5 列，无法绘制 PM2.5 趋势图")

# ==================== 新增：O₃ 多站点对比曲线 ====================
if 'O3' in df.columns and not df_filtered['O3'].isna().all():
    st.subheader("🌫️ 历史 O₃ 趋势 - 多站点对比")
    daily_o3_list = []
    for stn in selected_stations:
        df_stn = df_filtered[df_filtered['station']==stn].copy()
        if df_stn.empty:
            continue
        df_stn = df_stn.set_index('datetime')
        if df_stn['O3'].isna().all():
            continue
        daily = df_stn[['O3']].resample('D').mean().dropna().reset_index()
        daily['station'] = stn
        daily_o3_list.append(daily)
    if daily_o3_list:
        df_daily_o3 = pd.concat(daily_o3_list, ignore_index=True)
        fig_o3, ax_o3 = plt.subplots(figsize=(12,4))
        for stn in selected_stations:
            d = df_daily_o3[df_daily_o3['station']==stn]
            if not d.empty:
                ax_o3.plot(d['datetime'], d['O3'], label=stn, alpha=0.8)
        ax_o3.set_xlabel("日期")
        ax_o3.set_ylabel("O₃ (µg/m³)")
        ax_o3.set_title("多站点 O₃ 日均值对比")
        ax_o3.legend()
        ax_o3.grid(True, alpha=0.3)
        st.pyplot(fig_o3)
    else:
        st.info("所选站点在时间范围内无 O₃ 有效数据")
else:
    st.info("数据中未找到 O₃ 列，无法绘制 O₃ 趋势图")

# ==================== 新增：站点空气质量等级分布（饼图，支持多站点） ====================
st.subheader("📊 站点空气质量等级分布")

if selected_stations:
    # 让用户选择要统计的多个站点（默认全部选中，但限制最多4个避免界面过长）
    stat_stations = st.multiselect(
        "请选择要统计的站点（最多4个）",
        selected_stations,
        default=selected_stations[:min(2, len(selected_stations))]  # 默认选前2个
    )
    # 限制最多4个
    if len(stat_stations) > 4:
        st.warning("最多同时显示4个站点的饼图，已自动截取前4个")
        stat_stations = stat_stations[:4]
else:
    stat_stations = []

if stat_stations:
    # 定义AQI等级函数
    def get_aqi_level(aqi):
        if aqi <= 50:
            return '优'
        elif aqi <= 100:
            return '良'
        elif aqi <= 150:
            return '轻度污染'
        else:
            return '其他'   # 包括中度、重度、严重污染

    # 颜色映射
    level_colors = {
        '优': '#2ecc71',
        '良': '#3498db',
        '轻度污染': '#f39c12',
        '其他': '#e74c3c'
    }

    # 计算每个站点的日均AQI和等级分布
    station_data = {}
    for stn in stat_stations:
        df_stn = df_filtered[df_filtered['station'] == stn].copy()
        if df_stn.empty:
            continue
        df_stn = df_stn.set_index('datetime')
        daily_aqi = df_stn['AQI'].resample('D').mean().dropna()
        if len(daily_aqi) == 0:
            continue
        levels = daily_aqi.apply(get_aqi_level)
        level_counts = levels.value_counts()
        # 确保四个类别都存在
        for level in level_colors.keys():
            if level not in level_counts.index:
                level_counts[level] = 0
        level_counts = level_counts[list(level_colors.keys())]  # 保持顺序
        station_data[stn] = {
            'counts': level_counts,
            'total_days': len(daily_aqi)
        }

    if station_data:
        # 使用 columns 并排显示饼图
        cols = st.columns(len(station_data))
        for i, (stn, data) in enumerate(station_data.items()):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(5, 4))
                # 绘制饼图：只返回 wedges 和 texts（没有 autopct 所以不返回 autotexts）
                wedges, texts = ax.pie(
                    data['counts'].values,
                    labels=None,             # 不直接在饼图上显示标签
                    autopct=None,            # 不直接在饼图上显示百分比
                    colors=[level_colors[l] for l in data['counts'].index],
                    startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                )
                # 图例：显示“等级 (百分比)”
                legend_labels = [
                    f"{level} ({data['counts'][level]/data['total_days']*100:.1f}%)"
                    for level in data['counts'].index if data['counts'][level] > 0
                ]
                ax.legend(
                    wedges, legend_labels,
                    title="空气质量等级",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=9
                )
                ax.set_title(f"{stn}\n(共{data['total_days']}天)", fontsize=12)
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("所选站点在时间范围内无有效日均AQI数据")
else:
    st.info("请先在侧边栏选择至少一个对比站点")
# ==================== 2. 污染物相关性热图（功能1） ====================
if show_corr:
    st.subheader("📊 污染物相关性矩阵")
    # 选择实际存在的污染物列（至少要有 AQI 和另一列）
    corr_cols = [c for c in ['AQI','PM2.5','O3'] if c in df_daily_primary.columns]
    if len(corr_cols) >= 2:
        corr_data = df_daily_primary[corr_cols].dropna()
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
    else:
        st.info("缺少足够污染物列（需要至少 AQI 和另一污染物）")

# ==================== 3. 预测功能 ====================
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
else:
    horizons = list(range(1, pred_days+1))
    X, y, _ = prepare_features(df_daily_primary, n_lags=5, forecast_horizons=horizons)
    if len(X) < 20:
        st.warning("有效样本不足，使用移动平均")
        last_7 = df_daily_primary['AQI'].iloc[-7:].mean()
        future_dates = [(df_daily_primary['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]
        future_preds = [last_7] * pred_days
    else:
        # 训练随机森林
        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=-1)
        rf_multi = MultiOutputRegressor(rf, n_jobs=-1)
        rf_multi.fit(X, y)
        latest_X = X.iloc[-1:].copy()
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
            elif model_option == "仅LightGBM":
                final_preds = lgb_preds
        else:
            final_preds = all_preds
        
        future_preds = final_preds.tolist()
        future_dates = [(df_daily_primary['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)]

# 绘制预测图
fig_pred, ax_pred = plt.subplots(figsize=(10,5))
ax_pred.plot(future_dates, future_preds, marker='o', color='red', linewidth=2, label='预测值')
ax_pred.set_xlabel("日期")
ax_pred.set_ylabel("AQI")
ax_pred.set_title(f"{primary_station} 站点未来{pred_days}天AQI预测")
ax_pred.grid(True, alpha=0.3)
ax_pred.legend()
st.pyplot(fig_pred)

# 预测表格
pred_df = pd.DataFrame({"日期": future_dates, "预测AQI": [f"{v:.1f}" for v in future_preds]})
st.table(pred_df)

# ==================== 智能农事建议 ====================
st.subheader("📋 智能农事建议")
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

st.subheader("🌱 农田作物臭氧与污染风险等级")
if len(df_daily_primary) > 0:
    last_day = df_daily_primary.iloc[-1]
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
    # 预览时只显示存在的列
    preview_cols = [c for c in ['datetime','AQI','PM2.5','O3'] if c in df_daily_primary.columns]
    st.dataframe(df_daily_primary[preview_cols].tail(20))

st.caption("数据来源：北京市空气质量监测站点 | 模型支持随机森林与LightGBM融合 | 置信区间基于随机森林树间标准差")
