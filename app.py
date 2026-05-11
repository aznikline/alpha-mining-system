"""
Alpha 因子挖掘系统 v3.0 - Streamlit Web 界面
启动命令: streamlit run app.py
"""
import sys
sys.path.insert(0, '.')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Alpha 因子挖掘系统 v3.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("📊 Alpha 因子挖掘系统 v3.0")
st.markdown("Notebook 优先的自动化因子研究平台")

# 侧边栏配置
with st.sidebar:
    st.header("配置")
    
    market = st.selectbox("市场", ["a_share", "us"], index=0)
    start_date = st.date_input("开始日期", value=pd.to_datetime("2019-01-01"))
    end_date = st.date_input("结束日期", value=pd.to_datetime("2024-12-31"))
    
    n_groups = st.slider("分组数", min_value=3, max_value=10, value=5)
    neutralize = st.checkbox("行业市值中性化", value=False)
    train_ratio = st.slider("训练集比例", min_value=0.5, max_value=0.9, value=0.8)
    
    st.divider()
    st.caption("v3.0 | 支持表达式 / 遗传编程 / 深度学习")

# 因子表达式输入
st.subheader("🧮 因子表达式")
default_expr = "rank(ts_mean(close,20)/close)"
expression = st.text_area("输入因子表达式", value=default_expr, height=100)

st.markdown("""
**常用算子参考:**
- 截面: `rank(x)`, `demean(x)`, `zscore(x)`
- 时序: `ts_mean(x, d)`, `ts_std(x, d)`, `ts_delta(x, d)`, `ts_skew(x, d)`, `ts_kurt(x, d)`
- 数学: `abs(x)`, `sign(x)`, `log(x)`, `power(x, n)`
- 条件: `if_else(cond, x, y)`
""")

# 运行按钮
if st.button("🚀 计算并评估", type="primary", use_container_width=True):
    
    with st.spinner("正在加载数据..."):
        from alpha_mining import DataHub, FactorEngine, Evaluator, prepare_features
        
        # 加载数据
        hub = DataHub(
            market=market,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        data = hub.get_daily_data()
        data = prepare_features(data)
        
        st.success(f"✅ 数据加载完成: {len(data)} 条记录，{data.index.get_level_values('code').nunique()} 只股票")
    
    with st.spinner("正在计算因子..."):
        engine = FactorEngine(data)
        factor = engine.compute(expression)
        valid_count = factor.count()
        
        st.success(f"✅ 因子计算完成，有效值: {valid_count} / {len(factor)}")
    
    with st.spinner("正在评估因子..."):
        evaluator = Evaluator(
            data,
            n_groups=n_groups,
            neutralize=neutralize,
            train_ratio=train_ratio
        )
        result = evaluator.evaluate_single(factor)
    
    # 显示评估指标
    st.subheader("📊 评估结果")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("训练集 Rank ICIR", f"{result.get('Rank ICIR_train', 0):.4f}")
    with col2:
        st.metric("训练集 Rank IC 均值", f"{result.get('Rank IC_mean_train', 0):.4f}")
    with col3:
        st.metric("训练集多空夏普", f"{result.get('夏普比率_train', 0):.2f}")
    with col4:
        st.metric("日换手率", f"{result.get('换手率', 0):.4f}")
    
    # 显示图表
    st.subheader("📈 可视化分析")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        evaluator.plot_ic_series(result, ax=ax1)
        st.pyplot(fig1, use_container_width=True)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        evaluator.plot_ls_curve(result, ax=ax2)
        st.pyplot(fig2, use_container_width=True)
    
    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        evaluator.plot_group_cumulative(result, ax=ax3)
        st.pyplot(fig3, use_container_width=True)
    
    # 月度 IC 热力图
    st.subheader("🔥 月度 IC 热力图")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    evaluator.plot_monthly_ic_heatmap(result, ax=ax4)
    st.pyplot(fig4, use_container_width=True)
    
    # 完整指标表
    with st.expander("查看完整评估指标"):
        # 过滤掉 _ 开头的内部字段
        display_result = {k: v for k, v in result.items() if not k.startswith('_')}
        df = pd.DataFrame([display_result]).T
        df.columns = ['数值']
        st.dataframe(df, use_container_width=True)

# 底部信息
st.divider()
st.caption("""
💡 **提示**: 
- 先尝试默认表达式，熟悉系统后再修改
- 表达式支持任意组合嵌套
- 分组数建议使用 5 或 10
- 中性化功能可以消除行业和市值敞口，提高因子稳健性
""")
