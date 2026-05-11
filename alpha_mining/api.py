"""
v3.0 高级 API (Notebook 快捷函数)
提供面向 Notebook 的极简调用接口，3-4 行代码完成完整工作流
"""
import pandas as pd
import matplotlib.pyplot as plt

# IPython 可选依赖
try:
    from IPython.display import display, Markdown
except ImportError:
    def display(obj, **kwargs):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string())
        else:
            print(obj)
    def Markdown(text):
        return text

from .data_hub import DataHub, prepare_features
from .factor_engine import FactorEngine
from .evaluator import Evaluator


def init_data(market: str = 'a_share',
              start: str = '2019-01-01',
              end: str = '2024-12-31',
              symbols: str = 'all',
              use_cache: bool = True) -> pd.DataFrame:
    """
    一键初始化数据，自动计算衍生特征
    
    Args:
        market: 市场类型 'a_share' (A股) 或 'us' (美股)
        start: 开始日期 YYYY-MM-DD
        end: 结束日期 YYYY-MM-DD
        symbols: 股票代码列表或 'all'
        use_cache: 是否使用缓存
    
    Returns:
        标准化行情数据，多层索引 (date, code)
    """
    print(f"📥 加载 {market} 市场数据: {start} ~ {end}")
    hub = DataHub(
        market=market,
        start_date=start,
        end_date=end
    )
    df = hub.get_daily_data(symbols=symbols, use_cache=use_cache)
    df = prepare_features(df)
    print(f"✅ 数据加载完成: {len(df)} 条记录，{df.index.get_level_values('code').nunique()} 只股票")
    return df


def quick_evaluate(expression: str, data: pd.DataFrame,
                   n_groups: int = 5,
                   neutralize: bool = False,
                   train_ratio: float = 0.8,
                   show_plots: bool = True) -> pd.DataFrame:
    """
    快速评估单个因子，自动打印指标并展示图表
    
    Args:
        expression: 因子表达式
        data: 行情数据 (init_data 返回的 DataFrame)
        n_groups: 分组数
        neutralize: 是否行业市值中性化
        train_ratio: 训练集比例
        show_plots: 是否显示图表
    
    Returns:
        评估结果单行 DataFrame
    """
    print(f"🧮 计算因子: {expression}")
    engine = FactorEngine(data)
    factor = engine.compute(expression)
    valid_count = factor.count()
    print(f"✅ 因子计算完成，有效值: {valid_count} / {len(factor)}")
    
    evaluator = Evaluator(data, n_groups=n_groups, neutralize=neutralize, train_ratio=train_ratio)
    result = evaluator.evaluate_single(factor)
    
    # 打印关键指标
    print("\n" + "=" * 60)
    print("📊 因子评估结果")
    print("=" * 60)
    metrics = [
        ("训练集 Rank IC", result.get('Rank IC_mean_train', 0)),
        ("训练集 Rank ICIR", result.get('Rank ICIR_train', 0)),
        ("测试集 Rank IC", result.get('Rank IC_mean_test', 0)),
        ("训练集多空年化收益", result.get('年化收益率_train', 0)),
        ("训练集多空夏普", result.get('夏普比率_train', 0)),
        ("日换手率", result.get('换手率', 0)),
    ]
    
    for name, value in metrics:
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    print("=" * 60 + "\n")
    
    # 显示图表
    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # IC 序列
        evaluator.plot_ic_series(result, ax=axes[0])
        
        # 多空曲线
        evaluator.plot_ls_curve(result, ax=axes[1])
        
        # 分组累计收益
        evaluator.plot_group_cumulative(result, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
    
    # 返回单行 DataFrame
    return pd.DataFrame([result])


def compare_factors(expressions: list, data: pd.DataFrame,
                    n_groups: int = 5,
                    neutralize: bool = False,
                    train_ratio: float = 0.8,
                    sort_by: str = 'Rank ICIR_train') -> pd.DataFrame:
    """
    批量评估多个因子并对比结果
    
    Args:
        expressions: 因子表达式列表
        data: 行情数据
        n_groups: 分组数
        neutralize: 是否中性化
        train_ratio: 训练集比例
        sort_by: 排序列
    
    Returns:
        多因子对比结果 DataFrame
    """
    print(f"🔄 批量评估 {len(expressions)} 个因子...")
    
    engine = FactorEngine(data)
    factors_dict = {}
    
    for i, expr in enumerate(expressions):
        name = f"alpha_{i+1:03d}"
        factors_dict[name] = engine.compute(expr)
        print(f"  [{i+1}/{len(expressions)}] {expr[:40]}...")
    
    evaluator = Evaluator(data, n_groups=n_groups, neutralize=neutralize, train_ratio=train_ratio)
    result_df = evaluator.evaluate(factors_dict)
    
    # 按指定列排序
    if sort_by in result_df.columns:
        result_df = result_df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    
    print(f"\n✅ 评估完成，共 {len(result_df)} 个有效因子，按 {sort_by} 降序排列")
    
    # 显示精简结果
    display_cols = ['factor_name', 'Rank ICIR_train', 'Rank IC_mean_train', 
                   'Rank IC_mean_test', '夏普比率_train', '年化收益率_train', '换手率']
    display_cols = [c for c in display_cols if c in result_df.columns]
    
    display(result_df[display_cols])
    
    return result_df
