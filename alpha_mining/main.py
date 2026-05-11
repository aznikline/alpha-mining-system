#!/usr/bin/env python3
"""
Alpha因子挖掘系统主程序 v2.1
支持：表达式因子、遗传编程因子、深度学习因子
"""
import os
import sys
import yaml
import argparse
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_hub import DataHub
from factor_engine import FactorEngine, get_default_factor_expressions
from evaluator import FactorEvaluator
from visualizer import FactorVisualizer


def load_config(config_path: str = './config.yaml') -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Alpha因子挖掘系统 v2.1')
    parser.add_argument('--config', '-c', default='./config.yaml', help='配置文件路径')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存，重新计算')
    parser.add_argument('--no-charts', action='store_true', help='不生成图表')
    parser.add_argument('--market', '-m', choices=['a_share', 'us'], help='市场类型，覆盖配置')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Alpha因子挖掘系统 v2.1")
    print("=" * 80)
    
    # 加载配置
    config = load_config(args.config)
    
    data_config = config.get('data', {})
    factor_config = config.get('factor_generation', {})
    eval_config = config.get('evaluation', {})
    filter_config = config.get('filter', {})
    output_config = config.get('output', {})
    
    # 市场参数覆盖
    market = args.market or config.get('market', 'a_share')
    print(f"\n市场设置: {market}")
    
    # 1. 初始化数据中心并获取数据
    print("\n[1/5] 初始化数据中心...")
    hub = DataHub(
        market=market,
        preferred_source=data_config.get('preferred_source'),
        fallback_sources=data_config.get('fallback_sources'),
        cache_dir=data_config.get('cache_dir', './data_cache'),
        start_date=data_config.get('start_date', '2020-01-01'),
        end_date=data_config.get('end_date', '2024-12-31'),
        tushare_token=data_config.get('tushare_token')
    )
    
    print(f"  拉取数据: {data_config.get('start_date')} ~ {data_config.get('end_date')}")
    df = hub.get_daily_data(
        symbols=data_config.get('symbols', 'all'),
        use_cache=not args.no_cache
    )
    
    print(f"  数据加载完成: {len(df)} 条记录")
    print(f"  时间范围: {df.index.get_level_values('date').min()} ~ {df.index.get_level_values('date').max()}")
    print(f"  股票数量: {df.index.get_level_values('code').nunique()}")
    
    # 2. 初始化因子引擎
    print("\n[2/5] 初始化因子引擎...")
    engine = FactorEngine(
        cache_dir=data_config.get('cache_dir', './factor_cache'),
        use_cache=not args.no_cache
    )
    
    # 3. 计算因子
    print("\n[3/5] 计算因子...")
    mode = factor_config.get('mode', 'expression')
    print(f"  因子生成模式: {mode}")
    
    factor_results = {}
    
    if mode == 'expression':
        expressions = factor_config.get('expressions')
        if not expressions:
            expressions = get_default_factor_expressions()
        print(f"  计算 {len(expressions)} 个表达式因子")
        factor_results = engine.evaluate_batch(expressions, df)
    
    elif mode == 'gplearn':
        gp_params = factor_config.get('gp', {})
        forward_return = df.groupby(level='code')['return_1d'].shift(-1)
        factor_results = engine.generate_gp_factors(
            df, forward_return,
            population_size=gp_params.get('population_size', 100),
            generations=gp_params.get('generations', 15),
            top_n=gp_params.get('top_n', 5)
        )
    
    elif mode == 'deep_alpha':
        deep_params = factor_config.get('deep_alpha', {})
        forward_return = df.groupby(level='code')['return_1d'].shift(-1)
        factor_results = engine.generate_deep_alpha_factors(
            df, forward_return,
            model_type=deep_params.get('model_type', 'dnn'),
            hidden_layers=deep_params.get('hidden_layers', [128, 64, 32]),
            num_output_factors=deep_params.get('num_output_factors', 10),
            epochs=deep_params.get('epochs', 30)
        )
    
    # 转换为evaluator需要的格式
    factor_dict = {name: fr.to_dict() for name, fr in factor_results.items()}
    valid_count = sum(1 for fr in factor_results.values() if fr.values.count() > 100)
    print(f"  成功计算因子: {valid_count}/{len(factor_results)}")
    
    # 4. 评估因子
    print("\n[4/5] 因子评估...")
    evaluator = FactorEvaluator(
        n_groups=eval_config.get('n_groups', 5),
        train_ratio=eval_config.get('train_ratio', 0.8),
        neutralize=eval_config.get('neutralize', False)
    )
    
    result_df = evaluator.evaluate(
        factor_dict,
        df,
        forward_return_col=eval_config.get('forward_return_col', 'return_1d')
    )
    
    # 筛选因子
    filtered_df = evaluator.filter_factors(
        result_df,
        min_train_icir=filter_config.get('min_train_icir', 0.3),
        min_test_ic_mean=filter_config.get('min_test_ic_mean', 0.01),
        min_ls_sharpe_train=filter_config.get('min_ls_sharpe_train', 0.5)
    )
    
    # 5. 输出结果
    print("\n[5/5] 生成结果...")
    visualizer = FactorVisualizer(
        output_dir=output_config.get('results_dir', './results')
    )
    
    # 打印摘要
    visualizer.print_summary(result_df)
    
    # 生成图表
    if not args.no_charts:
        visualizer.generate_all_charts(
            result_df,
            top_n=output_config.get('top_n_charts', 5)
        )
    
    # 保存结果
    if output_config.get('save_full_results', True):
        output_path = os.path.join(output_config.get('results_dir', './results'), 'factor_results.csv')
        
        # 移除序列列用于保存
        save_df = result_df.copy()
        for col in save_df.columns:
            if col.startswith('_'):
                del save_df[col]
        
        save_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n  结果已保存至: {output_path}")
    
    print("\n" + "=" * 80)
    print("因子挖掘完成！")
    print(f"  总共计算: {len(result_df)} 个因子")
    print(f"  通过筛选: {len(filtered_df)} 个因子")
    print(f"  图表输出: {os.path.abspath(output_config.get('results_dir', './results'))}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
