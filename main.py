#!/usr/bin/env python3
"""
Alpha因子挖掘系统 - 主程序
"""
import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np

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
    parser = argparse.ArgumentParser(description='Alpha因子挖掘系统')
    parser.add_argument('--config', '-c', default='./config.yaml', help='配置文件路径')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存，重新计算')
    parser.add_argument('--no-charts', action='store_true', help='不生成图表')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Alpha因子挖掘系统 v1.0")
    print("=" * 80)
    
    # 加载配置
    config = load_config(args.config)
    
    data_config = config.get('data', {})
    factor_config = config.get('factor', {})
    eval_config = config.get('evaluation', {})
    filter_config = config.get('filter', {})
    output_config = config.get('output', {})
    
    # 1. 初始化数据中心
    print("\n[1/5] 初始化数据中心...")
    hub = DataHub(
        preferred_source=data_config.get('preferred_source', 'akshare'),
        fallback_sources=data_config.get('fallback_sources', ['tushare', 'baostock']),
        cache_dir=data_config.get('cache_dir', './data_cache'),
        start_date=data_config.get('start_date', '2020-01-01'),
        end_date=data_config.get('end_date', '2024-12-31'),
        tushare_token=data_config.get('tushare_token')
    )
    
    # 获取数据
    print("  正在加载行情数据...")
    df = hub.get_daily_data(
        symbols=data_config.get('symbols', 'all_a'),
        use_cache=not args.no_cache
    )
    
    print(f"  数据加载完成: {len(df)} 条记录")
    print(f"  日期范围: {df.index.get_level_values('date').min()} 至 {df.index.get_level_values('date').max()}")
    print(f"  股票数量: {df.index.get_level_values('code').nunique()}")
    
    # 2. 初始化因子引擎
    print("\n[2/5] 初始化因子计算引擎...")
    engine = FactorEngine(
        cache_dir=factor_config.get('cache_dir', './factor_cache'),
        use_cache=not args.no_cache
    )
    
    # 获取因子表达式
    expressions = factor_config.get('expressions')
    if expressions is None or len(expressions) == 0:
        print("  使用默认因子表达式...")
        expressions = get_default_factor_expressions()
    
    print(f"  待计算因子数量: {len(expressions)}")
    
    # 3. 批量计算因子
    print("\n[3/5] 计算因子值...")
    factor_dict = engine.evaluate_batch(expressions, df)
    
    valid_count = sum(1 for f in factor_dict.values() if f.count() > 0)
    print(f"  成功计算因子: {valid_count}/{len(factor_dict)}")
    
    # 4. 因子评估
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
    
    # 5. 结果可视化
    print("\n[5/5] 生成结果图表...")
    visualizer = FactorVisualizer(
        output_dir=output_config.get('results_dir', './results')
    )
    
    # 打印摘要
    visualizer.print_summary(result_df)
    
    if not args.no_charts:
        visualizer.generate_all_charts(result_df, top_n=output_config.get('top_n_charts', 5))
    
    # 保存结果
    if output_config.get('save_full_results', True):
        # 移除序列数据，避免序列化问题
        save_df = result_df.copy()
        for col in save_df.columns:
            if col.startswith('_'):
                del save_df[col]
        
        output_path = os.path.join(output_config.get('results_dir', './results'), 'factor_results.csv')
        save_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n  结果已保存至: {output_path}")
    
    print("\n" + "=" * 80)
    print("因子挖掘完成！")
    print(f"  总共计算: {len(result_df)} 个因子")
    print(f"  通过筛选: {len(filtered_df)} 个因子")
    print(f"  结果文件: {output_config.get('results_dir', './results')}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
