#!/usr/bin/env python3
"""
Alpha因子挖掘系统快速测试脚本
使用模拟数据快速验证核心功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from data_hub import DataHub
from factor_engine import FactorEngine, get_default_factor_expressions
from evaluator import FactorEvaluator
from visualizer import FactorVisualizer

def main():
    print("=" * 60)
    print("Alpha因子挖掘系统 - 快速测试")
    print("=" * 60)
    
    # 1. 初始化数据中心（使用模拟数据）
    print("\n[1/4] 初始化数据中心...")
    hub = DataHub(
        preferred_source='akshare',
        start_date='2023-01-01',
        end_date='2024-12-31',
        cache_dir='./data_cache'
    )
    
    # 直接使用模拟数据（避免API调用）
    print("  使用模拟数据...")
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ',
               '600519.SH', '000333.SZ', '002415.SZ', '000568.SZ', '000651.SZ']
    
    # 生成模拟数据
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
    all_data = []
    np.random.seed(42)
    
    for symbol in symbols:
        base_price = 10 + np.random.random() * 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        returns[0] = 0
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'date': dates,
            'code': symbol,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'amount': prices * np.random.randint(1000000, 10000000, len(dates)),
            'turn': np.random.uniform(0.01, 0.1, len(dates)),
        })
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        all_data.append(df)
    
    raw_df = pd.concat(all_data, ignore_index=True)
    df = hub.prepare_features(raw_df)
    df = df.set_index(['date', 'code']).sort_index()
    
    print(f"  数据加载完成: {len(df)} 条记录")
    print(f"  日期范围: {df.index.get_level_values('date').min()} 至 {df.index.get_level_values('date').max()}")
    print(f"  股票数量: {df.index.get_level_values('code').nunique()}")
    
    # 2. 因子计算
    print("\n[2/4] 计算因子...")
    engine = FactorEngine(use_cache=False)
    expressions = get_default_factor_expressions()[:5]  # 先测前5个
    
    factor_dict = engine.evaluate_batch(expressions, df)
    valid_count = sum(1 for f in factor_dict.values() if f.count() > 0)
    print(f"  成功计算因子: {valid_count}/{len(factor_dict)}")
    
    # 3. 因子评估
    print("\n[3/4] 因子评估...")
    evaluator = FactorEvaluator(n_groups=5, train_ratio=0.8, neutralize=False)
    result_df = evaluator.evaluate(factor_dict, df, forward_return_col='return_1d')
    
    # 打印结果
    print("\n" + "-"*60)
    print("因子评估结果:")
    print("-"*60)
    
    display_df = result_df[['factor_expr', 'RankIC_mean_train', 'RankICIR_train', 
                            '夏普比率_train', '年化收益率_train', '换手率']].copy()
    display_df = display_df.sort_values('RankICIR_train', ascending=False)
    
    for col in ['RankIC_mean_train', '年化收益率_train']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    
    for col in ['RankICIR_train', '夏普比率_train', '换手率']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    
    display_df['factor_expr'] = display_df['factor_expr'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
    
    print(display_df.to_string(index=False))
    
    # 4. 生成图表
    print("\n[4/4] 生成图表...")
    visualizer = FactorVisualizer(output_dir='./results')
    visualizer.print_summary(result_df)
    visualizer.generate_all_charts(result_df, top_n=3)
    
    print("\n" + "=" * 60)
    print("✅ 快速测试完成！")
    print(f"  结果已保存至: {os.path.abspath('./results')}/")
    print("=" * 60)

if __name__ == '__main__':
    main()
