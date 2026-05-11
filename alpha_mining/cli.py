"""
v3.0 命令行接口 (CLI)
用于批量因子生成、定时任务、自动化回测
"""
import argparse
import os
import sys
import yaml
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from .data_hub import DataHub
from .factor_engine import FactorEngine
from .evaluator import Evaluator
from .utils import ensure_dir
from .data_hub import prepare_features


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)


def run_batch(config: dict, expressions: list = None):
    """批量运行因子挖掘"""
    print("=" * 70)
    print("🚀 Alpha 因子挖掘系统 v3.0 - 批量运行模式")
    print("=" * 70)
    
    # 1. 加载数据
    data_config = config.get('data', {})
    factor_config = config.get('factor_generation', {})
    eval_config = config.get('evaluation', {})
    output_config = config.get('output', {})
    
    market = config.get('market', 'a_share')
    start_date = data_config.get('start_date', '2019-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    
    print(f"\n📥 步骤 1/4: 加载 {market} 市场数据")
    print(f"   时间范围: {start_date} ~ {end_date}")
    
    hub = DataHub(
        market=market,
        preferred_source=data_config.get('preferred_source'),
        fallback_sources=data_config.get('fallback_sources'),
        start_date=start_date,
        end_date=end_date
    )
    
    df = hub.get_daily_data(symbols=data_config.get('symbols', 'all'))
    df = prepare_features(df)
    print(f"   数据加载完成: {len(df)} 条记录，{df.index.get_level_values('code').nunique()} 只股票")
    
    # 2. 计算因子
    print(f"\n🧮 步骤 2/4: 计算因子")
    mode = factor_config.get('mode', 'expression')
    print(f"   因子生成模式: {mode}")
    
    engine = FactorEngine(df, cache_dir='./factor_cache')
    
    if mode == 'expression' and expressions:
        # 命令行传入的表达式优先
        factor_dict = engine.batch_compute(expressions)
    else:
        factor_dict = engine.compute_from_config(config)
    
    print(f"   完成计算: {len(factor_dict)} 个因子")
    
    # 3. 评估因子
    print(f"\n📊 步骤 3/4: 因子评估")
    evaluator = Evaluator(
        df,
        n_groups=eval_config.get('n_groups', 5),
        train_ratio=eval_config.get('train_ratio', 0.8),
        neutralize=eval_config.get('neutralize', False)
    )
    
    result_df = evaluator.evaluate(factor_dict)
    print(f"   评估完成: {len(result_df)} 个有效因子")
    
    # 4. 输出结果
    print(f"\n💾 步骤 4/4: 保存结果")
    results_dir = output_config.get('results_dir', './results')
    ensure_dir(results_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 CSV 报告
    csv_path = os.path.join(results_dir, f'factor_report_{timestamp}.csv')
    save_cols = [c for c in result_df.columns if not c.startswith('_')]
    result_df[save_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   报告已保存: {csv_path}")
    
    # 保存图表
    if output_config.get('save_plots', True):
        top_n = min(5, len(result_df))
        sorted_results = result_df.sort_values('Rank ICIR_train', ascending=False).head(top_n)
        
        print(f"   生成 Top {top_n} 因子图表...")
        
        for _, row in sorted_results.iterrows():
            factor_name = row['factor_name']
            safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in factor_name)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            try:
                evaluator.plot_ic_series(row, ax=axes[0, 0])
                evaluator.plot_ls_curve(row, ax=axes[0, 1])
                evaluator.plot_group_cumulative(row, ax=axes[1, 0])
                evaluator.plot_monthly_ic_heatmap(row, ax=axes[1, 1])
            except:
                pass
            
            plt.suptitle(f'因子评估报告: {factor_name}', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            plot_path = os.path.join(results_dir, f'{safe_name}_{timestamp}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   图表已保存至: {results_dir}/")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("✅ 运行完成！Top 5 因子摘要:")
    print("=" * 70)
    
    display_cols = ['factor_name', 'Rank ICIR_train', 'Rank IC_mean_train', 
                   '夏普比率_train', '年化收益率_train', '换手率']
    display_cols = [c for c in display_cols if c in result_df.columns]
    
    print(result_df.sort_values('Rank ICIR_train', ascending=False)[display_cols].head(5).to_string(index=False))
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Alpha 因子挖掘系统 v3.0')
    
    # 通用参数
    parser.add_argument('--config', '-c', default='./config.yaml', help='配置文件路径')
    parser.add_argument('--market', '-m', choices=['a_share', 'us'], help='市场类型，覆盖配置文件')
    
    # 运行模式
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # run 命令
    run_parser = subparsers.add_parser('run', help='运行因子挖掘')
    run_parser.add_argument('--mode', choices=['expression', 'gplearn', 'deep_alpha'], help='因子生成模式')
    run_parser.add_argument('--expressions', '-e', nargs='+', help='因子表达式列表')
    run_parser.add_argument('--no-plots', action='store_true', help='不生成图表')
    
    # info 命令
    subparsers.add_parser('info', help='显示系统信息')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        print("=" * 50)
        print("Alpha 因子挖掘系统 v3.0")
        print("=" * 50)
        print("\n支持的因子生成模式:")
        print("  - expression: 手动表达式模式 (默认)")
        print("  - gplearn: 遗传编程自动发现因子")
        print("  - deep_alpha: 深度网络生成隐式因子")
        print("\n支持的市场:")
        print("  - a_share: A股市场")
        print("  - us: 美股市场")
        print("\n支持的数据源:")
        print("  - akshare, tushare, baostock (A股)")
        print("  - yfinance (美股)")
        print("\n快速开始:")
        print("  python -m alpha_mining run --expressions \"rank(ts_mean(close,20)/close)\"")
        print("=" * 50)
        return
    
    elif args.command == 'run':
        config = load_config(args.config)
        
        # 命令行参数覆盖配置
        if args.market:
            config['market'] = args.market
        if args.mode:
            config.setdefault('factor_generation', {})['mode'] = args.mode
        if args.no_plots:
            config.setdefault('output', {})['save_plots'] = False
        
        run_batch(config, args.expressions)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
