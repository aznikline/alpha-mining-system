"""
Alpha因子挖掘系统 - 结果可视化
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from utils import ensure_dir

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class FactorVisualizer:
    """因子结果可视化"""
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = ensure_dir(output_dir)
    
    def generate_all_charts(self, result_df: pd.DataFrame, top_n: int = 5):
        """
        生成所有图表
        
        Args:
            result_df: 评估结果DataFrame
            top_n: 生成前N个因子的详细图表
        """
        # 排序（按ICIR降序）
        sorted_df = result_df.sort_values('RankICIR_train', ascending=False)
        
        # 生成总体对比图表
        self._generate_summary_chart(sorted_df)
        
        # 生成Top因子详细图表
        for i, (_, row) in enumerate(sorted_df.head(top_n).iterrows()):
            print(f"  生成因子图表 [{i+1}/{top_n}]: {row['factor_expr'][:40]}...")
            self._generate_factor_detail_charts(row)
    
    def _generate_summary_chart(self, df: pd.DataFrame):
        """生成因子总体对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ICIR分布
        ax = axes[0, 0]
        icir_data = df['RankICIR_train'].dropna()
        ax.hist(icir_data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('训练集RankICIR分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('RankICIR')
        ax.set_ylabel('因子数量')
        
        # 2. 训练集vs测试集IC均值
        ax = axes[0, 1]
        ax.scatter(df['RankIC_mean_train'], df['RankIC_mean_test'], 
                   alpha=0.7, s=80, edgecolors='black')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', alpha=0.5, label='y=x')
        ax.set_title('训练集vs测试集RankIC均值', fontsize=14, fontweight='bold')
        ax.set_xlabel('训练集RankIC均值')
        ax.set_ylabel('测试集RankIC均值')
        ax.legend()
        
        # 3. 夏普比率分布
        ax = axes[1, 0]
        sharpe_data = df['夏普比率_train'].dropna()
        ax.hist(sharpe_data, bins=20, alpha=0.7, color='forestgreen', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('训练集多空夏普比率分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('夏普比率')
        ax.set_ylabel('因子数量')
        
        # 4. 年化收益率vs最大回撤
        ax = axes[1, 1]
        scatter = ax.scatter(df['最大回撤_train'].abs(), df['年化收益率_train'],
                           c=df['夏普比率_train'], cmap='viridis',
                           s=80, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='夏普比率')
        ax.set_title('风险收益散点图', fontsize=14, fontweight='bold')
        ax.set_xlabel('最大回撤(绝对值)')
        ax.set_ylabel('年化收益率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'factor_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: factor_summary.png")
    
    def _generate_factor_detail_charts(self, row: pd.Series):
        """生成单个因子的详细图表"""
        expr = row['factor_expr']
        
        # 生成文件名（简化表达式）
        safe_name = ''.join(c if c.isalnum() else '_' for c in expr[:50])
        safe_name = safe_name[:80]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. IC时间序列
        ax = axes[0, 0]
        ic_series = row['_rank_ic_series']
        if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
            ic_series.rolling(20, min_periods=5).mean().plot(ax=ax, linewidth=2, label='20日滚动均值')
            ax.axhline(y=ic_series.mean(), color='red', linestyle='--', label=f'均值: {ic_series.mean():.3f}')
            ax.fill_between(ic_series.index, 
                          ic_series.mean() - 2*ic_series.std(),
                          ic_series.mean() + 2*ic_series.std(),
                          alpha=0.2, color='gray', label='±2σ')
            ax.set_title(f'RankIC时间序列\nICIR: {row["RankICIR_train"]:.2f}', fontsize=12, fontweight='bold')
            ax.set_ylabel('RankIC')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        # 2. 多空净值曲线
        ax = axes[0, 1]
        ls_returns = row['_ls_returns']
        if isinstance(ls_returns, pd.Series) and len(ls_returns) > 0:
            nav = (1 + ls_returns).cumprod()
            nav.plot(ax=ax, linewidth=2, color='darkblue')
            ax.set_title(f'多空组合净值曲线\n年化: {row["年化收益率_train"]:.1%} | 夏普: {row["夏普比率_train"]:.2f}', 
                       fontsize=12, fontweight='bold')
            ax.set_ylabel('净值')
            ax.tick_params(axis='x', rotation=45)
        
        # 3. 分组累计收益
        ax = axes[1, 0]
        group_returns = row['_group_returns']
        if isinstance(group_returns, pd.DataFrame) and len(group_returns) > 0:
            group_nav = (1 + group_returns).cumprod()
            colors = plt.cm.RdYlGn(np.linspace(0, 1, len(group_nav.columns)))
            for i, col in enumerate(sorted(group_nav.columns)):
                label = f'Group {col+1}' if col == 0 else f'Group {col+1}' if col == len(group_nav.columns)-1 else f'Group {col+1}'
                group_nav[col].plot(ax=ax, linewidth=2, color=colors[i], label=label)
            ax.set_title('分组累计净值曲线', fontsize=12, fontweight='bold')
            ax.set_ylabel('净值')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        # 4. IC热力图（月度）
        ax = axes[1, 1]
        if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
            ic_df = ic_series.to_frame('ic')
            ic_df['year'] = ic_df.index.year
            ic_df['month'] = ic_df.index.month
            ic_pivot = ic_df.pivot_table(index='year', columns='month', values='ic', aggfunc='mean')
            
            sns.heatmap(ic_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.2f', ax=ax)
            ax.set_title('月度RankIC均值热力图', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'因子分析报告: {expr[:60]}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'factor_{safe_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, result_df: pd.DataFrame):
        """打印因子评估摘要"""
        print("\n" + "="*80)
        print("因子挖掘结果摘要")
        print("="*80)
        
        sorted_df = result_df.sort_values('RankICIR_train', ascending=False)
        
        print(f"\n总因子数: {len(result_df)}")
        print(f"有效因子数: {(result_df['RankICIR_train'] > 0).sum()}")
        print(f"优秀因子数(ICIR>0.5): {(result_df['RankICIR_train'] > 0.5).sum()}")
        
        print("\n" + "-"*80)
        print("Top 10 因子 (按训练集ICIR排序):")
        print("-"*80)
        
        display_cols = ['factor_expr', 'RankIC_mean_train', 'RankICIR_train', 
                        'RankIC_mean_test', '夏普比率_train', '年化收益率_train', '换手率']
        display_df = sorted_df[display_cols].head(10).copy()
        
        # 格式化显示
        for col in ['RankIC_mean_train', 'RankIC_mean_test', '年化收益率_train']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        
        for col in ['RankICIR_train', '夏普比率_train']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
        
        if '换手率' in display_df.columns:
            display_df['换手率'] = display_df['换手率'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        
        # 截断过长的表达式
        display_df['factor_expr'] = display_df['factor_expr'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        
        print(display_df.to_string(index=False))
        
        print("\n" + "="*80)
