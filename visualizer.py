"""
Alpha因子挖掘系统 - 结果可视化 (v2.1)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from utils import ensure_dir

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class FactorVisualizer:
    """因子结果可视化器"""
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = ensure_dir(output_dir)
    
    def generate_all_charts(self, result_df: pd.DataFrame, top_n: int = 5):
        """
        生成所有图表
        
        Args:
            result_df: 评估结果DataFrame
            top_n: 生成前N个最优因子的详细图表
        """
        if len(result_df) == 0:
            print("[Visualizer] 无有效因子，跳过图表生成")
            return
        
        # 按ICIR排序
        sorted_df = result_df.sort_values('RankIC_IR_train', ascending=False)
        
        # 总体对比图
        self._generate_summary_chart(sorted_df)
        
        # Top因子详细图表
        actual_top_n = min(top_n, len(sorted_df))
        for i, (_, row) in enumerate(sorted_df.head(actual_top_n).iterrows()):
            print(f"  生成因子图表 [{i+1}/{actual_top_n}]: {row['factor_name']}")
            self._generate_factor_detail_charts(row)
    
    def _generate_summary_chart(self, df: pd.DataFrame):
        """生成总体对比图表"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. RankICIR分布
        ax1 = fig.add_subplot(gs[0, 0])
        icir_data = df['RankIC_IR_train'].dropna()
        if len(icir_data) > 0:
            ax1.hist(icir_data, bins=15, alpha=0.7, color='#2ecc71', edgecolor='black')
            ax1.axvline(y=0, color='#e74c3c', linestyle='--', alpha=0.8)
            ax1.axvline(y=0.3, color='#f39c12', linestyle='--', alpha=0.8, label='0.3阈值')
            ax1.set_title('训练集RankIC_IR分布', fontsize=12, fontweight='bold')
            ax1.set_xlabel('RankIC_IR')
            ax1.set_ylabel('因子数量')
            ax1.legend()
        
        # 2. 训练集vs测试集RankIC均值
        ax2 = fig.add_subplot(gs[0, 1])
        if 'RankIC_mean_train' in df.columns and 'RankIC_mean_test' in df.columns:
            train_vals = df['RankIC_mean_train'].dropna()
            test_vals = df['RankIC_mean_test'].dropna()
            if len(train_vals) > 0 and len(test_vals) > 0:
                min_val = min(train_vals.min(), test_vals.min(), -0.1)
                max_val = max(train_vals.max(), test_vals.max(), 0.1)
                ax2.scatter(train_vals, test_vals, alpha=0.7, s=80, 
                           color='#3498db', edgecolors='black', zorder=5)
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
                ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax2.set_title('训练集vs测试集RankIC均值', fontsize=12, fontweight='bold')
                ax2.set_xlabel('训练集RankIC均值')
                ax2.set_ylabel('测试集RankIC均值')
                ax2.legend()
        
        # 3. 夏普比率分布
        ax3 = fig.add_subplot(gs[1, 0])
        sharpe_data = df['夏普比率_train'].dropna() if '夏普比率_train' in df.columns else []
        if len(sharpe_data) > 0:
            ax3.hist(sharpe_data, bins=15, alpha=0.7, color='#9b59b6', edgecolor='black')
            ax3.axvline(x=0.5, color='#e74c3c', linestyle='--', alpha=0.8, label='0.5阈值')
            ax3.set_title('训练集多空夏普比率分布', fontsize=12, fontweight='bold')
            ax3.set_xlabel('夏普比率')
            ax3.set_ylabel('因子数量')
            ax3.legend()
        
        # 4. 收益-回撤散点图
        ax4 = fig.add_subplot(gs[1, 1])
        if '年化收益率_train' in df.columns and '最大回撤_train' in df.columns:
            ret = df['年化收益率_train'].dropna()
            dd = df['最大回撤_train'].dropna()
            if len(ret) > 0 and len(dd) > 0:
                sharpe = df.loc[ret.index, '夏普比率_train'].dropna()
                scatter = ax4.scatter(dd.abs(), ret, c=sharpe, cmap='viridis', 
                                     s=100, alpha=0.7, edgecolors='black')
                ax4.set_xlabel('最大回撤(绝对值)')
                ax4.set_ylabel('年化收益率')
                ax4.set_title('风险收益散点图', fontsize=12, fontweight='bold')
                plt.colorbar(scatter, ax=ax4, label='夏普比率')
        
        plt.savefig(os.path.join(self.output_dir, 'factor_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_factor_detail_charts(self, row: pd.Series):
        """生成单个因子的详细图表"""
        factor_name = row.get('factor_name', 'unknown')
        safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in factor_name)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. IC时间序列 + 滚动均值
        ax1 = fig.add_subplot(gs[0, 0])
        rank_ic = row.get('_rank_ic_series', pd.Series(dtype=float))
        if isinstance(rank_ic, pd.Series) and len(rank_ic) > 0:
            rank_ic_rolling = rank_ic.rolling(20, min_periods=5).mean()
            ax1.plot(rank_ic_rolling.index, rank_ic_rolling.values, 
                    linewidth=2, color='#2980b9', label='20日滚动均值')
            ax1.axhline(y=rank_ic.mean(), color='#e74c3c', linestyle='--', 
                       linewidth=2, label=f'均值: {rank_ic.mean():.3f}')
            ax1.fill_between(rank_ic.index,
                           rank_ic.mean() - 2*rank_ic.std(),
                           rank_ic.mean() + 2*rank_ic.std(),
                           alpha=0.2, color='gray', label='±2σ区间')
            ax1.set_title(f'{factor_name} - RankIC时间序列', fontsize=12, fontweight='bold')
            ax1.set_ylabel('RankIC')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
        
        # 2. 多空净值曲线
        ax2 = fig.add_subplot(gs[0, 1])
        ls_returns = row.get('_ls_returns', pd.Series(dtype=float))
        if isinstance(ls_returns, pd.Series) and len(ls_returns) > 0:
            nav = (1 + ls_returns.dropna()).cumprod()
            ax2.plot(nav.index, nav.values, linewidth=2.5, color='#27ae60')
            ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title(f'{factor_name} - 多空组合净值曲线', fontsize=12, fontweight='bold')
            ax2.set_ylabel('净值')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加绩效标签
            ann = f"年化: {row.get('年化收益率_train', 0):.1%}\n" \
                 f"夏普: {row.get('夏普比率_train', 0):.2f}\n" \
                 f"回撤: {row.get('最大回撤_train', 0):.1%}"
            ax2.text(0.02, 0.98, ann, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 3. 分组累计收益
        ax3 = fig.add_subplot(gs[1, 0])
        group_returns = row.get('_group_returns', pd.DataFrame())
        if isinstance(group_returns, pd.DataFrame) and len(group_returns) > 0:
            colors = plt.cm.RdYlGn(np.linspace(0, 1, len(group_returns.columns)))
            for i, q in enumerate(sorted(group_returns.columns)):
                group_nav = (1 + group_returns[q].dropna()).cumprod()
                ax3.plot(group_nav.index, group_nav.values, linewidth=1.5, 
                        color=colors[i], label=f'Group {q+1}')
            ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title(f'{factor_name} - 分组累计收益对比', fontsize=12, fontweight='bold')
            ax3.set_ylabel('净值')
            ax3.legend(loc='upper left', fontsize=9)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 月度IC热力图
        ax4 = fig.add_subplot(gs[1, 1])
        if isinstance(rank_ic, pd.Series) and len(rank_ic) > 0:
            ic_df = rank_ic.to_frame('ic')
            ic_df['year'] = ic_df.index.year
            ic_df['month'] = ic_df.index.month
            ic_pivot = ic_df.pivot_table(index='year', columns='month', values='ic', aggfunc='mean')
            
            sns.heatmap(ic_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.2f',
                       ax=ax4, cbar_kws={'label': 'RankIC'}, annot_kws={'size': 8})
            ax4.set_title(f'{factor_name} - 月度RankIC均值热力图', fontsize=12, fontweight='bold')
            ax4.set_xlabel('月份')
            ax4.set_ylabel('年份')
        
        plt.savefig(os.path.join(self.output_dir, f'factor_{safe_name}_detail.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, result_df: pd.DataFrame):
        """打印因子评估摘要"""
        print("\n" + "=" * 80)
        print("Alpha因子挖掘结果摘要")
        print("=" * 80)
        
        if len(result_df) == 0:
            print("\n无有效因子结果")
            return
        
        sorted_df = result_df.sort_values('RankIC_IR_train', ascending=False)
        
        print(f"\n总因子数: {len(result_df)}")
        valid_icir = (result_df['RankIC_IR_train'] > 0).sum()
        good_icir = (result_df['RankIC_IR_train'] > 0.3).sum()
        excellent_icir = (result_df['RankIC_IR_train'] > 0.5).sum()
        print(f"ICIR > 0因子数: {valid_icir} ({valid_icir/len(result_df):.1%})")
        print(f"ICIR > 0.3因子数: {good_icir} ({good_icir/len(result_df):.1%})")
        print(f"ICIR > 0.5因子数: {excellent_icir} ({excellent_icir/len(result_df):.1%})")
        
        print("\n" + "-" * 80)
        print("Top 10 因子 (按训练集RankIC_IR排序):")
        print("-" * 80)
        
        display_cols = ['factor_name', 'RankIC_IR_train', 'RankIC_mean_train', 
                        'RankIC_mean_test', '夏普比率_train', '年化收益率_train', '换手率']
        display_df = sorted_df[display_cols].head(10).copy()
        
        # 格式化
        for col in ['RankIC_IR_train', 'RankIC_mean_train', 'RankIC_mean_test', '换手率']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else '')
        if '夏普比率_train' in display_df.columns:
            display_df['夏普比率_train'] = display_df['夏普比率_train'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else '')
        if '年化收益率_train' in display_df.columns:
            display_df['年化收益率_train'] = display_df['年化收益率_train'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else '')
        
        print(display_df.to_string(index=False))
        print("\n" + "=" * 80)
