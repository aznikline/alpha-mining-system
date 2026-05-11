"""
Alpha 因子挖掘系统 - 因子评估器 v3.0
内置 Notebook 友好的绘图方法，支持训练/测试集划分、行业市值中性化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Optional, Tuple


class Evaluator:
    """因子评估器 v3.0"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 n_groups: int = 5,
                 train_ratio: float = 0.8,
                 neutralize: bool = False):
        """
        初始化评估器
        
        Args:
            data: 行情数据，多层索引 (date, code)
            n_groups: 分组数 (5或10)
            train_ratio: 训练集比例
            neutralize: 是否启用行业市值中性化
        """
        self.data = data
        self.n_groups = n_groups
        self.train_ratio = train_ratio
        self.neutralize = neutralize
        
        # 划分训练/测试集日期
        dates = sorted(set(data.index.get_level_values('date')))
        split_idx = int(len(dates) * train_ratio)
        self.train_dates = set(dates[:split_idx])
        self.test_dates = set(dates[split_idx:])
        
        # 缓存评估结果
        self._results_cache = {}
    
    def _neutralize_factor(self, factor: pd.Series) -> pd.Series:
        """行业市值中性化，取残差作为因子值"""
        if not self.neutralize:
            return factor
        
        factor_df = factor.to_frame('factor')
        
        # 准备中性化特征
        if 'group' in self.data.columns:
            factor_df['group'] = self.data['group']
        
        # 使用市值代理：成交量或者收盘价
        if 'log_cap' in self.data.columns:
            factor_df['log_cap'] = self.data['log_cap']
        elif 'log_volume' in self.data.columns:
            factor_df['log_cap'] = self.data['log_volume']
        elif 'close' in self.data.columns:
            factor_df['log_cap'] = np.log(self.data['close'])
        
        factor_df = factor_df.dropna()
        
        # 按日期做截面中性化
        def _date_neutral(df):
            if len(df) < 10:
                return df['factor']
            
            y = df['factor'].values.astype(float)
            X_cols = [c for c in ['log_cap'] if c in df.columns]
            
            # 构建特征矩阵
            X_list = [np.ones_like(y)]
            for col in X_cols:
                X_list.append(df[col].values.astype(float).reshape(-1, 1))
            
            # 行业哑变量
            if 'group' in df.columns:
                group_dummies = pd.get_dummies(df['group'], drop_first=True).values
                X_list.append(group_dummies.astype(float))
            
            X = np.hstack(X_list)
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                return pd.Series(resid, index=df.index, name='factor')
            except:
                return df['factor'] - df['factor'].mean()
        
        neutralized = factor_df.groupby(level='date', group_keys=False).apply(_date_neutral)
        return neutralized
    
    def _calculate_ic_series(self, factor: pd.Series, forward_return: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """计算 IC 和 Rank IC 时间序列"""
        df = pd.DataFrame({'factor': factor, 'ret': forward_return}).dropna()
        
        ic_values = {}
        rank_ic_values = {}
        
        for date, group in df.groupby(level='date'):
            if len(group) < 10:
                continue
            
            try:
                ic, _ = pearsonr(group['factor'], group['ret'])
                rank_ic, _ = spearmanr(group['factor'], group['ret'])
                ic_values[date] = ic
                rank_ic_values[date] = rank_ic
            except:
                continue
        
        ic_series = pd.Series(ic_values, name='IC').sort_index()
        rank_ic_series = pd.Series(rank_ic_values, name='Rank IC').sort_index()
        
        return ic_series, rank_ic_series
    
    def _calculate_group_returns(self, factor: pd.Series, forward_return: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """计算分组收益率和多空收益"""
        df = pd.DataFrame({'factor': factor, 'ret': forward_return}).dropna()
        
        group_returns = {}
        
        for date, group in df.groupby(level='date'):
            if len(group) < self.n_groups * 2:
                continue
            
            try:
                group['q'] = pd.qcut(group['factor'], self.n_groups, labels=False, duplicates='drop')
                daily_ret = group.groupby('q')['ret'].mean()
                group_returns[date] = daily_ret
            except:
                continue
        
        if len(group_returns) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        group_df = pd.DataFrame(group_returns).T.sort_index()
        
        # 多空收益 = 最高组 - 最低组
        max_q = group_df.columns.max()
        min_q = group_df.columns.min()
        ls_return = group_df[max_q] - group_df[min_q]
        
        return group_df, ls_return
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """计算组合绩效指标"""
        returns = returns.dropna()
        if len(returns) < 10:
            return {}
        
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        calmar = abs(annual_return / max_dd) if max_dd < 0 else 0
        win_rate = (returns > 0).mean()
        
        return {
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '夏普比率': sharpe,
            '最大回撤': max_dd,
            '卡玛比率': calmar,
            '日均收益率': returns.mean(),
            '日胜率': win_rate,
            '回测天数': len(returns),
        }
    
    def evaluate_single(self, factor: pd.Series,
                       forward_return_col: str = 'return_1d') -> Dict:
        """
        评估单个因子
        
        Args:
            factor: 因子值 Series
            forward_return_col: 下一期收益率列名
        
        Returns:
            包含所有评估指标和序列的字典
        """
        # 中性化
        factor = self._neutralize_factor(factor)
        
        # 对齐收益率，shift(-1) 使用下一期真实收益
        forward_return = self.data.groupby(level='code')[forward_return_col].shift(-1)
        forward_return = forward_return.reindex(factor.index)
        
        # 计算 IC
        ic_series, rank_ic_series = self._calculate_ic_series(factor, forward_return)
        
        # 训练集统计
        train_ic = rank_ic_series[rank_ic_series.index.isin(self.train_dates)]
        test_ic = rank_ic_series[rank_ic_series.index.isin(self.test_dates)]
        
        result = {}
        
        # IC 相关指标
        if len(train_ic) > 0:
            result['Rank IC_mean_train'] = train_ic.mean()
            result['Rank IC_std_train'] = train_ic.std()
            result['Rank ICIR_train'] = train_ic.mean() / train_ic.std() if train_ic.std() > 0 else 0
        
        if len(test_ic) > 0:
            result['Rank IC_mean_test'] = test_ic.mean()
            result['Rank IC_std_test'] = test_ic.std()
            result['Rank ICIR_test'] = test_ic.mean() / test_ic.std() if test_ic.std() > 0 else 0
        
        # 分组和多空
        group_returns, ls_returns = self._calculate_group_returns(factor, forward_return)
        
        train_ls = ls_returns[ls_returns.index.isin(self.train_dates)]
        test_ls = ls_returns[ls_returns.index.isin(self.test_dates)]
        
        # 多空绩效
        train_perf = self._calculate_performance_metrics(train_ls)
        test_perf = self._calculate_performance_metrics(test_ls)
        
        for k, v in train_perf.items():
            result[f'{k}_train'] = v
        for k, v in test_perf.items():
            result[f'{k}_test'] = v
        
        # 换手率
        factor_ranked = factor.groupby(level='date').rank(pct=True)
        rank_change = factor_ranked.groupby(level='code').diff().abs()
        result['换手率'] = rank_change.groupby(level='date').mean().mean()
        
        # 保存序列用于绘图
        result['_ic_series'] = ic_series
        result['_rank_ic_series'] = rank_ic_series
        result['_group_returns'] = group_returns
        result['_ls_returns'] = ls_returns
        
        return result
    
    def evaluate(self, factor_dict: Dict[str, pd.Series],
                forward_return_col: str = 'return_1d') -> pd.DataFrame:
        """
        批量评估多个因子
        
        Args:
            factor_dict: {因子名: 因子值 Series}
            forward_return_col: 下一期收益率列名
        
        Returns:
            评估结果 DataFrame，每行一个因子
        """
        results = []
        
        for name, factor in factor_dict.items():
            try:
                result = self.evaluate_single(factor, forward_return_col)
                result['factor_name'] = name
                results.append(result)
            except Exception as e:
                print(f"⚠️  因子 {name} 评估失败: {e}")
                continue
        
        result_df = pd.DataFrame(results)
        
        # 把 factor_name 移到第一列
        if 'factor_name' in result_df.columns:
            cols = ['factor_name'] + [c for c in result_df.columns if c != 'factor_name']
            result_df = result_df[cols]
        
        return result_df
    
    # ==================== 内置绘图方法 (Notebook 友好) ====================
    
    def plot_ic_series(self, result: Dict, ax=None):
        """
        绘制 Rank IC 时间序列 + 滚动均值 + ±2σ 通道
        
        Args:
            result: evaluate_single 返回的结果字典
            ax: matplotlib Axes 对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        rank_ic = result.get('_rank_ic_series', pd.Series(dtype=float))
        if len(rank_ic) == 0:
            ax.text(0.5, 0.5, '无有效 IC 数据', ha='center', va='center')
            return ax
        
        # 20日滚动均值
        rolling_mean = rank_ic.rolling(20, min_periods=5).mean()
        
        ax.plot(rank_ic.index, rank_ic.values, alpha=0.3, color='gray', label='日 Rank IC')
        ax.plot(rolling_mean.index, rolling_mean.values, linewidth=2, color='#2980b9', label='20日滚动均值')
        ax.axhline(y=rank_ic.mean(), color='#e74c3c', linestyle='--', 
                   linewidth=2, label=f'均值: {rank_ic.mean():.3f}')
        ax.axhspan(rank_ic.mean() - 2*rank_ic.std(), rank_ic.mean() + 2*rank_ic.std(), 
                   alpha=0.1, color='gray', label='±2σ 通道')
        
        ax.set_title('Rank IC 时间序列', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rank IC')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3)
        
        return ax
    
    def plot_ls_curve(self, result: Dict, ax=None):
        """
        绘制多空组合净值曲线
        
        Args:
            result: evaluate_single 返回的结果字典
            ax: matplotlib Axes 对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        ls_returns = result.get('_ls_returns', pd.Series(dtype=float))
        if len(ls_returns) == 0:
            ax.text(0.5, 0.5, '无有效多空收益数据', ha='center', va='center')
            return ax
        
        nav = (1 + ls_returns.dropna()).cumprod()
        ax.plot(nav.index, nav.values, linewidth=2.5, color='#27ae60')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # 添加绩效标签
        ann_text = f"年化: {result.get('年化收益率_train', 0):.1%}\n" \
                   f"夏普: {result.get('夏普比率_train', 0):.2f}\n" \
                   f"最大回撤: {result.get('最大回撤_train', 0):.1%}"
        
        ax.text(0.02, 0.98, ann_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        ax.set_title('多空组合净值曲线', fontsize=12, fontweight='bold')
        ax.set_ylabel('净值')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3)
        
        return ax
    
    def plot_group_cumulative(self, result: Dict, ax=None):
        """
        绘制分组累计收益对比图
        
        Args:
            result: evaluate_single 返回的结果字典
            ax: matplotlib Axes 对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        group_returns = result.get('_group_returns', pd.DataFrame())
        if len(group_returns) == 0:
            ax.text(0.5, 0.5, '无有效分组收益数据', ha='center', va='center')
            return ax
        
        # 颜色渐变：红 -> 黄 -> 绿
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(group_returns.columns)))
        
        for i, q in enumerate(sorted(group_returns.columns)):
            group_nav = (1 + group_returns[q].dropna()).cumprod()
            ax.plot(group_nav.index, group_nav.values, linewidth=1.5, 
                   color=colors[i], label=f'Group {q+1}')
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('分组累计收益对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('净值')
        ax.legend(loc='upper left', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3)
        
        return ax
    
    def plot_monthly_ic_heatmap(self, result: Dict, ax=None):
        """
        绘制月度 IC 均值热力图
        
        Args:
            result: evaluate_single 返回的结果字典
            ax: matplotlib Axes 对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        rank_ic = result.get('_rank_ic_series', pd.Series(dtype=float))
        if len(rank_ic) == 0:
            ax.text(0.5, 0.5, '无有效 IC 数据', ha='center', va='center')
            return ax
        
        # 构造年月透视表
        ic_df = rank_ic.to_frame('ic')
        ic_df['year'] = ic_df.index.year
        ic_df['month'] = ic_df.index.month
        ic_pivot = ic_df.pivot_table(index='year', columns='month', values='ic', aggfunc='mean')
        
        # 热力图
        sns.heatmap(ic_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.2f',
                   ax=ax, cbar_kws={'label': 'Rank IC'}, annot_kws={'size': 9})
        
        ax.set_title('月度 Rank IC 均值热力图', fontsize=12, fontweight='bold')
        ax.set_xlabel('月份')
        ax.set_ylabel('年份')
        
        return ax
