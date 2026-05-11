"""
Alpha因子挖掘系统 - 因子评估器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr, pearsonr


class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, n_groups: int = 5, train_ratio: float = 0.8, neutralize: bool = False):
        """
        初始化评估器
        
        Args:
            n_groups: 分组数量，默认5组
            train_ratio: 训练集比例，默认0.8
            neutralize: 是否进行行业市值中性化
        """
        self.n_groups = n_groups
        self.train_ratio = train_ratio
        self.neutralize = neutralize
    
    def _split_train_test(self, dates: pd.Index) -> Tuple[pd.Index, pd.Index]:
        """划分训练集和测试集日期"""
        unique_dates = sorted(dates.unique())
        split_idx = int(len(unique_dates) * self.train_ratio)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]
        return train_dates, test_dates
    
    def _neutralize_factor(self, factor: pd.Series, df: pd.DataFrame) -> pd.Series:
        """行业市值中性化"""
        if not self.neutralize:
            return factor
        
        factor_df = factor.to_frame('factor')
        
        # 合并行业和市值数据
        if 'group' in df.columns:
            factor_df['group'] = df['group']
        if 'log_market_cap' in df.columns:
            factor_df['log_mcap'] = df['log_market_cap']
        
        # 按日期进行截面回归取残差
        def _regress_resid(date_df):
            y = date_df['factor']
            X_cols = []
            
            if 'log_mcap' in date_df.columns:
                X_cols.append('log_mcap')
                X = date_df[X_cols].values
            else:
                X = np.ones((len(y), 1))
            
            # 添加虚拟变量（行业）
            if 'group' in date_df.columns and date_df['group'].nunique() > 1:
                group_dummies = pd.get_dummies(date_df['group'], drop_first=True)
                X = np.column_stack([X, group_dummies.values])
            
            # 回归
            try:
                beta = np.linalg.lstsq(X.astype(float), y.values.astype(float), rcond=None)[0]
                resid = y.values - X.astype(float) @ beta
                return pd.Series(resid, index=date_df.index, name='factor')
            except:
                return y
        
        neutralized = factor_df.groupby(level='date').apply(_regress_resid)
        if isinstance(neutralized, pd.Series):
            return neutralized
        else:
            return neutralized.iloc[:, 0]
    
    def _calculate_ic(self, factor: pd.Series, forward_return: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算IC序列和Rank IC序列
        
        Args:
            factor: 因子值序列
            forward_return: 下一期收益率
        
        Returns:
            (ic_series, rank_ic_series)
        """
        # 对齐索引
        df = pd.DataFrame({'factor': factor, 'ret': forward_return})
        df = df.dropna()
        
        ic_dict = {}
        rank_ic_dict = {}
        
        for date, group in df.groupby(level='date'):
            if len(group) < 10:
                continue
            
            try:
                ic, _ = pearsonr(group['factor'], group['ret'])
                rank_ic, _ = spearmanr(group['factor'], group['ret'])
                ic_dict[date] = ic
                rank_ic_dict[date] = rank_ic
            except:
                continue
        
        ic_series = pd.Series(ic_dict, name='IC').sort_index()
        rank_ic_series = pd.Series(rank_ic_dict, name='RankIC').sort_index()
        
        return ic_series, rank_ic_series
    
    def _calculate_group_returns(self, factor: pd.Series, forward_return: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        计算分组收益率
        
        Returns:
            group_return_df: 每组每日收益率
            ls_return: 多空组合收益率（Top - Bottom）
        """
        df = pd.DataFrame({'factor': factor, 'ret': forward_return})
        df = df.dropna()
        
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
        
        group_return_df = pd.DataFrame(group_returns).T.sort_index()
        
        # 多空收益
        if len(group_return_df.columns) >= 2:
            max_q = group_return_df.columns.max()
            min_q = group_return_df.columns.min()
            ls_return = group_return_df[max_q] - group_return_df[min_q]
        else:
            ls_return = pd.Series(dtype=float)
        
        return group_return_df, ls_return
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """计算绩效指标"""
        if len(returns.dropna()) < 10:
            return {}
        
        returns = returns.dropna()
        
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0
        
        return {
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown,
            '卡玛比率': calmar,
            '日均收益率': returns.mean(),
            '日胜率': (returns > 0).mean(),
        }
    
    def _calculate_turnover(self, factor: pd.Series) -> float:
        """计算因子换手率"""
        # 按股票分组，计算因子排名变化
        factor_df = factor.to_frame('factor')
        factor_df['rank'] = factor_df.groupby(level='date')['factor'].rank(pct=True)
        
        prev_rank = factor_df.groupby(level='code')['rank'].shift(1)
        rank_change = (factor_df['rank'] - prev_rank).abs()
        
        return rank_change.groupby(level='date').mean().mean()
    
    def evaluate_single(self, factor_expr: str, factor: pd.Series, df: pd.DataFrame, 
                        forward_return_col: str = 'return_1d') -> Dict:
        """
        评估单个因子
        
        Args:
            factor_expr: 因子表达式
            factor: 因子值序列
            df: 行情数据
            forward_return_col: 下一期收益率列名
        
        Returns:
            评估结果字典
        """
        result = {
            'factor_expr': factor_expr,
            'n_obs': factor.count(),
            'n_dates': factor.index.get_level_values('date').nunique(),
        }
        
        # 中性化处理
        factor = self._neutralize_factor(factor, df)
        
        # 下一期收益率（需要shift）
        forward_return = df.groupby(level='code')[forward_return_col].shift(-1)
        
        # 划分训练集测试集
        train_dates, test_dates = self._split_train_test(factor.index.get_level_values('date'))
        
        train_mask = factor.index.get_level_values('date').isin(train_dates)
        test_mask = factor.index.get_level_values('date').isin(test_dates)
        
        # 计算IC
        ic_series, rank_ic_series = self._calculate_ic(factor, forward_return)
        
        # 训练集IC
        train_ic = ic_series[ic_series.index.isin(train_dates)]
        test_ic = ic_series[ic_series.index.isin(test_dates)]
        
        train_rank_ic = rank_ic_series[rank_ic_series.index.isin(train_dates)]
        test_rank_ic = rank_ic_series[rank_ic_series.index.isin(test_dates)]
        
        result['IC_mean_train'] = train_ic.mean()
        result['IC_std_train'] = train_ic.std()
        result['ICIR_train'] = train_ic.mean() / train_ic.std() if train_ic.std() > 0 else 0
        result['IC_mean_test'] = test_ic.mean()
        result['IC_std_test'] = test_ic.std()
        result['ICIR_test'] = test_ic.mean() / test_ic.std() if test_ic.std() > 0 else 0
        
        result['RankIC_mean_train'] = train_rank_ic.mean()
        result['RankIC_std_train'] = train_rank_ic.std()
        result['RankICIR_train'] = train_rank_ic.mean() / train_rank_ic.std() if train_rank_ic.std() > 0 else 0
        result['RankIC_mean_test'] = test_rank_ic.mean()
        result['RankIC_std_test'] = test_rank_ic.std()
        result['RankICIR_test'] = test_rank_ic.mean() / test_rank_ic.std() if test_rank_ic.std() > 0 else 0
        
        # 计算分组和多空收益
        group_returns, ls_returns = self._calculate_group_returns(factor, forward_return)
        
        train_ls = ls_returns[ls_returns.index.isin(train_dates)]
        test_ls = ls_returns[ls_returns.index.isin(test_dates)]
        
        # 多空组合绩效
        train_metrics = self._calculate_performance_metrics(train_ls)
        test_metrics = self._calculate_performance_metrics(test_ls)
        
        for k, v in train_metrics.items():
            result[f'{k}_train'] = v
        for k, v in test_metrics.items():
            result[f'{k}_test'] = v
        
        # 换手率
        result['换手率'] = self._calculate_turnover(factor)
        
        # 保存序列数据用于可视化
        result['_ic_series'] = ic_series
        result['_rank_ic_series'] = rank_ic_series
        result['_group_returns'] = group_returns
        result['_ls_returns'] = ls_returns
        
        return result
    
    def evaluate(self, factor_dict: Dict[str, pd.Series], df: pd.DataFrame,
                 forward_return_col: str = 'return_1d') -> pd.DataFrame:
        """
        批量评估因子
        
        Args:
            factor_dict: {expr: factor_series} 字典
            df: 行情数据
            forward_return_col: 下一期收益率列名
        
        Returns:
            评估结果DataFrame，每行一个因子
        """
        results = []
        
        print(f"[Evaluator] 开始评估 {len(factor_dict)} 个因子...")
        
        for i, (expr, factor) in enumerate(factor_dict.items()):
            print(f"  评估 [{i+1}/{len(factor_dict)}]: {expr[:50]}...")
            try:
                result = self.evaluate_single(expr, factor, df, forward_return_col)
                results.append(result)
            except Exception as e:
                print(f"  评估失败 [{expr}]: {e}")
                continue
        
        result_df = pd.DataFrame(results)
        print(f"[Evaluator] 完成 {len(result_df)} 个因子评估")
        
        return result_df
    
    def filter_factors(self, result_df: pd.DataFrame,
                      min_train_icir: float = 0.5,
                      min_test_ic_mean: float = 0.02,
                      min_ls_sharpe_train: float = 1.0) -> pd.DataFrame:
        """
        筛选因子
        
        Args:
            result_df: 评估结果
            min_train_icir: 最小训练集ICIR
            min_test_ic_mean: 最小测试集IC均值
            min_ls_sharpe_train: 最小训练集多空夏普
        
        Returns:
            筛选后的结果
        """
        mask = (
            (result_df['RankICIR_train'] >= min_train_icir) &
            (result_df['RankIC_mean_test'] >= min_test_ic_mean) &
            (result_df['RankIC_mean_test'] * result_df['RankIC_mean_train'] > 0) &  # 方向一致
            (result_df['夏普比率_train'] >= min_ls_sharpe_train)
        )
        
        filtered = result_df[mask].copy()
        print(f"[Evaluator] 筛选通过: {len(filtered)}/{len(result_df)} 个因子")
        
        return filtered
