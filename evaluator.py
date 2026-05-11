"""
Alpha因子挖掘系统 - 因子评估器 (v2.1)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr, pearsonr


class FactorEvaluator:
    """因子评估器 (v2.1)"""
    
    def __init__(self, n_groups: int = 5, train_ratio: float = 0.8, neutralize: bool = False):
        self.n_groups = n_groups
        self.train_ratio = train_ratio
        self.neutralize = neutralize
    
    def _split_train_test_dates(self, df_index):
        """划分训练集和测试集日期"""
        dates = sorted(set(df_index.get_level_values('date')))
        split_idx = int(len(dates) * self.train_ratio)
        return dates[:split_idx], dates[split_idx:]
    
    def _neutralize_factor(self, factor: pd.Series, df: pd.DataFrame) -> pd.Series:
        """行业市值中性化"""
        if not self.neutralize:
            return factor
        
        factor_df = factor.to_frame('factor')
        
        if 'group' in df.columns:
            factor_df['group'] = df['group']
        if 'log_market_cap' in df.columns:
            factor_df['log_mcap'] = df['log_market_cap']
        elif 'log_volume' in df.columns:
            factor_df['log_mcap'] = df['log_volume']
        else:
            factor_df['log_mcap'] = np.log(df['volume'] + 1)
        
        def regression_resid(date_df):
            y = date_df['factor'].values
            
            # 构建特征矩阵
            X_cols = []
            if 'log_mcap' in date_df.columns:
                X_cols.append(date_df['log_mcap'].values.reshape(-1, 1))
            
            # 行业虚拟变量
            if 'group' in date_df.columns and date_df['group'].nunique() > 1:
                group_dummies = pd.get_dummies(date_df['group'], drop_first=True).values
                X_cols.append(group_dummies)
            
            if len(X_cols) == 0:
                return y - y.mean()
            
            X = np.hstack([np.ones((len(y), 1)), *X_cols])
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X.astype(float), y.astype(float), rcond=None)
                resid = y - X @ beta
                return resid
            except:
                return y - y.mean()
        
        neutralized = factor_df.groupby(level='date').apply(regression_resid)
        if isinstance(neutralized, pd.DataFrame):
            neutralized = neutralized.iloc[:, 0]
        
        return neutralized
    
    def _calculate_ic(self, factor: pd.Series, forward_return: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """计算IC和Rank IC序列"""
        df = pd.DataFrame({'factor': factor, 'ret': forward_return})
        df = df.dropna()
        
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
        rank_ic_series = pd.Series(rank_ic_values, name='RankIC').sort_index()
        
        return ic_series, rank_ic_series
    
    def _calculate_group_returns(self, factor: pd.Series, forward_return: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """计算分组收益率和多空收益"""
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
        
        if len(group_returns) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        group_df = pd.DataFrame(group_returns).T.sort_index()
        
        # 多空收益
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
    
    def evaluate_single(self, factor_result, df: pd.DataFrame,
                       forward_return_col: str = 'return_1d') -> Dict:
        """
        评估单个因子
        
        Args:
            factor_result: FactorResult 对象或字典
            df: 行情数据
            forward_return_col: 下一期收益率列名
        
        Returns:
            包含评估结果和元信息的字典
        """
        # 提取因子值和元信息
        if hasattr(factor_result, 'values'):
            factor_values = factor_result.values
            meta = factor_result.to_dict()['meta'] if hasattr(factor_result, 'to_dict') else {}
            factor_name = factor_result.name
        else:
            factor_values = factor_result.get('values', pd.Series())
            meta = factor_result.get('meta', {})
            factor_name = meta.get('name', 'unknown')
        
        result = {
            'factor_name': factor_name,
            **meta
        }
        
        # 中性化处理
        factor_values = self._neutralize_factor(factor_values, df)
        
        # 计算下一期收益率（shift -1）
        forward_return = df.groupby(level='code')[forward_return_col].shift(-1)
        forward_return = forward_return.reindex(factor_values.index)
        
        # 划分训练测试集
        train_dates, test_dates = self._split_train_test_dates(factor_values.index)
        train_mask = factor_values.index.get_level_values('date').isin(train_dates)
        test_mask = factor_values.index.get_level_values('date').isin(test_dates)
        
        # 计算IC
        ic_series, rank_ic_series = self._calculate_ic(factor_values, forward_return)
        
        # 训练集IC统计
        train_ic = ic_series[ic_series.index.isin(train_dates)]
        test_ic = ic_series[ic_series.index.isin(test_dates)]
        
        train_rank_ic = rank_ic_series[rank_ic_series.index.isin(train_dates)]
        test_rank_ic = rank_ic_series[rank_ic_series.index.isin(test_dates)]
        
        if len(train_rank_ic) > 0:
            result['RankIC_mean_train'] = train_rank_ic.mean()
            result['RankIC_std_train'] = train_rank_ic.std()
            result['RankIC_IR_train'] = train_rank_ic.mean() / train_rank_ic.std() if train_rank_ic.std() > 0 else 0
        
        if len(test_rank_ic) > 0:
            result['RankIC_mean_test'] = test_rank_ic.mean()
            result['RankIC_std_test'] = test_rank_ic.std()
            result['RankIC_IR_test'] = test_rank_ic.mean() / test_rank_ic.std() if test_rank_ic.std() > 0 else 0
        
        # 分组和多空
        group_returns, ls_returns = self._calculate_group_returns(factor_values, forward_return)
        
        train_ls = ls_returns[ls_returns.index.isin(train_dates)]
        test_ls = ls_returns[ls_returns.index.isin(test_dates)]
        
        # 多空绩效
        train_perf = self._calculate_performance_metrics(train_ls)
        test_perf = self._calculate_performance_metrics(test_ls)
        
        for k, v in train_perf.items():
            result[f'{k}_train'] = v
        for k, v in test_perf.items():
            result[f'{k}_test'] = v
        
        # 换手率
        factor_ranked = factor_values.groupby(level='date').rank(pct=True)
        rank_change = factor_ranked.groupby(level='code').diff().abs()
        result['换手率'] = rank_change.groupby(level='date').mean().mean()
        
        # 保存序列用于可视化
        result['_ic_series'] = ic_series
        result['_rank_ic_series'] = rank_ic_series
        result['_group_returns'] = group_returns
        result['_ls_returns'] = ls_returns
        
        return result
    
    def evaluate(self, factor_dict: Dict, df: pd.DataFrame,
                forward_return_col: str = 'return_1d') -> pd.DataFrame:
        """
        批量评估因子
        
        Args:
            factor_dict: {factor_name: FactorResult或dict}
            df: 行情数据
        
        Returns:
            评估结果DataFrame
        """
        results = []
        total = len(factor_dict)
        
        print(f"[FactorEvaluator] 开始评估 {total} 个因子...")
        
        for i, (name, factor_result) in enumerate(factor_dict.items()):
            print(f"  评估 [{i+1}/{total}]: {name}")
            try:
                result = self.evaluate_single(factor_result, df, forward_return_col)
                results.append(result)
            except Exception as e:
                print(f"  评估失败 [{name}]: {e}")
                continue
        
        result_df = pd.DataFrame(results)
        print(f"[FactorEvaluator] 完成 {len(result_df)} 个因子评估")
        
        return result_df
    
    def filter_factors(self, result_df: pd.DataFrame,
                       min_train_icir: float = 0.3,
                       min_test_ic_mean: float = 0.01,
                       min_ls_sharpe_train: float = 0.5) -> pd.DataFrame:
        """
        筛选有效因子
        
        Args:
            result_df: 评估结果
            min_train_icir: 最小训练集RankIC_IR
            min_test_ic_mean: 最小测试集RankIC均值
            min_ls_sharpe_train: 最小训练集多空夏普
        
        Returns:
            筛选后的因子
        """
        if len(result_df) == 0:
            return result_df
        
        # 训练集ICIR达标
        mask1 = result_df['RankIC_IR_train'] >= min_train_icir
        
        # 测试集IC均值达标，且方向与训练集一致
        if 'RankIC_mean_test' in result_df.columns:
            same_sign = (result_df['RankIC_mean_train'] * result_df['RankIC_mean_test']) > 0
            mask2 = (result_df['RankIC_mean_test'].abs() >= min_test_ic_mean) & same_sign
        else:
            mask2 = mask1
        
        # 多空夏普达标
        if '夏普比率_train' in result_df.columns:
            mask3 = result_df['夏普比率_train'] >= min_ls_sharpe_train
        else:
            mask3 = mask1
        
        filtered = result_df[mask1 & mask2 & mask3].copy()
        print(f"[FactorEvaluator] 筛选通过: {len(filtered)}/{len(result_df)} 个因子")
        
        return filtered
