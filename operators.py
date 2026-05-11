"""
Alpha因子挖掘系统 - 算子库
包含截面算子和时序算子
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple


# ============= 截面算子（按日期分组计算） =============

def rank(x: pd.Series) -> pd.Series:
    """截面排序，输出0-1之间的秩"""
    return x.groupby(level='date').rank(pct=True)


def scale(x: pd.Series, scale: float = 1.0) -> pd.Series:
    """截面标准化，使和为scale"""
    def _scale(s):
        return s / s.abs().sum() * scale
    return x.groupby(level='date').transform(_scale)


def zscore(x: pd.Series) -> pd.Series:
    """截面Z-score标准化"""
    def _zscore(s):
        return (s - s.mean()) / s.std()
    return x.groupby(level='date').transform(_zscore)


def demean(x: pd.Series) -> pd.Series:
    """截面去均值"""
    return x - x.groupby(level='date').transform('mean')


# ============= 时序算子（按股票分组，滑动窗口计算） =============

def ts_mean(x: pd.Series, d: int) -> pd.Series:
    """d日移动平均"""
    return x.groupby(level='code').rolling(window=d, min_periods=max(1, d//2)).mean().droplevel(0)


def ts_std(x: pd.Series, d: int) -> pd.Series:
    """d日移动标准差"""
    return x.groupby(level='code').rolling(window=d, min_periods=max(1, d//2)).std().droplevel(0)


def ts_min(x: pd.Series, d: int) -> pd.Series:
    """d日最小值"""
    return x.groupby(level='code').rolling(window=d, min_periods=max(1, d//2)).min().droplevel(0)


def ts_max(x: pd.Series, d: int) -> pd.Series:
    """d日最大值"""
    return x.groupby(level='code').rolling(window=d, min_periods=max(1, d//2)).max().droplevel(0)


def ts_delta(x: pd.Series, d: int) -> pd.Series:
    """d日变化量 = x - shift(x, d)"""
    return x - shift(x, d)


def ts_pct_change(x: pd.Series, d: int) -> pd.Series:
    """d日变化率"""
    return x.groupby(level='code').pct_change(periods=d).droplevel(0)


def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """x与y的d日滚动相关系数"""
    df = pd.DataFrame({'x': x, 'y': y})
    def _corr(g):
        return g['x'].rolling(d, min_periods=max(1, d//2)).corr(g['y'])
    return df.groupby(level='code').apply(_corr).droplevel(0)


def ts_cov(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """x与y的d日滚动协方差"""
    df = pd.DataFrame({'x': x, 'y': y})
    def _cov(g):
        return g['x'].rolling(d, min_periods=max(1, d//2)).cov(g['y'])
    return df.groupby(level='code').apply(_cov).droplevel(0)


def shift(x: pd.Series, d: int) -> pd.Series:
    """滞后d期"""
    return x.groupby(level='code').shift(d).droplevel(0)


def ts_zscore(x: pd.Series, d: int) -> pd.Series:
    """d日滚动Z-score"""
    return (x - ts_mean(x, d)) / ts_std(x, d)


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """d日内时序排名（0-1）"""
    def _ts_rank(s):
        return s.rolling(d, min_periods=max(1, d//2)).apply(lambda x: x.rank(pct=True).iloc[-1])
    return x.groupby(level='code').apply(_ts_rank).droplevel(0)


def ts_skew(x: pd.Series, d: int) -> pd.Series:
    """d日滚动偏度 - 衡量分布不对称性"""
    return x.groupby(level='code').rolling(window=d, min_periods=max(1, d//2)).skew().droplevel(0)


def ts_kurt(x: pd.Series, d: int) -> pd.Series:
    """d日滚动峰度 - 衡量分布尾部厚度"""
    return x.groupby(level='code').rolling(window=d, min_periods=max(1, d//2)).kurt().droplevel(0)


def residual(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """
    滚动回归残差 - x ~ y，时序动态中性化
    用于剔除基准影响，提取纯粹Alpha
    """
    df = pd.DataFrame({'x': x, 'y': y})
    
    def _calc_resid(g):
        # 滚动窗口计算残差
        def window_resid(window):
            if len(window) < 3:
                return np.nan
            window_x = window[:, 0]
            window_y = window[:, 1]
            # 简单线性回归: x = a + b*y + e
            A = np.vstack([np.ones(len(window_y)), window_y]).T
            try:
                coeff, _ = np.linalg.lstsq(A, window_x, rcond=None)[0:2]
                resid = window_x[-1] - (coeff[0] + coeff[1] * window_y[-1])
                return resid
            except:
                return np.nan
        
        values = g.values
        # 滑动窗口处理
        resids = np.full(len(g), np.nan)
        for i in range(d-1, len(g)):
            window = values[i-d+1:i+1]
            resids[i] = window_resid(window)
        
        return pd.Series(resids, index=g.index)
    
    result = df.groupby(level='code').apply(_calc_resid)
    return result.droplevel(0) if isinstance(result.index, pd.MultiIndex) else result


# ============= 逻辑与条件算子 =============

def if_else(cond: pd.Series, x: pd.Series, y: pd.Series) -> pd.Series:
    """条件选择：如果cond为True返回x，否则返回y"""
    return pd.Series(np.where(cond, x, y), index=x.index, name=x.name)


# ============= 基础数学算子 =============

def abs(x: pd.Series) -> pd.Series:
    """绝对值"""
    return np.abs(x)


def sign(x: pd.Series) -> pd.Series:
    """符号函数"""
    return np.sign(x)


def log(x: pd.Series) -> pd.Series:
    """对数"""
    return np.log(x)


def power(x: pd.Series, a: float) -> pd.Series:
    """幂函数"""
    return np.power(x, a)


def sqrt(x: pd.Series) -> pd.Series:
    """平方根"""
    return np.sqrt(x)


def add(x: pd.Series, y: pd.Series) -> pd.Series:
    """加法"""
    return x + y


def sub(x: pd.Series, y: pd.Series) -> pd.Series:
    """减法"""
    return x - y


def mul(x: pd.Series, y: pd.Series) -> pd.Series:
    """乘法"""
    return x * y


def div(x: pd.Series, y: pd.Series) -> pd.Series:
    """除法（避免除以0）"""
    return x / y.replace(0, np.nan)


# ============= 导出算子字典 =============

OPERATORS = {
    # 截面算子
    'rank': rank,
    'scale': scale,
    'zscore': zscore,
    'demean': demean,
    
    # 时序算子
    'ts_mean': ts_mean,
    'ts_std': ts_std,
    'ts_min': ts_min,
    'ts_max': ts_max,
    'ts_delta': ts_delta,
    'ts_pct_change': ts_pct_change,
    'ts_corr': ts_corr,
    'ts_cov': ts_cov,
    'shift': shift,
    'ts_zscore': ts_zscore,
    'ts_rank': ts_rank,
    'ts_skew': ts_skew,
    'ts_kurt': ts_kurt,
    'residual': residual,
    
    # 逻辑算子
    'if_else': if_else,
    
    # 数学算子
    'abs': abs,
    'sign': sign,
    'log': log,
    'power': power,
    'sqrt': sqrt,
}
