"""
Alpha因子挖掘系统 - 因子计算引擎
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
import joblib

from operators import OPERATORS
from utils import ensure_dir, get_cache_key, progress_bar


class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self, cache_dir: str = './factor_cache', use_cache: bool = True):
        self.cache_dir = ensure_dir(cache_dir)
        self.use_cache = use_cache
        
        # 算子上下文
        self._context = {**OPERATORS}
        # 后续会动态添加数据列
    
    def _get_cache_path(self, expr: str) -> str:
        """获取因子缓存路径"""
        cache_key = get_cache_key(expr=expr)
        return os.path.join(self.cache_dir, f"factor_{cache_key}.pkl")
    
    def evaluate(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """
        计算单个因子表达式
        
        Args:
            expr: 因子表达式，如 "rank(ts_mean(close,20) / close)"
            df: 行情数据，多级索引 [date, code]
        
        Returns:
            因子值序列，索引与df相同
        """
        # 尝试从缓存加载
        cache_path = self._get_cache_path(expr)
        if self.use_cache and os.path.exists(cache_path):
            try:
                return joblib.load(cache_path)
            except:
                pass
        
        # 构建求值上下文
        context = self._context.copy()
        
        # 添加数据列到上下文
        for col in df.columns:
            if col not in context:
                context[col] = df[col]
        
        # 安全求值
        try:
            # 使用eval求值，限制上下文为我们提供的算子
            result = eval(expr, {"__builtins__": {}}, context)
            
            # 确保结果是Series
            if isinstance(result, np.ndarray):
                result = pd.Series(result, index=df.index, name=expr)
            elif isinstance(result, pd.Series):
                result.name = expr
            
            # 处理Inf和NaN
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # 缓存结果
            if self.use_cache:
                joblib.dump(result, cache_path)
            
            return result
            
        except Exception as e:
            print(f"因子表达式计算失败 [{expr}]: {e}")
            return pd.Series(np.nan, index=df.index, name=expr)
    
    def evaluate_batch(self, expressions: List[str], df: pd.DataFrame, 
                      parallel: bool = False) -> Dict[str, pd.Series]:
        """
        批量计算多个因子
        
        Args:
            expressions: 因子表达式列表
            df: 行情数据
            parallel: 是否使用并行计算
        
        Returns:
            {expr: factor_series} 字典
        """
        results = {}
        
        print(f"[FactorEngine] 开始计算 {len(expressions)} 个因子...")
        
        for i, expr in enumerate(expressions):
            progress_bar(i + 1, len(expressions), prefix="计算因子")
            results[expr] = self.evaluate(expr, df)
        
        print()
        return results
    
    def generate_factor_combinations(self, 
                                   base_exprs: List[str], 
                                   max_complexity: int = 2) -> List[str]:
        """
        生成因子组合（简单的排列组合，非GP）
        
        Args:
            base_exprs: 基础因子表达式
            max_complexity: 最大组合层数
        
        Returns:
            扩展后的因子表达式列表
        """
        extended = list(base_exprs)
        
        if max_complexity >= 2:
            # 添加二元组合
            for i, e1 in enumerate(base_exprs):
                for e2 in base_exprs[i:]:
                    extended.append(f"rank(({e1}) + ({e2}))")
                    extended.append(f"rank(({e1}) - ({e2}))")
                    extended.append(f"rank(({e1}) * ({e2}))")
        
        return list(set(extended))  # 去重


def get_default_factor_expressions() -> List[str]:
    """获取默认的因子表达式列表"""
    return [
        # 动量类因子
        "rank(ts_delta(close, 5))",
        "rank(ts_delta(close, 20))",
        "ts_zscore(return_1d, 20)",
        
        # 反转类因子
        "-rank(ts_mean(return_1d, 5))",
        "-rank(ts_mean(return_1d, 20))",
        
        # 波动率类因子
        "-rank(ts_std(return_1d, 20))",
        "rank(high_low_ratio)",
        
        # 成交量类因子
        "rank(ts_zscore(log_volume, 20))",
        "rank(ts_zscore(turn, 5))",
        "-rank(ts_corr(close, log_volume, 20))",
        
        # 量价关系
        "rank(close_vwap_diff)",
        "rank(ts_mean(close_vwap_diff, 5))",
        
        # 技术指标类
        "rank(ts_mean(close, 10) / close)",
        "rank(ts_mean(close, 20) / close)",
        "rank(ts_mean(close, 60) / close)",
        "(rank(ts_mean(close, 5)) - rank(ts_mean(close, 20)))",
        
        # 高低价位置
        "rank((close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001))",
    ]
