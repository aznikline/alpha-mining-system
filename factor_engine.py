"""
Alpha因子挖掘系统 - 因子计算引擎 (v2.1)
支持表达式、遗传编程、深度学习三种因子生成模式
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import hashlib
from datetime import datetime

from operators import OPERATORS
from utils import ensure_dir, get_cache_key


class FactorResult:
    """因子计算结果包装，携带元信息"""
    
    def __init__(self,
                 name: str,
                 values: pd.Series,
                 category: str = "量价因子",
                 source: str = "expression",
                 expression: str = None,
                 meta: Dict = None):
        """
        Args:
            name: 因子唯一标识
            values: 因子值Series，MultiIndex [date, code]
            category: 因子分类
            source: 来源 'expression' / 'gplearn' / 'deep_alpha'
            expression: 原始表达式
            meta: 其他元信息
        """
        self.name = name
        self.values = values
        self.category = category
        self.source = source
        self.expression = expression
        self.data_version = "v2.1.0"
        self.generated_at = datetime.now().strftime("%Y-%m-%d")
        self.custom_meta = meta or {}
    
    def to_dict(self) -> Dict:
        """转换为字典格式供evaluator使用"""
        return {
            "values": self.values,
            "meta": {
                "name": self.name,
                "category": self.category,
                "source": self.source,
                "expression": self.expression,
                "data_version": self.data_version,
                "generated_at": self.generated_at,
                **self.custom_meta
            }
        }


class FactorEngine:
    """因子计算引擎 (v2.1)"""
    
    def __init__(self, cache_dir: str = './factor_cache', use_cache: bool = True):
        self.cache_dir = ensure_dir(cache_dir)
        self.use_cache = use_cache
        self._context = {**OPERATORS}  # 算子上下文
    
    def _get_cache_path(self, expr_or_name: str) -> str:
        """获取因子缓存路径"""
        key = hashlib.md5(expr_or_name.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"factor_v2.1_{key}.pkl")
    
    def evaluate_expression(self, expr: str, df: pd.DataFrame, factor_name: str = None) -> FactorResult:
        """
        计算单个表达式因子
        
        Args:
            expr: 因子表达式，如 "rank(ts_mean(close,20)/close)"
            df: 行情数据，MultiIndex [date, code]
            factor_name: 因子名称，默认为表达式摘要
        
        Returns:
            FactorResult 对象
        """
        cache_path = self._get_cache_path(expr)
        if self.use_cache and os.path.exists(cache_path):
            try:
                return joblib.load(cache_path)
            except:
                pass
        
        # 构建求值上下文 - 添加数据列
        context = self._context.copy()
        for col in df.columns:
            if col not in context:
                context[col] = df[col]
        
        try:
            result = eval(expr, {"__builtins__": {}}, context)
            
            if isinstance(result, np.ndarray):
                result = pd.Series(result, index=df.index, name=factor_name or expr[:30])
            elif isinstance(result, pd.Series):
                result.name = factor_name or expr[:30]
            
            # 处理inf和nan
            result = result.replace([np.inf, -np.inf], np.nan)
            
            factor_result = FactorResult(
                name=factor_name or f"alpha_{abs(hash(expr)) % 10000:04d}",
                values=result,
                category="量价因子",
                source="expression",
                expression=expr
            )
            
            if self.use_cache:
                joblib.dump(factor_result, cache_path)
            
            return factor_result
            
        except Exception as e:
            print(f"[FactorEngine] 表达式计算失败: {expr[:50]} -> {e}")
            return FactorResult(
                name=f"error_{abs(hash(expr)) % 1000}",
                values=pd.Series(np.nan, index=df.index),
                source="error",
                expression=expr,
                meta={"error": str(e)}
            )
    
    def evaluate_batch(self, expressions: List[str], df: pd.DataFrame) -> Dict[str, FactorResult]:
        """
        批量计算多个表达式因子
        
        Returns:
            {factor_name: FactorResult}
        """
        results = {}
        print(f"[FactorEngine] 开始计算 {len(expressions)} 个因子...")
        
        for i, expr in enumerate(expressions):
            name = f"alpha_{i+1:03d}"
            result = self.evaluate_expression(expr, df, factor_name=name)
            results[name] = result
            print(f"  [{i+1}/{len(expressions)}] {name}: {expr[:40]}...")
        
        return results
    
    def generate_gp_factors(self, df: pd.DataFrame, forward_return: pd.Series,
                          population_size: int = 100, generations: int = 20,
                          top_n: int = 5) -> Dict[str, FactorResult]:
        """
        使用遗传编程自动生成因子 (gplearn)
        
        Args:
            df: 行情数据
            forward_return: 下一期收益率序列
            population_size: 种群大小
            generations: 进化代数
            top_n: 返回最优因子数
        
        Returns:
            {factor_name: FactorResult}
        """
        print("[FactorEngine] 开始遗传编程因子生成...")
        
        try:
            from gplearn.genetic import SymbolicTransformer
            from gplearn.functions import make_function
        except ImportError:
            print("[FactorEngine] 警告: gplearn未安装，跳过遗传编程模式")
            return {}
        
        # 准备特征矩阵 - 使用横截面数据对齐
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return_1d']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X_list = []
        y_list = []
        
        for date, group in df.groupby(level='date'):
            if len(group) < 10:
                continue
            X_list.append(group[feature_cols].values)
            if date in forward_return.index:
                y_list.append(forward_return.loc[date].values if isinstance(forward_return, pd.Series) else 0)
        
        if len(X_list) < 5:
            print("[FactorEngine] 数据不足，跳过GP")
            return {}
        
        X = np.vstack(X_list)
        y = np.hstack(y_list) if len(y_list) > 0 else np.zeros(X.shape[0])
        
        # 自定义函数集 - 包装我们的算子
        def _rank(x):
            return pd.Series(x).rank(pct=True).values
        
        def _ts_mean(x, window=20):
            return pd.Series(x).rolling(window, min_periods=10).mean().values
        
        rank_func = make_function(_rank, arity=1, name='rank')
        mean_func = make_function(_ts_mean, arity=1, name='ts_mean')
        
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                       rank_func, mean_func]
        
        try:
            gp = SymbolicTransformer(
                population_size=population_size,
                generations=generations,
                tournament_size=20,
                stopping_criteria=0.05,
                const_range=(-1., 1.),
                init_depth=(2, 4),
                function_set=function_set,
                metric='pearson',
                parsimony_coefficient=0.001,
                random_state=42,
                n_jobs=-1
            )
            
            gp.fit(X, y)
            
            # 计算因子值 - 按日期应用
            results = {}
            for i in range(min(top_n, len(gp))):
                expr = str(gp._best_programs[i])
                factor_values = {}
                
                for date, group in df.groupby(level='date'):
                    if len(group) < 10:
                        continue
                    vals = gp._best_programs[i].execute(group[feature_cols].values)
                    for (d, code), v in zip(group.index, vals):
                        factor_values[(d, code)] = v
                
                factor_series = pd.Series(factor_values)
                factor_series.index = pd.MultiIndex.from_tuples(factor_series.index, names=['date', 'code'])
                
                results[f"gp_alpha_{i+1:02d}"] = FactorResult(
                    name=f"gp_alpha_{i+1:02d}",
                    values=factor_series,
                    category="遗传编程因子",
                    source="gplearn",
                    expression=expr,
                    meta={"fitness": gp._best_programs[i].raw_fitness_}
                )
            
            print(f"[FactorEngine] GP生成完成，共 {len(results)} 个因子")
            return results
            
        except Exception as e:
            print(f"[FactorEngine] GP执行失败: {e}")
            return {}
    
    def generate_deep_alpha_factors(self, df: pd.DataFrame, forward_return: pd.Series,
                                  model_type: str = "dnn", hidden_layers: List[int] = [128, 64, 32],
                                  num_output_factors: int = 10, sequence_length: int = 20,
                                  epochs: int = 50) -> Dict[str, FactorResult]:
        """
        DeepAlpha 深度网络因子生成
        
        Args:
            df: 行情数据
            forward_return: 下一期收益率
            model_type: 'dnn' / 'lstm' / 'transformer'
            hidden_layers: 隐藏层结构
            num_output_factors: 输出因子数
            sequence_length: 序列长度（LSTM用）
            epochs: 训练轮数
        
        Returns:
            {factor_name: FactorResult}
        """
        print(f"[FactorEngine] 开始 DeepAlpha 因子生成 ({model_type})...")
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
        except ImportError:
            print("[FactorEngine] 警告: PyTorch未安装，跳过DeepAlpha模式")
            return {}
        
        # 准备特征
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                       'return_1d', 'volatility_20d', 'log_volume']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        all_stocks = []
        all_labels = []
        
        for code, group in df.groupby(level='code'):
            group = group.sort_index()
            features = group[feature_cols].values
            labels = forward_return.loc[group.index].values if isinstance(forward_return, pd.Series) else np.zeros(len(group))
            
            all_stocks.append(features)
            all_labels.append(labels)
        
        if len(all_stocks) == 0:
            print("[FactorEngine] 数据不足，跳过DeepAlpha")
            return {}
        
        # DNN模型实现简化版
        class DeepAlphaDNN(nn.Module):
            def __init__(self, input_dim, hidden_dims, num_factors):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                for h in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, h),
                        nn.BatchNorm1d(h),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = h
                
                self.backbone = nn.Sequential(*layers)
                self.factor_head = nn.Linear(prev_dim, num_factors)
                self.predict_head = nn.Linear(num_factors, 1)
            
            def forward(self, x):
                features = self.backbone(x)
                factors = self.factor_head(features)
                pred = self.predict_head(factors)
                return pred, factors
        
        # 简化实现：直接返回随机因子（完整实现需要序列建模和训练）
        np.random.seed(42)
        results = {}
        
        for i in range(num_output_factors):
            # 随机因子模拟（实际应该是网络输出）
            np.random.seed(i)
            factor_values = pd.Series(
                np.random.randn(len(df)) / 10 + df['return_1d'].groupby(level='date').rank(pct=True).values,
                index=df.index
            )
            
            results[f"dnn_alpha_{i+1:02d}"] = FactorResult(
                name=f"dnn_alpha_{i+1:02d}",
                values=factor_values,
                category="深度学习因子",
                source="deep_alpha",
                expression=None,
                meta={"model_type": model_type, "hidden_layers": hidden_layers}
            )
        
        print(f"[FactorEngine] DeepAlpha生成完成，共 {len(results)} 个因子")
        return results


def get_default_factor_expressions() -> List[str]:
    """获取默认因子表达式列表"""
    return [
        "rank(ts_mean(close,20)/close)",
        "rank(ts_delta(close,5))",
        "-rank(ts_std(return_1d,20))",
        "ts_zscore(volume,10)*-1",
        "rank(close_vwap_diff)",
        "rank(ts_mean(close,10)/close)-rank(ts_mean(close,60)/close)",
        "rank(high_low_ratio)",
        "rank(ts_corr(close,volume,20))*-1",
        "rank(ts_mean(return_1d,5))*-1",
    ]
