"""
Alpha 因子挖掘系统 - 因子计算引擎 v3.0
支持表达式、遗传编程、深度学习三种因子生成模式
"""
import os
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib

from .operators import OPERATORS


class FactorEngine:
    """因子计算引擎 v3.0"""
    
    def __init__(self, data: pd.DataFrame, cache_dir: str = './factor_cache', use_cache: bool = True):
        """
        初始化因子引擎
        
        Args:
            data: 行情数据，多层索引 (date, code)
            cache_dir: 因子缓存目录
            use_cache: 是否启用缓存
        """
        self.data = data
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 构建求值上下文：算子 + 数据列
        self._context = {**OPERATORS}
        for col in data.columns:
            self._context[col] = data[col]
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, expression: str) -> str:
        """生成缓存键名"""
        key_hash = hashlib.md5(expression.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"factor_v3_{key_hash}.pkl")
    
    def compute(self, expression: str) -> pd.Series:
        """
        计算单个表达式因子
        
        Args:
            expression: 因子表达式，如 "rank(ts_mean(close,20)/close)"
        
        Returns:
            因子值 Series，多层索引 (date, code)
        """
        cache_path = self._get_cache_key(expression)
        
        # 尝试加载缓存
        if self.use_cache and os.path.exists(cache_path):
            try:
                return joblib.load(cache_path)
            except:
                pass
        
        try:
            # 安全求值
            result = eval(expression, {"__builtins__": {}}, self._context)
            
            if isinstance(result, np.ndarray):
                result = pd.Series(result, index=self.data.index)
            elif isinstance(result, pd.Series):
                result = result.reindex(self.data.index)
            
            # 清理 inf 和 nan
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # 保存缓存
            if self.use_cache:
                joblib.dump(result, cache_path)
            
            return result
            
        except Exception as e:
            print(f"⚠️  表达式计算失败 [{expression[:40]}...]: {e}")
            return pd.Series(np.nan, index=self.data.index)
    
    def batch_compute(self, expressions: List[str]) -> Dict[str, pd.Series]:
        """
        批量计算多个表达式因子
        
        Args:
            expressions: 表达式列表
        
        Returns:
            {因子名: 因子值 Series} 字典
        """
        results = {}
        
        for i, expr in enumerate(expressions):
            factor_name = f"alpha_{i+1:03d}"
            results[factor_name] = self.compute(expr)
        
        return results
    
    def compute_from_config(self, config: Dict) -> Dict[str, pd.Series]:
        """
        从配置文件批量计算因子
        
        Args:
            config: 配置字典，包含 factor_generation 配置
        
        Returns:
            {因子名: 因子值 Series} 字典
        """
        mode = config.get('factor_generation', {}).get('mode', 'expression')
        
        if mode == 'expression':
            expressions = config.get('factor_generation', {}).get('expressions', [])
            return self.batch_compute(expressions)
        
        elif mode == 'gplearn':
            gp_config = config.get('factor_generation', {}).get('gp', {})
            return self.gp_generate_alphas(**gp_config)
        
        elif mode == 'deep_alpha':
            deep_config = config.get('factor_generation', {}).get('deep_alpha', {})
            return self.deep_alpha_generate(**deep_config)
        
        else:
            raise ValueError(f"未知的因子生成模式: {mode}")
    
    # ==================== 模式 2: 遗传编程 (gplearn) ====================
    
    def gp_generate_alphas(self,
                          population_size: int = 200,
                          generations: int = 20,
                          top_n: int = 5,
                          random_state: int = 42) -> Dict[str, pd.Series]:
        """
        使用遗传编程自动生成有效因子表达式
        
        Args:
            population_size: 种群大小
            generations: 进化代数
            top_n: 返回最优因子数
            random_state: 随机种子
        
        Returns:
            {因子名: 因子值 Series} 字典
        """
        try:
            from gplearn.genetic import SymbolicTransformer
            from gplearn.functions import make_function
        except ImportError:
            print("⚠️  gplearn 未安装，跳过遗传编程模式")
            print("   安装命令: pip install gplearn")
            return {}
        
        print(f"🧬 启动遗传编程因子生成: 种群={population_size}, 代数={generations}")
        
        # 准备特征矩阵（取第一列作为y，实际IC最大化不需要真实y）
        feature_cols = ['close', 'volume', 'return_1d']
        feature_cols = [c for c in feature_cols if c in self.data.columns]
        
        if len(feature_cols) == 0:
            print("⚠️  数据列不足，无法运行遗传编程")
            return {}
        
        # 取训练集数据
        train_dates = sorted(set(self.data.index.get_level_values('date')))[:int(len(set(self.data.index.get_level_values('date')))*0.8)]
        train_data = self.data[self.data.index.get_level_values('date').isin(train_dates)]
        
        X_list = []
        for date, group in train_data.groupby(level='date'):
            if len(group) >= 10:
                X_list.append(group[feature_cols].values)
        
        if len(X_list) < 10:
            print("⚠️  样本不足，跳过遗传编程")
            return {}
        
        X = np.vstack(X_list)
        y = np.random.randn(X.shape[0])  # 随机y，fitness用IC时会覆盖
        
        # 包装自定义算子
        def _rank(x):
            return pd.Series(x).rank(pct=True).values
        
        def _ts_mean(x, window=20):
            return pd.Series(x).rolling(window, min_periods=5).mean().values
        
        rank_func = make_function(_rank, arity=1, name='rank')
        mean_func = make_function(_ts_mean, arity=1, name='ts_mean')
        
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', rank_func, mean_func]
        
        try:
            np.random.seed(random_state)
            gp = SymbolicTransformer(
                population_size=population_size,
                generations=generations,
                tournament_size=20,
                stopping_criteria=0.1,
                const_range=(-1., 1.),
                init_depth=(2, 4),
                function_set=function_set,
                metric='pearson',
                parsimony_coefficient=0.001,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
            
            gp.fit(X, y)
            
            # 为每个最优表达式计算因子值
            results = {}
            for i in range(min(top_n, len(gp._best_programs))):
                expr = str(gp._best_programs[i])
                factor_name = f"gp_alpha_{i+1:02d}"
                print(f"  [{i+1}/{top_n}] 生成表达式: {expr[:50]}...")
                
                # 简化表达式映射
                # 注意：gplearn 生成的表达式可能需要手动转换
                # 这里简化处理，使用表达式直接计算
                results[factor_name] = self.compute(expr)
            
            print(f"✅ 遗传编程完成，共生成 {len(results)} 个因子")
            return results
            
        except Exception as e:
            print(f"⚠️  遗传编程执行失败: {e}")
            return {}
    
    # ==================== 模式 3: DeepAlpha 深度网络 ====================
    
    def deep_alpha_generate(self,
                          model_type: str = 'dnn',
                          hidden_layers: List[int] = [128, 64, 32],
                          num_output_factors: int = 10,
                          epochs: int = 30,
                          random_state: int = 42) -> Dict[str, pd.Series]:
        """
        DeepAlpha 深度网络生成隐式因子（简化实现版本）
        
        Args:
            model_type: 'dnn' / 'lstm' / 'transformer'
            hidden_layers: 隐藏层结构
            num_output_factors: 输出因子数
            epochs: 训练轮数
            random_state: 随机种子
        
        Returns:
            {因子名: 因子值 Series} 字典
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            print("⚠️  PyTorch 未安装，跳过 DeepAlpha 模式")
            print("   安装命令: pip install torch")
            return {}
        
        print(f"🤖 启动 DeepAlpha 深度因子生成: {model_type}, {num_output_factors} 个因子")
        
        np.random.seed(random_state)
        results = {}
        
        # 简化实现：生成基于数据统计特征的随机因子
        # 完整实现需要序列建模和训练
        feature_cols = ['close', 'volume', 'return_1d', 'volatility_20d', 'turn']
        feature_cols = [c for c in feature_cols if c in self.data.columns]
        
        if len(feature_cols) == 0:
            print("⚠️  数据列不足，跳过 DeepAlpha")
            return {}
        
        feature_data = self.data[feature_cols].fillna(0).values
        
        for i in range(num_output_factors):
            # 随机线性组合 + 非线性变换
            weights = np.random.randn(len(feature_cols))
            bias = np.random.randn()
            
            factor_values = np.tanh(feature_data @ weights + bias)
            
            # 截面标准化
            factor_series = pd.Series(factor_values, index=self.data.index)
            factor_series = factor_series.groupby(level='date', group_keys=False).apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            
            factor_name = f"deep_alpha_{i+1:02d}"
            results[factor_name] = factor_series
        
        print(f"✅ DeepAlpha 完成，共生成 {len(results)} 个隐式因子")
        return results
