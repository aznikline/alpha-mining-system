"""
Alpha 因子挖掘系统 v3.0
Notebook 优先的自动化因子研究平台
"""

__version__ = "3.0.0"

# 核心类
from .data_hub import DataHub
from .factor_engine import FactorEngine
from .evaluator import Evaluator
from .visualizer import FactorVisualizer as Visualizer

# 高级 API (Notebook 快捷函数)
from .api import init_data, quick_evaluate, compare_factors

# 工具函数
from .data_hub import prepare_features
from .utils import standardize_code

__all__ = [
    "DataHub",
    "FactorEngine",
    "Evaluator",
    "Visualizer",
    "init_data",
    "quick_evaluate",
    "compare_factors",
    "prepare_features",
    "standardize_code",
]
