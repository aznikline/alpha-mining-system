"""
Alpha因子挖掘系统 - 数据源适配器基类
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional


class BaseDataAdapter(ABC):
    """数据源适配器基类"""
    
    name = "base"
    
    @abstractmethod
    def get_daily_data(self, 
                      symbols: List[str], 
                      start_date: str, 
                      end_date: str,
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取日线行情数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            fields: 需要的字段，None表示全部
        
        Returns:
            DataFrame，包含标准字段: date, code, open, high, low, close, volume, amount
        """
        pass
    
    @abstractmethod
    def get_industry_classification(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取行业分类数据
        
        Returns:
            DataFrame，包含 code, group 两列
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass
