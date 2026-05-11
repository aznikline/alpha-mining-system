"""
Alpha因子挖掘系统 - YFinance数据源适配器
美股等海外市场数据
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from .base_adapter import BaseDataAdapter
from utils import standardize_code


class YFinanceAdapter(BaseDataAdapter):
    """YFinance数据源适配器"""
    
    name = "yfinance"
    
    def __init__(self):
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import yfinance
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def get_daily_data(self,
                      symbols: List[str],
                      start_date: str,
                      end_date: str,
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取日线行情数据"""
        import yfinance as yf
        
        all_data = []
        
        if symbols == 'all_a' or symbols == ['all_a']:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if len(df) == 0:
                    continue
                
                df = df.reset_index()
                
                # 重命名字段
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                df['code'] = symbol
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df = df.sort_values('date')
                df['return_1d'] = df['close'].pct_change()
                
                if 'amount' not in df.columns:
                    df['amount'] = df['close'] * df['volume']
                
                # 计算vwap
                if 'amount' in df.columns and 'volume' in df.columns:
                    df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
                
                all_data.append(df)
                
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        return result
    
    def get_industry_classification(self, symbols: List[str]) -> pd.DataFrame:
        """获取行业分类"""
        import yfinance as yf
        
        result = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                industry = info.get('sector', 'unknown')
                result.append({
                    'code': symbol,
                    'group': industry
                })
            except:
                result.append({
                    'code': symbol,
                    'group': 'unknown'
                })
        
        return pd.DataFrame(result)
