"""
Alpha因子挖掘系统 - YFinance数据源适配器
美股等海外市场数据，免费无需注册
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from .base_adapter import BaseDataAdapter


class YFinanceAdapter(BaseDataAdapter):
    """YFinance数据源适配器 - 美股专用"""
    
    name = "yfinance"
    priority = 0  # 美股首选
    market_support = ['us']  # 仅支持美股
    
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
    
    def fetch(self,
             market: str,
             symbols: List[str],
             start_date: str,
             end_date: str,
             fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取美股日线行情数据"""
        import yfinance as yf
        
        all_data = []
        
        if symbols == 'all' or symbols == ['all']:
            # 简化：标普500成分股示例
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V']
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if len(df) == 0:
                    continue
                
                df = df.reset_index()
                df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                      'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                
                df['code'] = f"{symbol}.US"
                df['amount'] = df['close'] * df['volume']
                df['return_1d'] = df['close'].pct_change()
                
                all_data.append(df)
                
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        return result
    
    def get_industry_classification(self, market: str, symbols: List[str]) -> pd.DataFrame:
        """获取GICS行业分类"""
        import yfinance as yf
        
        result = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                industry = info.get('sector', 'unknown')
                result.append({
                    'code': f"{symbol}.US",
                    'group': industry
                })
            except:
                result.append({
                    'code': f"{symbol}.US",
                    'group': 'unknown'
                })
        
        return pd.DataFrame(result)
