"""
Alpha因子挖掘系统 - 统一数据抽象层 (v2.1)
支持多市场、多源自动降级
"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

from .data_adapters import (
    BaseDataAdapter,
    AkshareAdapter,
    TushareAdapter,
    BaostockAdapter,
    YFinanceAdapter,
)
from .utils import ensure_dir, save_parquet_cache, load_parquet_cache, standardize_code


class DataHub:
    """统一数据入口类 (v2.1)"""
    
    def __init__(self,
                 market: str = 'a_share',
                 preferred_source: str = None,
                 fallback_sources: List[str] = None,
                 cache_dir: str = './data_cache',
                 start_date: str = '2018-01-01',
                 end_date: str = '2025-01-01',
                 tushare_token: str = None):
        """
        初始化数据中心
        
        Args:
            market: 市场类型 'a_share' (A股) 或 'us' (美股)
            preferred_source: 首选数据源，None则按市场默认
            fallback_sources: 降级数据源列表，None则按市场默认
            cache_dir: 本地缓存目录
            start_date: 数据开始日期
            end_date: 数据结束日期
            tushare_token: Tushare Token
        """
        self.market = market
        self.cache_dir = ensure_dir(cache_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.tushare_token = tushare_token
        
        # 按市场设置默认数据源优先级
        if preferred_source is None:
            if market == 'a_share':
                preferred_source = 'akshare'
                fallback_sources = fallback_sources or ['tushare', 'baostock']
            elif market == 'us':
                preferred_source = 'yfinance'
                fallback_sources = fallback_sources or []
            else:
                raise ValueError(f"不支持的市场类型: {market}")
        
        self.preferred_source = preferred_source
        self.fallback_sources = fallback_sources
        
        # 初始化适配器
        self._adapters: Dict[str, BaseDataAdapter] = {}
        self._init_adapters()
        
        # 缓存数据
        self._cached_data = None
        self._data_version = "v2.1.0"
    
    def _init_adapters(self):
        """初始化所有支持当前市场的数据源适配器"""
        adapter_map = {
            'akshare': AkshareAdapter,
            'tushare': TushareAdapter,
            'baostock': BaostockAdapter,
            'yfinance': YFinanceAdapter,
        }
        
        # 只初始化支持当前市场的适配器
        for source in [self.preferred_source] + self.fallback_sources:
            if source in adapter_map:
                adapter_cls = adapter_map[source]
                # 检查适配器是否支持当前市场
                if self.market not in adapter_cls.market_support:
                    continue
                
                if source == 'tushare':
                    self._adapters[source] = adapter_cls(token=self.tushare_token)
                else:
                    self._adapters[source] = adapter_cls()
    
    def _get_cache_path(self, symbols, start_date, end_date) -> str:
        """获取缓存文件路径"""
        import hashlib
        symbols_str = str(symbols)
        key = hashlib.md5(f"{symbols_str}_{start_date}_{end_date}_{self.market}".encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"data_v{self._data_version}_{key}.parquet")
    
    def get_daily_data(self,
                       symbols = 'all',
                       fields = '*',
                       use_cache: bool = True) -> pd.DataFrame:
        """
        获取全市场日线数据
        
        Args:
            symbols: 股票代码列表或'all'表示全部
            fields: 字段列表或'*'表示全部
            use_cache: 是否使用缓存
        
        Returns:
            MultiIndex [date, code] 的DataFrame
        """
        if use_cache and self._cached_data is not None:
            return self._cached_data
        
        # 尝试从缓存加载
        cache_path = self._get_cache_path(symbols, self.start_date, self.end_date)
        if use_cache and os.path.exists(cache_path):
            print(f"[DataHub] 从缓存加载数据: {os.path.basename(cache_path)}")
            df = load_parquet_cache(cache_path)
            if df is not None and len(df) > 0:
                df = df.set_index(['date', 'code']).sort_index()
                self._cached_data = df
                return df
        
        # 依次尝试各数据源
        df = None
        tried_sources = []
        
        for source in [self.preferred_source] + self.fallback_sources:
            if source not in self._adapters:
                continue
            
            adapter = self._adapters[source]
            if not adapter.is_available():
                continue
            
            tried_sources.append(source)
            print(f"[DataHub] 尝试从 {source} 拉取数据...")
            
            try:
                df = adapter.fetch(
                    market=self.market,
                    symbols=symbols,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fields=fields
                )
                
                if df is not None and len(df) > 0:
                    print(f"[DataHub] {source} 成功获取 {len(df)} 条数据")
                    break
            except Exception as e:
                print(f"[DataHub] {source} 拉取失败: {e}")
                continue
        
        # 全部失败时使用模拟数据兜底
        if df is None or len(df) == 0:
            print(f"[DataHub] 所有数据源均失败，生成模拟数据")
            df = self._generate_mock_data(symbols)
        
        # 计算衍生特征
        df = self.prepare_features(df)
        
        # 设置多级索引
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index(['date', 'code']).sort_index()
        
        # 保存缓存
        if use_cache:
            save_parquet_cache(df.reset_index(), cache_path)
            self._cached_data = df
        
        return df
    
    def _generate_mock_data(self, symbols) -> pd.DataFrame:
        """生成模拟数据"""
        if symbols == 'all' or symbols == ['all']:
            if self.market == 'a_share':
                symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH',
                          '000858.SZ', '600519.SH', '000333.SZ', '002415.SZ']
            else:
                symbols = ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US']
        
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        all_data = []
        np.random.seed(42)
        
        for symbol in symbols:
            base_price = 10 + np.random.random() * 190
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_price * (1 + returns).cumprod()
            
            df = pd.DataFrame({
                'date': dates,
                'code': symbol,
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 50000000, len(dates)),
                'amount': prices * np.random.randint(1000000, 50000000, len(dates)),
                'turn': np.random.uniform(0.005, 0.05, len(dates)),
                'group': 'unknown',
            })
            
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生特征列（调用模块级独立函数）"""
        return prepare_features(df)


# 模块级独立函数，供外部导入使用
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算衍生特征列
    
    Args:
        df: 原始行情数据
    
    Returns:
        添加衍生列后的DataFrame
    """
    df = df.copy()
    
    # 处理 MultiIndex 情况
    if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names and 'code' in df.index.names:
        df = df.reset_index()
        has_multiindex = True
    else:
        has_multiindex = False
    
    # 价格数据按股票分组对齐
    if 'date' in df.columns and 'code' in df.columns:
        # 计算多期收益率
        df = df.sort_values(['code', 'date'])
        df['return_1d'] = df.groupby('code')['close'].pct_change()
        df['return_5d'] = df.groupby('code')['close'].pct_change(5)
        df['return_20d'] = df.groupby('code')['close'].pct_change(20)
        
        # 波动率
        df['volatility_20d'] = df.groupby('code')['return_1d'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        
        # 高低价比率
        df['high_low_ratio'] = df['high'] / df['low']
        
        # VWAP相关
        if 'amount' in df.columns and 'volume' in df.columns:
            df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
            df['close_vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
        
        # 成交量对数
        df['log_volume'] = np.log(df['volume'] + 1)
        
        # 换手率均线
        if 'turn' in df.columns:
            df['turn_5d_avg'] = df.groupby('code')['turn'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
        
        # 市值代理
        df['market_cap'] = df['close'] * 1e9
        df['log_market_cap'] = np.log(df['market_cap'])
        
        # 行业分类补全
        if 'group' not in df.columns:
            df['group'] = 'unknown'
    
    # 恢复 MultiIndex
    if has_multiindex:
        df = df.set_index(['date', 'code']).sort_index()
    
    return df
