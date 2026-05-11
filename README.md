# Alpha 因子挖掘系统 v2.1

一个模块化、可扩展的自动化 Alpha 因子挖掘平台，参考 WorldQuant / BigQuant 架构设计。

## 核心特性

### 📊 数据层
- **多数据源支持**：Akshare（免费A股）、Tushare（专业数据）、Baostock（稳定数据）、YFinance（美股）
- **自动降级机制**：首选数据源失败时自动尝试备选
- **多市场支持**：A股(`a_share`)、美股(`us`)自动路由
- **Parquet本地缓存**：重复使用无需重复下载
- **数据标准化**：统一字段命名、格式、复权处理

### 🔧 因子引擎
- **三种因子生成模式**
  - 手动表达式：类 WorldQuant 表达式语法
  - 遗传编程：gplearn 自动进化有效因子
  - DeepAlpha：DNN/LSTM 深度学习隐式因子

- **24个内置算子**
  - 截面算子：`rank`, `zscore`, `demean`, `scale`
  - 时序算子：`ts_mean`, `ts_std`, `ts_delta`, `ts_corr`, `ts_skew`, `ts_kurt`, `residual`
  - 数学算子：`abs`, `sign`, `log`, `power`, `sqrt`, `add`, `sub`, `mul`, `div`
  - 条件算子：`if_else`

### 📈 因子评估
- **IC分析**：Pearson / Spearman 双指标
- **IC_IR稳定性**：IC 均值/标准差
- **5-10分组回测**：分层年化收益对比
- **多空组合绩效**：年化收益、夏普、最大回撤、卡玛比率
- **换手率分析**：因子稳定性评估
- **训练/测试集严格切分**：避免过拟合
- **行业市值中性化**：可选

### 🎨 可视化输出
- IC 时间序列图（±2σ通道）
- 多空净值曲线图
- 分组累计收益对比图
- 月度IC热力图
- 因子整体对比汇总图

---

## 快速开始

### 安装依赖

```bash
pip install pandas numpy scipy matplotlib seaborn pyyaml joblib akshare
pip install gplearn  # 可选：遗传编程模式
pip install torch     # 可选：DeepAlpha 深度学习模式
```

### 运行

```bash
# 使用默认配置
python main.py

# 指定配置
python main.py --config my_config.yaml

# 强制重新计算（不使用缓存）
python main.py --no-cache

# 跳过图表生成
python main.py --no-charts

# 指定市场（覆盖配置文件）
python main.py --market us
```

### 配置文件说明

```yaml
# 市场类型
market: a_share  # 或 'us' 美股

# 因子生成模式
factor_generation:
  mode: expression  # expression / gplearn / deep_alpha
  
  # 模式1：手动表达式因子
  expressions:
    - "rank(ts_mean(close,20)/close)"
    - "-rank(ts_std(return_1d,20))"
    - "ts_skew(return_1d,20)"  # 新增高阶算子
  
  # 模式2：遗传编程
  gp:
    population_size: 100
    generations: 15
    top_n: 5
  
  # 模式3：DeepAlpha 深度学习
  deep_alpha:
    model_type: dnn
    hidden_layers: [128, 64, 32]
    num_output_factors: 10

# 评估配置
evaluation:
  n_groups: 5
  train_ratio: 0.8
  neutralize: false  # 行业市值中性化开关
```

---

## 因子表达式参考

| 表达式 | 因子含义 | 预期IC方向 |
|--------|----------|------------|
| `rank(ts_mean(close,20)/close)` | 价格偏离均线程度 | 负 |
| `ts_delta(close,5)` | 5日动量 | 正/负 |
| `-rank(ts_std(return_1d,20))` | 波动率反转 | 正 |
| `ts_zscore(volume,10)*-1` | 成交量异常 | 正 |
| `rank(close_vwap_diff)` | 价格与成交均价差 | 负 |
| `ts_skew(return_1d,20)` | 收益率偏度 | 负 |
| `residual(close, log_volume, 20)` | 成交量解释后的残差收益 | 正 |

---

## 项目结构

```
alpha_mining/
├── main.py                  # 主程序入口
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖列表
├── README.md               # 本文档
│
├── data_hub.py             # 统一数据抽象层（v2.1多市场）
├── factor_engine.py        # 因子计算引擎（v2.1元信息系统）
├── evaluator.py            # 因子评估器（v2.1中性化增强）
├── visualizer.py           # 结果可视化模块
├── operators.py            # 24个算子库（v2.1新增高阶算子）
├── utils.py                # 工具函数
│
├── data_adapters/          # 数据源适配器（v2.1市场感知）
│   ├── base_adapter.py
│   ├── akshare_adapter.py
│   ├── tushare_adapter.py
│   ├── baostock_adapter.py
│   └── yfinance_adapter.py
│
├── data_cache/             # 行情数据缓存（自动生成）
├── factor_cache/           # 因子值缓存（自动生成）
└── results/                # 输出结果（自动生成）
```

---

## v2.1 版本升级要点

### 1. 因子元信息系统
新增 `FactorResult` 类统一包装，每个因子携带完整溯源信息：
- `name` / `category` / `source` - 唯一标识
- `expression` / `data_version` / `generated_at` - 可复现
- `custom_meta` - 自定义拓展字段

### 2. 新增高阶算子
| 算子 | 说明 |
|------|------|
| `ts_skew(x, d)` | 滚动窗口偏度，捕捉不对称性 |
| `ts_kurt(x, d)` | 滚动窗口峰度，捕捉厚尾 |
| `residual(x, y, d)` | 滚动回归残差，x对y做中性化 |

### 3. 多市场架构
- `market` 参数：`a_share` 或 `us`
- 适配器自动过滤，仅支持对应市场的数据源生效
- A股默认优先级：Akshare > Tushare > Baostock
- 美股默认：YFinance

### 4. 遗传编程因子
- 使用 gplearn 符号回归自动发现因子
- 自定义函数集，复用已有算子实现
- 适应度函数：IC 值最大化
- 输出 Top N 最优表达式

### 5. DeepAlpha 深度因子
- 简化版 DNN 端到端因子生成
- 支持自定义网络结构
- 输出 N 个隐式因子向量
- 完整训练流程可扩展

---

## 有效因子筛选标准

| 指标 | 最低阈值 | 优秀阈值 |
|------|----------|----------|
| 训练集 RankIC_IR | > 0.3 | > 0.5 |
| 测试集 RankIC 均值 | > 0.01 | > 0.03 |
| 多空夏普（训练） | > 0.5 | > 1.0 |
| 方向一致性 | 训练/测试同号 | - |
| 日换手率 | < 0.3 | < 0.15 |

---

## 部署与集成

- GitHub: https://github.com/aznikline/alpha-mining-system
- Vercel Serverless: API 模式支持
- 本地Docker部署: `docker run -v ...`

---

## License

MIT License
