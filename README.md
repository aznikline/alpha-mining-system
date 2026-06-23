# Alpha 因子挖掘系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/aznikline/alpha-mining-system?style=social)](https://github.com/aznikline/alpha-mining-system)

> A modular, extensible automated Alpha factor mining platform for A-share and US equity markets, inspired by WorldQuant / BigQuant.

一个模块化、可扩展的自动化 Alpha 因子挖掘平台，参考 WorldQuant / BigQuant 架构设计。支持手动表达式、遗传编程、深度学习三种因子生成模式，覆盖数据加载 → 因子计算 → 有效性评估 → 可视化报告的完整研究工作流。

## 核心特性

### 📊 数据层
- **多数据源支持**：Akshare（免费A股）、Tushare（专业数据）、Baostock（稳定数据）、YFinance（美股）
- **自动降级机制**：首选数据源失败时自动尝试备选
- **多市场支持**：A股(`a_share`)、美股(`us`)自动路由
- **Parquet 本地缓存**：重复使用无需重复下载
- **数据标准化**：统一字段命名、格式、复权处理

### 🔧 因子引擎
- **三种因子生成模式**
  - 手动表达式：类 WorldQuant 表达式语法
  - 遗传编程：gplearn 自动进化有效因子
  - DeepAlpha：DNN/LSTM 深度学习隐式因子

- **24 个内置算子**
  - 截面算子：`rank`, `zscore`, `demean`, `scale`
  - 时序算子：`ts_mean`, `ts_std`, `ts_delta`, `ts_corr`, `ts_skew`, `ts_kurt`, `residual`
  - 数学算子：`abs`, `sign`, `log`, `power`, `sqrt`, `add`, `sub`, `mul`, `div`
  - 条件算子：`if_else`

### 📈 因子评估
- **IC 分析**：Pearson / Spearman 双指标
- **IC_IR 稳定性**：IC 均值/标准差
- **5-10 分组回测**：分层年化收益对比
- **多空组合绩效**：年化收益、夏普、最大回撤、卡玛比率
- **换手率分析**：因子稳定性评估
- **训练/测试集严格切分**：避免过拟合
- **行业市值中性化**：可选

### 🎨 可视化输出
- IC 时间序列图（±2σ 通道）
- 多空净值曲线图
- 分组累计收益对比图
- 月度 IC 热力图
- 因子整体对比汇总图

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

可选模式按需安装：

```bash
pip install gplearn      # 遗传编程模式
pip install torch         # DeepAlpha 深度学习模式
pip install streamlit     # Web 界面
```

### 命令行运行（v3.0 标准包入口）

```bash
# 查看系统信息与可用模式
python -m alpha_mining info

# 用默认 config.yaml 运行
python -m alpha_mining run

# 直接传入因子表达式（覆盖配置）
python -m alpha_mining run -e "rank(ts_mean(close,20)/close)" "-rank(ts_std(return_1d,20))"

# 指定市场 / 生成模式
python -m alpha_mining run --market us --mode expression

# 不生成图表
python -m alpha_mining run --no-plots

# 指定配置文件
python -m alpha_mining run -c my_config.yaml
```

### Streamlit Web 界面

```bash
streamlit run app.py
```

### Notebook 优先工作流

推荐在 Notebook 中使用 v3.0 高级 API，一行代码完成加载 / 评估：

```python
from alpha_mining import init_data, quick_evaluate, compare_factors

# 一行加载数据（含标准化 + 衍生特征）
data = init_data(market='a_share', start='2019-01-01', end='2024-12-31')

# 一行评估单个因子
result = quick_evaluate("rank(ts_mean(close,20)/close)", data, n_groups=5)

# 对比多个因子
compare_factors(["rank(ts_delta(close,5))", "-rank(ts_std(return_1d,20))"], data)
```

完整示例见 [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb)。

### 配置文件说明

完整配置见 [`config.yaml`](config.yaml)，主要分段：

```yaml
market: a_share            # a_share (A股) / us (美股)

data:
  preferred_source: null   # null = 按市场自动选择；或指定 akshare/tushare/baostock/yfinance
  fallback_sources: [tushare, baostock]  # 失败时自动降级
  start_date: "2019-01-01"
  end_date: "2024-12-31"
  symbols: all             # all = 全市场；或 ["000001.SZ", "000002.SZ"]

factor_generation:
  mode: expression          # expression / gplearn / deep_alpha
  expressions:              # 模式 1
    - "rank(ts_mean(close,20)/close)"
    - "-rank(ts_std(return_1d,20))"
  gp:                       # 模式 2
    population_size: 200
    generations: 20
    top_n: 5
  deep_alpha:               # 模式 3
    model_type: dnn         # dnn / lstm / transformer
    hidden_layers: [128, 64, 32]
    num_output_factors: 10

evaluation:
  n_groups: 5
  train_ratio: 0.8
  neutralize: false        # 行业市值中性化

filter:                    # 因子筛选阈值
  min_train_icir: 0.5
  min_test_ic_mean: 0.02
  min_ls_sharpe_train: 1.0
```

---

## 因子表达式参考

| 表达式 | 因子含义 | 预期 IC 方向 |
|--------|----------|------------|
| `rank(ts_mean(close,20)/close)` | 价格偏离均线程度 | 负 |
| `ts_delta(close,5)` | 5 日动量 | 正/负 |
| `-rank(ts_std(return_1d,20))` | 波动率反转 | 正 |
| `ts_zscore(volume,10)*-1` | 成交量异常 | 正 |
| `rank(close_vwap_diff)` | 价格与成交均价差 | 负 |
| `ts_skew(return_1d,20)` | 收益率偏度 | 负 |
| `residual(close, log_volume, 20)` | 成交量解释后的残差收益 | 正 |

---

## 项目结构

```
alpha-mining-system/
├── alpha_mining/                # v3.0 标准包
│   ├── __init__.py             # 包入口，导出高级 API
│   ├── __main__.py             # python -m alpha_mining 入口
│   ├── cli.py                  # 命令行接口 (run / info)
│   ├── api.py                  # Notebook 高级 API
│   ├── data_hub.py             # 统一数据抽象层（多市场）
│   ├── factor_engine.py        # 因子计算引擎
│   ├── evaluator.py            # 因子评估器（中性化增强）
│   ├── visualizer.py           # 结果可视化模块
│   ├── operators.py            # 24 个算子库
│   ├── utils.py                # 工具函数
│   └── data_adapters/          # 数据源适配器（市场感知）
│       ├── base_adapter.py
│       ├── akshare_adapter.py
│       ├── tushare_adapter.py
│       ├── baostock_adapter.py
│       └── yfinance_adapter.py
├── notebooks/
│   └── 01_quickstart.ipynb     # Notebook 优先工作流示例
├── api/
│   └── index.py                # Vercel Serverless Function 入口
├── app.py                      # Streamlit Web 界面
├── config.yaml                 # 默认配置
├── requirements.txt
├── vercel.json
├── LICENSE
└── README.md
```

运行时自动生成：`data_cache/`（行情缓存）、`factor_cache/`（因子缓存）、`results/`（输出报告与图表），均已在 `.gitignore` 中排除。

---

## v3.0 升级要点

相对 v2.1 的主要变化：

1. **标准 Python 包结构** — 从顶层脚本改为 `alpha_mining/` 包，入口统一为 `python -m alpha_mining`。
2. **Notebook 优先工作流** — 新增高级 API（`init_data` / `quick_evaluate` / `compare_factors`），一行代码完成加载与评估。
3. **命令行接口 (CLI)** — `run` / `info` 子命令，支持 `-e` 传入表达式、`--market` / `--mode` 覆盖配置，便于批量与自动化回测。
4. **Streamlit Web 界面** — `streamlit run app.py` 提供交互式因子探索。
5. **Serverless API** — `api/index.py` 提供轻量 HTTP 接口。
6. **因子元信息系统** — `FactorResult` 统一包装，携带 `name` / `category` / `source` / `expression` / `data_version` / `generated_at` 等完整溯源信息。
7. **高阶算子** — `ts_skew` / `ts_kurt` / `residual`。
8. **多市场架构** — `market` 参数路由 A 股 / 美股，适配器自动按市场过滤数据源。

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
- Vercel Serverless：API 模式支持（见 `vercel.json` / `api/index.py`）
- 本地 Docker：`docker run -v $(pwd)/config.yaml:/app/config.yaml ...`

---

## 相关项目

本仓是量化系列的一部分，配套项目分工：

- **[`alpha`](https://github.com/aznikline/alpha)** — OpenAlpha factor discovery → qmt 执行层的 integration bridge。alpha-mining-system 负责**挖掘**因子（遗传编程 / DeepAlpha），alpha 负责**桥接**到执行框架（`SignalAlphaFactor` adapter，含 116 个测试 + Protocol fallback 可独立运行）。
- **qmt / ptrade** — 作者维护的量化执行层（私有）。alpha 的 bridge 以它们为 integration target；不公开以隔离交易逻辑。

---

## License

[MIT License](LICENSE) © 2026 aznikline
