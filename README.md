# PINN 与 Signature–BSDE 的期权定价对比

在 Black–Scholes 模型下，比较浮动执行价回望、向上敲出及向下敲出期权的三类方法，并对照解析价格：

1. 充分状态 PINN：前馈网络读取定价所需的充分状态。
2. LSTM–PINN：从路径学习表示，使用实验性的 PDE 风格残差。
3. Signature–LSTM–BSDE：分段 logsignature、LSTM 与 BSDE 损失。

数学联系、边界条件及现有近似见 [summary.md](summary.md)。展示入口为 [comparison.ipynb](comparison.ipynb)。

## 目录

```text
comparison.ipynb                 解析解、三方法价格和 loss 曲线对比
summary.md                      PDE/PPDE–BSDE 联系及实现限制
scripts/
  analytic_bs.py                欧式、回望、障碍解析定价
  state_pinn/                   充分状态 PINN，三个产品脚本
  lstm_pinn/                    LSTM–PINN，三个产品脚本
  signature_bsde/               Signature–BSDE，三个产品脚本
  run_comparison.py             统一训练与结果汇总入口
  references.py                 公共离散/bridge MC 和标准误
  artifacts.py                  逐轮 loss、指标及模型保存
  report.py                     从已保存结果生成图表
docs/                           原始论文及来源说明
outputs/                        本地生成结果，默认不提交 Git
refined/                        本地旧入口兼容层，不随新仓库发布
```

工作区中的旧实验、课程资料、模型权重和虚拟环境不属于本次发布范围；`scripts/` 中的九个实现是新入口。原始路径 LSTM–BSDE `refined/bs_lookback_bsde.py` 作为历史实验保留，不属于三方法对比。

## 环境

训练已在 Python **3.9.18**、PyTorch **1.9.0+cpu**、Signatory 分发包 **1.2.6.1.9.0**、NumPy **1.26.4** 上验证。建议使用单独的 Python 3.9 虚拟环境。

Signatory 的二进制包必须匹配 PyTorch 版本，详见[官方安装说明](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html)。以下为对应当前已验证版本的安装命令；需要先安装 PyTorch，再安装 Signatory：

```bash
python -m pip install "torch==1.9.0+cpu" -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt
python -c "import torch,signatory; print(torch.__version__,signatory.__version__)"
```

PyTorch 历史安装方式见[官方页面](https://pytorch.org/get-started/previous-versions/)。`requirements.txt` 中的 Signatory 对应上述 PyTorch 版本，不适用于任意新版本 PyTorch。平台缺少对应 wheel 时可能需要本地 C++ 编译环境。

运行 notebook 或生成图表还需：

```bash
python -m pip install -r requirements-report.txt
```

绘图环境和训练环境可以分开。已有 `.venv_signatory` 时，在绘图环境执行统一入口并传入 `--python .venv_signatory/Scripts/python.exe`，训练子进程会使用该解释器。Notebook 会优先检测这个本地环境。

## 运行

所有命令均从仓库根目录执行。先做一次小规模流程验证：

```bash
python -m scripts.run_comparison --products lookback barrier_upout barrier_downout --epochs 2 --batch-size 16 --steps 8 --lag 2 --mc-paths 256 --eval-paths 16 --output-dir outputs/smoke
```

正式实验示例：

```bash
python -m scripts.run_comparison --products lookback barrier_upout barrier_downout --epochs 3000 --batch-size 256 --steps 128 --lag 8 --seeds 1 2 3 --output-dir outputs/experiment_01
python -m scripts.report outputs/experiment_01
```

公共参考价格的平价关系与 bridge 监控一致性检查：

```bash
python -m unittest discover -s tests -v
```

这只是实验配置示例，不保证达到指定精度。使用 `--option-type put` 切换 put；市场参数包括 `--s0 --r --sigma --T --strike --up-barrier --down-barrier`。`--steps` 必须能被 `--lag` 整除。当前回望解析公式不支持零利率；统一入口对此给出错误提示。`--device cuda` 需要自行配置匹配的 CUDA 训练环境。

单独运行方法，可用 `--help` 查看该方法全部超参数：

```bash
python -m scripts.state_pinn.bs_barrier_upout_pinn --epochs 1000 --mc-anchor-weight 0 --output-dir outputs/single_state
python -m scripts.lstm_pinn.bs_barrier_upout_lstm_pinn --epochs 1000 --mc-anchor-weight 0 --output-dir outputs/single_lstm
python -m scripts.signature_bsde.bs_barrier_upout_signature_bsde --epochs 1000 --output-dir outputs/single_bsde
```

统一入口默认关闭障碍 PINN 的 MC 价格锚点；单方法脚本保留原始默认权重，公平对比时请显式设为零。统一入口固定共同市场、训练轮数、batch、学习率及模拟网格，其余网络结构和损失权重保留各方法默认值并保存；这不是等参数量或等算力实验。

打开 `comparison.ipynb` 并运行全部单元格即可展示。Notebook 使用短训练配置（100 轮、32 步）演示，已有输出时读取保存的数据并显示实际配置；更改参数后应指定新的输出目录。统一训练入口会拒绝覆盖已有配置的实验目录。

## 输出位置

使用 `--output-dir outputs/experiment_01` 后：

```text
outputs/experiment_01/
  config.json                   统一配置及运行状态
  references.json               公共解析价、MC 价格与标准误
  references.log                参考价格计算日志
  comparison.csv                每个产品、方法、随机种子的价格及误差
  figures/                      report 或 notebook 生成的价格图和 loss 图
  <product>/<method>/seed_<n>/
    history.csv                 每一轮的总 loss 及各损失分量
    metrics.json                价格、误差、耗时、版本和完整训练参数
    model.pt                    state_dict、参数、市场和合约配置
    train.log                   训练日志
    command.json                实际调用参数
```

`relative_error` 是比例（0.01 表示 1%）；解析价为零时保存为空，改看绝对误差。模型 checkpoint 不包含优化器状态，供推断使用，不支持原样续训。不同方法的 loss 定义不同，图表按方法分开显示。

障碍 BSDE 目前学习离散监控支付；PINN 面向连续 PDE。Notebook 同时展示连续解析价、离散 MC 及 bridge MC，避免混淆监控差异和训练误差。LSTM–PINN 的路径时间差分尚不是严格 PPDE 导数，详见数学说明。Notebook 中保存的短训练输出只说明流程可运行。

## 来源

原论文：[Solving path dependent PDEs with LSTM networks and path signatures](https://arxiv.org/abs/2011.10630)，Marc Sabate-Vidales、David Šiška、Lukasz Szpruch（2020）。[论文 PDF](docs/2011.10630_Solving_path_dependent_PDEs_with_LSTM_networks_and_path_signatures.pdf) 已归档。

论文配套代码：[msabvid/Deep-PPDE](https://github.com/msabvid/Deep-PPDE)，其 GPL-3.0 许可副本保存在 `docs/Deep-PPDE-LICENSE.txt`。本项目的三方法实验由工作区现有实现整理而来，未将整个原始代码库打包进此仓库。
