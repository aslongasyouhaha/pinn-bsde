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

这只是实验配置示例，不保证达到指定精度。使用 `--option-type put` 切换 put；市场参数包括 `--s0 --r --sigma --T --strike --up-barrier --down-barrier`。`--steps` 必须能被 `--lag` 整除。当前回望解析公式不支持零利率；统一入口对此给出错误提示。`--device cuda` 需要自行配置匹配的 CUDA 训练环境。

单独运行方法，可用 `--help` 查看该方法全部超参数：

```bash
python -m scripts.state_pinn.bs_barrier_upout_pinn --epochs 10001 --output-dir outputs/single_state
python -m scripts.lstm_pinn.bs_barrier_upout_lstm_pinn --epochs 1000 --mc-anchor-weight 0 --output-dir outputs/single_lstm
python -m scripts.signature_bsde.bs_barrier_upout_signature_bsde --epochs 1000 --output-dir outputs/single_bsde
```

统一入口默认保留充分状态障碍 PINN 的 MC 价格锚点（权重 100）；LSTM–PINN 仍保持此前的统一入口配置（权重 0）。可用 `--mc-anchor-weight` 显式覆盖。锚点属于 MC 辅助监督，解释初始价格误差时应同时查看未锚定状态的误差。统一入口固定共同市场、训练轮数、batch、学习率及模拟网格，其余网络结构和损失权重保留各方法默认值并保存；这不是等参数量或等算力实验。

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

## 当前复核：充分状态障碍 PINN

这一版只使用 $(t,S)$ 与前馈网络，不涉及 LSTM。保留原来的 PDE、终端、障碍、非负及价格上界损失，以及 MC 起始锚点。连续障碍锚点使用精确 GBM 网格与 Brownian bridge 条件存续概率计算，没有将解析价格用作训练标签。采样不断更新，公共 MC 参考使用独立种子。

本次只恢复统一入口中的充分状态障碍 MC 锚点（默认权重 100），将单方法默认预算设为 10,001 轮、batch 128，并加入独立网格评估。未将没有稳定改善结果的价格表示或采样改动纳入最终算法。LSTM–PINN、Signature–BSDE 和回望算法未改。

~~~bash
python -m scripts.run_comparison --methods state_pinn --products barrier_upout barrier_downout --epochs 10001 --batch-size 128 --mc-paths 200000 --output-dir outputs/state_pinn_review
python -m scripts.state_pinn.evaluate outputs/state_pinn_review/barrier_upout/state_pinn/seed_1
python -m scripts.state_pinn.evaluate outputs/state_pinn_review/barrier_downout/state_pinn/seed_1
python -m scripts.report outputs/state_pinn_review
~~~

若在绘图环境中调用已有训练环境，增加 --python .venv_signatory/Scripts/python.exe。评估脚本应由训练环境运行。训练完成后自动保存 surface_validation.csv 和 validation.json，检查未设锚点的 30/29 个价格状态、障碍边界和终端支付。

当前单 seed、默认 call 合约的复核结果：向上敲出初始价格误差约 3.00%，向下敲出约 0.21%；独立网格价格相对 L2 误差分别约 13.60% 和 0.14%。向上敲出的近到期边界、终端误差仍较大，尚不能声称整个价格函数收敛。Notebook 保存了实际结果与曲线。

## 向上敲出：终端采样单项对照

只改变终端采样、保持网络、PDE 采样、损失权重、随机种子和 10,001 轮训练预算不变。旧采样只覆盖到 1.14，新采样覆盖障碍 1.2 内侧；两者消耗相同数量的随机数。

~~~bash
python -m scripts.run_comparison --methods state_pinn --products barrier_upout --terminal-sampling full --epochs 10001 --batch-size 128 --mc-paths 200000 --output-dir outputs/state_pinn_terminal_sampling
python -m scripts.state_pinn.evaluate outputs/state_pinn_terminal_sampling/barrier_upout/state_pinn/seed_1 --compare-to outputs/state_pinn_review/barrier_upout/state_pinn/seed_1
~~~

默认 --terminal-sampling legacy 保留原配置；full 仅作为单项对照。固定网格终端 RMSE 从 0.0446 降到 0.0208，但边界 RMSE 从 0.0101 增至 0.0288，内部价格相对 L2 误差从 13.60% 增至 21.53%。初始价格误差从 3.00% 增至 5.32%。因此这项改动尚未整体改善求解，不作为默认修复。下一步需单独验证仍被 PDE 采样遗漏的近障碍区域。

本地对照数据、曲线保存在 outputs/state_pinn_terminal_sampling/barrier_upout/state_pinn/seed_1/terminal_sampling_comparison.csv 和 terminal_sampling_curves.csv。Notebook 已嵌入实测结果。

## 向上敲出：PDE 采样范围对照

两组均使用补全的终端采样、相同种子和 10,001 轮预算。为使近障碍差分不越界，两组都将空间步长限制在到障碍距离的 45% 以内，并用双精度计算 PDE 差分；模型参数仍保持 float32。两组之间只改变 PDE 采样范围，默认配置未改。

~~~bash
python -m scripts.run_comparison --methods state_pinn --products barrier_upout --terminal-sampling full --pde-sampling legacy --pde-precision float64 --epochs 10001 --batch-size 128 --mc-paths 200000 --output-dir outputs/state_pinn_pde_control
python -m scripts.run_comparison --methods state_pinn --products barrier_upout --terminal-sampling full --pde-sampling full --pde-precision float64 --epochs 10001 --batch-size 128 --mc-paths 200000 --output-dir outputs/state_pinn_pde_full
~~~

相同安全差分下，完整 PDE 覆盖使初始价格误差从 5.32% 降至 2.65%，但内部网格相对 L2 误差从 21.53% 升至 67.04%，边界 RMSE 从 0.0288 升至 0.0314，终端 RMSE 从 0.0208 升至 0.0330。这项改动也不替换原默认配置。

在实验组最终模型的一批固定诊断样本上，加权终端损失与边界损失的梯度余弦约 -0.945，呈强烈反向；PDE 与终端项约 -0.771。它提示当前优化在终端、边界条件之间存在明显竞争，不表示这些条件在数学上不相容，也不能从一次诊断推出全部训练过程的因果关系。

Notebook 读取两组的 validation.json、surface_validation.csv 展示对照。原默认模式的短运行 checkpoint 已确认与改动前完全一致。

## 向上敲出：学习率衰减对照

初始学习率保持 5e-4，仍每 1,000 轮衰减，只把衰减因子从 0.5 改为 0.9。10,001 轮结束时，学习率从约 4.88e-7 提高到 1.74e-4。分别在原采样配置和补全采样配置下比较，默认设置未改。

~~~bash
python -m scripts.run_comparison --methods state_pinn --products barrier_upout --lr-gamma 0.9 --epochs 10001 --batch-size 128 --mc-paths 200000 --output-dir outputs/state_pinn_lr_decay_legacy
python -m scripts.run_comparison --methods state_pinn --products barrier_upout --terminal-sampling full --pde-sampling full --pde-precision float64 --lr-gamma 0.9 --epochs 10001 --batch-size 128 --mc-paths 200000 --output-dir outputs/state_pinn_lr_decay
~~~

原采样配置下，内部网格相对 L2 误差由 13.60% 降至 10.54%，边界 RMSE 由 0.0101 降至 0.00794，终端 RMSE 由 0.0446 降至 0.0343，初始价格误差由 3.00% 小幅降至 2.85%。补全采样配置下，内部和边界误差也下降，但终端、初始价格误差上升，仍有取舍。

每一对实验除衰减因子外配置相同，前 1,000 轮的全部 loss 记录与最终评估 MC 完全一致。两个采样背景使用的 PDE 差分精度不同，应在各背景内部比较学习率影响。这是单 seed 结果，不据此替换默认配置或声称完整收敛。

统一入口可显式覆盖 --lr-step 和 --lr-gamma，未指定时保留各方法原有默认值。Notebook 自动生成学习率及 loss 对照图和 outputs/state_pinn_lr_decay/lr_decay_comparison.csv。

## 来源

原论文：[Solving path dependent PDEs with LSTM networks and path signatures](https://arxiv.org/abs/2011.10630)，Marc Sabate-Vidales、David Šiška、Lukasz Szpruch（2020）。[论文 PDF](docs/2011.10630_Solving_path_dependent_PDEs_with_LSTM_networks_and_path_signatures.pdf) 已归档。

论文配套代码：[msabvid/Deep-PPDE](https://github.com/msabvid/Deep-PPDE)，其 GPL-3.0 许可副本保存在 `docs/Deep-PPDE-LICENSE.txt`。本项目的三方法实验由工作区现有实现整理而来，未将整个原始代码库打包进此仓库。
