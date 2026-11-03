# RTAT 代码与数据说明

本仓库包含完整 RTAT 模型结构、训练、显式微调、路径与组件消融接口、预测、空间评估、多种子汇总和数据准备代码。`external_data/` 附带与实验校验值一致的六组外部特征和重构日历。

不公开训练权重、历史预测文件、原始运行日志和私人实验记录。流量基准数组请从 [ProSTformer 原仓库](https://github.com/yanxiao1930/ProSTformer_code) 获取，原仓库 README 提供 Taxi/Bike 数据下载入口。所有权重、日志和结果都可以由新训练在本地生成。

## 安装与检查

推荐 Python 3.10；依赖参考版本为 PyTorch 2.6.0 和 NumPy 2.2.6。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/forward.py
```

前向示例使用随机权重和合成数据，只验证运行，不代表论文预测结果。完整训练建议使用 CUDA GPU。

## 数据准备

把流量数组放进 `benchmark/data_nyctaxi/` 与 `benchmark/data_nycbike/`，随后运行：

```bash
python prepare_data.py --flow-root benchmark --external-root external_data --output-root data
python train.py --data-root data --dataset nyctaxi --minutes 30 --validate-only
```

若只准备一组，在第一条命令后增加 `--dataset nyctaxi --minutes 30`。输出目录必须尚不存在。代码会核对数据 SHA256；不应绕过检查，把不同数组当成相同实验输入。

## 训练与消融

```bash
python train.py --data-root data --dataset nyctaxi --minutes 30 --variant full --seed 0 --output runs/taxi30_full_seed0
python run_suite.py --data-root data --output-root runs/path_taxi30 --dataset nyctaxi --minutes 30 --variants backbone tod fixed_residual gated_residual full --seeds 0 1 2 --dry-run
```

第二条命令先预览五种结构 × 三个种子的运行命令；删除 `--dry-run` 后才会顺序训练。遇到失败立即停止，已存在的运行目录会被拒绝。汇总用：

```bash
python summarize.py --runs-root runs/path_taxi30 --output results/path_taxi30_summary.json
```

微调通过 `--init-from` 显式指定同一数据集的 30 分钟 checkpoint。直接训练与微调是不同协议。预测及空间评估的完整命令见英文 README。新训练将生成权重、归一化参数和运行记录；这些本地生成文件已被 `.gitignore` 排除。

本仓库支持运行新实验，未认证所有历史表格在不同软件与硬件环境下数值完全一致。历史运行记录、权重和第三方基线实现未附带。日历为重构标签，气象为滞后的回顾性 ERA5 数据。详细范围见 `docs/VERIFICATION.md`，数据署名与来源见 `external_data/ATTRIBUTION.md`。
