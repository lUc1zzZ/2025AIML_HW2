# CIFAR-10 CNN Training (HW2)

本仓库实现了一个用于 CIFAR-10 图像分类的 PyTorch 实验套件，包含：可配置的卷积网络（可调宽度、dropout、归一化方式）、训练/评估工具、k-fold 交叉验证、网格搜索脚本以及模型集成评估工具。

**目标**：比较 dropout、不同 normalization、超参数（通过交叉验证）对泛化性能的影响，并尝试改进（增宽/加深、正则化、模型集成）。

注意：data由于过大无法上传

---

## 快速开始

1. 创建并激活 Python 虚拟环境并安装依赖：

```powershell
conda create -n hw2 python=3.12
conda activate hw2
pip install -r requirements.txt
```

2. 训练示例（单次训练）：

```powershell
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --k-folds 1 --augment --output-dir outputs --run-name baseline_bn_dropout0.3
```

3. 快速 smoke-test：

```powershell
python train.py --epochs 3 --batch-size 8 --lr 0.01 --dropout 0.0 --norm none --output-dir test_output --run-name quick_test
```

---

## 主要文件

- `train.py`：训练入口，包含 k-fold、保存 checkpoint、绘图和日志。
- `model.py`：可配置的 ConvNet，参数包括 `--dropout`、`--norm`（batch/group/instance/none）、`--base-channels`。
- `utils.py`：训练/评估循环、绘图、CSV/JSON 保存工具（含 label smoothing 支持）。
- `ensemble_eval.py`：加载多个 checkpoint，做 softmax 平均并评估测试精度。
- `grid_search.py` / `summarize_cv.py` / `auto_experiment.py` / `auto_ensemble.py`：负责网格搜索、汇总与自动化实验/集成。

所有训练输出（模型权重、学习曲线、cv 结果）默认保存在 `outputs/` 下，每个 run 目录包含 `args.json` 和训练记录（history CSV/图像等）。

---

## 推荐实验与命令

### 1) Baseline

BatchNorm + dropout=0.3（带训练增强）：

```powershell
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --output-dir outputs --run-name baseline_bn_dropout0.3
```

### 2) Dropout 对比

只改变 `--dropout` 值做对比：

```powershell
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.0 --norm batch --augment --output-dir outputs --run-name drop0
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.1 --norm batch --augment --output-dir outputs --run-name drop0.1
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --output-dir outputs --run-name drop0.3
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.5 --norm batch --augment --output-dir outputs --run-name drop0.5
```

### 3) Normalization 比较

比较 `batch/group/instance/none`：

```powershell
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --output-dir outputs --run-name norm_batch
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm group --augment --output-dir outputs --run-name norm_group
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm instance --augment --output-dir outputs --run-name norm_instance
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm none --augment --output-dir outputs --run-name norm_none
```

> 注：`GroupNorm` 的 group 数在代码中按 `min(8, channels)` 自动设置。

### 4) k-fold 交叉验证 + 网格搜索

- 准备 `grid.json`（超参组合），运行网格搜索（示例 3-fold）：

```powershell
python grid_search.py --grid grid.json --kfolds 3 --workers 1 --output-dir outputs --run-prefix gs1 --epochs 20 --batch-size 128 --python-exe python
```

- 汇总 CV 结果并选择最佳组合：

```powershell
python summarize_cv.py --outputs outputs --out-csv outputs/cv_summary.csv
```

- 用最佳超参在全训练集上训练更长的 epoch：

```powershell
python train.py --epochs 40 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --weight-decay 0.0005 --label-smoothing 0.0 --output-dir outputs --run-name final_best
```

或使用自动化脚本一键完成（网格搜索 → 汇总 → 重训练）：

```powershell
python auto_experiment.py --grid grid.json --kfolds 3 --workers 1 --output-dir outputs --epochs-grid 20 --epochs-final 40 --batch-size 128 --augment
```

### 5) 增强模型与正则化

- 增宽/加深：通过 `--base-channels` 控制，例如 `--base-channels 48`。
- 正则化：调大 `--weight-decay` 或使用 `--label-smoothing`（例如 `--label-smoothing 0.1`）。

### 6) 模型集成（ensemble）

- 推荐起点（成本低）: 相同架构与超参，仅改变 `--seed` 训练 N 个模型（如 N=3），再对它们 softmax 概率求平均。
- 若资源允许，可混合不同超参或不同架构增加多样性，通常带来更大提升。

自动化训练并评估（仓库内置）：

```powershell
python auto_ensemble.py --n-models 3 --epochs 20 --batch-size 128 --dropout 0.3 --norm batch --output-dir outputs --base-run-name ensemble_auto --base-channels 32
```

手动训练后按 checkpoint 路径显式指定：

```powershell
python ensemble_eval.py --model-paths outputs\\ensemble_m1_*\\best.pt outputs\\ensemble_m2_*\\best.pt outputs\\ensemble_m3_*\\best.pt --norm batch --dropout 0.3 --base-channels 32 --output-dir outputs --run-name my_ensemble
```

---