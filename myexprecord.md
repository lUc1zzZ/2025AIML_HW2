# CIFAR-10 CNN Training (HW2)

本仓库实现了一个用于 CIFAR-10 图像分类的 PyTorch 实验套件，包含：可配置的卷积网络（可调宽度、dropout、归一化方式）、训练/评估工具、k-fold 交叉验证、网格搜索脚本以及模型集成评估工具。

**目标**：比较 dropout、不同 normalization、超参数（通过交叉验证）对泛化性能的影响，并尝试附加改进（增宽/加深、正则化、模型集成）。

**快速开始**

1. 创建并激活 Python 环境（推荐 Python 3.8+）

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 训练示例（单次训练）

```powershell
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --k-folds 1 --augment --output-dir outputs --run-name baseline_bn_dropout0.3
```
快速测试
python train.py --epochs 3 --batch-size 8 --lr 0.01 --dropout 0.0 --norm none --output-dir test_output --run-name quick_test

主要文件

- `train.py`: 训练入口，支持命令行参数、数据增强、k-fold CV、保存模型与绘制学习曲线。
- `model.py`: 卷积网络实现，支持 `dropout` 与多种 normalization。
- `utils.py`: 训练/评估循环与绘图工具。
- `requirements.txt`: 依赖列表。


训练时请保存学习曲线（loss/accuracy）、记录超参数，并在报告中包含模型结构、训练曲线、交叉验证结果与最终测试准确率。可在 `outputs/` 目录下查找生成的图表和模型权重。


1.基线实验（baseline）
得到一个参考模型（BatchNorm + dropout=0.3）
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --k-folds 1 --augment --output-dir outputs --run-name baseline_bn_dropout0.3

2.Dropout 对比（不同 dropout 强度）
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.0 --norm batch --augment --output-dir outputs --run-name drop0_noaug
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.1 --norm batch --augment --output-dir outputs --run-name drop0.1
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --output-dir outputs --run-name drop0.3
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.5 --norm batch --augment --output-dir outputs --run-name drop0.5

3.Normalization 方法对比
比较 BatchNorm / GroupNorm / InstanceNorm / 无归一化 的影响
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --output-dir outputs --run-name norm_batch
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm group --augment --output-dir outputs --run-name norm_group
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm instance --augment --output-dir outputs --run-name norm_instance
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm none --augment --output-dir outputs --run-name norm_none
注意：GroupNorm 的 group 数自动用 min(8, channels)（在代码中实现）。

4.使用交叉验证进行超参数选择（示例：寻找最佳 dropout）
目的：通过 k-fold 找到在验证集上表现最好的超参数（例如 dropout）

python auto_experiment.py --grid grid.json --kfolds 3 --workers 1 --output-dir outputs --epochs-grid 20 --epochs-final 40 --batch-size 128 --augment
脚本用途：自动做 网格搜索（k-fold CV）→ 汇总 → 从最佳 run 读取超参 → 在整个训练集上用该超参重训练最终模型。
必备文件：grid_search.py、summarize_cv.py、train.py、grid.json（超参数网格）都应在当前工作目录。
输出：所有 run 存在 --output-dir（默认 outputs）下；最终会在该目录下生成 cv_summary.csv 与最终 final_<run> 目录及模型文件。
参数说明（常用）
--grid：超参数网格文件（JSON），必须指定，例如 grid.json。
--kfolds：交叉验证折数（k），例如 3 或 5。
--workers：在 grid_search.py 中并行 worker 数（建议单 GPU 时设 1）。
--output-dir：结果保存目录（默认 outputs）。
--epochs-grid：网格搜索阶段每个组合训练的 epoch（筛选时可短，如 10–20）。
--epochs-final：选出最佳超参后最终训练的 epoch（通常更大，如 30–100）。
--batch-size：训练 batch size。
--augment：如果指定则在训练时使用数据增强（训练子集）。
--python-exe：若需指定 Python 可执行程序（默认 python）



5.增加模型宽度/深度（在本代码里用 --base-channels 控制宽度）
例：把 base_channels 从 32 提升到 48 或 64：
python train.py --epochs 25 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --base-channels 48 --output-dir outputs --run-name wide48
python train.py --epochs 25 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --augment --base-channels 64 --output-dir outputs --run-name wide64
说明：更大的模型通常训练时间更长，可能需要降低学习率或增加正则化。

6.使用额外正则化（weight decay 已有，可尝试更大或使用 label smoothing）
修改命令（改变 --weight-decay）
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --weight-decay 0.0005 --augment --output-dir outputs --run-name wd5e-4
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --weight-decay 0.001 --augment --output-dir outputs --run-name wd1e-3
label smoothing
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --label-smoothing 0.1 --output-dir outputs --run-name label_smooth_0.1
Label smoothing


7.模型集成（训练多个稍有差异的模型并做投票/平均）
步骤：
训练 N 个模型（例如 N=3）使用不同随机种子或不同初始化/超参，分别保存在 outputs/ensemble_model1/ 等目录。
写一个小脚本 ensemble_eval.py 来加载这几个模型，对测试集取 softmax 平均或多数投票，再计算最终准确率。
简单训练命令（训练三个模型）
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --seed 1 --output-dir outputs --run-name ensemble_m1
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --seed 2 --output-dir outputs --run-name ensemble_m2
python train.py --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --seed 3 --output-dir outputs --run-name ensemble_m3
运行集成评估：
假设你已经有多个模型权重文件（例如 `outputs/ensemble_m1_xxx/fold1/best_fold1.pt` 或 `outputs/ensemble_m1_xxx/best.pt`），可以运行：
```powershell
python ensemble_eval.py --model-paths outputs\ensemble_m1_*/best.pt outputs\ensemble_m2_*/best.pt outputs\ensemble_m3_*/best.pt --norm batch --dropout 0.3 --base-channels 32 --output-dir outputs --run-name my_ensemble
```
或者显式指定每个模型的 checkpoint 路径：
```powershell
python ensemble_eval.py --model-paths outputs\ensemble_m1_2023*/best.pt outputs\ensemble_m2_2023*/best.pt outputs\ensemble_m3_2023*/best.pt --norm batch --dropout 0.3 --base-channels 32 --output-dir outputs --run-name my_ensemble
```
脚本会平均 softmax 概率并输出总体测试准确率，同时把结果保存在 `outputs/ensemble_my_ensemble.json`。
自动化脚本
python auto_ensemble.py --n-models 3 --epochs 20 --batch-size 128 --dropout 0.3 --norm batch --output-dir outputs --run-prefix ensemble_auto --base-channels 32




Best val acc: 0.7748
Test loss: 0.6322 test acc: 0.7775
Best val acc: 0.8778
Test loss: 0.3837 test acc: 0.8714
Best val acc: 0.8594
Test loss: 0.4191 test acc: 0.8564
Best val acc: 0.7748
Test loss: 0.6322 test acc: 0.7775
Best val acc: 0.5656
Test loss: 1.2576 test acc: 0.5557
Best val acc: 0.7748
Test loss: 0.6322 test acc: 0.7775
Best val acc: 0.7354
Test loss: 0.8289 test acc: 0.7060
Best val acc: 0.6998
Test loss: 0.9085 test acc: 0.6784
Best val acc: 0.5552
Test loss: 1.2170 test acc: 0.5491
Best val acc: 0.8354
Test loss: 0.4934 test acc: 0.8338
Best val acc: 0.8496
Test loss: 0.4527 test acc: 0.8450
Best val acc: 0.7748
Test loss: 0.6322 test acc: 0.7775
Best val acc: 0.7896
Test loss: 0.5997 test acc: 0.7914
Best val acc: 0.7938
Test loss: 0.6308 test acc: 0.7955



