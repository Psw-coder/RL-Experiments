# Breakout RPO / PPO 实验

实现论文 [Reflective Policy Optimization](https://arxiv.org/abs/2406.03678) 公式 (5)–(7)，参考[作者 Atari 设置](https://github.com/Edgargan/RPO)。使用 PyTorch 2.9.1 + CUDA 12.8（支持 Blackwell/sm_120），默认 GPU 训练。

## 训练与查看进度

```bash
cd /mnt/data/users/pengzhonghua/psw_temp/breakout_rpo
bash run.sh
```

终端中不带参数启动会显示交互菜单：`1` 为 RPO（回车默认），`2` 为 PPO，`q` 退出；无效输入会重新提示。带参数或无终端输入时不弹出菜单，算法默认取配置中的 `algorithm`（旧配置默认为 RPO）。

也可以直接指定算法，适用于后台运行和批量实验：

```bash
bash run.sh --algorithm ppo --run-name ppo_seed0
bash run.sh --algorithm rpo --run-name rpo_seed0
# CPU 小规模流程验证
bash run.sh --algorithm ppo --device cpu --num-envs 1 --rollout-length 8 --total-timesteps 16
```

PPO 使用裁剪策略目标、标准化 GAE 优势和与作者代码一致的折扣回报价值目标（默认 `critic_target=discounted`；如需统一 `GAE + V`，可加 `--critic-target gae`），总损失为策略损失 + 0.5 × 价值均方误差 − 熵系数 × 策略熵；不计算相邻状态反思项，`beta` 和 `next_clip_coef` 不参与 PPO。网络、Atari 预处理及其他默认超参数与 RPO 共用，便于对比。PPO 看板不显示反思曲线，原始日志中的 `reflective_loss` 为 0。

自动生成的运行名带 `ppo_` / `rpo_` 前缀，配置和 checkpoint 均保存算法标识；两种算法都可使用下方同一评估命令。

每次启动都会打印完整生效配置，包括超参数、GPU 型号、batch 大小、实际总步数和输出目录；同时保存为该次运行的 `config.json`。

终端每次更新显示：进度百分比、步数、最近 100 个完整游戏的平均原始得分、已完成游戏数、loss、策略熵、近似 KL、FPS、耗时和预计剩余时间。

打开终端打印的 `runs/<运行名>/dashboard.html` 即可查看曲线；浏览器每 10 秒自动刷新，无需 TensorBoard 或联网。也可以直接查看同目录的 `curves.png`。如果通过远程服务器查看，需要把 HTML 和 PNG 放在同一目录，或使用编辑器直接预览服务器上的 PNG。

六组曲线包括：

- 最近 100 个完整游戏的平均原始分数（未完成游戏时显示等待，不伪造 0 分）。
- 总损失、PPO 损失、反思损失（乘 β 前）和价值损失。
- 策略熵。
- 近似 KL 和裁剪比例（所有 minibatch 更新前的统计均值）。
- 价值函数 explained variance（采样时的价值预测对折扣回报的拟合程度）。
- 环境步数/秒（包含之前的绘图开销）。

首个更新后、每隔 `plot_every` 个更新和结束时刷新图像。默认 `plot_every=10`，以减少绘图开销；如需每次更新绘图：

```bash
bash run.sh --run-name breakout_seed0 --plot-every 1
```

训练参数在 `config.json` 中。默认：rollout=256，16 个并行环境，4 epochs，学习率=2.5e-4，γ=0.99，GAE λ=0.95，β=3.0，critic 目标为折扣回报（`critic_target=discounted`，与作者 Atari 代码一致）。两个概率比各自裁剪到 `[0.9,1.1]`；每轮 4 个 minibatch；熵系数 0.01；梯度范数上限 0.5。

默认 5000 万个环境步数，按完整 rollout 向上取整；每步重复 4 帧。命令行可覆盖步数、种子、环境数、rollout、设备和绘图频率。需要在可访问 GPU 的终端中执行；默认不会静默回退到 CPU。

## 文件与输出

核心文件：`train.py`（训练）、`rpo.py`（共用模型/RPO 目标）、`ppo.py`（PPO 目标）、`atari.py`（预处理）、`monitor.py`（绘图）、`evaluate.py`（评估）、`compare.py`（β 对照）、`repro_compare.py`（PPO vs RPO 复现对照）、`runtime.py`/`env.sh`（项目内存储配置）。

每次运行输出在 `runs/<运行名>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 完整生效配置 |
| `metrics.jsonl` | 每次更新的原始指标 |
| `dashboard.html` | 自动刷新看板 |
| `curves.png` | 六组训练曲线 |
| `latest.pt` | 最近保存的模型/优化器/步数 |

默认每 100 次更新及正常结束时保存模型。图像、HTML、checkpoint 均通过临时文件原子替换。异常退出会尽可能更新最后的曲线及状态；尚未实现精确断点续训。

## 评估

```bash
source env.sh
python -B evaluate.py runs/<运行名>/latest.pt --episodes 10
```

默认 GPU、采样动作，可加 `--deterministic` 使用最大概率动作。评估使用完整游戏、未裁剪奖励，结果写入模型同目录 `evaluation.json`。

## 环境与算法说明

现有 `.venv` 已安装依赖及 Breakout ROM。重建使用 `bash setup.sh`（Linux x86_64、Python 3.13/3.14、CUDA 12.8 wheel、联网），安装版本见 `requirements.lock.txt`。默认 `torch 2.9.1+cu128`；旧的 cu124 wheel 不支持 Blackwell/sm_120。代码、环境、ROM、下载缓存、临时文件和训练输出均在本项目目录内；使用已有系统解释器及 GPU 驱动。

RPO 目标为 PPO surrogate 加 β 倍相邻状态双概率比 surrogate，分别裁剪两个比率并保留两者梯度。配对严格限于同一环境的相邻时间步，屏蔽终止、截断和 rollout 边界。Actor 使用标准化 GAE，PPO 和 RPO 的 critic 默认使用与作者代码一致的折扣回报目标（`critic_target=discounted`）；如需统一使用 `GAE + V` 的 lambda-return，可加 `--critic-target gae`；时间上限截断使用最终观测 bootstrap。

Atari 使用灰度 84×84、4 帧堆叠、4 帧重复及最大池化、随机 NOOP 和 FIRE reset，sticky actions p=0.25（与作者 `BreakoutNoFrameskip-v4` 一致）。训练丢命终止和符号奖励仅用于学习，记录分数为完整游戏原始分数。现代 Gymnasium/ALE 与原论文环境版本不同，不宣称逐位复现论文结果。

此前已在 RTX 4090 上按 16×256 rollout、4 epochs 完成 3 次更新（12,288 步），配置打印、指标记录、最近 100 局均分计算、曲线图及完成状态均通过检查；结果示例位于 `runs/monitor_check/`。这只是流程验证，不是长期训练结果。此次改为 sticky actions 0.25、折扣回报 critic 和作者初始化后，建议先跑一次短程 smoke test。

## 统一 GAE critic 的反思项对照（可选）

默认已使用论文/作者的 `critic_target=discounted`。如需做统一 `GAE + V` 的受控对照，显式指定 `--critic-target gae`：

```bash
source env.sh
python -B compare.py --name unified_seed0 --critic-target gae
```

该命令从头串行训练两组 5000 万步实验，两组均使用 RPO 路径、GAE+V、seed=0，仅 β=0/3 不同。β=0 的策略目标及梯度等同 PPO。各组训练完成后以 seed=10000、采样动作评估 100 个完整游戏。两组完成后生成 `runs/unified_seed0/report.md`、`summary.json` 和 `comparison.png`。`status.json` 记录当前阶段或失败原因，`beta0_train.log`/`beta3_train.log` 记录训练日志；各组目录含独立看板和 checkpoint。运行时保存并执行源码快照，避免后续编辑影响排队中的实验。

按历史速度两组训练约需 10 小时，另加评估时间。这是单训练种子的受控对照；重复不同种子可使用 `--seed 1 --name unified_seed1`。旧结果不混入新实验。

## PPO vs RPO 论文复现对照

```bash
source env.sh
python -B repro_compare.py --name repro_seed012
```

该命令按论文/作者 Atari 设置（`critic_target=discounted`、sticky actions 0.25、作者网络初始化），对 PPO 和 RPO 各训练 3 个 seed（默认 0/1/2，每个 5000 万步），汇总每个 run 最后一个更新的 `return_mean_100`（最后 100 个完整游戏的平均原始得分）及跨 seed 均值和标准差，生成 `runs/repro_seed012/report.md`、`summary.json` 和 `comparison.png`。可加 `--eval-episodes 100` 额外用采样动作评估 100 个完整游戏。论文 Table 2 的 Breakout 参考值为 PPO 233、RPO 306（3 seeds，最后 100 局）。

注意：作者公开的 Atari 代码把 `[T,N]` 展平后用 `index+1` 配对，实际配到的是相邻 worker 的同一时刻；本实现按论文语义配合同一 worker 的下一时刻。若需要复刻作者代码的这一实现细节，需要另加 pairing 开关。

运行单元测试：`source env.sh && python -B -m unittest discover -s tests -v`。

# RL-Experiments
