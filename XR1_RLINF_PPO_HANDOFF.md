# XR-1 + RLinf PPO：交接与实验状态（2026-09-18）

这份文档是给新对话/新维护者的快速入口。它记录当前代码实际实现的
XR-1 RoboCasa365 PPO 框架、已验证/未验证的结论、结果位置与后续实验准则。
不要把“作业正常结束”误解成“性能已提升”：当前长跑尚在进行，最终只能以
完整的独立评测和训练曲线判断。

## 0. 项目背景与本分支范围

上游 `README.md` 的 Xiaomi-Robotics-1（XR-1）是一个 Vision-Language-Action
foundation model：Qwen3-VL VLM 与较小的 DiT/MoT action generator 结合，原始
训练分为大规模 pre-training 和 embodiment/instruction post-training。上游发布了
RoboCasa365 对齐 checkpoint `checkpoints/Xiaomi-Robotics-1-RoboCasa365`，它是本
PPO 实验的起点；不是从随机初始化训练。

本分支增加的是**在线、环境回报驱动的 RLinf PPO post-training adapter**，目标是
回答“在固定 XR-1 checkpoint、RoboCasa365 稀疏成功回报和给定算力下，单纯 PPO
还能带来多少提升”。它不改写上游监督式 post-training 配方，也不应把 README 中的
官方 benchmark 数字当成当前 PPO checkpoint 的分数。

## 1. 当前结论和正在运行的实验

### 已验证

- XR-1 能通过 RLinf 的 embodied PPO 接口训练，当前策略是**全量训练**
  (`actor.model.trainable_mode: full`)，且 value head 一同训练。
- `flow_hutchinson` 的 rollout、缓存 likelihood、PPO actor backward 和 FSDP
  均已在 GPU 上跑通。关键是关闭 FSDP 的自动嵌套包装
  (`XR1_RLINF_DISABLE_AUTO_WRAP=1`)，保留最外层 FSDP。
- 1-update 和 10-update 的 single-task integration run 都以 `COMPLETED (0:0)`
  结束，且没有 `Non-finite grad norm`/`Skipping optimizer step`。
- RoboCasa365 评测的 EGL/MuJoCo `read_pixels` 原生 abort 可以用“每个 episode
  后重启 Python worker”绕过。OpenDrawer 完整 5/5 评测成功（见第 9 节）。

### 不能据此宣称的结论

- 10 updates、且 reward 为零的 smoke **不证明** PPO 提升了成功率。
- `flow_hutchinson` 是发布版五步 Euler ODE 的离散反积分 + Hutchinson trace
  估计，不是连续 ODE 的精确解析 likelihood。
- 单任务 smoke 的 KL/clip/EV 仅是链路健康指标，不是全任务泛化能力证据。

### 进行中的主实验

| 项目 | 值 |
| --- | --- |
| Slurm job | `106535` (`xr1-flow-target50-1k-r256`) |
| 目标 | 纯 PPO 在 XR-1 + RoboCasa365 `target50` 的能力边界 |
| 状态（最近更新时） | `RUNNING`，节点 `gnho020` |
| 更新数 | 1000 (`runner.max_steps`，即 1000 次 PPO update；不是 1000 个 action/episode) |
| 任务 | 所有 `target50`，随机采样，无 task filter |
| 策略训练范围 | full model + value head |
| actor LR / critic LR | `1e-9` / `1e-4` |
| rollout / episode 配置 | 1 environment，64 / 64 |
| checkpoint | 250、500、750、1000 update |
| 输出目录 | `results/xr1_ppo_flow_target50_full_1000u_lr1e9_rootfsdp/` |

状态查询：

```bash
squeue -j 106535 -o "%.18i %.12T %.30j %.10M %.20R %.20b"
tail -n 120 logs/xr1_rlinf_ppo_106535.log
```

## 2. 仓库与关键入口

仓库根目录是本文件所在目录（下文用 `$REPO` 表示）。核心文件：

| 作用 | 文件 |
| --- | --- |
| XR-1 到 RLinf 的 PPO policy adapter | `xr1/mibot/rlinf/ppo_policy.py` |
| adapter 使用与约束 | `xr1/mibot/rlinf/README.md` |
| 受版本控制的 RLinf 配置镜像 | `xr1/mibot/rlinf/configs/` |
| 标准 RoboCasa365 PPO 配置 | `configs/xr1_robocasa365_ppo_full.yaml` |
| flow likelihood 模型覆盖 | `configs/xr1_ppo_flow.yaml` |
| flow smoke 组合配置 | `configs/xr1_robocasa365_ppo_flow_smoke.yaml` |
| Slurm 训练提交脚本 | `scripts/run_xr1_rlinf_ppo.sbatch` |
| policy / PPO 数学单测 | `xr1/tests/test_rlinf_ppo_policy.py`、`xr1/tests/test_rlinf_ppo_math.py` |
| 任务调度式评测 worker | `eval_robocasa365/dynamic_eval.py` |
| 评测 worker 启动/重启逻辑 | `scripts/launch_robocasa365.sh` |
| 隔离评测的 Slurm 提交 | `scripts/run_robocasa365_policy_eval_isolated.sbatch` |

依赖的 Python 环境是 `$REPO/.conda-robocasa365/bin/python`。RLinf 是
`$REPO/third_party/RLinf`；运行本地测试时必须显式设置 `PYTHONPATH`。RLinf 是外部
checkout（不会随本仓库上传）；训练 sbatch 在启动时把 `configs/` 内的四个配置镜像同步
到其 Hydra config 路径。

## 3. 数据流与 PPO 结构

```text
RoboCasa365 observation
  -> obs_to_batch / XR-1 multimodal processor
  -> released XR-1 VLM + DiT conditional five-step ODE
  -> normalized 60-D action chunks -> RoboCasa environment
  -> rollout batch (action, native XR-1 batch, old logprob, value, reward)
  -> RLinf PPO (GAE, ratio clipping, actor + value losses)
  -> XR1PPOPolicy.default_forward() recomputes current logprob/value
```

`XR1PPOPolicy` 的职责：

1. 将 RLinf 的 flat `forward_inputs` 还原为 XR-1 native multimodal batch；
2. 以 XR-1 生成的 5-step ODE action 进行 rollout；
3. 为 PPO 提供每个 action coordinate 的 logprob 和每个 chunk 的 value；
4. 保存/恢复必要的 rollout diagnostics，使 RLinf 在时间和 batch 维度切分后仍可
   计算 old/new policy ratio；
5. 处理 action horizon、mask、60D normalized action 到环境 action 的适配。

当前 `full` 模式会将 `xr1_model` 全部参数设为 trainable；`action_expert`
模式才会冻结 language/choice 模块并仅保留动作相关模块和 value head。两种模式
由 `set_trainable_mode()` 控制。主实验是 **full**，不是 action-expert-only。

## 4. 为什么需要 flow likelihood

### 旧的 Gaussian surrogate

旧实现把 XR-1 ODE 输出当作均值，并另加一个 diagonal Gaussian action policy。
这方便计算 `log_prob`，但并不是 XR-1 实际 sampling distribution：XR-1 本身从
Gaussian noise 经过 conditional ODE 生成 action，再在 action space 额外采样
Gaussian 会改变所执行的策略。旧 100-update run 的典型症状是
`approx_kl≈0.264`、clip fraction `≈0.261`、最大 logprob 差 `≈12.678`，critic EV
接近零或为负，不能作为可信提升基线。

### `flow_hutchinson`

当前模式把 sampler 视为 probability-flow ODE：

1. rollout 从标准 Gaussian 初始噪声，经发布的 5-step Euler ODE 得到 action；
2. actor update 固定已执行 action，反向 Euler 积分回 base noise；
3. log density = 标准 Gaussian base logprob + change-of-variables Jacobian；
4. Jacobian divergence 用 Rademacher Hutchinson probes 估计；
5. probes 缓存在 rollout 内，actor 重算时复用，避免 estimator noise 被误当成 PPO
   policy shift。

重要约束：

- flow 模式必须 `clip_normalized_action: null`，否则 clipping 改变了密度；
- `algorithm.entropy_bonus: 0.0`，因为当前没有可用的 flow entropy estimator；
- `flow_trace_samples: 1` 是当前吞吐/方差折中；增加 probes 前先确认长跑健康；
- 流 likelihood 需要 Hessian-vector product（divergence 对模型参数反传），代价比
  surrogate 更高。

## 5. 已修复的训练问题与原因

### A. trace probe 的 RLinf batch 路由错误

RLinf 对每个 `forward_inputs` tensor 按第 0 维切分。初版 probes 是
`[ode_step, trace, batch, ...]`，会被误认为 batch 为 ode step，导致
`split_with_sizes ... got [1]`。现在落盘时转成 batch-first：

```python
forward_inputs["xr1_trace_probes"] = probes.movedim(2, 0).cpu()
# [batch, ode_step, trace, ...]
```

读取时在 `_flow_logprobs()` 恢复内部 `[step, trace, batch, ...]` 布局。

### B. CUDA efficient/flash attention 没有二阶导

Hutchinson divergence 的训练反传触发 attention backward-of-backward；CUDA
memory-efficient/flash SDP 会报：

```text
RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
```

`_flow_logprobs()` 仅在 likelihood 路径强制 math SDP：

```python
torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_mem_efficient=False, enable_math=True
)
```

普通 rollout generation 仍可使用快速 attention kernel。

### C. 嵌套 FSDP 的二阶 backward 状态错误

math SDP 后，自动包装的 Transformer child FSDP 仍会在二阶导时报：

```text
ValueError: expected to be in states [FORWARD_BACKWARD] but current state is IDLE
```

训练提交时设置 `XR1_RLINF_DISABLE_AUTO_WRAP=1`。这让 RLinf 的 outer FSDP 仍存在，
但不再把每个 Transformer 层套成 child FSDP；1/10 update GPU 运行已验证这条路径。

### D. 小于 episode 的 rollout 曾产生 NaN

一次 `rollout_steps=16` 试验没有收集完整轨迹，出现 `num_trajectories=0`、NaN
advantages，并跳过 optimizer。主实验使用 64。注意运行中 `num_trajectories=0` 仍
可能只是“本 update 没刚好结束 episode”的统计；真正的健康判据是 advantages、loss
和 grad norm 必须是有限数，且没有 `Skipping optimizer step`。

### E. 长跑中的 renderer worker 重启上限

第一次全任务长跑 `106403` 在 step 114 退出，但 PPO 数值并未失败。RoboCasa renderer
子进程大约每 13 updates 因 pipe EOF 退出，现有 venv 能重启并继续；默认
`RLINF_ROBOCASA_MAX_RESTARTS=8` 在第 9 次重启时抛出 `worker exceeded restart limit`。
重跑 `106535` 明确设为 `RLINF_ROBOCASA_MAX_RESTARTS=256`，以覆盖 1k-update 期间
预期的约 77 次可恢复 renderer restart。仍须监控是否出现非 renderer/PPO 错误；这不是
把数值训练异常忽略掉。

## 6. 配置语义：特别是“100 步”

`runner.max_steps=100` 表示 **100 次 PPO update iteration**，不是 100 个动作，
也不是 100 个 episode。每个 update 再收集
`env.train.max_steps_per_rollout_epoch` 个环境控制步骤（本次为 64），每个控制决策
由 XR-1 生成 action chunk。因此实际执行的低层 action 数还会乘以 chunk horizon。

主配置的关键默认值：

| 参数 | 当前主实验值/来源 |
| --- | --- |
| `task_soup` | `target50` |
| `total_num_envs` | 1 |
| `max_episode_steps` | 64 |
| `max_steps_per_rollout_epoch` | 64 |
| PPO epochs | flow smoke 覆盖为 1 |
| clip ratio | 0.2 |
| GAE | gamma 0.99，lambda 0.95 |
| actor precision | bf16 |
| FSDP | no_shard + orig params；训练时禁用 auto wrap |
| model ODE steps | 5 |

## 7. 已有 smoke 结果

| 作业 | 内容 | 结论 |
| --- | --- | --- |
| `106179` | flow 1 update，默认嵌套 FSDP | FAILED：FSDP `IDLE` 二阶 backward 问题 |
| `106182` / `106183` | 1/10 update，rollout 16 | 正常结束但无效：NaN advantage，optimizer skipped |
| `106184` | 1 update，TurnOnMicrowave，rollout 64 | 完成有效 backward；KL 0.102，clip frac 0.25，grad norm 1858，EV -29.006 |
| `106185` | 10 updates，TurnOnMicrowave，rollout 64 | 完成 10/10，最后 KL -0.039，clip frac 0，logprob abs diff `5.92e-4`，EV -0.953，return 0 |

smoke checkpoint：

```text
results/xr1_ppo_flow_turnonmicrowave_10u_r64_rootfsdp/
  xr1_robocasa365_ppo_full/checkpoints/global_step_10/
```

解释：logprob diff 在约 `1e-4` 到 `1e-3` 是当前 rollout/update 重算的一致性
诊断；KL 的单批估计可能轻微为负。EV 长期为负、reward 始终为零、KL 持续偏大或
clip fraction 持续过高，才是需要停止或调参的信号。

## 8. 训练命令与安全操作

### 单元测试

```bash
cd "$REPO"
PYTHONPATH=xr1:third_party/RLinf:. .conda-robocasa365/bin/python -m pytest -q \
  -p no:cacheprovider xr1/tests/test_rlinf_ppo_policy.py xr1/tests/test_rlinf_ppo_math.py
```

已验证输出为 `3 passed`。修改 likelihood、probe shape 或 PPO loss 前必须重跑。

### 复现主训练提交

```bash
cd "$REPO"
env \
  XR1_RLINF_CONFIG=xr1_robocasa365_ppo_flow_smoke \
  XR1_RLINF_MAX_STEPS=1000 \
  XR1_RLINF_SAVE_INTERVAL=250 \
  XR1_RLINF_TRAIN_ENVS=1 \
  XR1_RLINF_ROLLOUT_STEPS=64 \
  XR1_RLINF_EPISODE_STEPS=64 \
  XR1_RLINF_TASK_SAMPLING=random \
  XR1_RLINF_DISABLE_AUTO_WRAP=1 \
  XR1_RLINF_LR=1.0e-9 \
  RLINF_ROBOCASA_MAX_RESTARTS=256 \
  XR1_RLINF_LOG_DIR="$REPO/results/xr1_ppo_flow_target50_full_1000u_lr1e9_rootfsdp_r256" \
  sbatch --job-name=xr1-flow-target50-1k-r256 scripts/run_xr1_rlinf_ppo.sbatch
```

`scripts/run_xr1_rlinf_ppo.sbatch` 支持的常用覆盖包括：

- `XR1_RLINF_MAX_STEPS`、`XR1_RLINF_SAVE_INTERVAL`、`XR1_RLINF_RESUME_DIR`；
- task/rollout：`TRAIN_ENVS`、`EPISODE_STEPS`、`ROLLOUT_STEPS`、
  `TASK_FILTER_INCLUDE`、`TASK_SAMPLING`；
- optimization：`LR`、`MICRO_BATCH`、`GLOBAL_BATCH`、`UPDATE_EPOCH`、
  `CLIP_GRAD`、`NORMALIZE_ADVANTAGES`；
- model：`TRAINABLE_MODE`、`ACTION_STD`、`CLIP_ACTION`；
- flow/FSDP：配置名使用 flow config，并设置 `DISABLE_AUTO_WRAP=1`。

每个 checkpoint 当前约 **40 GiB**。因此 1000-update job 设为 250-step 保存间隔；
不要不加评估地将 `save_interval=10` 用于长跑（约 4 TiB）。

### 监控清单

在日志或 TensorBoard 追踪：

1. `actor/approx_kl`：不应持续急剧上升；单批小负值并不异常；
2. `actor/clip_fraction`：持续接近 1 表明更新太大，长期为 0 且 reward 无变化则
   可能学习太慢/信号太弱；
3. `actor/xr1_logprob_abs_diff` 与 max diff：应维持很小，突增优先检查 policy
   replay consistency；
4. `actor/grad_norm`、policy/total loss：必须有限，绝不能出现 skipped optimizer；
5. `critic/explained_variance`、value loss：稀疏奖励初期可能差，但长期必须观察是否
   改善；
6. `rewards`、`return`、task-level success：这是最关键的性能信号。

发生下面任一情况，应停止继续烧算力并保留日志/checkpoint 诊断：NaN/Inf、FSDP
错误、logprob diff 明显恶化、KL/clip 持续失控，或长段训练无任何非零回报。

## 9. 独立 RoboCasa365 评测

训练日志不是成功率。独立评测必须检查聚合结果中：

```text
completed_tasks == expected_tasks
episodes == expected_episodes
```

否则 success rate 无效。旧 target50 eval `105451` 只完成 2/50 tasks、10/250
episodes，不能作为模型分数；根因是 render path 的
`robosuite ... read_pixels -> Fatal Python error: Aborted`。

修复机制：`launch_robocasa365.sh` 默认 `WORKER_MAX_JOBS=1`，
`dynamic_eval.py worker --max-jobs` 每完成一个 episode 后退出，shell 为下一个 job
重启新的 Python/MuJoCo/EGL process。单纯 `env.close()` 不足以释放该进程级资源。

已验证的完整小评测：

| 作业 | checkpoint/task | 结果 |
| --- | --- | --- |
| `105593` | pretrain step100，OpenDrawer，5 fixed seeds | 5/5，100%，completed tasks/episodes 均完整 |

结果目录：
`eval_results/robocasa365-full_step100-procfresh-openDrawer-105593/`。

主长跑至少在 250/500/1000 checkpoint 各做一次小型固定任务评测，再对最优 checkpoint
运行完整 target50 × 5 episodes。示例（单 GPU、小评测）：

```bash
NUM_LANES=1 EVAL_TASKS=OpenDrawer NUM_TRIALS=5 \
  sbatch --gres=gpu:1 scripts/run_robocasa365_policy_eval_isolated.sbatch
```

实际评测时必须将脚本的 checkpoint/output 参数指向待评 checkpoint，并为每次评测使用
新的结果目录。不要覆盖 pretrain 或其他 checkpoint 的 aggregate。

## 10. 推荐的实验决策树

1. **现在**：让 `106535` 至少跑到 step 250，确认 checkpoint、有限指标和是否出现
   非零 reward；不要只看作业状态。
2. **step 250**：对一个固定小任务集合做隔离评测；若训练指标恶化或 success 不低于
   pretrain，则暂停 1k 并分析 LR、reward coverage 和 critic。
3. **step 500/1000**：同一评测协议复测；仅对有趋势的 checkpoint 跑全 target50。
4. **若全程 reward 接近零**：这是纯 PPO + 稀疏二元成功 reward 的探索上限证据，
   不应仅把步数继续放大。下一轮应改变 reward/探索或采用 action-expert-only 的
   对照，而不是把“PPO 无效”归因给 flow 代码。
5. **若 KL/clip 不稳**：先降 actor LR、检查 logprob consistency、保持 trace probes
   固定；不要恢复 Gaussian surrogate 来掩盖目标分布不一致。
6. **若有成功趋势但 full fine-tune 不稳**：做同预算 `trainable_mode=action_expert`
   对照，分开回答“更稳定”与“更高成功率”。

## 11. 新对话最小上下文提示词

可以把下面内容直接给新对话：

```text
请先阅读 XR1_RLINF_PPO_HANDOFF.md，再检查 job 106535 的 squeue/sacct、
logs/xr1_rlinf_ppo_106535.log 和 results/xr1_ppo_flow_target50_full_1000u_lr1e9_rootfsdp_r256。
这是 XR-1 RoboCasa365 的 RLinf PPO：当前用 flow_hutchinson likelihood、full model、
XR1_RLINF_DISABLE_AUTO_WRAP=1。不要把作业 COMPLETED 当成功；报告 KL、clip fraction、
logprob diff、grad finite、critic EV、return，并以完整隔离评测的 aggregate 判定 success。
不要改用 Gaussian surrogate，除非明确作为独立 ablation。
```
