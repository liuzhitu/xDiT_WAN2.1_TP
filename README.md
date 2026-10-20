# xDiT_WAN2.1_TP

基于 [xDiT](https://github.com/xdit-project/xDiT) 的 **Wan2.1 视频生成模型张量并行与混合并行适配**。为 Wan2.1 T2V 的 FFN、SelfAttention 和文本 CrossAttention 实现 TP，复用 xDiT 原生加载、进程组与推理流程，并适配 Ring、Ulysses、DP 组合。

**已在 4×RTX 4090 上完成 Wan2.1-T2V-1.3B 的 TP4 性能测量：DiT 加速 1.52×，生成流水线加速 1.51×；TP2×Ulysses2、TP2×Ring2、TP2×DP2 三项端到端兼容性检查通过。**

完整实现位于 [`codex/wan21-tp`](https://github.com/liuzhitu/xDiT_WAN2.1_TP/tree/codex/wan21-tp) 分支。默认分支 `main` 用于展示项目，运行时请检出实现分支。1.3B / 14B 共用分片实现，实测结果来自 **1.3B**。

## 性能结果

2026-10-19，Wan2.1-T2V-1.3B，832×480、81 帧、50 steps。单卡与 TP4 各预热一次（17 帧、50 steps），再正式生成一次；两边均 GPU 驻留，关闭 CPU offload、量化、compile 和缓存。

| 指标 | 原生 xDiT 单卡 | 本项目 TP4 | 加速比 |
| --- | ---: | ---: | ---: |
| DiT CUDA 时间合计 | 231.977 s | 152.387 s | **1.52×** |
| Denoising CUDA 时间 | 232.441 s | 152.926 s | **1.52×** |
| 生成流水线 wall time | 237.663 s | 157.776 s | **1.51×** |
| 每卡峰值 allocated | 17.935 GiB | 16.059 GiB | — |

TP4 时间及显存取各 rank 的最大值。生成流水线计时不含模型加载、MP4 编码和落盘；GPU 输出快照的采集开销包含在内。每种配置只有一个正式样本，结果代表本次观测；每卡峰值 allocated 降低约 10.46%，不代表四卡总显存减少，也不代表加载峰值降低。

实验使用 UniPC、seed=42、guidance=5、flow_shift=3，Self/CrossAttention 均为 `FLASH`；Transformer 参数主要为 BF16、部分为 FP32，实测 DiT 输入为 BF16。四卡拓扑同 NUMA 内为 `NODE`、跨 NUMA 为 `SYS`。完整环境、来源与数值指标见 [实测记录](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/docs/wan_tp/results.md)。

**数值对照单独记录：** 采集的首步 DiT 输出和最终 latent 在 TP ranks 间零误差一致；相对单卡的对照状态为 `metrics_only`，未设定验收容差。最终 latent 的 relative L2 为 0.258719，不能将本轮性能结果表述为单卡数值对齐已通过。

## 混合并行验证

以下三项均在四张 RTX 4090 上完成原生 runner 的加载、预热、生成、解码、输出检查、保存和清理。每项一次预热、一次正式运行，输入为 **832×480、17 帧、10 steps、无视频 token padding**。

| 配置 | 正式 pipeline 时间（最慢 rank） | 正式视频数 | 结果 |
| --- | ---: | ---: | --- |
| TP2 × Ulysses2 | 5.160 s | 1 | passed |
| TP2 × Ring2 | 5.060 s | 1 | passed |
| TP2 × DP2 | 7.540 s | 2 | passed |

全部 12 条 rank 记录为 `passed`，视频形状均为 `[17,480,832,3]`、无 NaN/Inf；DP 两个副本分别处理不同 prompt，最终收集两段视频。这里验证的是端到端兼容性，未做单卡数值对照；不同输出数量的耗时不直接比较。该短流程结果与上方 81 帧 TP4 性能实验分别记录。

## 实现内容

| 模块 | 本项目新增能力 |
| --- | --- |
| 原生 runner 接入 | 1.3B / 14B 共用加载完成后的分片步骤，复用原生 TP 组与推理入口 |
| FFN TP | 第一层 Linear 按输出通道切分，第二层按输入通道切分，保留原生 GELU / Dropout |
| SelfAttention TP | Q/K/V 按 heads 对应通道分片，输出投影归约，沿用原生 RoPE / USP |
| 文本 CrossAttention TP | 分片 Q/K/V 与输出投影，保留完整文本序列及原生文本条件语义 |
| 全维 Q/K RMSNorm | 跨 TP ranks 汇总 FP32 平方和，按完整投影维度归一化，affine weight 随通道切分 |
| TP × SP | 从原生 SP 成员建立 Ring / Ulysses 子组，适配混合 USP 的成员布局及子组销毁 |
| TP × DP | 保持副本独立，仅由代表 rank 提交视频，避免 TP 复制输出 |
| 视频 padding K/V | 按有效长度裁剪计算侧 K/V，维持等长通信；完全无效的 Ring 块明确报错 |

embedding、普通归一化、调制和最终输出 head 仍在 TP ranks 上复制。

## 接入与计算流程

```text
xDiT 原生加载 Pipeline → 初始化 runtime / 进程组 → 原生参数放置
  → Wan2.1 T2V 加载完成步骤：分片 FFN、attn1、attn2
  → 原生 options 配置 → runner 推理 → scheduler → VAE 解码与输出收集
```

调用位置为 `xFuserWan21T2VModel._post_load_and_state_initialization()`。维度、heads、层数和 dtype 从实际模型读取；`TP=1` 保留原生路径。加载阶段仍持有完整 Transformer，分片使用独立连续存储释放原权重引用。

FFN 的数据流为 `[...,D] → Column [...,M/P] → GELU/Dropout → Row 部分结果 [...,D] → TP SUM AllReduce`。Column 切权重的输出轴和 bias；Row 切输入轴，完整 bias 仅由组内 rank 0 在本地 Linear 中贡献一次，再归约。

Attention 在本地 heads 上计算。Q/K RMSNorm 先跨 TP 汇总统计，再执行原生 attention；SelfAttention 沿用 RoPE 和视频 USP，CrossAttention 保持完整文本 K/V，不执行视频序列交换。每种 Attention 有两次统计 AllReduce 和一次输出 AllReduce，加上 FFN，一次 block forward 共 **7 次 TP AllReduce**。

TP 切通道，SP 切视频序列；沿用原生 `tp-sp-pp-cfg-dp` rank 排列。同一 TP 组内 SP 坐标相同，SelfAttention heads 需整除 `TP×Ulysses`，CrossAttention heads 只需整除 TP。输出经原生 Head、SP gather/unpad 和 VAE，收集时由 `SP=0、TP=0、CFG=0` 的代表提供真实视频。

## 使用与文档

```bash
git clone --branch codex/wan21-tp https://github.com/liuzhitu/xDiT_WAN2.1_TP.git
cd xDiT_WAN2.1_TP
git rev-parse HEAD
```

使用具备 Wan / Diffusers、FlashAttention 和分布式运行依赖的 Linux GPU 环境，以及 Diffusers 格式的 Wan2.1 T2V 权重。通过 xDiT 原生 `xdit` / `xfuser.runner` 入口选择模型，并设置 `tensor_parallel_degree`、`ulysses_degree`、`ring_degree` 和 `data_parallel_degree`；各并行维度乘积应匹配参与推理的进程数。本仓库不包含模型权重。

- [实测结果与证据边界](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/docs/wan_tp/results.md)：TP4 性能、三组合运行记录与数值误差。
- [设计与接口契约](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/docs/wan_tp/README.md)：分片语义、组成员、padding 和支持限制。
- [单卡 / TP 对照协议](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/docs/wan_tp/benchmark.md)：早期采集协议；本轮实际配置与结果以实测记录为准。

## 支持边界

- 新增 TP 面向 Wan2.1 T2V 1.3B / 14B 推理；14B 共用实现但尚未实测。I2V、VACE、Wan2.2 和训练不在新增支持范围。
- 已实测 TP4，以及 TP2×Ulysses2 / Ring2 / DP2 的上述 workload。混合 USP、视频 padding、共享 ranks 的并行 VAE、进程内重复初始化等仍需独立验证。
- 当前使用独立 Q/K/V 投影；融合投影和 Transformer GEMM 量化布局未适配。TP 与 FSDP、Wan T2V 的 CFG 并行 / PipeFusion 遵循原生限制。普通 CFG guidance 可在本次实验中使用。
- 独立 VAE ranks 的加载与调度未适配；offload、compile、cache、稀疏后端不由本次驻留模式结果覆盖。
- TP4 加速比相对同版本原生 xDiT 单卡，不是相对原生 Ulysses4 / Ring4 的性能结论。

## 推荐源码阅读

以下链接指向实现分支：

| 文件 | 阅读目的 |
| --- | --- |
| [runner_models/wan.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/models/runner_models/wan.py) | 能力声明与加载完成后的 TP 调用位置 |
| [transformers/wan_tp.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/models/transformers/wan_tp.py) | 参数切分、Row bias、全维 RMSNorm 与本地 heads |
| [distributed/parallel_state.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/core/distributed/parallel_state.py) | TP / SP 组成员及子组生命周期 |
| [transformers/transformer_wan.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/models/transformers/transformer_wan.py) | 序列、RoPE、有效视频长度与 gather/unpad |
| [layers/usp.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/layers/usp.py) | Ulysses / Ring 计算侧 K/V 裁剪 |

## 上游与参考

基于 xDiT commit `97a4425c68071e1fdf04d40f468015d30be60f41`。xDiT 的通用框架、加载器和原生并行设施来自上游；本 fork 新增上述 Wan TP 与组合适配。视频 padding 处理参考 [xDiT PR #912](https://github.com/xdit-project/xDiT/pull/912) 的有效长度传递与逐块裁剪思路。沿用仓库已有许可证，见 [LICENSE.txt](LICENSE.txt)。
