# Wan2.1 TP 云端实测记录

记录更新：2026-10-20。实验日期：2026-10-19。状态：**1.3B TP4 性能采集完成，三项混合并行端到端兼容性通过；单卡数值对照为 metrics_only。** 本次文档整理未重新运行推理或测试。

## 结果来源与版本

结果来自用户保存的原始 `report.json`（实验目录名 `wan_tp4_4090_20261019_090634`），以及三份 `summary.json`（实验目录名 `wan_tp_compat_20261019_092155`）。本记录摘录其中的配置、指标和状态，不发布 GPU UUID、服务器连接信息或本地绝对路径。

- 单卡基线：上游 xDiT `97a4425c68071e1fdf04d40f468015d30be60f41`。
- TP 源码：`f0d6074c653f304152b6fba5b9c885123db7fb36`；报告同时记录了未跟踪的 benchmark / 文档 / Docker 文件。因此该 SHA 标识生产源码基点，不是完整实验目录的可复现快照。
- 模型：Wan2.1-T2V-1.3B，Diffusers 格式；声明的 revision 为 `0fad780a534b6463e45facd96134c9f345acfa5b`。这是运行时提供的来源标签，没有对权重内容做哈希核验。
- 硬件：4×RTX 4090，每卡 24564 MiB；驱动 595.71.05。同 NUMA 的 GPU0/1 与 GPU2/3 为 NODE，跨 NUMA 为 SYS，报告未显示 NVLink。
- 依赖：PyTorch `2.8.0+cu128`、Diffusers `0.35.2`、Transformers `4.57.1`、Accelerate `1.4.0`、YunChang `0.6.4`、DistVAE `0.1.0`、FlashAttention `2.8.3.post1`、huggingface-hub `0.36.2`。

## TP4 性能

两边固定 832×480、81 帧、50 steps；各一次 17 帧 / 50 steps 预热，再一次正式生成。UniPCMultistepScheduler、seed=42、guidance=5、flow_shift=3，Self/CrossAttention 均为 FLASH。SP、DP、CFG parallel degree 均为 1，不启用并行 VAE、offload、compile、量化或缓存。

Prompt：`A red fox walks through a snowy forest, cinematic camera movement.`；负提示词：`blurry, low quality, static, subtitles`。

Transformer 参数以 BF16 为主，另有 FP32 参数；单卡参数字节为 BF16 2,831,739,008 / FP32 12,509,184，TP 每 rank 为 BF16 742,667,648 / FP32 12,509,184。文本编码器参数为 BF16，观测到的 DiT 输入为 BF16；benchmark 没有覆盖原生 autocast 设置。

| 指标 | 单卡 | TP4（最大 rank） | 加速比 | 四卡并行效率 |
| --- | ---: | ---: | ---: | ---: |
| pipeline wall time | 237.663077 s | 157.776254 s | 1.5063× | 37.66% |
| pipeline 原生 CUDA 区间 | 237.497813 s | 158.014578 s | 1.5030× | 37.58% |
| DiT CUDA 时间合计 | 231.977367 s | 152.386605 s | 1.5223× | 38.06% |
| denoising CUDA 区间 | 232.441469 s | 152.926172 s | 1.5200× | 38.00% |
| 峰值 allocated | 17.935 GiB | 16.059 GiB / rank | — | — |
| 峰值 reserved | 20.268 GiB | 19.568 GiB / rank | — | — |

加速比为单卡时间 / TP4 时间，并行效率为加速比 / 4。每卡 allocated 峰值降低 10.46%；各卡峰值不一定同时发生，不能把此结果解释为总显存或加载峰值降低。

生成计时排除加载、MP4 编码及落盘；GPU 快照复制和保留内存包含在测量中，CPU 保存及误差比较不计入生成时间。Denoising 从第一次 DiT 开始计到最后一次 scheduler 更新，不含前置准备。每种配置正式生成 n=1，不能由此推断稳定尾延迟或任意卡数扩展性。该性能结果不比较原生 SP4。

## 数值对照

| 采集对象 | 相对单卡 relative L2 | 相对单卡 max abs | TP ranks 间 max abs |
| --- | ---: | ---: | ---: |
| 首步 DiT 输出 1 | 0.00539320 | 0.07983398 | 0 |
| 首步 DiT 输出 2 | 0.00537181 | 0.07812500 | 0 |
| 最终 latent | 0.25871900 | 3.69372368 | 0 |

采集张量均为有限值，TP ranks 间按零容差检查一致。对单卡未指定 atol / rtol，报告 `status` 和 `acceptance` 均为 `metrics_only`，`allclose` 为 null。最终 latent 与单卡存在明显差异；本记录没有将其认定为验收通过，也没有仅凭现有指标判定误差成因。

## 混合并行兼容性

三个组合均使用上述 1.3B 模型、四张 RTX 4090、FLASH 后端、GPU 驻留模式。每项一次预热和一次正式运行，两轮均为 832×480、17 帧、10 steps，seed=42、guidance=5、flow_shift=3。视频 token 数 7800，无 SP padding。

| case | TP / Ulysses / Ring / DP | 最慢 rank 正式 pipeline 时间 | 正式视频数 | 汇总状态 |
| --- | --- | ---: | ---: | --- |
| tp2_ulysses2 | 2 / 2 / 1 / 1 | 5.159637 s | 1 | passed |
| tp2_ring2 | 2 / 1 / 2 / 1 | 5.059875 s | 1 | passed |
| tp2_dp2 | 2 / 1 / 1 / 2 | 7.540447 s | 2 | passed |

三份汇总及全部 12 条 rank 记录为 passed，failures 均为空。输出视频为 `[17,480,832,3]`、float32、有限值。DP 第二条 prompt 为 `A small boat floats on a calm blue lake, cinematic camera movement.`；两副本处理不同请求，收集两段视频，无重复或缺失的 prompt 元数据。

本轮覆盖原生 runner 的生成、解码、输出分发/收集与正常退出。该协议不采集单卡 latent 对照，不证明像素或语义等价；DP 的两视频耗时不与其他组合的单视频耗时直接相除。81 帧 / 50 steps 的混合并行数值和性能对照不在这三份短流程报告中。

## 尚未覆盖

14B、混合 USP、视频 padding、共享 ranks 的并行 VAE、独立 VAE ranks、进程内销毁后重新初始化，以及其他 offload / compile / cache / 稀疏后端配置均不由本记录证明。功能设计与不变量见 [设计记录](README.md)。
