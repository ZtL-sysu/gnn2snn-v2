# 对 Claude 文献树 v1 的逐行核查

- 核对对象：[docs/literature-tree/literature-tree-v1.md](../docs/literature-tree/literature-tree-v1.md)（未修改原文件）
- 核对人：Gemini，2026-11-16
- 判定只用四个词：**正确**、**错误**、**不准确**（大体对但有偏差）、**无法核实**。证据来自本目录 `01`–`05`、[questions.md](questions.md)，检索依据见 [search-log.md](search-log.md)。
- “草稿位置”写的是草稿中的小节和行号（按 v1 文件行号）。

## 先说结论：最需要改的五处

1. **“没有图网络免训练转换”不成立。** One-Spike SNN（IEEE TETC 2025）第 IV-C 节就是对预训练 Text GCN 的免训练转换；PMSM（2025）转换了 3 层 GCN（需重训量化源网络）。草稿漏掉了这两篇。视觉图（ViG、DGCNN）的转换确实没找到。
2. **Spiking PointCNN 不是转换。** 它是换成脉冲层后用替代梯度从头训练的；真正的点云转换先例是 Lan 等（ICCV 2023），而且它必须把 MaxPool 改成 AvgPool 重训源网络——这恰好说明 max 聚合是转换的难点。
3. **“max 聚合本身不再算空白”是错的。** 草稿说 `arXiv:2409.03368` 把 max 池化“改成神经元层”，原文其实是把 MaxPool 挪到脉冲神经元之前（层序交换，依赖 ReLU 与 $\max z>0$）；其余几种脉冲 max 池化都假设非负、率编码、长时间步。ViG 的 $\max_j(\tilde x_j-\tilde x_i)$ 作用于有符号差值，这些办法都不能直接用。
4. **“图上多跳扩散未解”说过头了。** AEGNN、DAGr、InkStream、Ripple / RIPPLE++、NeutronRT 都专门研究并缓解了多跳扩散（多为精确增量）。真正没找到的是：在视觉图上**逐层阈值化、带截断误差缓冲**的近似增量推理。
5. **几处元数据要改。** AEGNN 的“最多降 200 倍”在 arXiv 更新版里改成了“11 倍”；EvGNN 正式发表在 IEEE TCASAI 2025；ETHEREAL 论文已在 arXiv 公开（25.6 µs/事件），但两份 PDF 都没有写 ESSERC 录用信息；GreedyViG 发表处是 CVPR 2024。

## 一、长远目标与路线图（第 20–27 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 路线图 M0（第 22 行） | 固定图·线性聚合：超像素图上的 GCN、GatedGCN，低时间步保真已做到 | 无法核实 | “已做到”指项目自己的实验结果，本仓库里没有可复查的数据。文献侧可佐证的是：One-Spike SNN 免训练转换了文本图 GCN（Table V，平均掉 0.90%），PMSM 在 T=1 无损转换了触觉图 GCN（需重训）。另外 GatedGCN 的边门控是 sigmoid 非线性，严格说不属于“线性聚合”。 |
| 路线图 M1（第 23 行） | 动态拓扑与 max 聚合：ViG、DGCNN 等逐层选邻居 ／ 未见转换工作 ／ 空白，值得做 | 正确 | SL-10～SL-18、SL-45：ViG 与 DGCNN 的转换均未找到；ViG 552 篇被引中也没有。 |
| 路线图 M2（第 24 行） | 非线性聚合事件化：门控、注意力不再用模拟计算 ／ Transformer 上已有解法 | 不准确 | Transformer 转换并没有做到“不用模拟计算”：SpikeZIP-TF 的 Softmax / LayerNorm 是对累积输入直接做浮点计算再取差（式 5），ECMT 与 DCGS 用浮点期望补偿模块，STA 免浮点但要 T=64～256。图注意力在可变邻域上做 Softmax、GatedGCN 的 sigmoid 门也都没有现成事件化方案。 |
| 路线图 M3（第 25 行） | 流式图：骨架、事件相机、点云序列 ／ 图上的多跳扩散未解 | 不准确 | 多跳扩散已被大量研究：AEGNN 第 4.2 节（k 跳子图）、DAGr（有向图 + max 池化截断，减少 91% 计算）、InkStream（max 选择性，Fig. 3）、RIPPLE++（受影响顶点 0.7%→37%）、NeutronRT。没找到的是视觉图上的**逐层阈值化近似增量**。见 [04-event-driven-inference.md](04-event-driven-inference.md)。 |
| 路线图 M4（第 26 行） | 公平能耗评估：与同比特量化网络成对比较 ／ 图网络上未见分析 | 正确 | Yan 等（IEEE TCAD 2026，`arXiv:2409.08290v6`）只在 CNN、ViT、GPT-2 上评测；未找到图网络上的同类分析（SL-68）。 |
| 路线图 M5（第 27 行） | 硬件实测：事件驱动图网络芯片已出现 ／ 跑的还不是脉冲网络 ／ 较远 | 正确 | ETHEREAL、EvGNN、EFGCN、AMPLE 跑的都是 4/8 位整数量化 GNN（SL-71）。 |

## 二、时间线一：ANN→SNN 转换（第 35–45 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 时间线一 2026（第 37 行） | Wang 等 AAAI 2026：免训练转换高性能脉冲 Transformer ／ 仍是 Transformer | 正确 | 标题 *Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformers*（AAAI 2026）。补充：其 MBE 神经元在 ViT 上 T=16 时 85.31%，T=10 时跌到 21.99%（Table 4），并非超低时间步。 |
| 时间线一 2025（第 38 行） | DCGS ICML 2025：差分编码加多阈值神经元，免训练、低时间步 ／ 未涉及图结构 | 正确 | 论文标题 *Differential Coding for Training-Free ANN-to-SNN Conversion*。需补充两点风险：官方代码对 kernel_size≠stride 的 MaxPool 直接退回浮点（`neurons/nonlinear_neuron.py` 第 44–45 行），仓库没有许可证（见 Q5）。 |
| 时间线一 2024（第 39 行） | SpikeZIP-TF ICML 2024：与激活量化 Transformer 精确等价，含脉冲化的 Softmax 和 LayerNorm ／ 需要先量化训练 | 不准确 | “需要先量化训练”对，但还需先做 50–100 轮 GELU→ReLU 微调（Appendix A.4）；所谓脉冲 Softmax / LayerNorm 实际是对累积输入调用浮点函数后输出相邻步差值（式 5），不是纯脉冲实现；推理需 T=32～128。 |
| 时间线一 2024（第 40 行） | STA ICLR 2024：免训练转换 Transformer，处理自注意力和归一化 ／ 近似而非等价 | 正确 | OpenReview `135eKq323G`。补充：ImageNet 上需 T=64～256。 |
| 时间线一 2023（第 41 行） | 校准偏移脉冲 ICLR 2023：在 QCFS 框架上进一步消除误差 ／ CNN | 正确 | `arXiv:2302.10685`。补充：每张测试图要先跑 ρ 步再重置跑 T 步，标称 T=2、ρ=4 实际耗 6 步（Appendix Algorithm 1）。 |
| 时间线一 2022（第 42 行） | QCFS ICLR 2022：量化激活训练源网络，超低时间步 ／ 需要重训源网络 | 正确 | `arXiv:2303.04347` 为 ICLR 2022 论文的 arXiv 版。补充：训练前须把所有 MaxPool 换成 AvgPool（Appendix A.1）。 |
| 时间线一 2021（第 43 行） | SNN Calibration（Li 等）、Deng & Gu：逐层校准发放率，分析转换误差 ／ 未核对原文 | 正确 | 已核对全文：Li 等 *A Free Lunch From ANN*（ICML 2021，`arXiv:2106.06984`）；Deng & Gu *Optimal Conversion of Conventional ANNs to SNNs*（ICLR 2021，`arXiv:2103.00476`）。补充：两者实验里含 MaxPool 的网络都先改成 AvgPool。 |
| 时间线一 2017–2020（第 44 行） | Rueckauer 等、Sengupta 等、Han 等：权重归一化、阈值平衡、减法复位 ／ 需要几百个时间步；未核对原文 | 不准确 | 已核对全文。方法概括对；时间步低估了：Rueckauer 等（Front. Neurosci. 2017）T=300～550，Sengupta 等（Front. Neurosci. 2019）ImageNet 上 T=2500，Han 等 RMP-SNN（CVPR 2020）T=512～4096。另外 Sengupta 与 Han 都要求源网络去掉 BN 与偏置。 |
| 时间线一 2015（第 45 行） | Cao 等、Diehl 等：发放率对应 ReLU 输出 ／ 浅层网络；未核对原文 | 正确 | 已核对：Cao 等 IJCV 2015（113(1):54–66）；Diehl 等 IJCNN 2015。两者都要求无偏置、平均池化。 |

## 三、时间线二：脉冲图神经网络（第 53–60 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 时间线二 2025（第 55 行） | SGNNBench：大规模图上脉冲图网络的统一评测 ／ 评测的是直接训练的模型；未核对原文 | 正确 | 已核对：NeurIPS 2025 Datasets and Benchmarks（`arXiv:2509.21342`），9 种直接训练 SGNN、18 个数据集（Table 1）。 |
| 时间线二 2024（第 56 行） | MK-SGN、Signal-SGN：骨架动作识别的脉冲图卷积网络 ／ 视觉任务，但需专门训练 | 正确 | 两者都是直接训练；MK-SGN 另用 CTR-GCN 教师做知识蒸馏（见 Q8）。正式发表处：MK-SGN 为 Neurocomputing 2026，Signal-SGN 为 ACM MM 2025。 |
| 时间线二 2024（第 57 行） | SpikeGCL ICLR 2024：脉冲化的图对比学习，1 比特表示 ／ 节点表示学习 | 正确 | `arXiv:2305.19306`，ICLR 2024。 |
| 时间线二 2024（第 58 行） | Dy-SIGN AAAI 2024：动态图节点分类 ／ 动态图，非视觉 | 正确 | AAAI 2024，38(15)。 |
| 时间线二 2023（第 59 行） | SpikeNet AAAI 2023：用脉冲网络代替 RNN 建模时序图 ／ 时序图，非视觉 | 正确 | AAAI 2023，37(7):8588–8596。 |
| 时间线二 2022（第 60 行） | SpikingGCN IJCAI 2022：端到端脉冲图卷积，明确不用预训练和转换；扩展到图像图分类 ／ 最接近的早期工作，路线相反 | 正确 | IJCAI 2022，pp. 2434–2440；第 3.3 节在 MNIST 像素网格图和 75 节点超像素图上做了分类。 |

## 四、时间线三：视觉图骨干与事件驱动推理（第 75–88 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 时间线三 2026（第 77 行） | ETHEREAL，ESSERC 2026：事件驱动图网络处理器，每事件 25 微秒 ／ 仅见作者发布的消息，未核对论文 | 不准确 | 论文已公开：`arXiv:2608.17787`（14 页）与 `arXiv:2609.15241`（5 页），已读全文。每事件延迟 25.6 µs；能耗长文版写 1.6 µJ、短文版写 1.7 µJ。两份 PDF 都没有写 ESSERC 录用信息，会议归属无法确认。芯片跑的是 4/8 位量化 DAGr-GNN，需量化感知训练。 |
| 时间线三 2024（第 78 行） | EvGNN：事件相机图网络的 FPGA 加速器 ／ 硬件端的评测参照 | 不准确 | 内容对；正式发表在 IEEE TCASAI 2025（DOI 10.1109/TCASAI.2024.3520905），2024 是预印本年份。N-CARS 87.8%、每事件 16 µs（摘要）。 |
| 时间线三 2024（第 79 行） | DAGr，Nature 2024：图像 CNN 加异步图网络处理事件，做低延迟检测 ／ 流式图的强教师和应用场景 | 正确 | Nature 629:1034–1040。代码为 GPL-3.0，见脉络三。 |
| 时间线三 2024（第 80 行） | GreedyViG，CVPR 2024：用轴向建图替代 kNN，ImageNet 81.1%–83.9% ／ 建图更便宜的教师 | 正确 | CVPR 2024；S/M/B 为 81.1% / 82.9% / 83.9%（Table 1）。补充：它是动态轴向建图（在轴向候选上按距离筛选），论文写 L2 距离而代码用 L1；许可证为仅限非商业研究（见脉络三）。 |
| 时间线三 2023（第 81 行） | MobileViG：静态建图，图不随输入变化 ／ 固定拓扑的教师，最易转换；未核对原文 | 正确 | 已核对原文与代码：SVGA 固定轴向图，聚合是逐元素 max（见 Q7）。CVPRW 2023。 |
| 时间线三 2022（第 82 行） | ViG，NeurIPS 2022：图像块当节点、按特征相似度连边 ／ 动态拓扑加 max 聚合的代表教师 | 正确 | 见 Q6：每层重算扩张 kNN、max-relative 聚合。 |
| 时间线三 2022（第 83 行） | AEGNN，CVPR 2022：新事件到来只重算受影响节点，计算量最多降 200 倍 ／ 流式图增量推理的基线 | 不准确 | CVPR 原版摘要确为“最多降 200 倍”（CVF 页面，SL-65），但 arXiv 更新版封面说明更新了第 5 节的计算量计算，摘要改为“比最好的异步方法低 11 倍”、延迟比普通 GNN 低 8 倍。两个版本数字冲突，引用时应注明版本。 |
| 时间线三 2022（第 84 行） | DeltaCNN，CVPR 2022：视频上只传帧间差分的稀疏 CNN 推理 ／ 差分推理在 CNN 上的对应物 | 正确 | 不需重训，最高 7 倍加速；逐层阈值 + 截断缓冲（第 3.3 节）。 |
| 时间线三 2021（第 85 行） | CTR-GCN，ICCV 2021：骨架识别的强基线 ／ 流式图教师 | 正确 | ICCV 2021；许可证 CC BY-NC 4.0（见脉络三）。 |
| 时间线三 2019（第 86 行） | DGCNN（Wang 等）：点云上的动态图卷积 EdgeConv ／ 动态拓扑教师；未核对原文 | 正确 | 已核对：ACM TOG 2019；ModelNet40 上 k=20 时 92.9%（Table 2）。 |
| 时间线三 2018（第 87 行） | ST-GCN，AAAI 2018：骨架序列的时空图卷积 ／ 固定拓扑的流式图教师 | 正确 | AAAI 2018；NTU-60 X-Sub / X-View 81.5% / 88.3%（Table 3）。 |
| 时间线三 2017（第 88 行） | Sigma-Delta 网络，ICLR 2017：层间只传激活变化的离散化，可转换预训练网络 ／ 差分推理的思想源头 | 正确 | OpenReview `Hy1b8W9eg`，ICLR 2017 Poster；第 4 节明确针对预训练网络，只优化层间尺度。 |

## 五、难点—解法树（第 94–103 行），重点核查“在图上的空白”一列

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 难点树 第 1 行（第 96 行） | 低时间步下误差大 ／ QCFS、SNN Calibration、DCGS ／ 无：本项目已验证可直接用 | 无法核实 | 解法与出处列正确；“本项目已验证”无仓库内证据可查。另需注意：QCFS 要重训源网络，DCGS 代码对重叠 MaxPool 退回浮点，二者都不能“直接用”在含 max 聚合的视觉图上。 |
| 难点树 第 2 行（第 97 行） | 负值与残差 ／ 带符号脉冲、残差路径单独处理 ／ DCGS 等 ／ 小 | 正确 | DCGS 的正负对称阈值、PMSM 的极性多脉冲都处理负值。补充：MRConv 里对有符号差值取 max 是负值与 max 的叠加问题，应归到 max 聚合那一行，不能算“小”。 |
| 难点树 第 3 行（第 98 行） | 注意力、归一化等非线性 ／ 脉冲化算子或时空近似 ／ SpikeZIP-TF、STA ／ 小：门控和图注意力可借用 | 不准确 | SpikeZIP-TF 的 Softmax / LayerNorm 仍是浮点计算（式 5），STA 需 T=64～256；可变大小邻域上的图注意力 Softmax、GatedGCN 的 sigmoid 门没有现成事件化方案，空白不算“小”。 |
| 难点树 第 4 行（第 99 行） | 动态拓扑：逐层按特征选邻居 ／ 未见 ／ 大：选邻居是离散决策，误差可能使其翻转 | 正确 | 转换工作未找到（Q2）；kNN 翻转的分析工具见 Q9（距离间隔 > 4ε 时不翻转；RaBitQ、ScaNN 的量化误差结论）。 |
| 难点树 第 5 行（第 100 行） | max 聚合 ／ 脉冲化 max 池化、侧抑制池化、改成神经元层 ／ 小到中：池化本身已有解法 | 不准确 | “改成神经元层”是误读：Bu 等（CVPR 2025）是把 MaxPool 挪到神经元之前（Theorem 2）。其余方法要求非负、率编码、T=16～550（Q10）。MRConv 对有符号差值取 max，且没有后接 ReLU，这些方法都不能直接套用；超低时间步下逐步取 max 还有 $\frac{k-1}{k+1}\epsilon$ 级的高估偏置。空白至少是“中”。 |
| 难点树 第 6 行（第 101 行） | 流式输入的时间冗余 ／ 帧间差分、局部重算 ／ Sigma-Delta、DeltaCNN、AEGNN ／ 中：变化随层数扩散到多跳邻域，未见阈值化处理 | 不准确 | 多跳扩散已有专门研究与缓解（AEGNN、DAGr、InkStream、RIPPLE++、NeutronRT，均为精确方法）；图上的阈值化也不是完全没有：InstantGNN（KDD 2022）在线性传播中用残差阈值 ε，Reforge（IPDPS 2026）按预算选择性重算。没找到的是多层非线性消息传递中逐层激活阈值 + 截断缓冲的做法。 |
| 难点树 第 7 行（第 102 行） | 能耗比较是否公平 ／ 与等价量化网络成对比较 ／ Yan 等 ／ 中：未见针对图网络的分析 | 正确 | Yan 等正式发表于 IEEE TCAD 2026；图上未见同类分析。还应纳入 Davidson & Furber、Lemaire 等、Dampfhoffer 等、Ottati 等的盈亏分析与 TopGQ 量化基线（脉络五）。 |
| 难点树 第 8 行（第 103 行） | 硬件上的真实收益 ／ 事件驱动图网络加速器 ／ EvGNN、ETHEREAL ／ 中：这些芯片跑的不是脉冲网络 | 正确 | 两者都跑量化 GNN。补充：支持分级脉冲的 Loihi 2、SENECA、SpiNNaker 2 可作为差分分级脉冲的硬件落点（脉络五）。 |

## 六、按四种情况判断选题（第 109–116 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 四种情况 第 1 行（第 111 行） | 固定图、线性聚合的低时间步转换 ／ 第二种：换题 ／ CNN 和 Transformer 上已有好解法 | 正确 | 归类对，而且理由比草稿写的更强：图上本身已有先例（One-Spike SNN 的 Text GCN 免训练转换、PMSM 的 GCN 转换），草稿没有提到。 |
| 四种情况 第 2 行（第 112 行） | 门控、注意力聚合的完全事件化 ／ 第二种：换题 ／ 脉冲 Transformer 转换已在解同样的算子 | 不准确 | Transformer 转换仍依赖浮点补偿或很长的时间步（见 M2 行），且图上的边门控与可变邻域注意力和 Transformer 的定长 Softmax 不同。说“同样的问题已有好解法”根据不足，更接近第三种。 |
| 四种情况 第 3 行（第 113 行） | 流式图上的阈值化差分推理 ／ 第三种 ／ 视频 CNN 和事件图上各有一种解法，图上的多跳扩散未解 | 不准确 | 归为第三种可以接受；但理由要改：多跳扩散已被精确增量方法研究过（InkStream、RIPPLE++、NeutronRT、DAGr），视频侧也不止一种解法（CBinfer、Skip-Conv、DeltaCNN、EvNets、Eventful Transformers）。空白应表述为“视觉图上逐层阈值化 + 误差缓冲的近似增量，以及阈值对 kNN 与 max 胜者的影响”。 |
| 四种情况 第 4 行（第 114 行） | 图网络的公平能耗对照 ／ 第三种 ／ 分析框架已有，图上未见 | 正确 | 分析框架见 Yan 等、Davidson & Furber 等；图上未见（SL-68）。 |
| 四种情况 第 5 行（第 115 行） | 动态拓扑与 max 聚合的转换 ／ 第三或第四种 ／ 选邻居未见解法，max 部分可借池化的解法 | 不准确 | 归类合理，但“max 部分可借池化的解法”不成立（见难点树第 5 行）；真正可借的是 Bu 等“对前缀累积的连续值用比较器取 max”这一思路，而不是脉冲池化本身。 |
| 四种情况 第 6 行（第 116 行） | 现有转换方法在各类视觉图模型上哪里失效 ／ 发现问题本身 ／ 文档把“在新数据上让大家看到新结论”列为贡献 | 无法核实 | 本轮没有打开 pengsida/learning_research 文档逐句核对。文献侧支持“先做普查”：Lan 等与 Adaptive Calibration 都是因 max 无法转换而改源网络，说明失效案例真实存在。 |

## 七、失效普查实验协议：教师表（第 142–148 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 教师表 GatedGCN（第 144 行） | CIFAR10 超像素 ／ 固定 ／ 门控求和 ／ 锚点：已知能保真 | 正确 | 数据与结构对（Benchmarking GNNs，JMLR 2022：GatedGCN 在 CIFAR10 超像素上 67.31%，Table 7）。“已知能保真”来自项目自身实验。 |
| 教师表 MobileViG（第 145 行） | ImageNet ／ 固定 ／ max ／ 对照：有 max、无动态拓扑 | 正确 | Q7：SVGA 固定图、逐元素 max。补充：只有最后一个阶段用图，前面阶段是 MBConv 卷积，max 聚合在整网中占比小。 |
| 教师表 ViG-Ti（第 146 行） | ImageNet ／ 逐层 kNN ／ max ／ 动态拓扑的主要考察对象 | 正确 | ViG-Ti 73.9%（Table 4）；每层扩张 kNN、k 从 9 增到 18（Q6）。 |
| 教师表 DGCNN（第 147 行） | ModelNet40 ／ 逐层 kNN ／ max ／ 动态拓扑在点云上是否重现 | 正确 | EdgeConv 每层按特征重算 kNN（k=20），max 聚合；AnTao97/dgcnn.pytorch 仓库附预训练权重（`pretrained/*.t7`），MIT 许可证（见脉络三）。 |
| 教师表 ST-GCN（第 148 行） | NTU RGB+D 60 ／ 固定 ／ 求和 ／ 流式输入的时间冗余 | 正确 | 固定骨架图、按 3 个分区加权求和。补充：CoST-GCN（Pattern Recognition 2023）已给出 ST-GCN 的逐帧连续推理版本，可作流式基线。 |

## 八、失效普查实验协议：判据表（第 177–181 行）

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 判据 H1（第 179 行） | 动态拓扑会导致失效：T=4 时 ViG-Ti 精度差距比 MobileViG 大 1 个百分点以上，且拓扑替换能挽回一半以上 | 无法核实 | 是待检验的假设，文献中没有对应实验。Q9 的距离间隔判据可用来预先估计哪些层容易翻转。 |
| 判据 H2（第 180 行） | 流式输入有可用的时间冗余：ST-GCN 帧间差分非零比例明显低于逐帧激活 | 无法核实 | 文献中未找到 ST-GCN 逐层帧间差分稀疏度的测量；CoST-GCN 只复用时间维缓存，没有报告差分稀疏度。 |
| 判据 H3（第 181 行） | 静态输入下事件驱动难以胜过量化：等精度下每标量累计事件数高于盈亏线 | 无法核实 | 尚未在图上测量。文献倾向支持：Yan 等导出 T=5 相对 3 位 QNN 的盈亏发放率上限约 5.7%，Ottati 等显示 45nm 下 T≥2 的卷积 SNN 就输给 INT8 ANN（脉络五）。 |

## 九、表格之外的论断

| 草稿位置 | 草稿原话（简写） | 判定 | 证据或更正 |
| --- | --- | --- | --- |
| 开头（第 7 行） | 三处空白最值得做：动态拓扑的转换、流式图上的差分推理、与量化网络的公平能耗对照 | 不准确 | 第一、三处成立；第二处应收窄为“视觉图上的阈值化近似差分推理”，因为图上的精确增量推理已有大量工作。 |
| 怎么读（第 14 行） | 本文没有通读任何一篇论文全文 | 正确 | 这正是本次核查发现多处误读的原因（Spiking PointCNN、`2409.03368`、AEGNN 数字）。 |
| 时间线一导语（第 33 行） | 图网络还没有进入这条主线 | 不准确 | One-Spike SNN（IEEE TETC 2025）把 Text GCN 转换写成了独立一节；PMSM 也有 GCN 转换实验。视觉图确实没有进入。 |
| 时间线一结语（第 47 行） | 门控和注意力聚合在图上的事件化已有现成思路可借 | 不准确 | 可借的是思路，但现有 Transformer 方案仍要浮点补偿或长时间步（见 M2 行）。 |
| 时间线二导语（第 51 行） | 检索中没有找到把现成图网络免训练转成脉冲网络的系统工作 | 错误 | One-Spike SNN 第 IV-C 节就是对预训练 Text GCN 的免训练转换，5 个数据集平均掉 0.90%（Table V）。 |
| 时间线二结语（第 62 行） | 视觉图任务只有骨架识别有人做，且都走直接训练 | 错误 | SpikingGCN 做了 MNIST 超像素图分类；点云上有 Spiking PointNet、Spiking Point Transformer、NI-SGCN 等，Lan 等（ICCV 2023）还做了 PointNet++ 转换；事件骨架有 CGSSN（IJCNN 2026）。 |
| 第二轮检索（第 66 行） | MSG NeurIPS 2024 的相关工作一节称脉冲网络在图领域仍处于早期，基本做法是把激活函数换成脉冲神经元 | 无法核实 | 本轮读了 MSG 的方法与实验，没有逐句核对其相关工作节的这句话。 |
| 第二轮检索（第 67 行） | Spiking PointCNN 用转换得到点云上的脉冲网络，是点云转换的先例 | 错误 | 见 Q4：它是替代梯度从头训练；邻居按三维坐标固定；真正的先例是 Lan 等（ICCV 2023）。 |
| 第二轮检索（第 68 行） | 点云上直接训练的有 Spiking PointNet（NeurIPS 2023）和 Spiking Point Transformer | 正确 | Spiking Point Transformer 发表于 AAAI 2025（`arXiv:2502.15811`）。 |
| 第二轮检索（第 69 行） | max 池化转换已有多种解法……改成神经元层。因此 max 聚合本身不再算空白 | 错误 | 见难点树第 5 行与 Q10。 |
| 时间线三导语（第 73 行） | 事件驱动的图网络已经有专用芯片 | 正确 | EvGNN（FPGA）、ETHEREAL（28nm 流片）。 |
| 未核实事项（第 124 行） | “未见图网络免训练转换”和“未见 ViG 转换”需要用被引列表排查 | 正确 | 已排查（Q3）：前者被 One-Spike SNN 推翻（它不在这五篇的被引列表里，是另行检索到的）；后者在 ViG 552 篇被引中确认未见。 |
| 未核实事项（第 125 行） | GreedyViG 的发表处待核实 | 正确 | 已核实：CVPR 2024。 |
| 未核实事项（第 126 行） | Spiking PointCNN 如何处理点的邻域需读全文 | 正确 | 已读全文：三维坐标扩张 kNN，固定（Q4）。 |
| 未核实事项（第 127 行） | 各教师模型的公开权重与许可证 | 正确 | 已核实，见 [03-vision-graph-backbones.md](03-vision-graph-backbones.md) 总览表。要注意的许可证：DCGS 无许可证；GreedyViG 仅限非商业研究；CTR-GCN 为 CC BY-NC 4.0；DAGr 为 GPL-3.0。 |
| 转换方法（第 152 行） | DCGS 官方代码的差分多阈值单元 | 不准确 | 可以用，但代码没有任何图算子，重叠 MaxPool 退回浮点，仓库无许可证（Q5）；工作量估计应把这些计入。 |
| 转换方法（第 154 行） | 同比特均匀量化，作为非脉冲对照 | 不准确 | 低位宽下朴素均匀量化在 GNN 上会崩：TopGQ（DAC 2026）Table 6 中 Reddit 上 GCN 的朴素 INT4 只有 3.79%，TopGQ 免训练为 93.05%。只放朴素量化会被质疑选了弱基线，应加 TopGQ。 |

## 十、草稿漏掉的重要工作

按与本方向的关系从近到远排列；括号内为所在文件。

**直接相关（必须引用或对比）**

1. One-Spike SNN（IEEE TETC 2025）：唯一的预训练 GCN 免训练转换先例（02、01）。
2. Lan 等，*Efficient Converted Spiking Neural Network for 3D and 2D Classification*（ICCV 2023）：PointNet / PointNet++ 转换先例，因 max 无法转换而改 AvgPool 重训（02）。
3. PMSM（arXiv 2025）：3 层 GCN 在 T=1 无损转换，需重训量化源网络（02）。
4. Adaptive Calibration（AAAI 2025）：免训练校准转换推广到 PointNet，同样把 max 换成 mean（02、01）。
5. NI-SGCN（AAAI 2025）：唯一脉冲化 EdgeConv 的工作，直接训练且把 max 换成 mean（02）。
6. InkStream（IPDPS 2025）：max / min 聚合下多跳扩散大幅收缩，对 ViG / DGCNN 直接有启发（04）。
7. Eventful Transformers（ICCV 2023）与 EvNets（ECCV 2022）：现成网络免重训改造成事件驱动推理，阈值 + 累积状态（04）。
8. Benchmarking GNNs 超像素基准、RG-CNNs 等事件图基线（03）。

**可借用的技术**

9. 图上的增量推理：Ripple（ICDCS 2025）、RIPPLE++（2026）、NeutronRT（2026）、Reforge（IPDPS 2026）、InstantGNN（KDD 2022）、CoST-GCN（Pattern Recognition 2023）（04）。
10. 视频差分推理：CBinfer（ICDSC 2017 / IEEE TCSVT 2020）、Skip-Conv（CVPR 2021）、MotionDeltaCNN（ICCV 2023）、Delta 网络（ICML 2017）（04）。
11. 转换侧：ECMT（ACM MM 2024）、MST（ICCV 2023）、SpikedAttention（NeurIPS 2024）、FTBC（ECCV 2024）、并行脉冲计算转换（ICML 2025）、Adaptive Fission（NeurIPS 2025）、Spiking-YOLO（AAAI 2020）、SpiCalib（2022）、Fast-SNN（TPAMI 2023），以及脉冲 max 池化三篇：Rueckauer 等（2017）、Li & Zeng（IJCAI 2022）、Gaurav 等（IJCNN 2022）（01）。
12. kNN 与 max 的误差理论：RaBitQ（SIGMOD 2024）、ScaNN（ICML 2020）、Vecchiato & Bruch（2026）、最大化偏置（Thrun & Schwartz 1993；van Hasselt 2010）（questions.md Q9、Q10）。

**教师候选（脉络三）**

13. Pyramid ViG、ViHGNN（ICCV 2023）、MobileViGv2（CVPRW 2024）、DVHGNN（CVPR 2025）、PointNeXt（NeurIPS 2022）、Point Transformer v1 / v3、2s-AGCN、InfoGCN、HD-GCN、BlockGCN。

**能耗与硬件（脉络五）**

14. Horowitz 2014 原表（ISSCC，第 14 页 Fig. 1.1.9）；质疑 SNN 能效的 Davidson & Furber（2021）、Lemaire 等（ICONIP 2022）、Dampfhoffer 等（IEEE TETCI 2023）、Ottati 等（IEEE JETCAS 2023）、Bhattacharjee 等（ICASSP 2024）。
15. GNN 量化基线：TopGQ（DAC 2026）、Degree-Quant（ICLR 2021）、SGQuant（ICTAI 2020）、DRA（NorCAS 2024）。
16. 支持分级脉冲的芯片：Loihi 2、SENECA、SpiNNaker 2、Tianjic；事件图硬件：EFGCN、AMPLE、EDA-GNN。
