# 脉络四：事件驱动与增量推理

- 核对人：Gemini，2026-11-16
- 范围：只计算变化部分的方法。分三组：（A）视频 / 序列上的差分推理（Sigma-Delta、Delta 网络、CBinfer、Skip-Conv、DeltaCNN、MotionDeltaCNN、EvNets、Eventful Transformers）；（B）图上的增量 / 异步推理（AEGNN、DAGr、EvGNN、CoST-GCN、InstantGNN、InkStream、Ripple / RIPPLE++、NeutronRT、Reforge）；（C）ZtL-sysu/gnn2snn 仓库里提到的两个系统（DynaGraph、Ingress）。
- 原始 PDF 均从 arXiv / 出版方页面下载，用 `pypdf` 抽取文本后核对；页码指 PDF 页码。检索过程见 [search-log.md](search-log.md) 的 SL-56 至 SL-66。
- 本脉络里的 ETHEREAL 芯片已在 [05-energy-and-hardware.md](05-energy-and-hardware.md) 详细记录，这里不重复列条目；DAGr、AEGNN 在脉络三里作为教师模型也有条目，这里只从“增量推理”角度再记一次。

## 先回答任务书里点名的三件事

1. **DynaGraph 和 Ingress 是什么。**
   - DynaGraph（Guan、Iyer、Kim，GRADES-NDA 2022）是**动态图神经网络的训练系统**，不是推理系统，也不做近似。它的三项技术是：跨门控缓存同一份图消息（cached message passing）、把多个时间步的输入图卷积合并成一次（timestep fusion）、按时间序列切分的分布式数据并行。论文第 1 页明确写“we focus on efficient dynamic GNN training”，第 7 页 5.4 节说明结果与 DGL / PyG Temporal 精度相同。**它没有阈值，也不丢弃小变化。**仓库里把它当作“共享消息缓存”的先例，这个定位是对的。
   - Ingress（Gong 等，PVLDB 14(9) 2021）是**通用的顶点中心增量图计算系统**：把批处理的顶点中心算法自动改写成增量版本，按算法性质在四种记忆化策略（不记忆、记边、记路径、记顶点）中自动选一种。它的示例里包括 PageRank、SSSP，也包括 GCN 前向和 mean 聚合的 GraphSAGE（第 1 页 Fig. 1、第 2 页）。它做的是**精确增量**，没有阈值。
   - 两者都不是脉冲网络，也都不处理视觉图。仓库把它们和 Sigma-Delta、DeltaCNN 并列为“差分 / 增量 / 缓存早有先例”的证据，这个用法站得住；但若把 DynaGraph 理解为“流式推理”或“带阈值的增量”，就错了。
2. **有没有人给图网络的增量更新加阈值（小变化不传）。**
   - 视频 CNN / RNN / ViT 上普遍有：Delta 网络（阈值 Θ）、CBinfer（像素变化阈值）、DeltaCNN（逐层截断阈值 ε）、EvNets（每个神经元阈值 h）、Eventful Transformers（top-r 或阈值 h 的 token 门）。
   - 图上：**只找到两类近似做法**——InstantGNN 在**线性解耦传播**（类 PPR 的特征传播）里用残差阈值 ε 控制误差；Reforge 在**服务查询**时按预算只重算一部分邻居的嵌入。其余图增量系统（InkStream、Ripple、RIPPLE++、NeutronRT、DynaGraph、Ingress）都强调**与全量重算结果完全一致**，不丢弃小变化。DAGr 依靠 max 池化“最大值没变就不往下传”来截断更新，也是精确的，不是阈值。
   - **没找到**：在多层、含非线性的消息传递图网络里，逐层对激活变化量设阈值、把小于阈值的变化截断并累积误差（DeltaCNN 式）的工作。这是在 SL-56～SL-66 的检索范围内的结论，不是穷尽。
3. **变化随层数扩散（k 跳扩散）有没有人研究过。** 有，而且是这条线的中心问题：
   - AEGNN 第 5 页 4.2 节：第 k 层的更新范围是新节点的 k 跳子图；第 7 页 Fig. 4(b) 给出逐层节省量。
   - InkStream 第 2 页 Fig. 3：层数从 2 到 5，理论受影响区域占全图的比例快速上升；但在 min / max 聚合下，100 次更新时理论受影响区域里真正被影响的节点平均只有 3%。
   - RIPPLE++ 第 3 页：Products 图（平均度 50.5）上，批大小从 1 到 100 时受影响顶点从 0.7% 涨到约 37%。
   - DeltaCNN 第 5 页 3.3 节：一个像素的更新经过三层 3×3 卷积会扩散到 49 个像素，所以每层激活后都要截断。
   - CBinfer 第 3 页：同样指出像素级更新会逐层扩散。
   - 缓解办法分四类：有向图只让旧事件指向新事件（DAGr、EvGNN）；利用 max / min 的选择性（InkStream、DAGr、RIPPLE++）；逐层阈值截断（DeltaCNN、EvNets、Eventful Transformers，均不在图上）；预算式选择性重算（Reforge）。

## 总览表

| 短名 | 年份 | 发表处 | 对象 | 是否精确 | 阈值 / 截断 | 是否需要重训 |
| --- | --- | --- | --- | --- | --- | --- |
| Reforge（原名 Omega） | 2026 | IPDPS 2026 | 大图 GNN 服务 | 近似 | 按预算选择性重算 | 否 |
| RIPPLE++ | 2026 | arXiv（ICDCS 2025 的扩展版） | 流式图 GNN | 精确 | 无 | 否 |
| NeutronRT | 2026 | arXiv | 流式图 GNN 嵌入 | 精确 | 无 | 否 |
| Ripple | 2025 | ICDCS 2025 | 流式图 GNN | 精确 | 无 | 否 |
| InkStream | 2025 | IPDPS 2025 | 流式图 GNN | 精确 | 无（利用 max/min 选择性） | 否 |
| EvGNN | 2025 | IEEE TCASAI 2025 | 事件相机 GNN 加速器 | — | 有向单跳图 | 是 |
| DAGr | 2024 | Nature 2024 | 事件相机检测 | 精确 | max 池化截断 | 是 |
| Eventful Transformers | 2023 | ICCV 2023 | 视频 ViT | 近似 | top-r / 阈值 | 多数情况否 |
| MotionDeltaCNN | 2023 | ICCV 2023 | 移动相机视频 CNN | 近似 | 继承 DeltaCNN | 否 |
| CoST-GCN | 2023 | Pattern Recognition 2023 | 骨架流式 ST-GCN | 近似（去掉补零） | 无 | 部分 |
| DeltaCNN | 2022 | CVPR 2022 | 视频 CNN | 近似 | 逐层 ε + 截断缓冲 | 否 |
| EvNets | 2022 | ECCV 2022 | 视频 CNN | 近似 | 每神经元阈值 h | 否 |
| AEGNN | 2022 | CVPR 2022 | 事件相机 GNN | 精确 | 无 | 是 |
| InstantGNN | 2022 | KDD 2022 | 动态图解耦 GNN | 近似 | 残差阈值 ε | 部分 |
| DynaGraph | 2022 | GRADES-NDA 2022 | 动态 GNN 训练 | 精确 | 无 | — |
| Skip-Conv | 2021 | CVPR 2021 | 视频 CNN | 近似 | 门（阈值或学习） | 部分 |
| Ingress | 2021 | PVLDB 2021 | 顶点中心增量图计算 | 精确 | 无 | — |
| CBinfer（期刊版） | 2020 | IEEE TCSVT 2020 | 视频 CNN | 近似 | 像素变化阈值 | 否 |
| CBinfer（会议版） | 2017 | ICDSC 2017 | 视频 CNN | 近似 | 像素变化阈值 | 否 |
| Sigma-Delta | 2017 | ICLR 2017 | 视频 CNN / MLP | 与取整网络等价 | 量化即截断 | 否（只调层间尺度） |
| Delta 网络 | 2017 | ICML 2017 | RNN | 近似 | 阈值 Θ | 部分 |

---

### Reforge（2026）

- 标题：Reforge: Low-Latency Distributed GNN Serving with Selective Embedding Recomputation
- 作者：Geon-Woo Kim, Donghyun Kim, Jeongyoon Moon, Henry Liu, Tarannum Khan, Anand Iyer, Daehyeok Kim, Aditya Akella
- 发表处：2026 IEEE International Parallel and Distributed Processing Symposium（IPDPS 2026），pp. 788–803（据 arXiv v2 第 14 页参考文献 [1]）。**名称冲突**：同一 arXiv 编号 2501.08547 的早期版本名为 Omega，Ripple 论文第 8 页即以 “Omega [16]” 引用它。
- 链接：https://arxiv.org/abs/2501.08547 ；代码：未公开（PDF 中未见仓库链接）
- 核对深度：全文
- 一句话：在大图上回答“新来一个查询节点，它的嵌入是什么”时，大部分邻居直接用预先算好的嵌入，只挑对精度影响大的少数邻居重新计算。
- 解决的难点：完全复用预计算嵌入会忽略新节点对旧节点的影响，导致掉点（第 4 页 Table I：快 10.7 倍但掉 6.3 个百分点）；完全重算又太慢。它用一个启发式在预算内挑选要重算的节点，并做跨机器的计算图并行以减少通信。
- 留下的局限：针对的是“新增查询节点”的服务场景，不是逐帧的特征变化；挑选依据是统计意义上的误差最小化，没有逐层的变化量阈值；面向社交 / 商品大图，与视觉图无关。
- 是否需要训练或微调：否（复用已训练模型）
- 涉及的算子：GraphSAGE、GCN、GCNII 等的 sum / mean 聚合，嵌入缓存，选择性重算
- 关键数字：Amazon 数据集上 SAGE 模型延迟比 DGL 全量低 159 倍（第 10 页 Fig. 10 附近正文）；GCNII + Yelp 上层数增加时延迟变化见第 13 页 Fig. 17。
- 与本方向的关系：可借用的技术（“预算式选择性重算”是阈值化之外另一种控制扩散的思路）
- 核对人与日期：Gemini，2026-11-16

### RIPPLE++（2026）

- 标题：RIPPLE++: An Incremental Framework for Efficient GNN Inference on Evolving Graphs
- 作者：Pranjal Naman, Parv Agarwal, Hrishikesh Haritas, Yogesh Simmhan
- 发表处：arXiv（第 1 页脚注说明是 ICDCS 2025 论文 Ripple 的扩展长文版）
- 链接：https://arxiv.org/abs/2601.12347 ；代码：https://github.com/dream-lab/ripple （据正文引用）
- 核对深度：全文
- 一句话：图的边或节点特征变化时，只沿受影响的邻域逐层传播“增量消息”，精确更新所有节点的嵌入，比每次在受影响邻域里重算更快。
- 解决的难点：把 Ripple 从只支持累加型聚合推广到 max 这类单调算子和 GAT 这类注意力算子（第 5 页）；给出逐跳的计算量模型（第 13 页）：重算需约 $a_l\delta d_{l-1}+a_l d_{l-1}d_l$ 次浮点运算，增量需约 $a_{l-1}\delta d_{l-1}+a_l d_{l-1}d_l$ 次，其中 $a_l$ 是第 l 跳受影响顶点数。
- 留下的局限：坚持精确结果，不做近似（第 17 页明确说不采样、不近似）；当 $a_{l-1}$ 接近 $a_l$（深层或高度数图）时节省消失；注意力的系数同时依赖源和目标节点，一跳变化会迫使下一跳所有入边重算。
- 是否需要训练或微调：否
- 涉及的算子：sum / mean / max 聚合、GAT 注意力、线性变换、ReLU
- 关键数字：单机最高约 56K 次更新 / 秒，比 InkStream 快 2.2–24 倍；分布式比重算基线吞吐高约 25 倍、通信低 20 倍（第 1 页摘要、第 5 页）；Products 图上批大小 1→100 时受影响顶点由 0.7% 升到约 37%，Arxiv 图由 0.06% 升到 2.2%（第 3 页 Fig. 4 正文）。
- 与本方向的关系：可借用的技术（它量化了多跳扩散，并且支持 max 聚合）
- 核对人与日期：Gemini，2026-11-16

### NeutronRT（2026）

- 标题：Incremental GNN Embedding Computation on Streaming Graphs
- 作者：Qiange Wang, Haoran Lv, Yanfeng Zhang, Weng-Fai Wong, Bingsheng He
- 发表处：arXiv（PDF 中未见会议信息；本地文件名里的 “ICDE26” 无法从 PDF 证实，记为未核实）
- 链接：https://arxiv.org/abs/2603.20622 ；代码：未公开
- 核对深度：全文
- 一句话：把 GNN 拆成一组细粒度算子并安全地重排顺序，使流式图上的嵌入更新只在受影响子图上做，结果与全邻居计算一致。
- 解决的难点：L 层 GNN 的一次更新会波及 L 跳，朴素做法连有效的中间结果也重算（第 1 页 Fig. 1）；它区分“传播路径”和“最后一层受影响顶点”，只算必要部分；对 GAT、门控 GCN 这类消息依赖目标节点的模型给出了正确性条件。
- 留下的局限：精确方法，无阈值；更新比例变大时优势迅速缩小（第 11 页：|ΔE|=1K 时比全量快 285.0 倍，|ΔE|=100M、约占原边集 16% 时只快 3.8 倍）。对“每帧几乎所有节点特征都变”的视觉图，这正是最差情况。
- 是否需要训练或微调：否
- 涉及的算子：消息、聚合、更新的细粒度算子，GCN / GraphSAGE / GAT / 门控 GCN
- 关键数字：冗余计算减少 64%–99%，比现有方案快 1.7–145.8 倍，单张 A5000 处理十亿级流式图（第 1–2 页摘要与引言）。
- 与本方向的关系：可借用的技术（门控 GCN 的增量正确性分析与本项目 GatedGCN 锚点直接相关）
- 核对人与日期：Gemini，2026-11-16

### Ripple（2025）

- 标题：Ripple: Scalable Incremental GNN Inferencing on Large Streaming Graphs
- 作者：Pranjal Naman, Yogesh Simmhan
- 发表处：45th IEEE International Conference on Distributed Computing Systems（ICDCS 2025）（第 1 页脚注）
- 链接：https://arxiv.org/abs/2505.12112 ；代码：https://github.com/dream-lab/ripple （据 RIPPLE++ 正文）
- 核对深度：全文
- 一句话：流式图上边或特征变化时，用增量消息逐层更新受影响顶点的嵌入，支持单机和分布式。
- 解决的难点：逐顶点推理会重复计算重叠的邻域，逐层全图推理太慢（Products 图一次更新约 20 秒，第 3 页）；它只对受影响邻域做精确增量。
- 留下的局限：只支持线性（累加型）聚合（第 4 页贡献 1）；精确、无阈值；与视觉图无关。
- 是否需要训练或微调：否
- 涉及的算子：sum / mean 聚合、线性变换、逐元素激活
- 关键数字：单机稀疏图 Arxiv 上约 28000 次更新 / 秒，Products 上约 1200 次 / 秒，延迟 0.1 ms–1 s；分布式比基线吞吐高约 30 倍、通信低 70 倍（第 1 页摘要）。
- 与本方向的关系：可借用的技术
- 核对人与日期：Gemini，2026-11-16

### InkStream（2025）

- 标题：arXiv 版：InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update；**会议版标题不同**：InkStream: Instantaneous GNN Inference on Dynamic Graphs via Incremental Update（据检索到的 IPDPS 2025 目录，未打开会议版 PDF）
- 作者：Dan Wu, Zhaoying Li, Tulika Mitra
- 发表处：2025 IEEE International Parallel and Distributed Processing Symposium（IPDPS 2025），pp. 1273–1285（据检索结果；arXiv 版 2023 年发布）
- 链接：https://arxiv.org/abs/2309.11071 ；代码：https://github.com/WuDan0399/InkStream （据 RIPPLE++ 正文引用）
- 核对深度：全文（arXiv 版）
- 一句话：利用 max / min 聚合“只有被选中的邻居才有贡献”这一性质，在流式图里只更新真正受影响的节点，并保证结果与重新推理完全相同。
- 解决的难点：k 层 GNN 的一次更新理论上要访问整个 k 跳邻域，GPU 放不下；它发现 max / min 聚合下理论受影响区域里大多数节点其实没受影响（第 2 页 Fig. 3(b)：100 次更新时平均只有 3%），用事件机制控制层间传播。
- 留下的局限：节省来自 max / min 的选择性，对 sum / mean 收益小；精确方法无阈值；在视觉图上每帧所有节点特征都可能变，选择性带来的节省未经验证。
- 是否需要训练或微调：否
- 涉及的算子：max / min / sum 聚合，GCN、GraphSAGE、GIN
- 关键数字：三个模型、四张大图上，CPU 集群加速 2.5–427 倍，两种 GPU 集群加速 2.4–343 倍，输出与最新快照上的推理完全相同（第 1 页摘要）。
- 与本方向的关系：直接相关（ViG、DGCNN 都用 max 聚合，它说明 max 的选择性能大幅压低多跳扩散）
- 核对人与日期：Gemini，2026-11-16

### EvGNN（2025）

- 标题：EvGNN: An Event-driven Graph Neural Network Accelerator for Edge Vision
- 作者：Yufeng Yang, Adrian Kneip, Charlotte Frenkel
- 发表处：IEEE Transactions on Circuits and Systems for Artificial Intelligence（TCASAI），2025（DOI 10.1109/TCASAI.2024.3520905，第 1 页页眉）
- 链接：https://arxiv.org/abs/2404.19489 ；代码：https://github.com/cogsys-tudelft/evgnn
- 核对深度：全文
- 一句话：事件相机 GNN 的 FPGA 加速器，用“只指向新事件”的有向图把每次更新限制在一跳，从而让多层网络能逐层并行处理。
- 解决的难点：无向事件图的更新会逐层扩散到 n 跳（第 2 页定义）；改成有向动态图后每个新事件只影响自身，存储也不必保存边。
- 留下的局限：跑的是 INT8 量化 GNN，不是脉冲网络；有向单跳的简化只适用于“每个事件是一个新节点”的事件图，不适用于节点固定、特征逐帧变化的超像素图或 ViG 图。
- 是否需要训练或微调：是（需按其有向图结构训练并量化）
- 涉及的算子：PointNetConv 风格的消息传递、max 聚合、INT8 量化
- 关键数字：Xilinx KV260 上 N-CARS 精度 87.8%，平均每事件延迟 16 µs（第 1 页摘要）；Table VI（第 10 页）中软件版 0.07 MFLOPs/ev、88.0%。同表把 AEGNN 记为 0.6 MFLOPs/ev、86.7%（注明来自开源代码复现），与 AEGNN 论文 Table 1 的 0.47 MFLOPs/ev、0.945 不一致，两者都记录。
- 与本方向的关系：可借用的技术（硬件端如何规避多跳扩散）
- 核对人与日期：Gemini，2026-11-16

### DAGr（2024）

- 标题：Low-latency automotive vision with event cameras
- 作者：Daniel Gehrig, Davide Scaramuzza
- 发表处：Nature（2024，Vol. 629，pp. 1034–1040）
- 链接：https://doi.org/10.1038/s41586-024-07409-w ；代码：https://github.com/uzh-rpg/dagr
- 核对深度：全文（PMC 开放全文 PMC11136662）
- 一句话：事件建成有向时空图、配合低帧率图像 CNN 做检测；新事件到来时只更新必要节点，并在 max 池化处截断“最大值没变”的更新。
- 解决的难点：AEGNN 的无向图让更新向过去事件回传、逐层膨胀；DAGr 只让旧事件指向新事件，再利用 max 池化的选择性截断，把异步更新计算量削减 91%（详见脉络三 DAGr 条目）。
- 留下的局限：截断是“max 结果没变”这种精确条件，不是阈值；浮点激活，不是脉冲网络；双向时空依赖的固定节点图不能直接套用。
- 是否需要训练或微调：是
- 涉及的算子：有向半径建图、样条卷积、体素 max 池化、ELU、BatchNorm
- 关键数字：体素 max 池化拦截约 73% 的下游更新，有向建图加池化合计减少 91% 计算（正文与 Fig. 4，数字与脉络三一致）。
- 与本方向的关系：直接相关
- 核对人与日期：Gemini，2026-11-16

### Eventful Transformers（2023）

- 标题：Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers
- 作者：Matthew Dutson, Yin Li, Mohit Gupta
- 发表处：IEEE/CVF International Conference on Computer Vision（ICCV 2023）（PDF 为 arXiv 版，会议归属据 CVF 检索，见 SL-65）
- 链接：https://arxiv.org/abs/2308.13494 ；代码：未核实
- 核对深度：全文
- 一句话：视频逐帧跑 ViT 时，每个块里只重新处理变化大的 token，其余沿用上一帧的结果。
- 解决的难点：用“参考缓冲 + 误差累积”的 token 门避免缓慢漂移被一直忽略；注意力里的 $QK^\top$ 和 $AV$ 用差分形式增量更新（第 4–6 页）；选择策略有 top-r（固定预算）和阈值 h（输入自适应）两种（第 6 页）。
- 留下的局限：自注意力相当于全连接图，一跳就扩散到所有 token，它靠门在每个块后截断；没有处理稀疏图上的多跳扩散，也没有处理按特征选邻居（kNN）这种离散决策。
- 是否需要训练或微调：多数情况否（第 1 页摘要：“often without any re-training”）；第 3 页称同类自适应 ViT 方法一般需要重训。
- 涉及的算子：token 门、自注意力、Softmax、LayerNorm、MLP
- 关键数字：ImageNet VID 上 ViTDet，分辨率 1024、r=768 时由 467.4 降到 122.3 GFlops（3.8 倍），mAP50 由 82.93 降到 81.25；分辨率 672 时省 3.7 倍、mAP50 降 0.85（第 8 页）。
- 与本方向的关系：可借用的技术（“全连接图 + 门截断”是最接近 ViG 的增量推理先例）
- 核对人与日期：Gemini，2026-11-16

### MotionDeltaCNN（2023）

- 标题：MotionDeltaCNN: Sparse CNN Inference of Frame Differences in Moving Camera Videos with Spherical Buffers and Padded Convolutions
- 作者：Mathias Parger, Chengcheng Tang, Thomas Neff, Christopher D. Twigg, Cem Keskin, Robert Wang, Markus Steinberger
- 发表处：IEEE/CVF International Conference on Computer Vision（ICCV 2023）（据 CVF 检索；PDF 为 arXiv 版）
- 链接：https://arxiv.org/abs/2210.09887 ；代码：未核实
- 核对深度：全文
- 一句话：把 DeltaCNN 推广到移动相机：用环形缓冲对齐前后帧，只处理新露出的区域和真正变化的像素。
- 解决的难点：相机移动时几乎所有像素都变，DeltaCNN 退化为稠密计算；球形缓冲和带填充的卷积让新区域无缝接入已有缓冲。
- 留下的局限：仍是 CNN；依赖前后帧可对齐。
- 是否需要训练或微调：否（使用作者提供的预训练权重，第 6 页）
- 涉及的算子：卷积、激活、截断阈值、环形缓冲
- 关键数字：移动相机视频上帧率比 DeltaCNN 最多高 90%（第 1 页摘要、第 2 页）。
- 与本方向的关系：背景
- 核对人与日期：Gemini，2026-11-16

### CoST-GCN（2023）

- 标题：Continual Spatio-Temporal Graph Convolutional Networks
- 作者：Lukas Hedegaard, Negar Heidari, Alexandros Iosifidis
- 发表处：Pattern Recognition，Vol. 140，109528，2023（据检索结果；PDF 为 arXiv 版）
- 链接：https://arxiv.org/abs/2203.11009 ；代码：https://github.com/lukashedegaard/continual-skeletons （第 6 页）
- 核对深度：全文
- 一句话：把 ST-GCN 改写成“来一帧算一帧”的连续推理网络，时间卷积缓存历史结果，不再每次重算整段序列。
- 解决的难点：骨架动作识别的在线推理中，滑窗输入造成大量重复计算。
- 留下的局限：节省来自时间维的缓存复用，空间图上每帧所有关节仍全部重算，没有阈值；直接搬权重会因去掉补零掉 0.3 个点，最好再微调（第 7 页 Table I）。
- 是否需要训练或微调：部分（可直接转换，微调后精度更好）
- 涉及的算子：空间图卷积（固定骨架图）、时间卷积、BatchNorm、ReLU、自注意力（CoAGCN、CoS-TR 变体）
- 关键数字：每次预测 FLOPs 最多降 109 倍，硬件上加速 26 倍，峰值显存降 52%（第 1 页摘要）；NTU RGB+D 60 X-View 上直接转换精度 −0.3（第 7 页 Table I）。
- 与本方向的关系：直接相关（ST-GCN 是失效普查的流式教师，这是它的增量推理先例，但不是差分推理）
- 核对人与日期：Gemini，2026-11-16

### DeltaCNN（2022）

- 标题：DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos
- 作者：Mathias Parger, Chengcheng Tang, Christopher D. Twigg, Cem Keskin, Robert Wang, Markus Steinberger
- 发表处：IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2022）
- 链接：https://arxiv.org/abs/2203.03996 ；代码：https://github.com/facebookresearch/DeltaCNN
- 核对深度：全文
- 一句话：视频逐帧推理时，各层只传帧间差分，差分太小就截断并存入缓冲，第一个在 GPU 上真正跑赢 cuDNN 的稀疏 CNN 框架。
- 解决的难点：线性层传差分、非线性层保存累积输入；截断误差放进缓冲，超过阈值才一次性发出，避免误差长期累积（第 4 页 Fig. 3）；每层阈值在小训练子集上从前往后自动调（第 5 页 3.3 节）。
- 留下的局限：针对规则网格卷积；“更新掩码随卷积逐层膨胀”在图上对应多跳扩散，但它没有处理按特征选邻居。
- 是否需要训练或微调：否（第 1 页摘要“without retraining”；阈值需在小子集上自动调）
- 涉及的算子：卷积、ReLU 等激活、池化、截断阈值 ε、累积缓冲
- 关键数字：最高 7 倍加速（第 1 页摘要）；单像素更新经三层 3×3 卷积扩散到 49 像素（第 5 页 3.3 节）；Human3.6M 姿态估计的速度与精度对比见第 8 页 Table 1。
- 与本方向的关系：直接相关（差分推理在 CNN 上的对应物，阈值 + 截断缓冲是可直接借到图上的机制）
- 核对人与日期：Gemini，2026-11-16

### EvNets（2022）

- 标题：Event Neural Networks
- 作者：Matthew Dutson, Yin Li, Mohit Gupta
- 发表处：European Conference on Computer Vision（ECCV 2022）（据检索结果；PDF 为 arXiv 版）
- 链接：https://arxiv.org/abs/2112.00891 ；代码：未核实
- 核对深度：全文
- 一句话：给每个神经元加上长期记忆的状态变量，只有累计变化超过阈值才向下游发送，可把现成网络直接改造成事件驱动推理。
- 解决的难点：Skip-Conv 一类“只看帧间差”的做法会漏掉缓慢累积的变化（第 7 页举例：从 0 每步增加 0.1δ 到 2δ）；EvNets 用累积量和阈值 $m_i=H(|d_i|-h_i)$ 解决（第 9 页式 6）。
- 留下的局限：针对 CNN；没有处理图结构和邻居选择。
- 是否需要训练或微调：否（第 1 页摘要“without re-training”；附录中的 HRNet 基线本身是在 JHMDB 上微调的）
- 涉及的算子：卷积、线性层、逐神经元阈值、累积器
- 关键数字：计算量约降一个数量级（第 1 页摘要、第 3 页）；每个模型选“PCK 或 mAP 下降不到 0.5%”的最高阈值，结果见第 12 页 Table 1，开销见 Table 2。
- 与本方向的关系：可借用的技术（逐神经元阈值 + 累积状态与脉冲神经元的“积分—发放”同构）
- 核对人与日期：Gemini，2026-11-16

### AEGNN（2022）

- 标题：AEGNN: Asynchronous Event-based Graph Neural Networks
- 作者：Simon Schaefer, Daniel Gehrig, Davide Scaramuzza
- 发表处：IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2022）；arXiv v3 是更新版，封面写明“更新了第 5 节的计算量计算”
- 链接：https://arxiv.org/abs/2203.17149 ；代码：项目页 https://uzh-rpg.github.io/aegnn/
- 核对深度：全文（arXiv 更新版）
- 一句话：把事件流当作不断生长的时空图，每来一个新事件只重算受影响的局部子图。
- 解决的难点：第 5 页 4.2 节推出：第 1 层只需更新新节点的 1 跳子图，之后每层多扩一跳，第 N 层更新 N 跳子图，结果与全图计算相同；在同步数据上训练，测试时以异步方式部署。
- 留下的局限：扩散是无条件的——只要拓扑上可达就重算，没有变化量阈值；深层的节省迅速变小（第 7 页 Fig. 4(b) 逐层节省）；靠下采样、半径和最大邻居数、图池化压制计算量。
- 是否需要训练或微调：是（训练普通 GNN；异步化本身不需要改权重）
- 涉及的算子：半径建图、样条卷积、max 图池化、ELU、BatchNorm
- 关键数字：**数字冲突**——CVPR 原版摘要称计算量“最多降 200 倍”（据检索到的 CVF 页面摘要，见 SL-65；未打开 CVF PDF）；arXiv 更新版摘要改为“比最好的异步方法低 11 倍”，延迟比普通 GNN 低 8 倍（第 1 页）。N-Cars 上 0.47 MFLOPs/ev、精度 0.945（第 6 页 Table 1）；N-Caltech101 上 25000 事件时普通 GNN 每事件 11.2×10³ MFLOPs，AEGNN 6.11 MFLOPs（第 7 页 Table 2）；Fig. 4(a)：GNN 比稠密 CNN 少 10 倍，异步化再少 30 倍。
- 与本方向的关系：直接相关
- 核对人与日期：Gemini，2026-11-16

### InstantGNN（2022）

- 标题：Instant Graph Neural Networks for Dynamic Graphs
- 作者：Yanping Zheng, Hanzhi Wang, Zhewei Wei, Jiajun Liu, Sibo Wang
- 发表处：Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining（KDD 2022）（第 1 页）
- 链接：https://arxiv.org/abs/2206.01379 ；代码：未核实
- 核对深度：全文
- 一句话：动态图上边陆续到来时，增量更新解耦式 GNN 的传播矩阵，用残差阈值控制近似误差，不必每个快照都重算。
- 解决的难点：快照式动态 GNN 反映变化有延迟、重复计算量大；它借用近似 PPR 的“估计值 + 残差”推送算法，只推送残差超过阈值 ε 的节点（第 4–5 页 Algorithm 1、2），并给出更新复杂度上界。
- 留下的局限：阈值作用在**线性特征传播**（解耦 GNN 的预处理）上，不是多层非线性消息传递的逐层激活；附带“何时重训”的自适应策略。这是图上唯一找到的“小于阈值不传”的做法，但机制与 DeltaCNN 式逐层截断不同。
- 是否需要训练或微调：部分（传播增量更新，分类器按自适应策略在必要时重训）
- 涉及的算子：线性特征传播（类 PPR）、残差推送、MLP 分类器
- 关键数字：大图上效率提高一到数个数量级，精度相当（第 2 页贡献、第 8 页 Fig. 1–3）。
- 与本方向的关系：可借用的技术（图上“残差 + 阈值”推送的误差分析可借）
- 核对人与日期：Gemini，2026-11-16

### DynaGraph（2022）

- 标题：DynaGraph: Dynamic Graph Neural Networks at Scale
- 作者：Mingyu Guan, Anand Padmanabha Iyer, Taesoo Kim
- 发表处：GRADES & NDA '22（ACM SIGMOD 附属研讨会），2022 年 6 月 12 日，费城（第 1 页页脚，DOI 10.1145/3534540.3534691）。**更正**：上一轮笔记写作 “SoCC 2022”，PDF 页脚显示不是。
- 链接：https://taesoo.kim/pubs/2022/guan:dynagraph.pdf ；代码：未公开
- 核对深度：全文
- 一句话：加速“图卷积 + 循环网络”式动态 GNN 训练的系统，靠缓存同一份图消息、合并多个时间步的图卷积、分布式数据并行来提速。
- 解决的难点：GraphRNN 的多个门各自重复做同一次图聚合；小图上逐时间步卷积 GPU 利用率低。
- 留下的局限：针对训练，不是推理；精确，无阈值，不丢小变化（第 7 页 5.4 节：精度与 DGL / PyG Temporal 相同）；不涉及脉冲或视觉图。
- 是否需要训练或微调：不适用（本身就是训练系统）
- 涉及的算子：图卷积、GRU / LSTM 门、消息缓存
- 关键数字：比 DGL 最多快 2.5 倍、比 PyG Temporal 最多快 2.7 倍（第 5 页）；GCRN-LSTM 两机加速 2.54 倍、八机 2.23 倍（第 7 页）。
- 与本方向的关系：背景（共享消息缓存的先例）
- 核对人与日期：Gemini，2026-11-16

### Skip-Conv（2021）

- 标题：Skip-Convolutions for Efficient Video Processing
- 作者：Amirhossein Habibian, Davide Abati, Taco S. Cohen, Babak Ehteshami Bejnordi
- 发表处：IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2021）（据检索结果）
- 链接：https://arxiv.org/abs/2104.11487 ；代码：未核实
- 核对深度：全文
- 一句话：卷积改为在帧间残差上计算，每层配一个门决定哪些位置的残差值得算。
- 解决的难点：背景区域的重复计算；门可以按残差幅值直接判断，也可以与卷积一起训练（第 1 页）。
- 留下的局限：只看帧间差，缓慢累积的变化会被持续忽略（EvNets 第 7 页指出此问题）；针对 CNN。
- 是否需要训练或微调：部分（幅值门不需要，学习门需要）
- 涉及的算子：卷积、门函数、残差
- 关键数字：EfficientDet 和 HRNet 上计算量降 3–4 倍且不掉点（第 1 页摘要）；HRNet-w32 上 MAC 降 2.51 / 4.33 / 7.92 倍时实际耗时降 1.48 / 1.91 / 4.09 倍（第 7 页 Table 2）。
- 与本方向的关系：背景
- 核对人与日期：Gemini，2026-11-16

### Ingress（2021）

- 标题：Automating Incremental Graph Processing with Flexible Memoization
- 作者：Shufeng Gong, Chao Tian, Qiang Yin, Wenyuan Yu, Yanfeng Zhang, Liang Geng, Song Yu, Ge Yu, Jingren Zhou
- 发表处：Proceedings of the VLDB Endowment（PVLDB），Vol. 14，No. 9，pp. 1613–1625，2021
- 链接：https://vldb.org/pvldb/vol14/p1613-gong.pdf ；代码：https://github.com/alibaba/libgrape-lite （ingress 分支）
- 核对深度：全文
- 一句话：把用户写好的批处理顶点中心图算法自动改写成增量版本，图变化时只修正受影响的结果。
- 解决的难点：增量图算法难写；它给出四种记忆化策略及各自的适用条件，并自动检查条件来选用（第 1–2 页）。
- 留下的局限：精确增量，无阈值；面向 PageRank、SSSP 等迭代图计算和 GCN 前向这类顶点中心程序，不涉及脉冲、视觉图或逐帧特征变化。
- 是否需要训练或微调：不适用
- 涉及的算子：顶点中心消息传递、min / sum 聚合、GCN 前向、mean 聚合 GraphSAGE
- 关键数字：比现有增量图系统平均快 15.93 倍，最高 147.14 倍（第 1 页摘要）。
- 与本方向的关系：背景（通用增量图计算的先例）
- 核对人与日期：Gemini，2026-11-16

### CBinfer 期刊版（2020）

- 标题：CBinfer: Exploiting Frame-to-Frame Locality for Faster Convolutional Network Inference on Video Streams
- 作者：Lukas Cavigelli, Luca Benini
- 发表处：IEEE Transactions on Circuits and Systems for Video Technology，Vol. 30，No. 5，pp. 1451–1465，2020（据检索结果；PDF 为 arXiv 1808.05488）
- 链接：https://arxiv.org/abs/1808.05488 ；代码：https://github.com/lukasc-ch/CBinfer
- 核对深度：全文
- 一句话：静态相机视频上，只对变化超过阈值的像素及其受影响区域重算卷积。
- 解决的难点：把会议版推广到语义分割、姿态检测和目标检测。
- 留下的局限：静态相机；变化区域随层扩散，靠阈值和池化压制。
- 是否需要训练或微调：否
- 涉及的算子：卷积、ReLU、池化、像素变化阈值
- 关键数字：Tegra X2 上语义分割平均加速 9.1 倍、精度损失 <0.1%；姿态检测加速 7.0 倍；目标检测运算量降 5 倍；能效 511 GOp/s/W（基线 70）（第 1 页摘要）。
- 与本方向的关系：背景
- 核对人与日期：Gemini，2026-11-16

### CBinfer 会议版（2017）

- 标题：CBinfer: Change-Based Inference for Convolutional Neural Networks on Video Data
- 作者：Lukas Cavigelli, Philippe Degen, Luca Benini
- 发表处：International Conference on Distributed Smart Cameras（ICDSC 2017）（据 DeltaCNN、Eventful Transformers 参考文献及检索；PDF 为 arXiv 1704.04313）。**更正**：上一轮下载的 `arXiv:1608.08404` 实际是一篇类星体天文论文，已排除。
- 链接：https://arxiv.org/abs/1704.04313 ；代码：https://github.com/lukasc-ch/CBinfer
- 核对深度：全文
- 一句话：检测帧间变化的像素，只更新受它们影响的卷积输出。
- 解决的难点：第 3 页指出像素级更新会随卷积层逐层扩散，但变化在输出端同样局部，噪声样的变化对深层影响小。
- 留下的局限：静态相机；阈值手调。
- 是否需要训练或微调：否
- 涉及的算子：卷积、像素变化阈值
- 关键数字：比 cuDNN 平均快 8.6 倍、精度损失 <0.1%、能效约为逐帧推理的 10 倍（第 1 页摘要）。
- 与本方向的关系：背景
- 核对人与日期：Gemini，2026-11-16

### Sigma-Delta（2017）

- 标题：Sigma-Delta Quantized Networks
- 作者：Peter O'Connor, Max Welling
- 发表处：International Conference on Learning Representations（ICLR 2017，Poster）（OpenReview 决定，见 SL-65；PDF 页眉仍是“Under review”）
- 链接：https://openreview.net/forum?id=Hy1b8W9eg ；https://arxiv.org/abs/1611.02024 ；代码：https://github.com/petered/sigma-delta
- 核对深度：全文
- 一句话：层与层之间只传激活变化量的取整脉冲，计算量随输入变化多少而不是网络大小而定，可把预训练网络直接改造过来。
- 解决的难点：时间差分和积分与线性运算可交换；用 herding 做离散化，使任一时刻的输出与“取整网络”相同，不随历史路径漂移；只优化层间尺度系数，在精度和计算量之间折中（第 5 页第 4 节）。
- 留下的局限：MLP / CNN；层间尺度需在小数据上优化；未涉及图结构。
- 是否需要训练或微调：否（不改权重，只调层间尺度）
- 涉及的算子：线性层、卷积、ReLU、取整、herding、时间差分 / 积分
- 关键数字：MNIST / Temporal-MNIST 结果见第 13 页 Table 1；预训练 VGG 在 ILSVRC2015 两段拼接视频上，第一段（静态背景）计算量约为原网络的 1/11，第二段（运动较多）约为 1/4（第 8 页 Fig. 5 图注）；正文另称视频上约可省 10 倍计算、精度有一定损失（第 8 页）。
- 与本方向的关系：直接相关（差分推理的思想源头，且明确面向预训练网络）
- 核对人与日期：Gemini，2026-11-16

### Delta 网络（2017）

- 标题：Delta Networks for Optimized Recurrent Network Computation
- 作者：Daniel Neil, Jun Haeng Lee, Tobi Delbruck, Shih-Chii Liu
- 发表处：Proceedings of the 34th International Conference on Machine Learning（ICML 2017），PMLR 70:2584–2593（据 PMLR 检索）
- 链接：https://arxiv.org/abs/1612.05571 ；代码：未公开
- 核对深度：全文
- 一句话：RNN 里每个神经元只有在激活变化超过阈值时才发送新值。
- 解决的难点：RNN 每步都要读写状态，内存访问是瓶颈；阈值化减少计算和访存，并保存“上次发送值”以防漂移。
- 留下的局限：RNN；阈值大时需要专门训练才能保持精度。
- 是否需要训练或微调：部分（现成网络可直接用，训练后效果更好）
- 涉及的算子：GRU、线性层、阈值 Θ
- 关键数字：WSJ 语音识别上现成网络 Θ=0.2 时约 5.5 倍加速、WER 10.8%（Θ=0 时 10.2%），训练后 5.7 倍同 WER（第 6 页）；TIDIGITS 上成本降 9 倍；驾驶转向网络中 RNN 部分约 100 倍（第 1 页摘要、第 6 页）。
- 与本方向的关系：背景
- 核对人与日期：Gemini，2026-11-16

## 这条线是怎么演化的

1. **2016–2017，思想成形。** Sigma-Delta 网络和 Delta 网络先后提出“只传变化量”：前者把它做成与取整网络等价的离散化，可套在预训练网络上；后者在 RNN 上用阈值丢掉小变化。CBinfer 把同样的想法落到静态相机视频的 CNN 上，并第一次明确写出“更新会随卷积层扩散”。
2. **2021–2023，在 GPU 上做实。** Skip-Conv 引入门，DeltaCNN 用“逐层阈值 + 截断误差缓冲”第一次在 GPU 上跑赢 cuDNN，EvNets 用每神经元的累积状态解决缓慢漂移，MotionDeltaCNN 处理移动相机，Eventful Transformers 把同样的机制搬到 ViT——自注意力等价于全连接图，变化一跳就扩散到全部 token，只能靠每块后的门截断。CoST-GCN 则在骨架 ST-GCN 上做了时间维的缓存复用。
3. **2022–2025，图上的增量推理分成两支。**
   - 事件相机一支：AEGNN 证明可以只更新 k 跳子图且结果精确，但深层节省迅速消失；DAGr 改用有向图并借 max 池化截断，EvGNN、ETHEREAL 把有向单跳结构做进 FPGA 和芯片。
   - 大图系统一支：Ingress、DynaGraph 做通用增量与缓存；InstantGNN 在线性传播里用残差阈值；InkStream 利用 max / min 的选择性；Ripple / RIPPLE++、NeutronRT 把精确增量推广到更多算子并量化多跳扩散；Reforge 用预算式选择性重算换取低延迟。
4. **还空着的地方。** 视频一支有阈值没有图，图一支有图但几乎都坚持精确、没有逐层阈值。在“节点固定、特征逐帧变化、按特征选邻居”的视觉图（ViG、DGCNN、骨架序列）上，**逐层阈值化差分 + 截断误差缓冲**的做法、以及阈值对 kNN 邻居选择和 max 胜者的影响，在本轮检索范围内都没找到。InkStream 与 DAGr 的结果提示 max 聚合本身会大幅抑制扩散，这对 ViG / DGCNN 是有利信号，但还没有人在这类模型上测过。
