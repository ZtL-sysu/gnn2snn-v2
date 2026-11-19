# 定向问题逐条回答

- 核对人：Gemini，2026-11-16
- 格式：每题三段——**结论**、**证据**、**搜索记录**。搜索记录编号见 [search-log.md](search-log.md)。
- 涉及论文的完整条目在 `01`–`05` 各脉络文件里，这里只摘与问题直接相关的部分。

---

## Q1：有没有把图神经网络从普通网络转换成脉冲网络的论文？

**结论**：是。但只找到固定图、线性聚合的先例，且大多要先改造或重训源网络；**真正免训练、直接转换现成 GNN 权重的只有一篇（One-Spike SNN，对象是文本图上的 GCN）**。没有找到视觉图网络（ViG、DGCNN 这类按特征选邻居、用 max 聚合的模型）的转换。

**证据**：

| 论文 | 图 / 模型 | 是否免训练 | 关键事实与出处 |
| --- | --- | --- | --- |
| One-Spike SNN（Hwang & Kung，IEEE TETC 2025，13(1):162–172） | 2 层 ReLU Text GCN，5 个文本图数据集 | 是 | 第 IV-C 节标题即 “GCN-to-SNN Conversion”：每层 GCN 拆成 $XW$ 与 $\tilde A(\cdot)$ 两个脉冲层；Table V（T=16、Q=1.3）平均掉点 0.90%。只适用于固定邻接、线性求和、ReLU。 |
| PMSM（Zhang 等，arXiv 2508.14520，2025） | 3 层 GCN，EvTouch-Objects 触觉图 | 否（须用极性量化激活 PQA 重训源 GCN） | Table IV、V：源 PQA-GCN 90.97%，转换后 T=1 也是 90.97%。 |
| Lan 等（ICCV 2023，pp. 9211–9220） | PointNet、PointNet++（点云，非严格意义的 GNN） | 部分（必须先把 MaxPool 换成 AvgPool 并重训源网络） | 第 4.1 节原话：“In the ANN-SNN conversion method, using Maxpooling will not work.”；Table 1–3：ScanObjectNN 上改用 AvgPool 的 PointNet++ 源网络只有 69.22%。 |
| Adaptive Calibration（AAAI 2025） | 改进版 PointNet（3D 分类与部件分割） | 部分（转换免训练，但源 PointNet 已把 `torch.max` 换成 `torch.mean` 再训练） | 官方仓库 `3dtasks/models/model.py`。 |
| DeNA-SGN（arXiv 2611.14775，2026） | 通用图节点分类 | 不算转换（训练期就用硬阈值 STE，再零失配编译到脉冲电路） | 见 [02-spiking-gnn.md](02-spiking-gnn.md)。 |

另：SpikingGCN（IJCAI 2022）明确走端到端训练路线；GC-SNN / GA-SNN（IJCAI 2021）在引言讨论了转换路线的局限后改走直接训练。

**搜索记录**：SL-01～SL-09、SL-42、SL-44～SL-48。

---

## Q2：有没有针对 ViG、DGCNN、PointNet++、ST-GCN 的转换工作？

**结论**：

| 模型 | 结论 |
| --- | --- |
| ViG（含 Pyramid ViG、ViHGNN、MobileViG、GreedyViG） | 否 |
| DGCNN（EdgeConv） | 否 |
| PointNet++ | 是（有条件：Lan 等 ICCV 2023 转换的是把 MaxPool 改成 AvgPool 并重训后的 PointNet++，不是官方 max 池化权重） |
| ST-GCN（含 2s-AGCN、CTR-GCN） | 否 |

**证据**：

- **ViG**：arXiv 三组检索式均 0 命中（SL-10～SL-12）；Crossref 前排 30 条与网页检索都没有交集（SL-13、SL-14）；ViG 在 OpenAlex 上的 552 篇被引里也没有一篇做脉冲化或转换（SL-45）。
- **DGCNN**：arXiv、Crossref、网页检索均未找到（SL-15～SL-18）。唯一把 EdgeConv 脉冲化的是 NI-SGCN（AAAI 2025，`arXiv:2502.19660`），它是直接训练的点云去噪网络，并且把 EdgeConv 的 max 聚合换成了 mean。
- **PointNet++**：Lan 等（ICCV 2023）Table 1–3 给出 PointNet++ 转换结果（ModelNet40 上源网络 89.45%，T=4/8/16 时 89.37% / 89.43% / 89.45%），但第 4.1 节说明必须先换成 AvgPool 并重训源网络。直接训练的有 Spiking PointNet（NeurIPS 2023，只做了 PointNet）、AMP2（AAAI 2026）Table 5 中的 Spiking PointNet++（T=1）。
- **ST-GCN**：arXiv 骨架 + 脉冲检索命中的 Signal-SGN、Signal-SGN++、MK-SGN、S3T-Former 等全部是直接训练（SL-22～SL-24）；ST-GCN / CTR-GCN 只作为全精度基线或蒸馏教师出现（SL-25）。另有 CoST-GCN（Pattern Recognition 2023）把 ST-GCN “转换”为逐帧连续推理网络，但它不是脉冲网络（SL-62，见 [04-event-driven-inference.md](04-event-driven-inference.md)）。

**搜索记录**：SL-07、SL-10～SL-26、SL-45、SL-62。

---

## Q3：DCGS、ViG、SpikingGCN、SpikeZIP-TF、STA 的全部被引列表里，与“图 + 脉冲”或“图 + 转换”有关的每一篇

**结论**：五份被引列表里**没有任何一篇“图 + 转换”**。“图 + 脉冲”只出现在 SpikingGCN 的被引列表里（41 篇），全部是从头设计并训练的脉冲图模型、数据集或芯片部署工作，摘要中都没有出现 conversion / converted / ANN-to-SNN 等词。另外四篇的被引列表里“图 + 脉冲”为 0（ViG 的 1 条规则命中是假阳性）。

**证据**（OpenAlex，2026-11-16 抓取；筛选规则见 search-log 第五节）：

| 被引对象 | OpenAlex 被引总数 | “图 + 脉冲”命中 | “图 + 转换”命中 |
| --- | --- | --- | --- |
| DCGS（`arXiv:2503.00301`） | 13 | 0 | 0 |
| ViG（NeurIPS 2022） | 552 | 1（假阳性：*A Syncretic Perception Architecture …*，Small 2026，光电突触储备池器件） | 0（宽松规则命中的 3 篇量化 / 攻击论文与脉冲转换无关） |
| SpikingGCN（IJCAI 2022） | 79（另并入同名期刊记录） | 41 | 0 |
| SpikeZIP-TF（ICML 2024） | 25 | 0 | 0 |
| STA（ICLR 2024） | 20（全文检索法，完整性不确定） | 0 | 0 |

SpikingGCN 被引列表中的 41 篇“图 + 脉冲”论文（按 OpenAlex 记录；“类型”按标题与摘要判断，除标注“全文”外均为仅摘要）：

| # | 年份 | 标题 | 发表处 | 类型 |
| --- | --- | --- | --- | --- |
| 1 | 2023 | Scaling Up Dynamic Graph Representation Learning via Spiking Neural Networks（SpikeNet） | AAAI | 直接训练的脉冲图模型 |
| 2 | 2024 | Dynamic Spiking Graph Neural Networks（Dy-SIGN） | AAAI | 直接训练的脉冲图模型 |
| 3 | 2024 | SiGNN: A spike-induced graph neural network for dynamic graph representation learning | Pattern Recognition | 直接训练的脉冲图模型 |
| 4 | 2024 | Adaptive Synaptic Scaling in Spiking Networks for Continual Learning and Enhanced Robustness | IEEE TNNLS | 规则命中，主题是脉冲网络持续学习 |
| 5 | 2024 | A novel label-aware global graph construction method and spiking-coded graph neural network for intelligent process fault diagnosis | Neurocomputing | 直接训练的脉冲图模型 |
| 6 | 2025 | Signal-SGN | ACM MM | 直接训练（全文，见 Q8） |
| 7 | 2023 | AM-SGCN: Tactile Object Recognition for Adaptive Multichannel Spiking Graph Convolutional Neural Networks | IEEE Sensors Journal | 直接训练的脉冲图模型 |
| 8 | 2024 | Dynamic Reactive Spiking Graph Neural Network（DRSGNN） | AAAI | 直接训练的脉冲图模型 |
| 9 | 2025 | MK-SGN | Neurocomputing | 直接训练 + 知识蒸馏（全文，见 Q8） |
| 10 | 2023 | Hyperparameter Optimization and Feature Inclusion in Graph Neural Networks for Spiking Implementation | ICMLA | 设计脉冲神经元网络做半监督图学习，不是转换 |
| 11 | 2023 | Improving Graph Collaborative Filtering via Spike Signal Embedding Perturbation | IEEE TCDS | 直接训练 |
| 12 | 2025 | ELAI-SGCN | Neural Networks | 直接训练的脉冲图模型 |
| 13 | 2025 | Graph Spiking Attention Network: Sparsity, Efficiency and Robustness | IEEE TPAMI | 直接训练的脉冲图模型 |
| 14 | 2026 | GT-SNT | AAAI | 直接训练（全文，见脉络二） |
| 15 | 2024 | VT-SGN: Spiking Graph Neural Network for Neuromorphic Visual-Tactile Fusion | IEEE Trans. Haptics | 直接训练的脉冲图模型 |
| 16 | 2024 | Efficient Spiking Variational Graph Autoencoders for Unsupervised Graph Representation Learning Tasks | IEEE Intelligent Systems | 直接训练的脉冲图模型 |
| 17 | 2025 | SSEL: spike-based structural entropic learning for spiking graph neural networks | Frontiers in Neuroscience | 直接训练的脉冲图模型 |
| 18 | 2024 | Spiking Neural Networks-Based Neighborhood Co-Occurrence Link Prediction | CCAI | 直接训练 |
| 19 | 2024 | Human poses recognition based on Spiking Pulse Graph Neural Networks | PeerJ Computer Science | 直接训练的脉冲图模型 |
| 20 | 2026 | HASNN: Hierarchical attention spiking neural network for dynamic graph representation learning | Knowledge-Based Systems | 直接训练的脉冲图模型 |
| 21 | 2025 | Edge-Private Spiking Graph Convolutional Learning | ICEDGE | 直接训练 |
| 22 | 2026 | Citation network datasets for benchmarking spiking graph neural networks on experimental neuromorphic hardware | npj Unconventional Computing | 数据集 / 基准 |
| 23 | 2023 | Efficient GCN Deployment with Spiking Property on Spatial-Temporal Neuromorphic Chips | ICONS | 把已有脉冲 GCN 映射到芯片，不是转换 |
| 24 | 2023 | A Graph is Worth 1-bit Spikes（SpikeGCL） | arXiv（ICLR 2024） | 直接训练（全文，见脉络二） |
| 25 | 2023 | EEGSN: Towards Efficient Low-latency Decoding of EEG with Graph Spiking Neural Networks | arXiv | 直接训练的脉冲图模型 |
| 26 | 2025 | Signal-SGN++ | arXiv | 直接训练（全文） |
| 27 | 2026 | When Prompting Meets Spiking: Graph Sparse Prompting via Spiking Graph Prompt Learning | arXiv | 直接训练 |
| 28 | 2026 | Scalable and Efficient Joint Spiking Embedding Predictive Architecture for Large-Scale Dynamic Graphs | arXiv | 直接训练 |
| 29 | 2024 | Unveiling the Potential of Spiking Dynamics in Graph Representation Learning through Spatial-Temporal Normalization | arXiv | 直接训练 |
| 30 | 2025 | Improving vision-language alignment with graph spiking hybrid Networks | arXiv | 直接训练 |
| 31 | 2025 | Enhancing Graph Transformers with SNNs and Mutual Information | LNCS | 直接训练 |
| 32 | 2025 | Spiking Heterogeneous Graph Attention Networks | arXiv | 直接训练（与 #37 为同一工作） |
| 33 | 2026 | Higher-Order Spiking and Graph Neural Network for Knowledge Tracing | ACM TOIS | 直接训练 |
| 34 | 2026 | Spiking Heterogeneous Graph Attention Networks | AAAI | 直接训练 |
| 35 | 2026 | E²SGNN: Reconciling Expression and Efficiency in Spiking Graph Neural Network | WWW | 直接训练的脉冲图模型 |
| 36 | 2026 | A novel spiking graph convolutional residual network for robust fault diagnosis of planetary gearboxes under strong noises | Structural Health Monitoring | 直接训练 |
| 37 | 2026 | Towards Fair Spiking Graph Neural Networks via Debiased Representation Learning and Fairness-Aware Bernoulli Encoding | SSRN | 直接训练 |
| 38 | 2026 | A Spike-Based Graph Surrogate for Modeling Finite Element Structures | PAMM | 直接训练 |
| 39 | 2026 | Neural Approaches to Educational Knowledge Graph Construction, Completion, and Reasoning | 学位论文（Athabasca University） | 其他 |
| 40 | 2026 | CGSSN: A Coupled Graph Spiking Skeleton Network for Human Action Recognition Using Neuromorphic Vision Sensors | IJCNN | 直接训练的脉冲图模型（事件骨架） |
| 41 | 2026 | Knowledge Graph Prerequisite Reasoning with Spiking Neural Networks for Adaptive Intelligent Tutoring | LNCS | 直接训练 |

**局限**：OpenAlex 的被引覆盖并不完整（例如 One-Spike SNN 与 PMSM 都没有出现在这五篇的被引列表里，它们是通过 SL-01、SL-07 找到的）；STA 只有 OpenReview 记录，被引列表是用全文检索近似得到的。

**搜索记录**：SL-44～SL-48。

---

## Q4：Spiking PointCNN（Electronics 2024）怎么处理邻域？邻居固定还是按特征算？用了哪种转换方法、多少时间步、掉多少精度？

**结论**：邻居是**按三维坐标固定**的扩张 kNN，不按特征重算；它**不是**把预训练权重转过去的转换方法，而是把模块换成 LIF 脉冲层后用替代梯度从头训练；默认 T=1（消融 T=1～4）；比原版全精度 PointCNN 低约 5.5 个点（ModelNet40）和 4.4 个点（ScanObjectNN）。

**证据**：

- 邻域：4 层 Spiking X-Conv 的 (K, D) 依次为 (8,1)、(12,2)、(16,2)、(16,2)，在三维坐标上做扩张 kNN；官方代码 `knn_graph(pos, K * self.dilation, ...)` 用的是坐标 `pos`（`DeadlyAbyss/Spiking-PointCNN`）。
- 训练方式：标题里的 “Converted” 指架构层面的模块替换；第 3.2–3.3.3 节与代码 `main_ModelNet.py`（600 轮）、`main_ScanObjectNN.py`（1000 轮）都是 `F.nll_loss` 反向传播从头训练，没有权重复制、阈值平衡或校准。论文式 13 写的替代梯度是参数化 Sigmoid，代码 `SpikingLayers.py` 实现为 `tanh`，且每层末尾 `out.mean(dim=0)` 把时间维折叠回浮点，两处不一致都记录。
- 时间步与精度：第 4.2.3 节 Table 3，ModelNet40 上 T=1/2/3/4 的 OA 为 84.28% / 85.78% / 85.94% / 84.97%；主表（Table 2、4）ModelNet40 86.7% OA、ScanObjectNN 74.1% OA。原版 PointCNN 为 92.2% 与 78.5%。正文另一处写 ScanObjectNN 最高 76.10%，与 Table 4 的 74.1% 不一致，两者都记录。
- 对 Claude 草稿的影响：草稿把它称为“点云转换的先例”不成立；真正的点云转换先例是它 Table 4 引用的 Lan 等（ICCV 2023）。

**搜索记录**：SL-31～SL-33。

---

## Q5：DCGS（ICML 2025）多阈值神经元有几级阈值？官方代码支持哪些层和算子，支不支持 max 池化？许可证？论文里的能耗怎么算？

**结论**：阈值是正负对称的 2 的幂级数 $\pm\theta/2^{i-1}$（$i=1..n$），共 $2n$ 级；代码默认 $n=8$（16 级）。支持 Conv2d / Linear、矩阵乘，以及靠浮点补偿处理的 GELU / LayerNorm / Softmax；**MaxPool 只在 kernel_size = stride 时脉冲化，否则直接退回浮点 max**；不支持任何图算子或 kNN。**仓库没有许可证**。能耗按“每个非零脉冲记一次 0.9 pJ 加法、ANN 每次乘加 4.6 pJ”估算，不计多阈值比较器和浮点补偿模块的开销。

**证据**（论文 `arXiv:2503.00301v3`，标题 *Differential Coding for Training-Free ANN-to-SNN Conversion*；代码 `h-z-h-cell/ANN-to-SNN-DCGS`，2026-11-16 默认分支，行号为文件内行号）：

- 阈值级数：论文方法节定义正负 $n$ 级阈值；实验取 $n\in\{1,4,6,8\}$，主表用 $n=8$（Appendix K Table 6、Appendix L Table 7）。`main.py` 第 34 行 `--num_thresholds` 默认值 8。
- 支持的层：`converter/converter.py` 第 53–81 行按神经元名替换 ReLU（IF、LIF、MTH 及其 `_with_neg`、`_diff`、`_line` 变体）；`neurons/nonlinear_neuron.py` 中 `exp_comp_neuron`（第 116 行起）用浮点期望补偿处理 GELU / LayerNorm / Softmax，`AtNeuron`（第 196 行起）、`MulNeuron`（第 296 行起）处理首步锚定与矩阵乘。
- MaxPool：`neurons/nonlinear_neuron.py` 第 22–45 行 `maxpool_neuron.forward` 一开始就判断 `if self.maxpool.kernel_size!=self.maxpool.stride: return self.maxpool(x)`，即重叠池化（如 ResNet 首层 3×3、步长 2）直接对累积值做浮点 max；只有 k=s 时才走 `UniformUnpooling` 的脉冲化路径。`converter/converter.py` 第 39–49 行另有 `change_maxpool_before_relu`，把 MaxPool 挪到 ReLU 之前；它是否在默认流程中被调用未核实。
- 图算子：代码中没有 kNN、topk 建图或任何图消息传递。论文 Table 4 的分割实验代码不在仓库里。
- 许可证：仓库目录没有 LICENSE 文件，README 也没有许可声明，按默认即保留所有权利。
- 能耗：`statics/sop.py` 第 45、49 行只对 Conv2d、Linear 统计突触操作数；`train_val_functions/val_functions.py` 第 92 行 `energy_list.append((i+1)*0.9/4.6*tmp)`，即能耗比 = SOP × 0.9 pJ ÷（MAC × 4.6 pJ），数字取自 Horowitz 2014 的 45nm 32 位浮点加法与乘加。多级阈值比较、带符号分级值的累加、浮点补偿模块与 k≠s 的浮点 MaxPool 都没有计入。

**搜索记录**：SL-41、SL-49。

---

## Q6：ViG 每一层都重新算邻居吗？用什么距离，k 是多少，有没有扩张？聚合算子的精确公式？

**结论**：每个 Grapher 层都按当前特征重新建 kNN 图；距离是通道方向 L2 归一化后的平方欧氏距离（等价于 $2-2\cos$），Pyramid ViG 再加相对位置偏置；各向同性 ViG 的 k 从 9 线性增到 18，Pyramid ViG 固定 k=9；有扩张（dilated kNN）；聚合是 max-relative 图卷积。

**证据**（论文 `arXiv:2206.00272`；代码 `huawei-noah/Efficient-AI-Backbones/vig_pytorch`）：

- 每层重算：`gcn_lib/torch_vertex.py` 第 121–130 行 `DyGraphConv2d.forward` 每次前向都调用建图。
- 距离：`gcn_lib/torch_edge.py` 第 147–160 行先 `F.normalize(x, p=2.0, dim=1)`，再算两两平方欧氏距离；Pyramid ViG 加 `relative_pos`，并在前三个阶段对候选节点做平均池化下采样（r = 4, 2, 1, 1）。
- k 与扩张：`vig.py` 第 106–116 行 `torch.linspace(k, 2*k, n_blocks)`（k=9），扩张率 $d=\min(\lfloor i/4\rfloor+1,\ d_{\max})$（各向同性 $d_{\max}=10$，Pyramid 为 5）；取 `topk(-dist, k*d)` 后每隔 d 个取 1 个。
- 聚合公式（max-relative）：先 $\tilde X=\mathrm{BN}(\mathrm{Conv}_{1\times1}(X))$，

$$m_i=\max_{j\in\mathcal N(i)}\left(\tilde x_j-\tilde x_i\right),\qquad h_i=\mathrm{GELU}\!\left(\mathrm{BN}\!\left(\mathrm{GConv}_{1\times1}^{g=4}\left([\tilde x_i\,\|\,m_i]\right)\right)\right),$$

  再 $y_i=x_i+\mathrm{DropPath}(\mathrm{BN}(\mathrm{Conv}_{1\times1}(h_i)))$，后接 FFN（Conv1×1–BN–GELU–Conv1×1–BN，带残差）。max 是逐通道取的，差值可正可负。

**搜索记录**：SL-50。

---

## Q7：MobileViG 的图是怎么固定的？聚合算子是不是取最大值？

**结论**：图完全固定、与输入无关——每个像素连到同行、同列上每隔 K=2 个像素的位置（用张量切片拼接实现循环平移，效果等同 `torch.roll`；不计算距离、不排序）。聚合**是**逐元素取最大值（max-relative），并且以 0 为初值，相当于对“自身减邻居”的差值先与 0 取大。

**证据**（论文 CVPRW 2023，`arXiv:2307.00395`，第 5 页 Algorithm 1；代码 `SLDGroup/MobileViG/models/mobilevig.py` 第 115–143 行 `MRConv4d`）：

- 固定图（SVGA）：只在最后的低分辨率阶段使用；沿行、列分别循环平移 2、4、6……个位置取邻居（代码用 `torch.cat` 拼接切片，步长 K=2，直到该方向的边长）。
- 聚合：`x_j = x - x`（全零初始化），对每个平移量 `x_j = torch.max(x_j, x - shift(x))`（`shift` 为上述循环平移，差值方向是自身减邻居，与 ViG 的邻居减自身相反）；之后 `Conv1×1([x, x_j]) → BN → GELU`。
- 含义：没有动态拓扑，但仍有 max 聚合，所以它是“有 max、无动态拓扑”的对照教师，草稿对它的定位正确。

**搜索记录**：SL-51。

---

## Q8：Signal-SGN 和 MK-SGN 是直接训练还是转换？有没有知识蒸馏？

**结论**：两者都是**直接训练**，都不是转换。**MK-SGN 用了知识蒸馏**（教师为预训练 4 流 CTR-GCN）；**Signal-SGN 没有用知识蒸馏**。

**证据**：

- Signal-SGN（ACM MM 2025，DOI `10.1145/3746027.3755246`；`arXiv:2408.01701`）：SGD + 交叉熵从头训练 150 轮，T=16 与 16 帧骨架对齐；全文无蒸馏损失。NTU-60 四流集成 86.1% / 93.1%（Section 4.2 Table 1）。
- MK-SGN（Neurocomputing 2026，DOI `10.1016/j.neucom.2025.131796`；`arXiv:2404.10210`）：SpikingJelly 替代梯度训练 60 轮；蒸馏包括软标签蒸馏（SKD）和内层特征蒸馏（IFD，对齐教师第 5、8 层与学生第 3、5 层）；消融显示蒸馏使 NTU-60 XSub / XView 从 76.1% / 82.6% 升到 78.5% / 85.6%（Section 4.2.2、Table 6、Table 9）。

**搜索记录**：SL-23、SL-25、SL-34。

---

## Q9：有没有研究“特征被量化或加噪声后，k 近邻结果还稳不稳定”的工作？

**结论**：是。近似最近邻检索、高维数据挖掘、点云对抗稳健性三方面都有可借用的结论；但**没有找到**研究“脉冲转换或时间步离散误差下，ViG / DGCNN 逐层 kNN 邻居会不会翻转”的论文。

**证据**：

- 可直接用的几何判据（初等推导）：若每个节点特征误差 $\|\hat x_u-x_u\|_2\le\varepsilon$，任意距离的变化不超过 $2\varepsilon$；所以只要第 k 与第 k+1 近邻的距离间隔 $d_{(k+1)}(i)-d_{(k)}(i)>4\varepsilon$，节点 i 的 top-k 集合一定不变。
- RaBitQ（Gao & Long，SIGMOD 2024，`arXiv:2405.12497`）：Theorem 3.2 给出量化内积估计 $O(1/\sqrt D)$ 的无偏误差界；Theorem 3.3 与 Fig. 6 显示查询侧约 4 bit 时召回率已收敛，1 bit 时骤降；PQ / OPQ 在 MSong 上距离相对误差超过 100%、Recall@100 不到 60%（Fig. 4）。
- ScaNN（Guo 等，ICML 2020，`arXiv:1908.10396`）：Theorem 3.2–3.4 证明沿原向量方向的幅值误差对 top-k 内积排序的危害大于正交误差，比值随维数 d 线性增长。对脉冲转换而言，阈值截断与残余膜电位主要造成的正是幅值方向的误差。
- Vecchiato & Bruch（`arXiv:2602.16673`，2026）：提出邻域稳定度指标 point-NSM，在 CLIP 等视觉特征上与 HNSW 检索精度的 Spearman 相关约 0.83–0.84，可用来预判哪一层最容易翻转。
- Beyer 等（ICDT 1999）Theorem 1：距离集中时最近与最远距离之比趋于 1，近邻间隔消失（只核对了定理陈述）。
- KNN-Defense（`arXiv:2506.06906`，2025）：点云对抗扰动会打乱 DGCNN、PointNet++ 的局部邻域。

**搜索记录**：SL-72、SL-73。

---

## Q10：有没有研究“取最大值这类操作对输入误差的敏感度”的结论可以借用？

**结论**：是。三类结论可借：（1）单次取 max 对输入的无穷范数误差是 1-Lipschitz 的，数值误差不放大，但胜者下标可能翻转；（2）对带零均值噪声的候选取 max 会系统性高估，且邻域越大越严重；（3）脉冲 max 池化的已有分析都假设非负、率编码、较长时间步，不能直接套到 ViG 的有符号相对差上。

**证据**：

- 1-Lipschitz：$|\max_j\hat x_j-\max_j x_j|\le\|\hat x-x\|_\infty$；当 $\|\hat x-x\|_\infty<\tfrac12(x_{(1)}-x_{(2)})$ 时胜者不变（初等推导）。EvGNN（IEEE TCASAI 2025）在 max 聚合事件图网络上做 INT8 训练后量化，N-CARS 精度从 87.9% 到 87.8%（第 6 页 Table IV）；对比 TopGQ（DAC 2026）Table 6 中度归一化求和的 GCN 在 Reddit 上朴素 INT4 量化跌到 3.79%。
- 最大化偏置：由 Jensen 不等式 $\mathbb E[\max_j(x_j+\eta_j)]\ge\max_j x_j$；k 个独立 $\mathcal U[-\epsilon,\epsilon]$ 误差的期望最大值为 $\frac{k-1}{k+1}\epsilon$（k=9 时 0.8ε，k=20 时约 0.905ε）。出处：Thrun & Schwartz（1993）、van Hasselt（NeurIPS 2010，Double Q-learning），经 *Self-correcting Q-Learning*（`arXiv:2012.01100`）交叉核对；“选胜者与取数值解耦”可消除偏置。
- 脉冲 max 池化：Li & Zeng（IJCAI 2022）第 4 页式 (11) 证明逐时刻放行脉冲时输出发放率落在 $[\max,\ \mathrm{sum}]$ 之间，并提出侧抑制池化；Rueckauer 等（Front. Neurosci. 2017）用滤波发放率做门控，需要数百步；Gaurav 等（IJCNN 2022）的 Loihi 电路需 T=50～120；Bu 等（CVPR 2025）第 5 页 Theorem 2 利用 $\max_j\mathrm{ReLU}(z_j)=\mathrm{ReLU}(\max_j z_j)$（$\max z>0$）把 MaxPool 挪到脉冲神经元之前、对膜前连续值用比较器取最大——它是**交换层序**，不是“改成神经元层”。ViG 的 $\max_j(\tilde x_j-\tilde x_i)$ 作用于有符号差值、之后没有 ReLU，这一交换条件不满足。

**搜索记录**：SL-74、SL-75。

---

## Q11：ICCV 2027、NeurIPS 2027、ICML 2027、CVPR 2028、ECCV 2028、AAAI 2028 的下一次截稿日期

**结论**：只有 ICML 2027 已在官网公布截稿日期（摘要 2027-01-14 AoE，全文 2027-01-22 AoE）；其余五个**未公布**。

**证据**（2026-11-16 直接访问官网）：

| 会议 | 截稿日期 | 官网信息 |
| --- | --- | --- |
| ICCV 2027 | 未公布 | `iccv.thecvf.com/Conferences/2027/Dates` 只列出 2027-10-03～04 研讨会与教程、10-05～07 主会（香港 HKCEC）；`/CallForPapers` 返回 404 |
| NeurIPS 2027 | 未公布 | `neurips.cc/Conferences/FutureMeetings` 只写 “2027 -- Europe”；`/Conferences/2027` 返回 404 |
| ICML 2027 | 摘要 2027-01-14（AoE）；全文 2027-01-22（AoE） | `icml.cc/Conferences/2027/Dates` 与 `/CallForPapers`；会议 2027-07-31～08-06，里约热内卢 |
| CVPR 2028 | 未公布 | `cvpr.thecvf.com/Conferences/2028` 返回 404（官网当前为 CVPR 2027） |
| ECCV 2028 | 未公布 | `eccv.ecva.net/Conferences/2028` 返回 404（官网当前为 ECCV 2026） |
| AAAI 2028 | 未公布 | `aaai.org/conference/aaai/` 的 Future Conferences 只列到 AAAI-27；`/aaai-28/` 返回 404 |

**搜索记录**：SL-76。
