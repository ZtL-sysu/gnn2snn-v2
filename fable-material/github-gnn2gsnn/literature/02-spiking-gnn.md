# 脉络二：脉冲图神经网络（2020–2026）

本文件整理所有将脉冲神经网络（SNN）与图结构、3D 点云邻域图、人体骨架图、图像超像素/特征图及事件流空间图相结合的代表性工作，严格区分**“ANN→SNN 转换”**（含免训练转换、改池化重训后转换、量化训练后转换、硬阈值编译映射）与**“从头直接训练”**（含结合教师网络知识蒸馏的直接训练），并覆盖评测基准与综述现状。

## 路线与模态总览表

| 训练范式分类 | 代表工作（年份） | 图模态 / 任务类型 | 是否需要重训源网络或从头训练 |
| --- | --- | --- | --- |
| **真免训练 ANN→SNN 权重转换** | `One-Spike SNN`（2025） | 文本词—文档静态图分类（Text GCN） | **否**（直接转换预训练 ReLU Text GCN） |
| **改池化算子重训源 ANN + 训练后校准转换** | `Adaptive Calibration`（2025）、`Lan 等 3D/2D 转换 SNN`（2023） | 3D 点云分类与部件分割（PointNet、PointNet++） | **部分**（须先把源 ANN 的 `MaxPool` 改为 `AvgPool` 重训，再做免训练校准转换） |
| **量化感知重训源 ANN + 无损映射转换** | `PMSM`（2025） | 事件触觉传感器图分类（3 层 GCN，`EvTouch-Objects`） | **部分**（须用极性量化激活 PQA 训练源 GCN，再零误差映射为 $T=1$ SNN） |
| **硬阈值 STE 训练 + 零失配神经形态编译** | `DeNA-SGN`（2026） | 通用静态图节点分类（SuperNeuroMAT 硬件电路） | **是**（训练期即采用硬阈值与二值特征聚合，非转换现成连续 GNN） |
| **误标为“转换”、实为替代梯度从头直接训练** | `Spiking PointCNN`（2024） | 3D 点云分类（三维坐标膨胀 $k$-NN + X-Conv） | **是**（标题含 Converted，但实为模块化替换为 LIF 后用替代梯度从头 SGD 训练） |
| **视觉图与事件图：从头直接训练（含蒸馏）** | `AMP2`（2026）、`MK-SGN`（2026，含 CTR-GCN 蒸馏）、`Signal-SGN++`（2025）、`Signal-SGN`（2025）、`Spiking Point Mamba`（2025）、`Spiking Point Transformer`（2025）、`NI-SGCN`（2025，脉冲动态 EdgeConv）、`E-3DSNN`（2025）、`SpikePoint`（2024）、`P2SResLNet`（2024，含 KPConv-SNN 转换基线）、`Spiking PointNet`（2023）、`SpikingGCN`（2022，含 MNIST 超像素图）、`TactileSGNet`（2020） | 骨架动作识别、3D 点云分类/检测/分割/去噪、图像超像素与 $k$-NN 特征图、事件相机与触觉图 | **是**（全部从头训练；仅 `MK-SGN` 引入预训练 CTR-GCN 做知识蒸馏教师） |
| **通用静态/动态图与评测基准：从头直接训练** | `SpikeGraphormer`（2026，含 Mini-ImageNet $k$-NN 图像图）、`GT-SNT`（2026）、`SGNNBench`（2025，评测基准）、`MSG`（2024）、`Loihi Transductive SGNN`（2024）、`SpikeGCL`（2024）、`Dy-SIGN`（2024）、`DRSGNN`（2024）、`SpikeNet`（2023）、`GC-SNN 与 GA-SNN`（2021） | 引文/共购/异配静态图、大规模时序动态图、图对比学习、图 Transformer 分词 | **是**（`SGNNBench` 评测的 9 种 SGNN 全部为从头直接训练，未收录任何转换法） |

> **关于综述文献的核查说明**：截至 2026-11-16，经 `arXiv API` 与 `Crossref REST API` 专项检索（见 [search-log.md](search-log.md) 条目 `SL-30`），学术界**尚未发表专门针对“脉冲图神经网络（Spiking GNN）”的独立综述论文**；目前承担系统性梳理与横向对比职能的核心文献为 NeurIPS 2025 数据集与基准轨道论文 `SGNNBench`。

---

## 逐篇文献核查记录（按发表年份从新到旧排列）

### DeNA-SGN（2026）

- 标题：DeNA-SGN: Deterministic Neighborhood Aggregation for Verified Spiking Graph Inference
- 作者：Sanjay Das, Xi Zhang, Emily Herron, Prasanna Date
- 发表处：arXiv（2026 年 11 月预印本，`arXiv:2611.14775`）
- 链接：https://arxiv.org/abs/2611.14775 ；代码：未公开
- 核对深度：全文
- 一句话：先在二值化节点特征上用布尔“或（OR）”与“多数表决（Majority）”确定性邻域聚合配合直通估计器（STE）训练硬阈值分类器，再零决策失配地编译为 7 个时间拍的 SuperNeuroMAT 脉冲神经形态电路。
- 解决的难点：消除了脉冲图推理中的随机采样编码噪声与软硬模型转换失配，并用共享抑制控制神经元精确实现与节点度数相关的多数表决聚合。
- 留下的局限：并非对现成浮点 ReLU/GELU 视觉图骨干做免训练转换——节点特征必须预先二值化，且分类器在训练期就使用硬阈值；同时在 `ogbn-arxiv` 上揭示出若硬件为空闲脉冲神经元持续供能，脉冲电路总能耗反而高于浮点基线。
- 是否需要训练或微调：是（需用直通估计器 STE 从头训练带符号权重的硬阈值分类器，随后确定性编译为脉冲电路）
- 涉及的算子：离线特征二值化投影、无参数闭邻域 1 跳 OR 聚合（$c_{if}\ge 1$）、由至多 $\lfloor\log_2 \Delta\rfloor$ 个共享二进制抑制控制神经元（权重 $-2^b$）实现的度数自适应 tie-inclusive Majority 聚合（$c_{if}\ge \lceil k_i/2\rceil$）、无偏置硬阈值阶跃激活与验证集输出阈值校准
- 关键数字：在 7 个图数据集共 41 个映射分类器、累计 **296,577** 次原生测试查询上，SuperNeuroMAT 脉冲仿真（7 ticks，第 4 tick 输出）与导出硬阈值 ANN 实现 **0 决策与脉冲模式失配**（Abstract 及 Section VI-A Table VI，第 8 页）；`Amazon Photo` 达 **83.93±0.93%**（电流读出）/ **71.61±2.22%**（严格单脉冲读出），`Wiki-CS` 达 **75.99±0.54%**（电流）/ **67.36±1.66%**（严格脉冲），`Cora` 达 **74.90±2.67%**（电流）/ **68.57±2.94%**（严格脉冲，同划分 GAT 参考为 `81.47±0.90%`，Section VI-A/C Table V & VIII，第 8–9 页）；按解析能耗模型 $E_{\text{ANN}}=4.6M+0.9G$ 对照 $E_{\text{SNN}}=0.9\varepsilon+0.1s+0.7N_{\text{dep}}$（Section IV-J 公式 14），`Cora` 与 `CiteSeer` 的建模 ANN/SNN 能耗比为 **102.35×** 与 **171.03×**，`Amazon Photo` 与 `Wiki-CS` 为 **5.41×** 与 **4.24×**，但在 `ogbn-arxiv`（电流 **58.36±0.69%** / 严格脉冲 **46.20±0.99%**，部署神经元 `339,135` 个）上因空闲神经元每拍时钟更新开销主导，能耗比跌至 **0.53×**（ANN `136.9 nJ` vs SNN `257.71 nJ`，即 SNN 反而更耗电，Section VI-E Table X，第 10–11 页）。
- 与本方向的关系：可借用的技术（其度数自适应抑制阈值构造以及“空闲神经元时钟功耗会吞噬脉冲收益”的定量反例对 M4 能耗盈亏分析极具参考价值）
- 核对人与日期：Gemini，2026-11-16

### SpikeGraphormer（2026）

- 标题：SpikeGraphormer: A High-Performance Graph Transformer with Spiking Graph Attention
- 作者：Yundong Sun, Dongjie Zhu, Yansong Wang, Zhaoshuo Tian, Ning Cao, Gregory O'Hare
- 发表处：Information Sciences（2026 年第 123828 号论文，DOI: `10.1016/j.ins.2026.123828`；预印本于 2024 年 3 月发布于 `arXiv:2403.15480`，常简称 `SpikeGT`）
- 链接：https://doi.org/10.1016/j.ins.2026.123828 （预印本：https://arxiv.org/abs/2403.15480 ）；代码：https://github.com/PHD-lanyu/SpikeGraphormer
- 核对深度：全文
- 一句话：将稀疏全精度 GCN 局部分支与去除 Softmax 的线性复杂度脉冲图注意力（SGA）全局分支并联，用于大规模图节点分类及基于 $k$-NN 构图的图像与文本分类。
- 解决的难点：用二值脉冲 $Q, K, V$ 的按列掩码与稀疏加法替代 $O(N^2)$ 点积 Softmax 注意力，把全局节点交互复杂度降为 $O(ND)$，显著降低显存开销（较标准自注意力省 10–20 倍）。
- 留下的局限：需要从头端到端训练；并联的局部图分支仍是常规浮点稀疏 GCN（公式 6），并非纯事件驱动网络；在图像任务上只是对 CNN 提取好的样本级特征建样本间 $k$-NN 图，而非图像块级视觉图骨干转换。
- 是否需要训练或微调：是（从头端到端梯度训练）
- 涉及的算子：LIF 脉冲神经元、无 Softmax 的脉冲图注意力（SGA，$Q^i((K^i)^\top \odot V^i)$）、稀疏浮点 GCN 聚合、样本特征 $k$-NN 建图（$k \in \{5,10,15,20\}$）
- 关键数字：`Cora` **84.8±0.7%**、`Chameleon` **44.8±4.2%**、`Squirrel` **42.6±2.4%**（Section 5.2 Table 2）；大规模图 `OGB-Proteins` ROC-AUC **79.62±0.90%**（显存 3.7 GB，Table 3），`Amazon2M` **88.12±0.06%**（显存 4.8 GB，Table 4）；在 `Mini-ImageNet` 图像样本 $k$-NN 图分类上达 **86.90±0.39%**（$k=15$，无图退化版亦达 86.84%，Section 5.5 Table 6）。
- 与本方向的关系：可借用的技术（SGA 去掉 Softmax 的线性脉冲图注意力设计，以及在 `Mini-ImageNet` 上建 $k$-NN 图但发现模型几乎不依赖显式图边的现象值得对照）
- 核对人与日期：Gemini，2026-11-16

### AMP2（2026）

- 标题：Activation-wise Propagation: A One-Timestep Strategy for Spiking Neural Networks
- 作者：Jian Song, Xiangfei Yang, Shangke Lyu, Donglin Wang
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2026，Vol. 40, Issue 3，DOI: `10.1609/aaai.v40i3.37187`；预印本 `arXiv:2502.12791`）
- 链接：https://doi.org/10.1609/aaai.v40i3.37187 （预印本：https://arxiv.org/abs/2502.12791 ）；代码：https://github.com/LockFaiz/AMP2
- 核对深度：全文
- 一句话：提出单时间步（$T=1$）逐激活混合传播策略，在训练与推理中通过神经元子群耦合消除多时间步循环展开，并在 2D 图像与 **3D 点云脉冲版 PointNet++** 上验证了单步推理精度。
- 解决的难点：把直接训练脉冲网络在 3D 点云分层架构（PointNet++）上的推理时间步压缩到 $T=1$，大幅降低多步展开的显存与延迟成本。
- 留下的局限：完全依赖从头替代梯度训练（非 ANN→SNN 转换），无法直接复用已训练好的浮点 PointNet++ 权重。
- 是否需要训练或微调：是（从头直接训练）
- 涉及的算子：单时间步逐激活混合传播脉冲单元、PointNet++ 最远点采样（FPS）、球查询（Ball Query）局部邻域分组、逐点 MLP 与局部对称池化
- 关键数字：在 `Spiking PointNet++` 骨干、单时间步 **$T=1$** 下，`ModelNet40` 总体精度（OA）达 **91.63%**、类别平均精度（mAcc）达 **89.02%**；真实扫描点云 `ScanObjectNN` 达 **78.81% OA / 76.23% mAcc**，显著超过同表引用的转换基线 `KPConv-SNN`（$T=40$ 时仅 70.50% / 43.90%）及多步直接训练基线（Section 4 Table 5）。
- 与本方向的关系：直接相关（提供了 `PointNet++` 在 $T=1$ 直接训练路线下的强脉冲基线，可与 Lan et al. ICCV 2023 的改池化转换版 `PointNet++` 形成鲜明对照）
- 核对人与日期：Gemini，2026-11-16

### GT-SNT（2026）

- 标题：GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization
- 作者：Huizhe Zhang, Jintang Li, Yuchang Zhu, Huazhen Zhong, Liang Chen
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2026；预印本 `arXiv:2504.11840`）
- 链接：https://arxiv.org/abs/2504.11840 ；代码：https://github.com/Zhhuizhe/GT-SNT
- 核对深度：全文
- 一句话：不把脉冲神经元当作省电激活替代品，而是用作“序列到离散词元”的无码本节点分词器（SNT），将多跳随机特征传播序列压缩为脉冲计数码本，从而驱动线性时间复杂度的图 Transformer。
- 解决的难点：解决了传统矢量量化（VQ）图模型中的“码本坍缩（codebook collapse）”与最近邻查表开销，实现 100% 有效码字利用率与 $O(NBd_v)$（$B \ll N$）线性全局注意力。
- 留下的局限：仅脉冲分词支路采用脉冲神经元，主干辅助 MPNN 与 `softmax` 注意力仍包含浮点线性层与指数运算；且必须从头训练。
- 是否需要训练或微调：是（采用替代梯度从头端到端训练）
- 涉及的算子：$T$ 步图传播算子、IF/LIF/PLIF 脉冲计数累加（$\hat{\mathbf{S}} = \sum_{t=0}^T \mathbf{S}^t$）、动态去重码本重构、节点—码字 Softmax 注意力（CGSA）、单层无归一化 GCN
- 关键数字：9 个数据集上平均超越已有脉冲图基线 **3.9%**（Section 5.1）：`Cora` **84.7±0.8%**、`CiteSeer` **74.0±0.5%**、`PubMed` **80.6±0.4%**、异配图 `Actor` **39.1±0.2%**、`ogbn-arxiv` **72.4±0.3%**、`ogbn-products` **74.8±0.4%**（Table 2）；在 `Coauthor-Physics` 上推理延迟仅 **0.02 s**（比图 Transformer 平均速度快 **130×**，理论能耗 **0.37 J**，Table 3）。
- 与本方向的关系：可借用的技术（利用多步脉冲计数构造局部拓扑感知离散桶的思想，对视觉图动态选邻居的粗粒度分桶有启发）
- 核对人与日期：Gemini，2026-11-16

### MK-SGN（2026）

- 标题：MK-SGN: A Spiking Graph Convolutional Network with Multimodal Fusion and Knowledge Distillation for Skeleton-based Action Recognition
- 作者：Naichuan Zheng, Hailun Xia, Zeyu Liang, Yuchen Du
- 发表处：Neurocomputing（2026 年 1 月正式发表，DOI: `10.1016/j.neucom.2025.131796`；预印本于 2024 年 4 月发布于 `arXiv:2404.10210`）
- 链接：https://doi.org/10.1016/j.neucom.2025.131796 （预印本：https://arxiv.org/abs/2404.10210 ）；代码：https://github.com/zhengnaichuan2022/MK-SGN
- 核对深度：全文
- 一句话：首个面向人体骨架动作识别的脉冲图卷积网络，融合关节/骨骼/关节运动/骨骼运动四模态脉冲编码、空间—时间脉冲图自注意力，并用预训练 4 流 CTR-GCN 教师对 6 层脉冲学生网络实施软标签与中间层特征知识蒸馏。
- 解决的难点：缓解了二值脉冲在三维骨架序列上的表征退化，通过跨架构知识蒸馏将 10 层浮点 CTR-GCN 的高阶时空语义迁移到轻量级 6 层脉冲图卷积网络中。
- 留下的局限：属于**从头直接训练 + 跨架构知识蒸馏**（需训练 60 轮），完全不是免训练权重转换；且与教师 CTR-GCN 相比在 `NTU-RGB+D 120` 等大类别集上仍有逾 20 个百分点的精度差距。
- 是否需要训练或微调：是（使用 SGD 与 SpikingJelly 替代梯度从头训练 60 轮，并依赖预训练好的 4 流 CTR-GCN 教师模型提供蒸馏监督）
- 涉及的算子：脉冲多模态融合（SMF，Conv2D+BN+LIF）、脉冲自适应图卷积（SA-GC，固定骨架邻接 + 可学习邻接矩阵）、脉冲时空图自注意力（STGA，无 Softmax 脉冲矩阵乘）、特征变换模块（FTM）与余弦相似度内层特征蒸馏（IFD）
- 关键数字：脉冲步 $S=4$ 下，`NTU-RGB+D`（60 类）XSub / XView 精度达 **78.5% / 85.6%**，单样本理论能耗仅 **0.596 mJ**（较教师 CTR-GCN 的 36.25 mJ 降低 **98.36%**，Section 4.2 Table 6）；`NTU-RGB+D 120` XSub / XSet 为 **67.8% / 69.5%**，`NW-UCLA` 为 **92.3%**；知识蒸馏消融（Section 4.2.2 Table 6 & Table 9）显示，加入软标签蒸馏（SKD）与内层特征蒸馏（IFD，对齐教师第 5、8 层与学生第 3、5 层）使 6 层多模态模型在 NTU-60 XSub/XView 上从 **76.1% / 82.6%** 提升至 **78.5% / 85.6%**（+2.4 / +3.0 个百分点）。
- 与本方向的关系：直接相关（视觉骨架图上的代表性脉冲工作；回答定向问题 Q2 与 Q8 的核心证据：并非转换，但明确使用了 CTR-GCN→SGN 知识蒸馏）
- 核对人与日期：Gemini，2026-11-16

### Signal-SGN++（2025）

- 标题：Signal-SGN++: Topology-Enhanced Time–Frequency Spiking Graph Network for Skeleton-Based Action Recognition
- 作者：Naichuan Zheng, Hailun Xia, Weiyi Li, Yuchen Du
- 发表处：arXiv（2025 年 12 月预印本，`arXiv:2512.22214`，投稿 *IEEE Transactions on Neural Networks and Learning Systems*）
- 链接：https://arxiv.org/abs/2512.22214 ；代码：未公开
- 核对深度：全文
- 一句话：在 `Signal-SGN` 基础上引入融合自适应骨架拓扑先验与 QK 特征相似度的 **Top-$k$ 选邻居 + 通道循环移位注意力（TSSA）**，以及基于勒让德多项式小波的低频拓扑感知时频融合模块（TATF）。
- 解决的难点：把骨架关节脉冲自注意力和跨频段融合的复杂度从 $O(V^2 d)$ 降为 $O(k V d)$，并用无乘法的左右循环通道移位增强局部动态关节交互表达。
- 留下的局限：仍需从头训练 150 轮；内部包含关节维度的复数快速傅里叶变换（FFT）与小波分解插值，不兼容纯加法脉冲硬件直接免训练迁移。
- 是否需要训练或微调：是（使用 SGD 与替代梯度从头直接训练 150 轮，无知识蒸馏）
- 涉及的算子：骨架脉冲编码（每通道 min-max 归一化 + Conv1D + BN + LIF）、1D 脉冲图卷积（1D-SGC）、Top-$k$ 拓扑—相似度融合选邻居 + 通道循环移位（TSSA，$k=8$）、FFT 频域脉冲复卷积（FSC）、勒让德多小波分解与低频 Top-$k_{\text{topo}}$ 均值聚合（TATF，$k_{\text{topo}}=6$）
- 关键数字：16 帧输入（$T=16$）下，单流（Joint）在 `NTU-RGB+D` Xs/Xv 达 **81.3% / 88.4%**（能耗 **0.369 mJ**，注：消融表 Table II 中 D3 配置报告 Xs 为 82.5%，与主表 Table I 单流 81.3% 存在文内表间微小出入，两处均予记录）；四流集成（4-ensemble）在 `NTU-RGB+D` Xs/Xv 达 **87.2% / 94.5%**（能耗 **1.476 mJ**），`NTU-RGB+D 120` Xs/Xt 达 **76.5% / 78.9%**，`NW-UCLA` 达 **96.3%**（Section V-C Table I）。
- 与本方向的关系：可借用的技术（TSSA 将静态拓扑先验矩阵与动态脉冲特征相似度加权后再取 Top-$k$ 邻居的设计，对缓解动态选邻居翻转有直接借鉴意义）
- 核对人与日期：Gemini，2026-11-16

### SGNNBench（2025）

- 标题：SGNNBench: A Holistic Evaluation of Spiking Graph Neural Network on Large-scale Graph
- 作者：Huizhe Zhang, Jintang Li, Yuchang Zhu, Huazhen Zhong / Li Kuang, Liang Chen（注：arXiv 元数据末位作者列为 Li Kuang，NeurIPS 检索页含 Huazhen Zhong，同属中山大学陈亮团队）
- 发表处：Advances in Neural Information Processing Systems Datasets and Benchmarks Track（NeurIPS 2025；预印本 `arXiv:2509.21342`）
- 链接：https://arxiv.org/abs/2509.21342 ；代码：https://github.com/Zhhuizhe/SGNNBench
- 核对深度：全文
- 一句话：首个面向大规模图的脉冲图神经网络统一评测基准，在 18 个同配、异配与大规模动态图数据集上系统对比了 9 种直接训练 SGNN 与 7 种全精度 GNN 的精度、显存、延迟和理论能耗。
- 解决的难点：统一了此前各篇 SGNN 各自为政的数据划分与能耗估算口径，首次定量揭示了现有 SGNN 在深层异配图上的严重退化以及部分 SGNN 能耗反超全精度 GNN 的隐患。
- 留下的局限：**9 种被测模型全部为从头直接训练的通用图节点分类模型**，既未收录任何 ANN→SNN 图转换方法，也未包含任何视觉图（点云、骨架、超像素/图像块图）任务。
- 是否需要训练或微调：是（基准内所有被测脉冲图模型均为从头训练）
- 涉及的算子：涵盖伯努利/泊松/直接线性脉冲编码，IF/LIF/PLIF 神经元，GCN/GAT/SAGE/SGC/图 Transformer/流形图卷积等消息传递算子
- 关键数字：评测 **9 种 SGNN**（`GC-SNN`, `GA-SNN`, `SpikingGCN`, `SpikeNet`, `SpikeGCL`, `SpikeGT`, `MSG`, `DRSGNN`, `SiGNN`）与 **7 种全精度 GNN**，横跨 **18 个数据集**（9 个静态同配图、6 个静态异配图、3 个百万边级动态图 `DBLP`/`Tmall`/`Patent`，Section 3.1 Table 1；注：正文 3.1 节文字曾顺带提及 Actor/Deezer，实际评测以 Table 1 的 18 个数据集为准）；在异配长程依赖图 `Roman-Empire` 上 9 种 SGNN 相对全精度 GNN 平均暴跌 **-31.27%**（最佳 SGNN 仅 51.32% vs GraphGPS 82.59%，Table 3）；能耗表（Table 6）揭示 `SiGNN` 在多个静态图上理论能耗甚至**高于**全精度 GCN/SGC。
- 与本方向的关系：直接相关（一方面印证主流图脉冲社区完全忽视了“免训练转换”与“视觉图”，另一方面其揭示的“编码方式与未脉冲化层会导致 SGNN 能耗反超 ANN”直接支撑本项目的公平能耗盈亏分析）
- 核对人与日期：Gemini，2026-11-16

### Signal-SGN（2025）

- 标题：Signal-SGN: A Spiking Graph Convolutional Network for Skeleton Action Recognition via Learning Temporal-Frequency Dynamics
- 作者：Naichuan Zheng, Yuchen Du, Hailun Xia, Zeyu Liang
- 发表处：Proceedings of the 33rd ACM International Conference on Multimedia（ACM MM 2025，DOI: `10.1145/3746027.3755246`；预印本于 2024 年 8 月发布于 `arXiv:2408.01701`）
- 链接：https://doi.org/10.1145/3746027.3755246 （预印本：https://arxiv.org/abs/2408.01701 ）；代码：https://github.com/zhengnaichuan2022/Signal-SGN
- 核对深度：全文
- 一句话：把人体骨架动作序列的自然帧时间轴直接映射为脉冲神经网络的时间步（$T=16$），将脉冲特征视为多维离散信号，结合一维脉冲图卷积、FFT 频域脉冲卷积与离散小波多尺度融合直接训练骨架识别模型。
- 解决的难点：避免了在静态化骨架张量外额外堆叠脉冲时间步维度造成的冗余，利用频域与多尺度小波特征显著拉高了无蒸馏脉冲骨架网络的识别精度。
- 留下的局限：完全依赖从头直接训练（150 轮），无免训练转换机制；频域 FFT 与小波变换引入非纯脉冲算子开销。
- 是否需要训练或微调：是（使用 SGD + 交叉熵损失从头训练 150 轮，**未使用知识蒸馏**）
- 涉及的算子：骨架坐标归一化脉冲编码、1D 脉冲图卷积（1D-SGC）、脉冲自注意力（SSA）、快速傅里叶变换频域脉冲卷积（FSC）、离散小波变换多尺度融合（MWTF）
- 关键数字：$T=16$（对应 16 帧骨架输入）下，`NTU-RGB+D` Xs/Xv 单流（Joint）精度为 **80.5% / 87.7%**（理论能耗 **0.372 mJ**），双流（Bone+Joint）为 **82.5% / 89.2%**（0.744 mJ），四流集成为 **86.1% / 93.1%**（**1.488 mJ**，比 ST-GCN 的 16.01 mJ 低一个数量级以上）；`NTU-RGB+D 120` Xs/Xt 四流集成为 **75.3% / 77.9%**，`NW-UCLA` 达 **95.9%**（Section 4.2 Table 1）。
- 与本方向的关系：直接相关（回答定向问题 Q2 与 Q8 的核心文献：直接训练、无蒸馏；其“把流式序列帧直接对齐脉冲步 $T$”的做法与 M3 流式图处理思路相通）
- 核对人与日期：Gemini，2026-11-16

### Spiking Point Mamba（2025）

- 标题：Efficient Spiking Point Mamba for Point Cloud Analysis
- 作者：Peixi Wu, Bosong Chai, Menghua Zheng, Wei Li, Zhangchi Hu, Jie Chen, Zheyu Zhang, Hebei Li, Xiaoyan Sun
- 发表处：IEEE/CVF International Conference on Computer Vision（ICCV 2025，DOI: `10.1109/ICCV51701.2025.02449`；预印本 `arXiv:2504.14371`）
- 链接：https://doi.org/10.1109/ICCV51701.2025.02449 （预印本：https://arxiv.org/abs/2504.14371 ）；代码：https://github.com/PeppaWu/SPM
- 核对深度：全文
- 一句话：提出首个面向 3D 点云分析的脉冲 Mamba 框架（SPM），结合分阶段动态最远点采样编码（HDE）、基于 $k$-NN 邻域与逐元素最大池化的脉冲嵌入层、沿时间步双向翻转的脉冲状态空间块，以及非对称 SNN 编码器—ANN 解码器掩码预训练。
- 解决的难点：突破了此前点云脉冲网络难以建模长程点序列依赖的瓶颈，并把点云脉冲网络拓展到部件分割（ShapeNetPart）与自监督掩码预训练。
- 留下的局限：需在双卡 RTX 4090 上从头迭代训练 300 轮；状态空间模块内部仍保留部分浮点矩阵乘法（公式 14 中 $E_{\text{MAC}} \cdot FL_1^n$），且 $k$-NN 仅在输入三维坐标层做一次静态分组，未涉及深层特征动态建图的转换。
- 是否需要训练或微调：是（从头直接训练 300 轮，或先做掩码自监督预训练再微调 150 轮）
- 涉及的算子：分层动态 FPS 采样编码（HDE）、三维坐标 $k$-NN 邻域分组、逐元素最大池化（EMP）与逐元素平均池化（EAP，用于门控支路防脉冲相乘信息丢失）、时间维翻转双向状态空间模型（SSM）、LIF 神经元
- 关键数字：$T=4$ 从头训练下，`ScanObjectNN` 三种变体 `OBJ-BG` / `OBJ-ONLY` / `PB-T50-RS` 总体精度分别达 **90.2% / 89.5% / 84.2%**（较 `Spiking Point Transformer` 分别提升 **+7.4 / +6.1 / +6.2** 个百分点），`ModelNet40` 无投票/有投票达 **92.3% / 92.7%**（Section 5.2 Table 1）；叠加掩码预训练后 `PB-T50-RS` 达 **86.5%**、`ModelNet40` 达 **93.1%**（Section 5.2.1）；`ShapeNetPart` 部件分割达 **82.29% Cls. mIoU / 84.8% Ins. mIoU**（Table 3）；$T=4$ 能耗 **5.4 mJ**（比 PointMamba 的 18.9 mJ 低 **3.5×**，$T=1$ 时 1.5 mJ 低 **12.6×**，Table 6）。
- 与本方向的关系：背景（代表 2025 年直接训练 3D 点云脉冲网络的精度上限参照；其发现“脉冲门控逐元素相乘会破坏词元关系、改用均值池化门控可挽回精度”的消融结论可资参考）
- 核对人与日期：Gemini，2026-11-16

### PMSM（2025）

- 标题：Quantization Meets Spikes: Nearly Lossless Conversion at the First Timestep via Polarity Multi-Spike Mapping
- 作者：Hangming Zhang, Zheng Li, Chenxiang Ma, Huajin Tang, Long Cheng, Kay Chen Tan, Qiang Yu
- 发表处：arXiv（2025 年 8 月预印本，`arXiv:2508.14520`）
- 版本说明（Claude 审查时补充，2026-11-17）：arXiv 首版（v1）标题为 *Quantization Meets Spikes: Lossless Conversion in the First Timestep via Polarity Multi-Spike Mapping*，作者为 Hangming Zhang、Zheng Li、Qiang Yu 三人，摘要只提 CIFAR 与 ImageNet。本条目的标题、七位作者和 GCN 触觉图实验对应的是后续版本；审查时 arXiv 限流，未能独立打开后续版本确认 GCN 实验，引用前请写明版本号并复核。
- 链接：https://arxiv.org/abs/2508.14520 ；代码：未公开
- 核对深度：全文
- 一句话：通过极性量化激活函数（PQA）训练源网络，建立量化整数激活与带正负极性多脉冲增广积分发放神经元（AIF）之间的精确双射映射，实现**第一个时间步（$T=1$）零精度损失转换**，并在 **3 层图卷积网络（GCN）事件触觉图分类**上完成实验验证。
- 解决的难点：解决了传统 ANN→SNN 转换无法处理负激活值、以及常规量化转换在极低时间步（尤其 $T=1$）下因残余膜电位导致精度崩溃的难题。
- 留下的局限：**不是免训练转换**——必须用自定义 PQA 激活函数（可学习正负缩放因子 $\alpha^*, \beta^*$）重新训练源 GCN 200 轮；且硬件端需要支持正负多级整数脉冲（分级脉冲）的神经形态芯片。
- 是否需要训练或微调：部分（转换映射本身无需微调，但**必须先用 PQA 极性量化激活函数从头训练或重训源 ANN/GCN**）
- 涉及的算子：极性量化激活（PQA，含正负双分支阶梯量化）、增广积分发放神经元（AIF，初始膜电位偏置 $\frac{1}{2}\vartheta^l_{\text{SNN}}+\epsilon$，软复位，支持正负多脉冲发放）、3 层标准图卷积（GCN，39 节点触觉图邻接聚合）
- 关键数字：在 36 类事件触觉图数据集 `EvTouch-Objects`（沿用 TactileSGNet 的 39 触觉单元空间图，构建 3 层 PQA-GCN，$L=8, \vartheta_0=8$，Appendix Table VII）上，源 PQA-GCN 精度为 **90.97%**，转换后 SNN 在 **$T=1$** 即取得 **90.97%**（**转换损失 0.00%**，突触操作数 0.41M SOPs，理论能耗仅 **0.37 $\mu$J**，Section 4.2 Table IV & Table V）；作为对比，常规单极性量化转换（QA）在 $T=1$ 崩溃至 **2.78%**（$T=64$ 才恢复至 86.11%），直接训练基线 `TactileSGNet` 在 $T=250$ 为 **89.44%**（Table IV & V）。
- 与本方向的关系：直接相关（回答定向问题 Q1 的关键证据之一：明确对 3 层 GCN 做了 ANN→SNN 权重转换并在 $T=1$ 无损保真，但前提是重训量化源网络且图结构固定、聚合线性）
- 核对人与日期：Gemini，2026-11-16

### Spiking Point Transformer（2025）

- 标题：Spiking Point Transformer for Point Cloud Classification
- 作者：Peixi Wu, Bosong Chai, Hebei Li, Menghua Zheng, Yansong Peng, Zeyu Wang, Xuan Nie, Yueyi Zhang, Xiaoyan Sun
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2025；预印本 `arXiv:2502.15811`）
- 链接：https://arxiv.org/abs/2502.15811 ；代码：https://github.com/PeppaWu/SPT
- 核对深度：全文
- 一句话：提出面向 3D 点云分类的脉冲点云 Transformer（SPT），用队列偏移动态点编码（Q-SDE）激发时间步间差异，并设计基于局部 $k$-NN 邻域向量减法注意力的双分支脉冲点注意力模块。
- 解决的难点：缓解了静态点云直接重复输入导致的脉冲序列完全雷同问题，并将 Point Transformer 的局部邻域向量自注意力改造为全脉冲驱动版本。
- 留下的局限：从头直接训练（300 轮），不支持预训练浮点 Point Transformer 的免训练转换；局部 $k$-NN 是在三维空间坐标上构建的固定几何邻域。
- 是否需要训练或微调：是（使用替代梯度从头直接训练）
- 涉及的算子：最远点采样（FPS）、三维坐标 $k$-NN 局部邻域建图、队列偏移动态编码（Q-SDE）、局部向量减法脉冲自注意力与全局分支脉冲点注意力（DB-SPA）、LIF 神经元
- 关键数字：$T=4$ 下，`ModelNet40` 达 **91.37% OA / 88.97% mAcc**（推理能耗 **5.19 mJ**，比浮点 Point Transformer 的 19.65 mJ 低 **3.79×**），`ModelNet10` 达 **94.50% OA**，`ScanObjectNN`（`PB-T50-RS`）达 **78.02% OA / 75.15% mAcc**（Section 4.2 Table 1 & Table 2）。
- 与本方向的关系：直接相关（Claude 草稿二轮补充点名的点云直接训练代表作，其局部 $k$-NN 向量减法脉冲注意力可作为 M2 图注意力事件化的对照）
- 核对人与日期：Gemini，2026-11-16

### NI-SGCN（2025）

- 标题：Noise-Injected Spiking Graph Convolution for Energy-Efficient 3D Point Cloud Denoising
- 作者：Zikuan Li, Qiaoyun Wu, Jialin Zhang, Kaijun Zhang, Jun Wang
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2025；预印本 `arXiv:2502.19660`）
- 链接：https://arxiv.org/abs/2502.19660 ；代码：https://github.com/Miraclelzk/NI-SGCN
- 核对深度：全文
- 一句话：将 DGCNN 的动态边卷积（EdgeConv）改造为面向 3D 点云去噪回归任务的噪声注入脉冲图卷积网络，**直接在二值脉冲特征空间上动态计算 $k$-NN 建图，并将 EdgeConv 的 `max` 池化替换为 `mean` 池化**。
- 解决的难点：填补了脉冲图卷积在连续三维坐标回归（点云去噪）任务上的空白，并通过向膜电位注入自适应高斯噪声缓解二值脉冲量化对精细几何位移估计的损失。
- 留下的局限：属于从头直接训练，并非对预训练 DGCNN 做免训练转换；但它的消融实验极其直观地暴露了**二值脉冲特征下 `max` 邻域聚合严重丢失几何细节**的现象。
- 是否需要训练或微调：是（使用三角窗替代梯度从头端到端训练）
- 涉及的算子：**脉冲特征空间动态 $k$-NN 建图**、脉冲相对边特征构造 $[s_i, s_j - s_i]$、噪声注入 LIF 神经元（NI-LIF）、**邻域 `mean` 聚合（替代原版 EdgeConv 的 `max` 聚合）**
- 关键数字：$T=4$ 下，在 `PU-Net` 测试集（10K 点，2% 噪声）上 Chamfer Distance（CD，$\times 10^{-4}$，越小越好）达 **1.798**、Point2Mesh（P2M）达 **0.659**（优于浮点 DGCNN/DMRDenoise 等基线，理论能耗仅 **6.12 mJ**，较浮点基线节省约 **86%**，Section 4.2 Table 1 & Section 4.3）；**关键聚合算子消融（Section 4.2 Table 2）**：在完全相同的脉冲动态 EdgeConv 下，使用原版 `MAX` 聚合时 CD 恶化至 **2.199**，改为 `MEAN` 聚合后 CD 显著改善至 **1.798**（降低 18.2% 误差）！
- 与本方向的关系：直接相关（回答定向问题 Q2 中 DGCNN 现状的关键文献；其“脉冲特征上算动态 $k$-NN + `max` 聚合显著劣于 `mean` 聚合”的实验结果是对本项目 M1 核心假设的极佳侧面印证）
- 核对人与日期：Gemini，2026-11-16

### E-3DSNN（2025）

- 标题：Efficient 3D Recognition with Event-driven Spike Sparse Convolution
- 作者：Xuerui Qiu, Man Yao, Jieyuan Zhang, Yuhong Chou, Ning Qiao, Shibo Zhou, Bo Xu, Guoqi Li
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2025；预印本 `arXiv:2412.07360`）
- 链接：https://arxiv.org/abs/2412.07360 ；代码：https://github.com/bollossom/E-3DSNN
- 核对深度：全文
- 一句话：放弃直接在无序点集上做脉冲采样，提出脉冲体素编码（SVC）与仅在中心位置有脉冲时触发计算的脉冲稀疏卷积（SSC），配合整数多脉冲神经元（I-LIF）构建覆盖 3D 分类、自动驾驶检测与语义分割的通用 3D 脉冲骨干。
- 解决的难点：解决了三维稠密脉冲卷积立方级膨胀及背景空体素冗余计算问题，首次把直接训练 3D SNN 推向大规模室外雷达检测与分割基准（KITTI、SemanticKITTI、nuScenes）。
- 留下的局限：走规则体素化稀疏 3D 卷积 + 从头直接训练路线，并非图网络转换；且训练时依赖整数发放神经元 I-LIF（$T=1, D=4$），推理时需展开为虚拟时间步。
- 是否需要训练或微调：是（在 4 张 A100 上从头直接训练 80–200 轮）
- 涉及的算子：3D 点云体素化、整数 LIF 神经元（I-LIF，$\lfloor\text{clip}\{U,0,D\}\rceil$）、子流形脉冲稀疏卷积（SSC，Rulebook 地址映射）、膜电位残差连接
- 关键数字：$T \times D = 1 \times 4$ 下，`ModelNet40` 达 **91.7%**（3.27M 参数，能耗 **0.04 mJ**；轻量版 1.87M 达 **91.5%**、**0.02 mJ**，Section 4.2 Table 1）；`KITTI` 验证集 Car 3D AP（Easy/Mod./Hard）达 **89.6% / 84.0% / 78.7%**（能耗 **3.4 mJ**，Table 2）；`SemanticKITTI` 验证集达 **69.2% mIoU**（能耗 **6.1 mJ**，较同架构 ANN 省电 **8.8×**，Table 3 & Table 4）；`nuScenes` 验证集达 **64.1 NDS**（能耗 **3.1 mJ**，Appendix Table 6）。
- 与本方向的关系：背景（3D 视觉脉冲网络的强性能标杆；其 I-LIF 整数多级发放与分级脉冲硬件映射同脉络一/五中的多阈值脉冲高度相通）
- 核对人与日期：Gemini，2026-11-16

### Adaptive Calibration（2025）

- 标题：Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks
- 作者：Ziqing Wang, Yuetong Fang, Jiahang Cao, Hongwei Ren, Renjing Xu
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2025 Oral；预印本 `arXiv:2412.16219`）
- 链接：https://arxiv.org/abs/2412.16219 ；代码：https://github.com/bic-L/burst-ann2snn
- 核对深度：全文（含主文 PDF 全文与官方 GitHub 仓库 `3dtasks/` 全部模型及转换源码核验）
- 一句话：提出免训练的统一 ANN→SNN 校准转换框架，利用灵敏度分析为各层自动搜索最优脉冲簇（burst-firing）上限与自适应发放阈值（AdaFire），并将评测拓展到改进型 **PointNet 3D 点云分类与部件分割**。
- 解决的难点：无须重训即可在低时间步下利用商业类脑芯片（如 Loihi 2、Xylo）已支持的脉冲簇机制快速释放残余膜电位，兼顾 2D 分类/检测/分割与 3D 点云转换。
- 留下的局限：经核查官方开源仓库 `3dtasks/models/model.py`（文件头注释写明 `# Improved PointNet for ANN2SNN 3d Tasks`），其 3D 点云转换**同样被迫将原版 PointNet（含 STN3d、STNkd 与主干特征层）中的全局 `torch.max` 池化全部替换为 `torch.mean` 平均池化后再训练源 ANN**，并未解决原版 `max` 聚合或动态 $k$-NN 图网络的免重训直接转换。
- 是否需要训练或微调：部分（转换阶段本身完全免训练，仅需少量校准样本做 KL 散度灵敏度搜索与偏置校准；但对于 3D 点云任务，源 PointNet 须先改为 `mean` 池化架构再预训练）
- 涉及的算子：BatchNorm 折叠、分层灵敏度引导的脉冲簇上限搜索（Burst-firing，`maxspike`）、自适应阈值发放（AdaFire）、偏置校准、**改用 `torch.mean` 全局池化的 PointNet（`PointNetCls` 与 `PointNetDenseCls`）**
- 关键数字：转换校准仅需约 **0.09 小时**（README 与 Section 4）；在 `CIFAR-10` / `CIFAR-100` / `ImageNet` 上分别降低 **70.1% / 60.3% / 43.1%** 能耗；主文 Figure 1 多任务雷达图报告在改进型 PointNet 3D 点云分类（`3D-Cls`，默认 $T=16, \text{maxspike}=4$）上转换 SNN 达约 **94.0%**（源 ANN 约 96.0%），3D 部件分割（`3D-Seg`，ShapeNet Part，默认 $T=8, \text{maxspike}=4$）达约 **74.75% mIoU**（源 ANN 约 75.5%）。
- 与本方向的关系：直接相关（继 Lan et al. ICCV 2023 之后又一篇把免训练校准转换推广到 3D 点云网络的工作，且源码实锤了现有转换框架遇到点集 `max` 池化时仍依赖“改源网络为 `mean` 池化”）
- 核对人与日期：Gemini，2026-11-16

### One-Spike SNN（2025）

- 标题：One-Spike SNN: Single-Spike Phase Coding With Base Manipulation for ANN-to-SNN Conversion Loss Minimization
- 作者：Sangwoo Hwang, Jaeha Kung
- 发表处：IEEE Transactions on Emerging Topics in Computing（IEEE TETC，2025 年 1–3 月刊，Vol. 13, No. 1, pp. 162–172，DOI: `10.1109/TETC.2024.3386893`；预印本 `arXiv:2403.08786`）
- 链接：https://doi.org/10.1109/TETC.2024.3386893 （预印本：https://arxiv.org/abs/2403.08786 ）；代码：未公开
- 核对深度：全文
- 一句话：**本次检索范围内找到的唯一一篇对预训练 ReLU 图卷积网络（Text GCN）做免训练 ANN→SNN 权重转换的工作**——将每层 GCN 拆解为“特征线性变换”与“邻接矩阵乘”两个级联脉冲层，并用基数可调（$Q=1.3$）的单脉冲相位编码将 5 个文本图数据集的平均转换精度损失压至 0.90%。
- 解决的难点：证明了在固定邻接矩阵、线性求和聚合的 ReLU GCN 上，无需重训源模型即可通过“双矩阵乘法拆层 + 阈值中点平移 + 小基数单脉冲相位编码”在 $T=16$ 内完成高精度低功耗转换。
- 留下的局限：① 仅适用于**静态固定拓扑 + 线性矩阵乘聚合（$\tilde{A}XW$）+ 纯 ReLU 激活**的浅层（2 层拆为 4 层）非视觉文本图（Text GCN），完全未涉及视觉图骨干中的动态 $k$-NN 选邻居、`max` 聚合、GELU/SiLU 或归一化算子；② 层间存在因残余电位引发的伪脉冲（noise spikes），需在层与层之间插入等待步 $w$ 抑制误差传播（Section V-C2）。
- 是否需要训练或微调：**否**（真正免训练转换：仅在训练集上统计最大激活值做通道/层归一化，直接复制预训练 Text GCN 权重）
- 涉及的算子：通道级/层级最大激活权重归一化（Eq. 2）、单脉冲相位编码（每个隐藏神经元最多发放 1 次脉冲，输入层采用多脉冲二进制相位编码，Section III-A）、发放阈值中点平移 $V_\theta \to \frac{1+1/Q}{2}$（Eq. 4）、相位基数调节（Base Manipulation，$Q=1.3$，膜电位衰减系数 $\lambda_j = Q$）、**GCN 双步拆层（2 层 Text GCN 拆为 4 层 SNN：奇数层算消息传递 $XW$，偶数层乘固定归一化邻接矩阵 $A$，Section IV-C）**
- 关键数字：在 Section IV-C（*GCN-to-SNN Conversion*）及 **Table V（`\label{tab:GCN}`，$T=16, Q=1.3$）** 中，对预训练 2 层 Text GCN 免训练转换的精确结果为：`20NG`（ANN 86.27% → SNN **85.90%**，掉点 0.37%，能效提升 $\alpha=\mathbf{5.91\times}$）、`R52`（ANN 93.49% → SNN **92.24%**，掉点 1.25%，$\alpha=\mathbf{11.11\times}$）、`R8`（ANN 97.06% → SNN **96.44%**，掉点 0.62%，$\alpha=\mathbf{6.59\times}$）、`Ohsumed`（ANN 68.42% → SNN **66.95%**，掉点 1.47%，$\alpha=\mathbf{6.24\times}$）、`MR`（ANN 75.93% → SNN **75.09%**，掉点 0.84%，$\alpha=\mathbf{4.53\times}$）；5 个图数据集**平均转换精度损失仅 0.90%**，按 45nm Horowitz 模型（4.6 pJ MAC vs 0.9 pJ AC，Eq. 6）估算能效提升 **4.53×–11.11×**。
- 与本方向的关系：**直接相关（核心必引先例！它直接推翻了 Claude v1 草稿中“未见任何图网络免训练转换”的全称判断，同时其“仅支持固定图 + 线性矩阵乘 + ReLU Text GCN”的清晰边界恰恰凸显了本项目聚焦“视觉图动态拓扑与非线性聚合免训练转换”的不可替代性）**
- 核对人与日期：Gemini，2026-11-16

### MSG（2024）

- 标题：Spiking Graph Neural Network on Riemannian Manifolds
- 作者：Li Sun, Zhenhao Huang, Qiqi Wan, Hao Peng, Philip S. Yu
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2024，Vol. 37，DOI: `10.52202/079017-1071`；预印本 `arXiv:2410.17941`）
- 链接：https://arxiv.org/abs/2410.17941 ；代码：论文内仅提供匿名评审仓 `https://anonymous.4open.science/r/MSG-16E9`（未见正式实名公开仓）
- 核对深度：全文
- 一句话：将脉冲图神经网络推广到双曲/球面等常曲率黎曼流形空间（MSG），利用指数映射与对数映射桥接切空间脉冲序列与流形点，并推导无需替代梯度、无需按时间步反向展开（BPTT-free）的流形微分（DvM）训练算法。
- 解决的难点：克服了欧氏空间脉冲图网络表达层次/环状图结构的几何失真，同时在反向传播时绕开了不可导阶跃函数和随 $T$ 线性增长的 BPTT 训练延迟。
- 留下的局限：面向同配无向静态图的节点分类与链路预测从头训练（作者在 Section 8 局限性中自述尚未解决有向图与异配图，且 `SGNNBench` 实测其在深层异配图上掉点严重）；流形指数/对数映射涉及大量超越函数浮点运算。
- 是否需要训练或微调：是（使用流形微分 DvM 从头训练）
- 涉及的算子：黎曼流形对数映射（$\operatorname{Log}_{\mathbf{z}}$）与指数映射（$\operatorname{Exp}_{\mathbf{z}}$）、切空间图卷积聚合、IF/LIF 脉冲神经元、推前/拉回（Pushforward / Pullback）雅可比矩阵反向梯度
- 关键数字：$T \in \{5, 15\}$、32 维双曲洛伦兹流形 $\mathbb{H}^{32}$ 下，节点分类精度在 `Computers` / `Photo` / `CS` / `Physics` 上分别达 **89.27±0.19% / 93.11±0.11% / 92.65±0.04% / 95.93±0.07%**，链路预测 AUC 分别达 **94.65% / 96.75% / 95.19% / 93.43%**（Section 6.2 Table 1）；理论推理能耗分别为 **0.047 / 0.043 / 0.026 / 0.029 mJ**，不到全精度黎曼 GNN 基线的 **1/20**（Table 3）。
- 与本方向的关系：背景（通用静态图直接训练里程碑；Claude 草稿二轮补充曾引用其相关工作综述判断）
- 核对人与日期：Gemini，2026-11-16

### Spiking PointCNN（2024）

- 标题：Spiking PointCNN: An Efficient Converted Spiking Neural Network under a Flexible Framework
- 作者：Yingzhi Tao, Qiaoyun Wu
- 发表处：Electronics（MDPI，2024 年 9 月，Vol. 13, Issue 18, Article 3626，DOI: `10.3390/electronics13183626`）
- 链接：https://www.mdpi.com/2079-9292/13/18/3626 ；代码：https://github.com/DeadlyAbyss/Spiking-PointCNN
- 核对深度：全文（通读 MDPI 正文全 5 节、公式 1–13、表 1–4，并逐文件核验官方开源仓库 `DeadlyAbyss/Spiking-PointCNN` 全部 7 个源码文件）
- 一句话：将 PointCNN 的局部坐标 $\mathcal{X}$-变换卷积模块在架构层面替换为嵌入 LIF 脉冲神经元的 `Spiking X-Conv`，并用**可调陡度参数的替代梯度函数从头反向传播训练** 3D 点云分类网络。
- 解决的难点：在保留 PointCNN 对无序局部点集做学习置换（$\mathcal{X}$-transformation）能力的同时引入脉冲激活，提升了合成点云与含噪真实扫描点云上的直接训练收敛性。
- 留下的局限：**极其关键的文献纠偏**：尽管论文标题和摘要使用了“Converted SNN”一词，但原文第 3.2–3.3.3 节及官方代码（`main_ModelNet.py` 从头训练 600 轮、`main_ScanObjectNN.py` 从头训练 1000 轮）证实，**它根本不是把预训练 ANN 权重转成 SNN 的转换方法**，而是“把网络模块替换成脉冲层后用替代梯度从头训练”；此外，官方代码 `SpikingLayers.py` 在每个脉冲层内部重复输入 $T$ 次后立刻执行 `out = out.mean(dim=0)` 折叠回连续浮点数，导致层间传递与矩阵乘（`torch.matmul(x_star, transform_matrix)`）实为浮点运算，且全文未报告任何能耗或发放率数据。
- 是否需要训练或微调：**是**（完全从头直接训练，无任何预训练 ANN 权重复制、阈值平衡或训练后校准）
- 涉及的算子：**三维空间坐标膨胀 $k$-NN 建图**（4 层 `Spiking X-Conv` 的 $(K, D)$ 依次为 $(8,1), (12,2), (16,2), (16,2)$，非按深层特征动态选邻居）、最远点采样（FPS，下采样率 0.375 与 0.334）、局部相对坐标中心化 $p_{i,k}-p_i$、坐标特征脉冲 MLP（$MLP'_\delta$）与 $\mathcal{X}$ 置换矩阵脉冲 MLP（$MLP'_{trans}$）、深度可分离一维卷积、全局均值池化（`global_mean_pool`）、硬复位 LIF 神经元与可调替代梯度（正文 Eq. 13 写为参数化 Sigmoid，开源代码 `SpikingLayers.py` 实现为温度缩放 `tanh`）
- 关键数字：时间步消融 $T \in \{1,2,3,4\}$（默认 $T=1$）：在 `ModelNet40` 上，$T=1,2,3,4$ 的 OA 分别为 **84.28% / 85.78% / 85.94% / 84.97%**，mACC 分别为 **81.85% / 84.48% / 83.14% / 80.71%**（Section 4.2.3 Table 3）；替代梯度陡度 $k=1$ 时 `ModelNet40` 取得主对比表最佳 OA **86.71% / 82.21% mACC**（Table 2 & Table 4，主表四舍五入记为 **86.7% / 82.2%**）；在真实扫描集 `ScanObjectNN`（代码加载 `PB_T50_RS` 旋转缩放扰动集）上主表报告 **74.1% OA / 73.3% mACC**（Section 4.3 Table 4；注：正文第 40 段文字另称最高达 76.10% OA，与表 4 的 74.1% 存在文内不一致，两处均予记录）；相比作者自建同结构无脉冲基线（84.68% OA，Table 1）高 1.26%，但相比原版全精度 PointCNN（ModelNet40 92.2%、ScanObjectNN 78.5%）分别低约 **5.5%** 与 **4.4%**。
- 与本方向的关系：直接相关（定向问题 Q4 的目标论文；纠正 Claude v1 将其误判为“点云 ANN→SNN 转换先例”的关键证据——真正的点云转换先例是该文表 4 引用的 `[20]` Lan et al., ICCV 2023）
- 核对人与日期：Gemini，2026-11-16

### Loihi Transductive SGNN（2024）

- 标题：Transductive Spiking Graph Neural Networks for Loihi
- 作者：Shay Snyder, Victoria Clerico, Guojing Cong, Shruti Kulkarni, Catherine Schuman, Sumedh R. Risbud, Maryam Parsa
- 发表处：arXiv（2024 年 4 月预印本，`arXiv:2404.17048`）
- 链接：https://arxiv.org/abs/2404.17048 ；代码：未公开
- 核对深度：全文
- 一句话：在面向 Intel Loihi 2 的开源异步框架 Lava 中用定点整数权重与脉冲时间依赖可塑性（STDP）实现直推式引文图分类，并用贝叶斯优化（Lava BO）自动搜索神经形态超参数。
- 解决的难点：摆脱了传统冯·诺依曼浮点仿真器（NEST）无法直接部署到异步定点类脑芯片的限制，验证了纯脉冲局部可塑性规则在图拓扑上的硬件兼容原型。
- 留下的局限：仅利用引文边连接而不处理高维连续视觉特征，且直推式 STDP 在 `Cora` 上的分类精度明显低于现代 GNN。
- 是否需要训练或微调：是（基于在线 STDP 突触可塑性学习与 Lava BO 超参数搜索）
- 涉及的算子：无漏电定点积分发放神经元、不应期门控（Refractory gating）、STDP 突触权重更新、异步通道事件通信
- 关键数字：在 `Cora` 验证集上，Lava 定点实现于 $t=14$ 时间步、零突触延迟下取得 **63.57%** 分类精度（对比 NEST 浮点延迟 1 配置的 68.57%，Section 4 Table 2），经 15 轮 Lava BO 搜索 2,450 组参数后测试精度为 **62.86%**，单篇论文网络重置开销从 NEST 的 9.73 s 降至 **123.6 ms**（Section 4）。
- 与本方向的关系：背景（类脑芯片 Loihi 2 软件栈 Lava 上的纯事件图推理早期探索，与脉络五硬件实测呼应）
- 核对人与日期：Gemini，2026-11-16

### SpikeGCL（2024）

- 标题：A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks（arXiv API 返回的标题；简称 SpikeGCL。本条早先标题字段有乱码，复查时已更正）
- 作者：Jintang Li, Huizhe Zhang, Ruofan Wu, Zulun Zhu, Baokun Wang, Changhua Meng, Zibin Zheng, Liang Chen
- 发表处：International Conference on Learning Representations（ICLR 2024；预印本于 2023 年 5 月发布于 `arXiv:2305.19306`）
- 链接：https://arxiv.org/abs/2305.19306 ；代码：https://github.com/EdisonLeeeee/SpikeGCL
- 核对深度：全文
- 一句话：将节点全维特征按通道均分为 $T$ 个子块依次送入脉冲图编码器，采用时间块间截断梯度（blockwise stop-gradient）局部对比损失学习紧凑的 1 比特节点表征，并在理论上证明了减法复位 IF 脉冲图网络对全精度 ReLU GCN 的率编码逼近误差界。
- 解决的难点：避免了对整张大图静态重复输入 $T$ 次的算力浪费，并通过分块局部对比训练消除了长序列替代梯度反向传播中的梯度消失。
- 留下的局限：本身属于自监督从头训练而非免训练转换；但**极具理论价值的是其附录 A 定理 1 证明了若用减法复位 IF 神经元按率编码逼近 $L$ 层最大度为 $D$ 的预训练 ReLU GCN，最坏误差上界随 $(\nu\sqrt{1+D})^L / (T\sqrt{T})$ 指数放大**！
- 是否需要训练或微调：是（分块自监督图对比学习从头训练 + 线性分类头微调）
- 涉及的算子：节点特征通道分块（Feature partitioning）、随机边丢弃与特征置换图增强、GCN/SAGE 邻域聚合、IF/LIF 神经元、时间块间 `detach` 梯度隔离对比损失
- 关键数字：1 比特脉冲表征在 `Computers` / `Photo` / `CS` / `Physics` / `ogbn-arxiv` / `ogbn-MAG` 上分别达 **88.9±0.3% / 93.0±0.1% / 92.8±0.1% / 95.2±0.6% / 70.9±0.1% / 32.0±0.3%**，达到全精度图对比学习基线 95%–99% 的精度（Section 5 Table 1）；在百万级异构图 `ogbn-MAG` 上参数量仅 **6.6 KB**（约为全精度平均值的 **1/37**，原文正文泛称为 ~1/60），理论能耗 **0.18 mJ**（Table 2）；**理论逼近界（Appendix A Theorem 1 Eq. 6）**：$\sup_{v} \|\widehat{z}_v - z^*_v\|_2 \le \frac{\sqrt{d}\kappa(\sqrt{1+D}\nu)^L}{T\sqrt{T}}$。
- 与本方向的关系：可借用的技术（附录 A 定理 1 直接给出了固定图 ReLU GCN 做减法复位 IF 转换时的度数—层数误差放大上界，可直接引入本项目理论分析章节！）
- 核对人与日期：Gemini，2026-11-16

### SpikePoint（2024）

- 标题：SpikePoint: An Efficient Point-based Spiking Neural Network for Event Cameras Action Recognition
- 作者：Hongwei Ren, Yue Zhou, Yulong Huang, Haotian Fu, Xiaopeng Lin, Jie Song, Bojun Cheng
- 发表处：International Conference on Learning Representations（ICLR 2024 Spotlight；预印本于 2023 年 10 月发布于 `arXiv:2310.07189`）
- 链接：https://arxiv.org/abs/2310.07189 ；代码：未公开（论文内未附代码仓链接）
- 核对深度：全文
- 一句话：把事件相机输出的异步事件流切窗映射为三维时空伪点云 $(x, y, t)$，用单阶段 FPS + $k$-NN 局部球分组、折叠高斯绝对距离补偿编码与恒等脉冲残差块直接训练超低功耗事件动作识别网络。
- 解决的难点：避免把稀疏异步事件压成稠密二维帧造成的时空细节丢失，解决了局部相对坐标含负值无法直接做无符号率编码导致的不对称失真。
- 留下的局限：明确采用替代梯度反向传播从头训练 300 轮（第 5 节结论写明“not converting ANNs to SNNs”）；局部特征与全局特征提取均使用 `MaxPool` 在二值脉冲或脉冲计数上池化。
- 是否需要训练或微调：是（使用 Adam + ATan 替代梯度 + MSE 投票损失从头训练 300 轮）
- 涉及的算子：滑动时间窗三维时空点云映射、最远点采样（FPS）、三维时空坐标 $k$-NN 分组、相对坐标标准差归一化与绝对值 + 组内最小坐标补偿编码 $[\Delta|x|,\Delta|y|,\Delta|z|,x_{\min},y_{\min},z_{\min}]$、无符号泊松率编码、脉冲后恒等残差连接（$\text{LIF}(I)+S^{l-1}$）、局部与全局 `MaxPool`、Parametric LIF 神经元
- 关键数字：$T=16$、1024 采样点下，`DVS128 Gesture` 达 **98.74%**（0.58M 参数，动态能耗 **0.82 mJ**、静态能耗 **0.756 mJ**，Section 4.4 Table 2 & Section 4.5 Table 7）；`Daily DVS` 达 **97.92%**（0.16M 参数，仅为重型 ANN 参数量的约 **0.3%**，Table 3）；`DVS Action` 达 **90.6%**（Table 4）；`HMDB51-DVS` 达 **55.6%**（超越图卷积基线 RG-CNN 的 51.5%，Table 5）；`UCF101-DVS` 达 **68.46%**（Table 6）。
- 与本方向的关系：可借用的技术（事件流三维点云分组中对局部负相对坐标做分布对称补偿的设计，对点云/事件图输入编码有参考价值）
- 核对人与日期：Gemini，2026-11-16

### Dy-SIGN（2024）

- 标题：Dynamic Spiking Framework for Graph Neural Networks
- 作者：Nan Yin, Mengzhu Wang, Zhenghan Chen, Giulia De Masi, Huan Xiong, Bin Gu
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2024，Vol. 38, No. 15；预印本 `arXiv:2401.05373`）
- 链接：https://mlanthology.org/aaai/2024/yin2024aaai-dynamic （预印本：https://arxiv.org/abs/2401.05373 ）；代码：未公开（附录匿名链接已注释）
- 核对深度：全文
- 一句话：面向离散时间快照动态图节点分类，设计跨层拓扑与初始特征信息补偿机制，并将多层反馈脉冲图网络的平均发放率视作不动点平衡态，用隐式微分变分训练替代高显存的 BPTT。
- 解决的难点：把动态图脉冲训练的显存占用砍到 `SpikeNet` 的一半左右，缓解了多层二值脉冲传递造成的原始图结构细节丢失。
- 留下的局限：处理的是社交/电商/专利等非视觉离散快照动态图，且需要从头训练 100 轮；平衡态推导依赖线性连续图卷积假设（Section 4 公式推导末行明确要求聚合函数 $g_A(\cdot)$ 线性）。
- 是否需要训练或微调：是（采用隐式微分变分法从头训练 100 轮）
- 涉及的算子：快照内 GCN 邻域聚合、首尾跨层特征/结构补偿反馈连接、LIF 脉冲神经元、不动点平衡方程隐式梯度反传、跨时间片状态传递与拼接分类
- 关键数字：在 80% 训练比例下，`DBLP`（28K 节点，27 时间步）Macro-F1 / Micro-F1 达 **74.67±0.5% / 74.96±0.2%**，`Tmall`（577K 节点，186 时间步）达 **61.89±0.1% / 65.82±0.2%**，`Patent`（2.73M 节点，25 时间步）达 **83.91±0.2% / 83.90±0.2%**（Section 4 Table 1 & Table 2），且在 `DBLP` 上训练显存约为 `SpikeNet` 的 **50%**（Section 5.2 Figure 2a）。
- 与本方向的关系：背景（通用动态图直接训练代表作，其“平衡态等价于缩放 ReLU 图卷积不动点”的推导仅在线性聚合下成立）
- 核对人与日期：Gemini，2026-11-16

### P2SResLNet（2024）

- 标题：Point-to-Spike Residual Learning for Energy-Efficient 3D Point Cloud Classification
- 作者：Qiaoyun Wu, Quanxiao Zhang, Chunyu Tan, Yun Zhou, Changyin Sun
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2024，Vol. 38, No. 6, pp. 6092–6099，DOI: `10.1609/aaai.v38i6.28425`）
- 链接：https://doi.org/10.1609/aaai.v38i6.28425 ；代码：未公开
- 核对深度：仅摘要（摘要与官方会议元数据直接核对；具体实验数值交叉核实自 `Spiking Point Transformer` 表 2、`Spiking Point Mamba` 表 1、`E-3DSNN` 表 1 与 `Spiking PointCNN` 表 4 的一致引用）
- 一句话：结合核点卷积（KPConv）几何思想设计空间感知核点脉冲神经元与 3D 脉冲残差块，直接训练低时间步 3D 点云分类脉冲网络，并在文中构建了早期点云卷积 ANN→SNN 转换对照基线 `KPConv-SNN`。
- 解决的难点：缓解了常规脉冲神经元在三维核点卷积深层残差传递中的空间几何退化，实现了 $T=1$ 至 $T=4$ 下在真实扫描集 ScanObjectNN 上的有效收敛。
- 留下的局限：主模型采用从头直接训练且参数量偏大（14.3M）；而文中对比的常规转换基线 **`KPConv-SNN` 即便拉长到 $T=40$ 时间步，在真实点云 ScanObjectNN 上精度仍崩溃至 43.9%**，直观暴露了未经专门校准的朴素转换在三维局部几何卷积上的严重失效。
- 是否需要训练或微调：是（主模型 `P2SResLNet` 从头直接训练；文中对照基线 `KPConv-SNN` 为转换基线）
- 涉及的算子：三维欧氏球邻域查询、空间感知核点卷积（KPConv）脉冲神经元、3D 脉冲残差连接
- 关键数字：据 `Spiking Point Mamba` Table 1、`Spiking Point Transformer` Table 2 与 `E-3DSNN` Table 1 一致引用，`P2SResLNet-B`（14.3M 参数）在 `ModelNet40` 上达 **88.7% OA**，在 `ScanObjectNN`（`PB-T50-RS`，$T=1$）上达 **74.5% OA**；而其报告的转换对照基线 **`KPConv-SNN`（$T=40$）** 在 `ModelNet40` 上仅为 **70.5% OA / 67.6% mAcc**，在 `ScanObjectNN` 上跌至 **43.9% OA / 38.7% mAcc**（亦见 `AMP2` Table 5 与 `Spiking PointCNN` Table 4；注：`Spiking Point Mamba` Table 1 第 23 行误将 `KPConv-SNN` 的 70.5 与 43.9 两列颠倒排印，对照 `SPT`/`AMP2`/`Spiking PointCNN` 三篇表格可确认 ModelNet40 为 70.5%、ScanObjectNN 为 43.9%）。
- 与本方向的关系：直接相关（提供了点云核点卷积朴素 ANN→SNN 转换在大时间步 $T=40$ 下仍然严重掉点的基准反例）
- 核对人与日期：Gemini，2026-11-16

### DRSGNN（2024）

- 标题：Dynamic Reactive Spiking Graph Neural Network
- 作者：Han Zhao, Xu Yang, Cheng Deng, Junchi Yan
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2024，Vol. 38, No. 15，DOI: `10.1609/aaai.v38i15.29640`）
- 链接：https://doi.org/10.1609/aaai.v38i15.29640 ；代码：https://github.com/hzhao98/DRSGNN
- 核对深度：仅摘要（摘要与官方 AAAI/GitHub 页面直接核对，横评精度与能耗数据核实自 `SGNNBench` 全文 Table 2–6）
- 一句话：为脉冲图网络引入每个神经元可独立优化的动态反应阈值（Dynamic Reactive Spiking Neurons），并自适应融合图位置脉冲特征以增强长程邻域交互建模。
- 解决的难点：打破了全图共用单一固定发放阈值的同质性限制，提升了小规模与中等规模静态图节点分类上的表达能力。
- 留下的局限：必须从头训练可学习动态阈值与图位置脉冲模块，无法用于现成预训练视觉图模型的免训练转换。
- 是否需要训练或微调：是（从头端到端直接训练）
- 涉及的算子：可学习动态阈值脉冲神经元、图位置编码脉冲融合、消息传递图卷积聚合
- 关键数字：据 `SGNNBench` 全文统一复现评测（Table 2 & Table 3），`DRSGNN` 在静态同配图 `Cora` / `CiteSeer` / `PubMed` / `Photo` 上达 **80.76% / 71.12% / 78.03% / 92.66%**，在浅层局部异配图 `Chameleon` / `Squirrel` 上达 **63.60% / 47.74%**（居 9 种 SGNN 前列），但在长程异配图 `Roman-Empire` 上仅为 **45.28%**（全精度 GraphGPS 为 82.59%）。
- 与本方向的关系：背景（通用静态图直接训练代表作，其节点/通道异质可调阈值的思想可与转换校准中的分层/分通道阈值设定对照）
- 核对人与日期：Gemini，2026-11-16

### Lan 等 3D/2D 转换 SNN（2023）

- 标题：Efficient Converted Spiking Neural Network for 3D and 2D Classification
- 作者：Yuxiang Lan, Yachao Zhang, Xu Ma, Yanyun Qu, Yun Fu
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2023，pp. 9211–9220，DOI: `10.1109/ICCV51070.2023.00845`）
- 链接：https://doi.org/10.1109/ICCV51070.2023.00845 ；代码：未公开
- 核对深度：全文
- 一句话：**首个系统性将 3D 点云网络（`PointNet` 与 `PointNet++`）通过 ANN→SNN 转换变为低延迟脉冲网络的工作**——先将源网络中的 `MaxPooling` 替换为 `AvgPooling` 训练源模型，再用 5–10 批样本做 KL 散度阈值校准、偏置修正，并在推理时引入动态指数衰减阈值与 2 比特自适应多脉冲发放。
- 解决的难点：将 3D 点云转换 SNN 的有效推理时间步从数十步压缩至 **$T=4\sim 16$**，在转换后 SNN 与“改用平均池化的源 ANN”之间实现了几乎零掉点（$<0.36\%$）。
- 留下的局限：**对本项目极富启发的致命局限**：作者在 **Section 4.1（第 9216 页左栏）** 明确承认*“In the ANN-SNN conversion method, using Maxpooling will not work. To better match SNN conversion, the Maxpooling layer in ANN needs to be replaced with Avgpooling layer.”*——为此**被迫修改源 `PointNet` / `PointNet++` 架构并重新训练源 ANN**，导致源 `PointNet++` 教师本身在真实扫描集 `ScanObjectNN` 上的精度从原版 `MaxPool` 的 ~77.9%–84.3% 暴跌至 **69.22%**！它回避而非解决了 `max` 聚合与现成预训练权重的免训练转换。
- 是否需要训练或微调：**部分**（转换阶段仅需 5–10 个 batch 训练数据做免训练校准；但**必须先把源 PointNet/PointNet++ 的 `MaxPooling` 全部改成 `AvgPooling` 并从头重训源 ANN**，不能直接转官方公开的 `MaxPool` 预训练权重！）
- 涉及的算子：**`MaxPooling` 强制替换为 `AvgPooling`**、BatchNorm 折叠（Eq. 4）、KL 散度 95 点网格搜索阈值初始化（Eq. 8）、输出均值误差偏置校准（Eq. 7）、按激活神经元比例调节的自适应动态阈值 $\tilde{V}_{th}^{(l)}(t) = (1 - \alpha e^{-(t-1)}\tau^{(l)}(t))V_{th}^{(l)}$（$\alpha=0.5$，Eq. 9–11）、2 比特自适应分级脉冲发放 $s^{(l)}(t)=\text{clip}(\lfloor \tilde{v}_{mem}(t)/\tilde{V}_{th}(t)\rfloor, 0, 2^n-1)$（$n=2$，最大脉冲值 3，软复位，Eq. 13–14）、PointNet++ 的 FPS 与多尺度球邻域分组
- 关键数字：全部出自 Section 4.2 **Table 1–3（pp. 9216–9217）**：① `ModelNet-10`（Table 1）：`PointNet`（AvgPool 源 ANN 92.81%）在 $T=4/8/16$ 达 **92.38% / 92.59% / 92.75%**；`PointNet++`（AvgPool 源 ANN 93.54%）在 $T=4/8/16$ 达 **93.18% / 93.54% / 93.54%**；② `ModelNet-40`（Table 2）：`PointNet`（源 ANN 88.17%）在 $T=4/8/16$ 达 **87.95% / 87.99% / 88.17%**；`PointNet++`（源 ANN 89.45%）在 $T=4/8/16$ 达 **89.37% / 89.43% / 89.45%**；③ `ScanObjectNN`（Table 3）：`PointNet`（源 ANN 66.56%）在 $T=4/8/16$ 达 **65.16% / 65.94% / 66.56%**；`PointNet++`（源 ANN 69.22%，常规转换 Baseline 在 $T=16$ 仅 58.02%）在 $T=4/8/16$ 达 **68.91% / 69.06% / 69.22%**。
- 与本方向的关系：**直接相关（Claude v1 完全漏掉的 3D 点云与 `PointNet++` 转换头号先例！其“因无法转换 MaxPool 而把 PointNet++ 改成 AvgPool 重训、致使教师在 ScanObjectNN 上狂掉约 9–15 个百分点”的事实，为本项目坚持“不改教师架构、免重训直接转换含 `max` 聚合的视觉图模型”提供了最强立项依据！）**
- 核对人与日期：Gemini，2026-11-16

### Spiking PointNet（2023）

- 标题：Spiking PointNet: Spiking Neural Networks for Point Clouds
- 作者：Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2023，Vol. 36；预印本 `arXiv:2310.06232`）
- 链接：https://arxiv.org/abs/2310.06232 ；代码：https://github.com/DayongRen/Spiking-PointNet
- 核对深度：全文
- 一句话：首个面向三维点云分类的直接训练脉冲神经网络——将 PointNet 的 ReLU 替换为 LIF 脉冲神经元，提出“单时间步训练、多时间步推理（Trained-less but Learning-more）”范式与膜电位微扰增广（MPP）。
- 解决的难点：缓解了点云脉冲网络在多时间步反向传播（BPTT）下的梯度消失/爆炸及高显存开销，仅需 $T=1$ 训练即可在 $T=4$ 推理时获得类似集成投票的精度提升。
- 留下的局限：**完全属于替代梯度从头直接训练（非 ANN→SNN 转换）**；骨干为无局部邻域图结构的逐点 MLP + 全局对称池化（PointNet），未涉及动态 $k$-NN 建图或局部图消息传递算子的免训练转换。
- 是否需要训练或微调：是（使用 `tanh` 型替代梯度 $k=5$ 在 $T=1$ 下从头直接训练，配合训练期初始膜电位随机扰动 MPP）
- 涉及的算子：逐点共享 MLP、T-Net 空间变换对齐、硬复位 LIF 脉冲神经元、训练期膜电位微扰（MPP，$\delta \in [0, 0.5]$）、全局对称最大池化、多时间步输出均值集成
- 关键数字：单步训练（$T_{\text{train}}=1$）配合膜电位微扰（MPP）后，在 `ModelNet10` 上 $T_{\text{test}}=1 / 4$ 精度达 **91.66% / 93.31%**（超越同结构浮点 PointNet 的 92.98%，Section 5.1 Table 4 / `tab:ab4` & Table 5 / `Comparison`）；在 `ModelNet40` 上 $T_{\text{test}}=1 / 2 / 4$ 精度达 **87.72% / 88.46% / 88.61%**（浮点 PointNet 为 89.20%，常规 $T=4$ BPTT 训练仅为 86.70%，Section 5.1 Table 5）；按 45nm Horowitz 模型（4.6 pJ MAC vs 0.9 pJ AC，平均发放率 18.7%）估算，`ModelNet10` 单次前向能耗在 $T=1$ 为 **$9.2 \times 10^6$ pJ**（比 PointNet 省电 **15.2×**），$T=4$ 为 **$3.7 \times 10^7$ pJ**（省电 **3.8×**，Section 5.2 Table 6）。
- 与本方向的关系：直接相关（回答定向问题 Q2 与核查 Claude 草稿二轮补充的核心文献：三维点云脉冲网络早期里程碑，但走的是从头直接训练路线且无图邻域拓扑）
- 核对人与日期：Gemini，2026-11-16

### SpikeNet（2023）

- 标题：Scaling Up Dynamic Graph Representation Learning via Spiking Neural Networks
- 作者：Jintang Li, Zulun Zhu, Zeang Liu, Ruofan Wu, Liang Chen, Zibin Zheng, Sheng-hua Zhong
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2023，Vol. 37, No. 7, pp. 8588–8596；预印本于 2022 年 8 月发布于 `arXiv:2208.10364`）
- 链接：https://arxiv.org/abs/2208.10364 ；代码：https://github.com/EdisonLeeeee/SpikeNet
- 核对深度：全文
- 一句话：用可学习阈值的 LIF 脉冲神经元取代传统时序图网络里的重量级 RNN（LSTM/GRU），结合时间邻域采样在百万节点级离散快照时序图上直接训练可扩展节点分类模型。
- 解决的难点：把动态图时序状态记忆从浮点循环网络替换为稀疏二值脉冲动力学，首次让脉冲图网络扩展到 273 万节点、1,396 万边的 `Patent` 时序图。
- 留下的局限：从头使用动态替代梯度训练；在每个时间步的邻域采样与特征线性聚合阶段仍包含浮点线性投影与采样开销。
- 是否需要训练或微调：是（基于动态自适应替代梯度从头端到端训练）
- 涉及的算子：时序随机游走/邻域采样（Temporal neighbor sampling）、GraphSAGE 式特征拼接与均值/求和线性聚合、带自适应阈值更新的 LIF 脉冲神经元
- 关键数字：在 `DBLP` / `Tmall` / `Patent` 三个大规模时序图（80% 训练集，Section 5.2 Table 2）上，Macro-F1 分别达 **74.86% / 62.40% / 83.90%**（注：`Dy-SIGN` 引用复现值在 DBLP 上记为 74.64% Macro-F1 / 74.86% Micro-F1，原论文 `aaai23.tex` Table 2 报告 DBLP 80% 划分为 Macro-F1 **74.34%**、Micro-F1 **75.58%**，`Tmall` Macro-F1 **62.40%** / Micro-F1 **66.10%**，`Patent` Macro-F1 **83.90%** / Micro-F1 **83.88%**，以原论文表 2 为准），训练参数量与推理速度显著优于 EvolveGCN 与 TGAT。
- 与本方向的关系：背景（时序动态图直接训练奠基工作，其把外部流式时间步与内部脉冲膜电位时间步对齐的思路与 `Signal-SGN` 及 M3 流式推理同源）
- 核对人与日期：Gemini，2026-11-16

### SpikingGCN（2022）

- 标题：Spiking Graph Convolutional Networks
- 作者：Zulun Zhu, Jiaying Peng, Jintang Li, Liang Chen, Qiang Yu, Siqiang Luo
- 发表处：Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence（IJCAI 2022，pp. 2434–2440；预印本 `arXiv:2205.02767`）
- 链接：https://arxiv.org/abs/2205.02767 ；代码：https://github.com/ZulunZhu/SpikingGCN
- 核对深度：全文（含主文及附录 A–C 完整核对）
- 一句话：将归一化图卷积预传播（$S=\tilde{A}^K X$）与伯努利脉冲编码、全连接线性层及 LIF 神经元结合进行端到端直接训练，**明确否定自动转换路线，并在 MNIST 网格图与 75 节点超像素图分类上验证了视觉图像图扩展性**。
- 解决的难点：给出了首个端到端简明脉冲图卷积范式，避免在每一层重复做稠密图矩阵乘，并首次把脉冲图卷积拓展到超像素图像图分类（B 样条几何核）。
- 留下的局限：① 引言与附录 A 以“预训练 ANN 的多种算子与 SNN 不兼容（引 Rueckauer et al., 2017）”为由直接放弃转换路线，每次换任务都需从头训练；② 在普通引文图上把图传播退化为输入前的无参预处理 $S=\tilde{A}^K X$，脉冲层本身只是单层线性分类器；③ 在规模稍大的 `ogbn-arxiv` 上精度跌至 55.8%（见 `SpikeGCL` Table 1）。
- 是否需要训练或微调：是（使用替代梯度从头直接训练）
- 涉及的算子：无参归一化邻接矩阵幂预聚合 $S=\tilde{A}^K X$、伯努利率编码器、全连接线性层、LIF 神经元；在图像超像素图任务上额外包含**可训练 B 样条图卷积核（SplineCNN 式伪坐标核）与 IF 脉冲编码器**
- 关键数字：`Cora` / `CiteSeer` / `PubMed` 节点分类精度达 **80.9±0.6% / 72.5±0.7% / 77.6±0.6%**，理论能耗极低（Section 3.2 Table 2 & Table 3）；**视觉图像图实验（Section 3.3 Table 4 & Table 5 及 Appendix C.4，直接反驳 Claude 草稿“视觉图只有骨架有人做”的说法）**：① `MNIST` 28×28 像素网格图（784 节点，4 层深度）达 **99.35%**（Table 4）；② **`MNIST` 75 超像素图分类**（沿用 MoNet / SplineCNN 超像素图设置，$T=20$，10 次运行均值）取得 **94.50%** 准确率，逼近全精度 `SplineCNN` 的 95.22% 并超越 `MoNet` 的 91.11%（Table 5）！
- 与本方向的关系：**直接相关（一方面它是本项目现有 M0 超像素图任务在“直接训练阵营”里的直接前辈——早在 2022 年就跑了 MNIST 75 超像素图；另一方面它在附录 A 断言“自动转换算子不兼容、精度比不上直接训练”，正好构成本项目免训练低步保真转换要正面回应的靶子）**
- 核对人与日期：Gemini，2026-11-16

### GC-SNN 与 GA-SNN（2021）

- 标题：Exploiting Spiking Dynamics with Spatial-temporal Feature Normalization in Graph Learning
- 作者：Mingkun Xu, Yujie Wu, Lei Deng, Faqiang Liu, Guoqi Li, Jing Pei
- 发表处：Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence（IJCAI 2021，pp. 3171–3177；预印本 `arXiv:2107.06865`）
- 链接：https://arxiv.org/abs/2107.06865 ；代码：未公开
- 核对深度：全文
- 一句话：静态引文图上最早一批直接训练脉冲图网络的工作，构建脉冲图卷积（GC-SNN）与脉冲图注意力（GA-SNN），并提出按节点邻域空间维度与时间步同步归一化的时空特征归一化（STFN）。
- 解决的难点：解决了深层/多步脉冲图传递中的发放率失衡与梯度难以收敛问题，把基于 BPTT 的直接训练引入 GCN 与 GAT。
- 留下的局限：从头 BPTT 训练开销大（无法扩展到 `ogbn-arxiv` 等大图，在 `SpikeGCL` 表 1 中因显存溢出 OOM 缺失）；且 `GA-SNN` 中的注意力系数 $\alpha_{ij}$ 仍由浮点 LeakyReLU + Softmax 算出后再乘二值脉冲（并非完全事件化的注意力）。
- 是否需要训练或微调：是（使用空间—时间反向传播 STBP 从头直接训练，$T=10\sim 20$）
- 涉及的算子：度归一化邻接聚合（GC-SNN）、浮点 Softmax 图注意力系数加权二值脉冲（GA-SNN）、时空特征归一化（STFN）、LIF 脉冲神经元
- 关键数字：在 `Cora` / `CiteSeer` / `PubMed` 标准划分下（$T=20$，Section 4.2 Table 1），`GC-SNN` 分别达 **80.7±0.5% / 70.3±0.6% / 78.2±0.5%**，`GA-SNN` 分别达 **81.0±0.6% / 70.1±0.6% / 78.6±0.4%**；消融显示去掉 STFN 后 `Cora` 精度下降约 2–4 个百分点（Section 4.3）。
- 与本方向的关系：背景（通用图脉冲网络早期奠基工作；`GA-SNN` 中注意力权重仍用模拟浮点计算的遗留问题恰对应路线图中的 M2）
- 核对人与日期：Gemini，2026-11-16

### TactileSGNet（2020）

- 标题：TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition
- 作者：Fuqiang Gu, Weicong Sng, Tasbolat Taunyazov, Harold Soh
- 发表处：IEEE/RSJ International Conference on Intelligent Robots and Systems（IROS 2020，pp. 9908–9914，DOI: `10.1109/IROS45743.2020.9341421`；预印本 `arXiv:2008.08046`）
- 链接：https://doi.org/10.1109/IROS45743.2020.9341421 （预印本：https://arxiv.org/abs/2008.08046 ）；代码：https://github.com/clear-nus/TactileSGNet
- 核对深度：全文
- 一句话：最早将脉冲神经网络与图卷积结合处理异步传感器事件流的工作之一——把机器手指尖 NeuTouch 事件触觉传感器的 39 个不规则触觉单元建成空间图（手工图、$k$-NN 图或最小生成树图），用拓扑自适应图卷积（TAGConv）+ LIF 神经元直接识别抓取物体。
- 解决的难点：无需把非欧排布的异步触觉脉冲强行插值填零成二维像素网格，直接利用传感器几何邻接消费原生事件流。
- 留下的局限：从头使用矩形窗替代梯度训练 100 轮，且需要将 5–6.5 秒触觉序列按 0.02 s 分箱为多达 $T=250\sim 325$ 个时间步（对比后来 `PMSM` 用转换法在同数据集实现 $T=1$ 达 90.97%）。
- 是否需要训练或微调：是（使用 Adam + 矩形替代梯度 + MSE 投票损失从头训练 100 轮）
- 涉及的算子：传感器三维物理坐标手工连边 / $k$-NN（$k=1\sim 8$）/ 最小生成树（MST+$\sigma_d$）静态建图、多跳多项式拓扑自适应图卷积（TAGConv，$\sum_{k=0}^K g_k \mathbf{A}^k$）、LIF 脉冲神经元、时间均值投票解码
- 关键数字：在 36 类抓取物体数据集 `EvTouch-Objects` 与 20 类容器盛水量数据集 `EvTouch-Containers` 上（10 轮平均，Section IV-D Table II & Section IV-E Table III），手工连边图达 **89.44±0.55% / 64.17±2.75%**（超越网格脉冲 CNN 的 88.40% / 60.17% 与标准脉冲 GCN 的 85.14% / 58.83%）；改用自动 $k$-NN 建图后进一步提升至 **89.65±0.83%**（$k=7$，`EvTouch-Objects`）与 **67.00±1.39%**（$k=4$，`EvTouch-Containers`）。
- 与本方向的关系：背景（异步物理事件传感器 + 脉冲图网络的先驱工作，也是 2025 年 `PMSM` 做 GCN→SNN $T=1$ 转换实验所采用的标准基准来源）
- 核对人与日期：Gemini，2026-11-16

---

## 这条线是怎么演化的

脉冲图神经网络（Spiking GNN）在 2020–2026 年间大致经历了四个清晰的演化阶段，而将整条演进脉络摊开后，本项目所瞄准的空白位置与边界也随之水落石出：

1. **第一阶段：从异步传感器与小规模引文图起步的“直接训练”原型期（2020–2022）**  
   最早的探索始于天然输出脉冲的异步触觉传感器（`TactileSGNet`, IROS 2020），随后 `GC-SNN / GA-SNN`（IJCAI 2021）与 `SpikingGCN`（IJCAI 2022）把替代梯度直接训练搬到标准引文图乃至 **MNIST 75 超像素图像图**上。值得注意的是，`SpikingGCN` 在当时明确把“ANN→SNN 自动转换算子不兼容、精度不如直接训练”写进论文作为走直接训练路线的理由——这一论断深刻影响了此后整个图脉冲社区，使得后续九成以上的工作默认扎堆在“从头设计并训练一个脉冲图网络”。

2. **第二阶段：面向大规模图与动态图的训练降本增效期（2023–2024）**  
   直接用 BPTT 训练脉冲图网络很快撞上显存墙和梯度消失墙。于是通用图支线集中攻克训练可扩展性：`SpikeNet`（AAAI 2023）用采样配合自适应阈值替代动态图里的 RNN；`SpikeGCL`（ICLR 2024）切分特征通道并按时间块截断梯度做 1 比特对比学习（同时在理论上推导了减法复位 IF 逼近 ReLU GCN 的误差上界）；`Dy-SIGN`（AAAI 2024）用不动点隐式微分省去一半显存；`MSG`（NeurIPS 2024）则把反向微分搬到黎曼流形切空间以彻底绕开 BPTT 和替代梯度；`SpikeGraphormer` 与后来的 `GT-SNT`（AAAI 2026）进一步把脉冲机制引入线性复杂度图 Transformer。但正如 `SGNNBench`（NeurIPS 2025）横评 9 种代表模型后所揭示的那样：这条线全部囿于节点分类从头训练，不仅在深层异配图上大幅掉点（如 `Roman-Empire` 平均落后 31.27%），甚至出现了因首层编码和未脉冲化算子导致理论能耗反超全精度 GNN 的现象。

3. **第三阶段：三维点云与人体骨架等视觉图直接训练的爆发期（2023–2026）**  
   与 Claude 初稿“视觉图只有骨架有人做”的印象截然不同，视觉三维点云与事件点云的脉冲网络自 2023 年底起快速繁荣：从 `Spiking PointNet`（NeurIPS 2023）、事件相机时空点云 `SpikePoint`（ICLR 2024）、核点残差 `P2SResLNet`（AAAI 2024）、容易因标题被误认为转换法但实为替代梯度直接训练的 `Spiking PointCNN`（Electronics 2024），一路演进到 2025–2026 年的体素稀疏卷积 `E-3DSNN`（AAAI 2025）、局部 $k$-NN 向量注意力 `Spiking Point Transformer`（AAAI 2025）、脉冲动态 EdgeConv 点云去噪 `NI-SGCN`（AAAI 2025）、`Spiking Point Mamba`（ICCV 2025）以及 $T=1$ 的 `AMP2`（AAAI 2026）；人体骨架线也从结合 CTR-GCN 蒸馏的 `MK-SGN`（2024 预印 / Neurocomputing 2026）演进到时频域与拓扑移位注意力直接训练的 `Signal-SGN`（ACM MM 2025）和 `Signal-SGN++`（2025）。**但上述视觉模型无一例外都要耗费上百轮 GPU 从头训练**；其中 `NI-SGCN` 在二值脉冲上跑动态 EdgeConv 时还专门通过消融证明：二值脉冲特征会让原版 `max` 邻域聚合严重丢失几何细节，被迫改成 `mean` 聚合才能挽回 18.2% 的去噪误差。

4. **第四阶段：图与点云转换路线的零星破冰——以及真正留给本项目的精确空白（2023–2026）**  
   细致排查表明，“从来没人转过图或点云网络”并不准确，前人其实做过三次非常具体的尝试，但每一次都停在了关键门槛前：
   - **点云改结构重训转换**：`Lan et al.`（ICCV 2023）与 `Adaptive Calibration`（AAAI 2025 Oral）转换了 `PointNet` / `PointNet++`，但两者都明确因为**无法直接转换 `MaxPooling`**，被迫先把源网络的 `MaxPool` 改成 `AvgPool` 重新训练源 ANN，导致教师 `PointNet++` 在 `ScanObjectNN` 上本身就从 ~80% 跌到 69.22%；而未改结构做朴素转换的 `KPConv-SNN`（$T=40$）在 `ScanObjectNN` 上更是只剩 43.9%。
   - **触觉固定图量化重训转换**：`PMSM`（2025）把 3 层触觉 GCN 转成了 $T=1$ 零损 SNN，但前提是用专门的极性量化激活函数（PQA）从头重训源 GCN。
   - **文本固定图线性 GCN 免训练转换**：`One-Spike SNN`（IEEE TETC 2025）真正免训练地转换了预训练 2 层 ReLU Text GCN（平均仅掉 0.90%），但它高度依赖**静态固定邻接矩阵 + 纯线性矩阵乘聚合 + 纯 ReLU**。

   **结论非常清晰**：真正尚未被攻克的空白，不是笼统的“任何图网络转换”，而是**在不修改现成预训练视觉图骨干架构、不重新训练源模型的前提下，解决含逐层特征动态选邻居（$k$-NN / 膨胀 $k$-NN）、`max` 邻域聚合与现代非线性算子（如 ViG、DGCNN、原版 MaxPool PointNet++、ST-GCN/CTR-GCN）的低时间步保真转换，以及流式视觉图上的阈值化差分传播与公平量化能耗对照**。
