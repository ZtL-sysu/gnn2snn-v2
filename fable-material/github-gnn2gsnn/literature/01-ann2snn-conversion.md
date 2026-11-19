# 脉络一：ANN→SNN 转换（2015–2026）

本文件整理自 2015 年奠基以来将预训练人工神经网络（ANN）转换为脉冲神经网络（SNN）的核心脉络，重点覆盖：（1）把推理时间步从数千步压至个位数的理论里程碑；（2）**完全无需重训源网络（Training-Free / Post-Training）**的校准与编码路线；（3）Transformer 中自注意力、LayerNorm、Softmax 与 GELU 等非线性的转换手段；（4）`MaxPool` 的四类脉冲化/前置规程；（5）多阈值、带符号、分级脉冲、脉冲簇与差分编码神经元；以及（6）检测、分割、视频与文本图等分类以外任务的转换。

## 关键技术维度总览表

| 路线类别 | 代表工作（年份） | 是否需要重训/微调源 ANN | 典型时间步 $T$ | 对非线性与 `MaxPool` 的处理方式 |
| --- | --- | --- | --- | --- |
| **早期率编码与软复位奠基** | `Cao 等`（2015）、`Diehl 等`（2015）、`SNN Toolbox`（2017）、`Spike-Norm`（2019）、`RMP-SNN`（2020） | 部分（多需去偏置、去 BN 或改 `AvgPool` 训练源网；`SNN Toolbox` 支持折叠 BN） | $250\sim 4096$ | ReLU 映射发放率；`SNN Toolbox` 用滑动滤波门控近似 `MaxPool`，其余多改用 `AvgPool` |
| **改激活重训 / 量化感知转换（QAT）** | `RTS`（2021）、`QCFS`（2022）、`COS`（2023）、`Fast-SNN`（2023）、`MST`（2023）、`SpikeZIP-TF`（2024） | **是**（需将源网激活替换为阈值/阶梯量化函数并重训或蒸馏微调） | $2\sim 64$ | CNN 线全部要求把 `MaxPool` 改成 `AvgPool` 或步幅卷积；`SpikeZIP-TF` 用浮点累积差分包装器处理 Softmax/LN，并将 GELU 改为 ReLU 微调 |
| **免重训训练后校准（CNN 为主）** | `SNNC-AP`（2021）、`FTBC`（2024）、`Adaptive Calibration`（2025）、`Inference-Scale Conversion`（2025）、`Parallel Conversion`（2025，含免训练分支）、`Adaptive Fission`（2025） | **否 / 部分**（仅需百余至千余无标签样本做前向校准；`SNNC-AP` 进阶版含逐层权重微调） | $4\sim 64$ | `Inference-Scale Conversion` 利用 $\max\text{ReLU}(z)=\text{ReLU}(\max z)$ 将 `MaxPool` 移至神经元之前用硬件比较器计算 |
| **免训练多阈值 / 分级差分 / 相位编码** | `ECMT`（2024）、`One-Spike SNN`（2025）、`DCGS`（2025）、`MBE 脉冲浮点转换`（2026） | **否**（直接复用开源预训练 CNN / ViT / EVA / Text GCN 权重） | $1\sim 16$ | `ECMT`/`DCGS` 用浮点期望补偿模块算 GELU/LN/Softmax（`DCGS` 源码对 $k\neq s$ 的 `MaxPool` 退回模拟浮点计算）；`MBE` 用多基指数衰减神经元 + IEEE-754 指数尾数拆分全脉冲逼近非线性 |
| **脉冲 `MaxPool` 与检测/分割专用转换** | `Spiking-YOLO`（2020）、`Burst + LIPooling`（2022）、`Spiking MaxPooling 电路`（2022）、`SpiCalib 与 MLIPooling`（2022） | **否**（直接转换含 LeakyReLU 与 `MaxPool` 的检测/分割网络） | $32\sim 8000$ | 侧抑制池化 `LIPooling` / `MLIPooling` 或 Loihi 绝对值配对电路 `AVAM`；带符号双阈值神经元 `IBT` 处理负激活 |

---

## 逐篇文献核查记录（按发表年份从新到旧排列）

### MBE 脉冲浮点转换（2026）

- 标题：Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformers（PDF 第 1 页与 arXiv API；本条早先误写为另一标题，复查时已更正）
- 作者：Jingya Wang, Xin Deng, Wenjie Wei, Dehao Zhang, Shuai Wang, Qian Sun, Jieyuan Zhang, Hanwen Liu, Ning Xie, Malu Zhang
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2026；预印本 `arXiv:2508.07710`）
- 链接：https://arxiv.org/abs/2508.07710 ；代码：未公开
- 核对深度：全文
- 一句话：用离线在合成实数点上预拟合的多基指数衰减（MBE）脉冲神经元传输精确浮点值，配合外积脉冲乘法及 IEEE-754 指数/尾数位拆分，首次在不重训源模型且不回退主机模拟浮点的前提下将 ViT、RoBERTa 与 GPT-2 中的 GELU、LayerNorm、Softmax 全部脉冲化。
- 解决的难点：攻克了免训练转换中二值脉冲无法精确执行动态两激活乘法（$Q\cdot K^\top$）以及指数、平方根倒数等超越非线性运算的瓶颈，无需接触任何源任务训练数据。
- 留下的局限：时间步压到极低时精度雪崩——在 `ViT-M/16` 上 $T=16$ 虽达 85.31%，但 $T=10$ 即暴跌至 21.99%、$T=8$ 崩至 1.17%（Table 4）；且全文仅处理连续算术非线性，完全未涉及离散 `MaxPool` 或动态 $k$-NN 排序选邻居。
- 是否需要训练或微调：否（仅在 $M=10{,}000$ 个合成标量点上离线拟合一次 $N=4$ 个基神经元的通用系数，对下游预训练模型完全零训练、零数据校准）
- 涉及的算子：多基指数衰减神经元（MBE）、Hadamard 外积脉冲浮点乘法、基于 $e^x=2^{x\log_2 e}$ 与 IEEE-754 阶码/尾数拆分的脉冲 Softmax 与脉冲 LayerNorm、GELU / SiLU / ReLU
- 关键数字：`ImageNet-1K` 上（Section 5.2 Table 1，第 6 页），$T=16$ 时 `ViT-B/16` 从源 ANN **83.44%** 转为 **83.00%**（掉 0.44%），`ViT-M/16` 从 **85.95%** 转为 **85.31%**（掉 0.64%；但 $T=10$ 为 **21.99%**、$T=8$ 为 **1.17%**，Section 5.4 Table 4，第 7 页）；CNN 在 $T=10$ 时 `ResNet-34` 达 **75.57%**（源 76.31%）、`VGG-16` 达 **72.61%**（源 73.37%）；NLP 上 $T=16$ 时 `RoBERTa-Large` 在 `SST-2` 达 **95.98%**（源 96.22%，Table 2），`GPT-2` 在 `WikiText-2` 困惑度为 **22.69**（源 22.34，Table 3）。
- 与本方向的关系：可借用的技术（为视觉图骨干或混合图—注意力模型中的 GELU、LayerNorm 与边注意力 Softmax 提供了纯脉冲免训练逼近方案，但需警惕其在 $T\le 8$ 下的精度崩溃）
- 核对人与日期：Gemini，2026-11-16

### Adaptive Fission（2025）

- 标题：Adaptive Fission: Post-training Encoding for Low-latency Spike Neural Networks
- 作者：Yizhou Jiang, Feng Chen, Yihan Li, Yuqian Liu, Haichuan Gao, Tianren Zhang, Ying Fang
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2025 Poster，OpenReview ID: `2zZzdAMyYi`）
- 链接：https://openreview.net/forum?id=2zZzdAMyYi ；代码：未公开
- 核对深度：仅摘要（已核实 OpenReview 官方 API 元数据、完整摘要与公开审稿记录；PDF 下载端触发站点人机校验拦截，按规范如实标注仅摘要）
- 一句话：提出一种广义群体编码式的训练后神经元“自适应裂变”技术，按误差敏感度只把少数高敏感神经元拆分为具有不同尺度与权重的二值脉冲子神经元群，在不重训的前提下用极小空间冗余换取大幅延迟与功耗下降。
- 解决的难点：在不依赖硬件原生多位分级脉冲的前提下，仅靠纯二值脉冲神经元按需分配等效量化分辨率，缓解传统率编码对高时间步的依赖。
- 留下的局限：会改变高敏感通道的神经元扇出拓扑结构；未涉及图神经网络中消息传递算子、不规则邻接或动态 $k$-NN 建图的转换。
- 是否需要训练或微调：否（纯训练后编码与神经元拆分，无需重训或微调）
- 涉及的算子：敏感度引导的神经元自适应裂变（群体多尺度二值脉冲编码）、按需阈值与权重分配
- 关键数字：据 OpenReview 官方摘要披露，在类脑硬件实测中可在不降低精度的前提下实现最高 **80%** 的推理延迟与功耗缩减（Abstract；具体各数据集逐表数值因未获取全文记为未核实）。
- 与本方向的关系：可借用的技术（若目标硬件不支持 DCGS 式的分级/多阈值脉冲，可用“高敏感通道神经元裂变”在纯二值脉冲芯片上模拟等效多级精度）
- 核对人与日期：Gemini，2026-11-16

### DCGS（2025）

- 标题：Differential Coding for Training-Free ANN-to-SNN Conversion（`arXiv:2503.00301v3` 与 PDF 第 1 页；本条早先误写为另一标题，复查时已更正）
- 作者：Zihan Huang, Wei Fang, Tong Bu, Peng Xue, Zecheng Hao, Wenxuan Liu, Yuanhong Tang, Zhaofei Yu, Tiejun Huang
- 发表处：Proceedings of the 42nd International Conference on Machine Learning（ICML 2025，PMLR Vol. 267，pp. 25235–25264；预印本 `arXiv:2503.00301`）
- 链接：https://arxiv.org/abs/2503.00301 （OpenReview：https://openreview.net/forum?id=vB8cBscR6o ）；代码：https://github.com/h-z-h-cell/ANN-to-SNN-DCGS
- 核对深度：全文（已逐行核对论文正文与附录 A–M 全部表格，以及官方 GitHub 仓库全部 Python 源码）
- 一句话：将首步密集激活锚定（AC）、每步对输出变化量做正负 $2n$ 级二次幂阈值编码的差分编码分级脉冲（DCGS），以及逐通道异步网格搜索阈值校准（CAT）相结合，把大型 CNN 与视觉 Transformer 的免训练转换推至 $T=2\sim 4$ 步近无损精度。
- 解决的难点：消除了传统率编码在 $T=2\sim 4$ 超低步下的严重量化截断误差与层间延迟失配，用“只对相邻时间步的增量差发放带符号分级脉冲”换取高稀疏度。
- 留下的局限：（1）**源码实查不支持通用 `MaxPool` 与图算子**：`neurons/nonlinear_neuron.py` 第 22–114 行写明只要 `kernel_size != stride`（如 ResNet 首层 $3\times 3, s=2$），就绕过脉冲化直接对累积激活调浮点 `self.maxpool(x)`，仅对规则无重叠网格 $k=s$ 实现 `UniformUnpooling`；（2）GELU、LayerNorm、Softmax 依赖 `exp_comp_neuron` 在每步用浮点重算前缀期望差（非纯脉冲电路）；（3）能耗估算把任意多阈值分级脉冲按单次 `0.9 pJ` 累加计费且漏计模拟非线性层与多比较器开销；（4）官方仓库**未设置任何开源许可证**（默认保留全部权利），且缺失论文汇报的分割代码。
- 是否需要训练或微调：否（完全免训练，仅用少量校准集前向搜索阈值缩放因子，支持 $\times 1$ 与 $\times 2$ 网格搜索预算）
- 涉及的算子：激活中心化首步稠密锚定（`AtNeuron`）、正负对称 $2n$ 级二次幂阈值差分分级脉冲神经元（$\pm\theta/2^{i-1}, i=1,\dots,n$）、逐通道异步阈值搜索（CAT）、浮点期望补偿模块（`exp_comp_neuron` 处理 GELU / LayerNorm / Softmax）、脉冲矩阵乘补偿（`ProductNeuron`）、规则网格 `MaxPool2d`（$k=s$ 时 `UniformUnpooling`，$k\neq s$ 时退回浮点 `MaxPool2d`）
- 关键数字：`ImageNet-1K` 上（$n=8$ 即正负共 16 级阈值，Appendix K Table 6 & Appendix L Table 7），`EVA02-Large`（源 **90.05%**）在 $T=2 / 4 / 8$ 达 **88.25% / 89.72% / 89.90%**（能耗比 ER 为 **0.31 / 0.64 / 1.15**）；`ViT-Large`（源 **85.84%**）在 $T=4$ 达 **85.45%**（ER=0.46）；`ViT-Base`（源 **84.54%**）在 $T=4$ 达 **83.98%**（ER=0.54）；`ViT-Small`（源 **81.38%**）在 $T=4 / 8$ 达 **81.11% / 81.43%**（ER=0.62 / 1.05）；`ResNet-34`（源 **76.42%**，$\times 2$ 搜索）在 $T=4 / 8$ 达 **76.24% / 76.39%**（ER=0.32 / 0.50）；`MS COCO` 检测 `Mask R-CNN`（源 45.7% mAP）在 $T=4$ 达 **44.1%**（Table 3，第 7 页）；`ADE20K` 分割 `EVA-02+UperNet`（源 53.1% mIoU）在 $T=4$ 达 **52.18%**（Table 4，第 8 页）。
- 与本方向的关系：直接相关（本项目最重要的免训练超低步转换参照基线；对其论文与源码差异的核查见定向问题 Q5）
- 核对人与日期：Gemini，2026-11-16

### Parallel Conversion（2025）

- 标题：Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation（PDF 第 1 页与 arXiv API；本条早先误写为另一标题，复查时已更正）
- 作者：Zecheng Hao, Qichao Ma, Kang Chen, Yi Zhang, Zhaofei Yu, Tiejun Huang
- 发表处：Proceedings of the 42nd International Conference on Machine Learning（ICML 2025，PMLR Vol. 267；预印本 `arXiv:2412.13610`）
- 链接：https://arxiv.org/abs/2412.13610 ；代码：https://github.com/hzc1208/Parallel_Conversion
- 核对深度：全文
- 一句话：利用转换任务中输入恒定的性质推导累积发放数闭式跳变边界，将串行时间步展开的 IF 神经元改写为可一步并行计算的并行脉冲神经元（PSN），并提供免重训的 `ReLU → ClipReLU → DA-QCFS` 贪心通道校准流水线。
- 解决的难点：打破了传统转换 SNN 必须在 GPU 上串行循环 $T$ 个时间步的仿真墙，用 $O(T)$ 前缀充电加 $O(\log T)$ 二分搜索在 $T\ge 32$ 时获得 19–38 倍 GPU 推理加速，同时兼容 QAT（QCFS）与免训练转换。
- 留下的局限：闭式并行跳变条件依赖“输入在全部 $T$ 个时间步内保持恒定或严格阶梯单调”的静态假设，无法直接迁移到动态重算拓扑或流式异步变化的视觉图输入；且免训练分支仍需 $T=16\sim 32$ 步才能逼近原精度。
- 是否需要训练或微调：部分（支持两种模式：一是直接对接需重训的 QCFS 模型；二是完全免训练的 `ReLU → ClipReLU → DA-QCFS` 训练后逐通道贪心误差校准）
- 涉及的算子：并行脉冲神经元（PSN，预计算累积权重矩阵 $\Lambda_{\text{PC}}^l$ 与二分搜索脉冲计数）、免训练剪裁阈值与通道残差贪心校准（`DA-QCFS`）、BatchNorm 折叠、`AvgPool`
- 关键数字：在 GPU 实测中（Section 5.3 Figure 2，第 8 页），当 $T\ge 32$ 时较串行 IF 仿真提速 **19×–38×**；对接 QCFS 源模型时，`ImageNet` `ResNet-34`（源 **74.30%**）在 $T=4 / 8$ 达 **72.90% / 74.32%**（Section 5.1 Table 3，第 7 页）；**纯免训练转换标准 TorchVision ReLU 模型**时（Section 5.2 Table 4，第 8 页），`ResNet-34`（源 **73.31%**）在 $T=16 / 32$ 达 **68.04% / 72.46%**，`ResNet-50`（源 **76.12%**）达 **73.59% / 75.71%**，`ResNet-101`（源 **77.38%**）达 **73.86% / 76.42%**。
- 与本方向的关系：可借用的技术（在大规模静态视觉图失效普查实验中，可借用其闭式并行仿真思想大幅加速多时间步率编码基线的 GPU 评测）
- 核对人与日期：Gemini，2026-11-16

### Inference-Scale Conversion（2025）

- 标题：Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and Low-Power Applications
- 作者：Tong Bu, Maohua Li, Zhaofei Yu
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2025；预印本 `arXiv:2409.03368`）
- 链接：https://arxiv.org/abs/2409.03368 ；代码：https://github.com/putshua/Inference-scale-ANN-SNN
- 核对深度：全文
- 一句话：提出仅需单次 ANN 推理级开销（RTX 4090 上 93 秒）的免训练转换框架：用局部学习率更新逐通道阈值，**利用交换律把 `MaxPool` 移到脉冲神经元之前由硬件比较器处理膜前连续值**，并估计首脉冲传播延迟做延迟区间统计，统一打通分类、分割、检测与视频分类。
- 解决的难点：无需像 `SNNC-AP` 那样对每个目标 $T$ 反复跑脉冲前向和权重微调，更巧妙绕开了二值脉冲流上做 `MaxPool` 的相位错位高估难题（注：论文第 5 页英文写 "replace all max-pooling layers with neuron layers"，结合紧随其后的 Theorem 2 实指**互换 `MaxPool` 层与神经元层的先后顺序**，即前置比较器池化而非用脉冲神经元拟合 `max`）。
- 留下的局限：**Theorem 2 的层序交换恒等式 $\max_i\text{ReLU}(z_i)=\text{ReLU}(\max_i z_i)$ 严格依赖先线性卷积、紧接 ReLU 再接 `MaxPool` 且 $\max z > 0$ 的 CNN 拓扑**；而在 ViG 的 `MRConv` 中，$\max_{j\in\mathcal{N}(i)}(x_j-x_i)$ 位于上一层激活之后、当前层线性映射之前且相对差值 $x_j-x_i$ 可正可负，不能直接套用层序对调，且该文仅支持 CNN，仍需 $T=32\sim 64$ 步。
- 是否需要训练或微调：否（直接加载 TorchVision 等开源预训练权重，仅需少量样本做推理级复杂度局部阈值搜索，一次校准通用于任意时间步 $T$）
- 涉及的算子：逐层/逐通道局部阈值平衡（CLTB）、**神经元前最大池化（Pre-Neuron Max Pooling，硬件比较器处理膜前连续值）**、首脉冲传播延迟估计与延迟求均值评估（Delayed Evaluation）、初始膜电位 $\theta^l/2$ 预充
- 关键数字：`ImageNet` 标准 TorchVision `ResNet-34`（源 **73.31%**）免训练转换仅耗时 **92.73 秒**（Appendix G Table S5，第 15 页），在 $T=32 / 64 / 128$ 达 **64.88% / 71.17% / 72.30%**（Table 2，第 7 页；若从无前置池化交换退化到基础局部平衡则在低步明显掉点）；密集预测与视频任务（Section 5.6 Table 3，第 8 页）：`Pascal VOC` `DeepLab-34`（源 58.95% mIoU）在 $T=64 / 128$ 达 **55.74% / 58.36%**，`MS COCO` 分割 `DeepLab-50`（源 63.01%）在 $T=128$ 达 **61.52%**，`MS COCO` 检测 `RetinaNet-50`（源 36.4% mAP）在 $T=64 / 128$ 达 **27.3% / 32.2%**，`S-Kinetics-400` 视频分类 `MC3-18`（源 63.52%）在 $T=64$ 达 **61.60%**。
- 与本方向的关系：直接相关（回答定向问题 Q10 及核查 Claude 草稿“改成神经元层”误读的关键文献；其“对连续累积量用硬件比较器取 `max` 再进脉冲编码”的思想可直接改造为图 `MRConv` 的节点前缀重构比较器规程）
- 核对人与日期：Gemini，2026-11-16

### Adaptive Calibration（2025）

- 标题：Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks
- 作者：Ziqing Wang, Yuetong Fang, Jiahang Cao, Hongwei Ren, Renjing Xu
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2025 Oral；预印本 `arXiv:2412.16219`）
- 链接：https://arxiv.org/abs/2412.16219 ；代码：https://github.com/bic-L/burst-ann2snn
- 核对深度：全文（含论文全文与官方开源仓库 2D/3D 代码核验）
- 一句话：用输出 KL 散度灵敏度分析为不同层自动分配最优脉冲簇发放上限（Burst-firing）并结合运行期自适应降阈值发放（AdaFire）与偏置校准，在约 5 分钟免训练校准内同时覆盖 2D 分类/检测/分割与改进版 PointNet 3D 点云任务。
- 解决的难点：同时解决了前期高激活截断（靠多层差异化脉冲簇上限释放）与初期沉默神经元欠发放（靠 AdaFire 动态微调阈值），并给出匹配商用芯片（Loihi 2、Xylo）脉冲簇指令的低开销部署路径。
- 留下的局限：核查其官方仓库 `3dtasks/models/model.py` 证实，为了转换 PointNet 分类与分割模型，作者**预先把原版 PointNet 中的全局 `torch.max` 对称池化全部硬改成了 `torch.mean` 平均池化并重新训练源模型**，仍未正面解决点集/图邻域 `max` 聚合算子的免重训转换。
- 是否需要训练或微调：部分（对本身不含 `MaxPool` 瓶颈的预训练 2D 网络完全免重训，校准仅约 0.09 小时；但对 3D PointNet 需先将 `max` 池化改为 `mean` 池化重训源 ANN）
- 涉及的算子：KL 灵敏度引导的分层脉冲簇（Burst-firing，`maxspike` 搜索）、自适应阈值衰减（AdaFire）、偏置校准、BatchNorm 折叠、改用 `mean` 池化的 PointNet
- 关键数字：校准耗时约 **0.09 小时**（Section 4）；在 `CIFAR-10` / `CIFAR-100` / `ImageNet` 上较基线降低 **70.1% / 60.3% / 43.1%** 能耗；多任务结果（Figure 1 及官方配置）：改进型 PointNet 3D 分类（$T=16, \text{maxspike}=4$）达约 **94.0%**（源约 96.0%），ShapeNet Part 3D 分割（$T=8$）达约 **74.75% mIoU**（源约 75.5%）。
- 与本方向的关系：直接相关（连接脉络一与脉络二的桥梁文献：既提供分层灵敏度分配脉冲位宽的工具，又从源码层面证实了现有转换法遇到几何 `max` 聚合时的回避做法）
- 核对人与日期：Gemini，2026-11-16

### One-Spike SNN（2025）

- 标题：One-Spike SNN: Single-Spike Phase Coding With Base Manipulation for ANN-to-SNN Conversion Loss Minimization
- 作者：Sangwoo Hwang, Jaeha Kung
- 发表处：IEEE Transactions on Emerging Topics in Computing（IEEE TETC 2025，Vol. 13, Issue 3，pp. 1020–1032，DOI: `10.1109/TETC.2025.3558454`；预印本 `arXiv:2403.08786`）
- 链接：https://doi.org/10.1109/TETC.2025.3558454 （预印本：https://arxiv.org/abs/2403.08786 ）；代码：未公开
- 核对深度：全文
- 一句话：把整数二次幂相位编码推广为非整数底数 $B \in (1,2)$（如 $B=1.5$）并加两遍扫描抑制冗余脉冲，使每个激活神经元在整个推理窗口内尽量只发 1 个脉冲，且**在 5 个文本图数据集上免训练转换了预训练 2 层 ReLU Text GCN**。
- 解决的难点：攻克了传统 $B=2$ 相位编码在连续实数上因表达空隙导致的严重转换掉点，无需修改或重训源模型即可兼顾高稀疏度与精度，并给出了面向固定静态线性 GCN 的首个免训练转换实测。
- 留下的局限：（1）每个神经元虽平均仅发约 1 个脉冲，但为了扫过分数底数相位阶梯仍需 $T=16\sim 22$ 个串行时钟拍；（2）每个时钟拍需将发放阈值乘以分数底数 $B^{-t}$，带来额外硬件控制复杂度；（3）仅适用于 `ReLU` + 线性加权求和（静态 Text GCN 邻接乘），不支持 `MaxPool`、GELU 或动态 $k$-NN 建图。
- 是否需要训练或微调：否（纯训练后校准每层 99.9% 分位数缩放并套用分数底数两阶段相位编码）
- 涉及的算子：分数底数（$B=1.5$）两遍单脉冲相位编码（Phase Coding）、99.9% 激活分位数阈值校准、ReLU、卷积与全连接、静态归一化邻接矩阵乘法（Text GCN）
- 关键数字：在 5 个文本词—文档图基准（`20NG`, `R8`, `R52`, `Ohsumed`, `MR`）上免训练转换预训练 2 层 ReLU Text GCN（$B=1.5, T=16$，Section 4.3 Table 2，第 8 页），平均精度仅从 **81.98%** 微降至 **81.08%**（掉点 **0.90%**，其中 `R8` 从 97.07% 转为 **96.98%**），较传统 $B=2$ 相位编码减损 **89%** 并节能 **66%**；`ImageNet` `ResNet-34`（$B=1.5, T=22$）达 **71.50%**（源 73.30%），45nm ASIC 综合较浮点基线节能 **5.9×**（Section 4.2 & 4.4）。
- 与本方向的关系：直接相关（回答定向问题 Q1“有没有人把普通图网络免训练转成脉冲网络”的核心阳性证据：在静态固定图线性 ReLU Text GCN 上做过，但视觉图与非线性聚合仍属空白）
- 核对人与日期：Gemini，2026-11-16

### SpikedAttention（2024）

- 标题：SpikedAttention: Training-Free and Fully Spike-Driven Transformer-to-SNN Conversion with Winner-Oriented Spike Shift for Softmax Operation
- 作者：Sangwoo Hwang, Seunghyun Lee, Dahoon Park, Donghun Lee, Jaeha Kung
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2024，Vol. 37，pp. 67422–67445，DOI: `10.52202/079017-2152`）
- 链接：https://proceedings.neurips.cc/paper_files/paper/2024/hash/7c9341ad0263428b5057d92f4d88dfa0-Abstract-Conference.html ；代码：https://github.com/sangwoohwang/SpikedAttention
- 核对深度：全文
- 一句话：通过维护 $Q, K$ 历史脉冲迹在脉冲域增量计算注意力点积，并用赢家导向脉冲移位（WOSS）电路在无需浮点指数单元的条件下全脉冲逼近 Softmax，还给出 45nm 硬件综合面积与功耗数据。
- 解决的难点：实现了自注意力 $Q K^\top$ 点积、Softmax 与加权 $V$ 的纯脉冲硬件流水线，证明专用 WOSS Softmax 单元仅占加速器总面积的 9.88%。
- 留下的局限：**标题虽写 Training-Free，但全文不支持标准 Transformer 的 `LayerNorm` 和 `GELU`**——论文第 8–10 页（Section 5.1–5.2 与 Section 6 Limitations）明确交代，被转源模型必须是**预先把 `LayerNorm` 全换成 `BatchNorm`、把 `GELU` 全换成 `ReLU` 重新训练好的定制变体**（如 `Swin-T (BN)` 与 `MA-BERT`），无法直接转换现成开源的 GELU/LN 视觉骨干！同时推理仍需 $T=16\sim 40$ 步。
- 是否需要训练或微调：部分（转换过程本身免训练，但**要求源 Transformer 预先以 `BatchNorm + ReLU` 替代 `LayerNorm + GELU` 完成训练**）
- 涉及的算子：分数底数（$B=1.15$）相位编码、基于脉冲迹（Spike Trace）的双脉冲流点积乘法、赢家导向脉冲移位 Softmax（`WOSS`）、BatchNorm 折叠、ReLU
- 关键数字：`ImageNet-1K` 上转换定制 `Swin-T (BN)`（源 **80.6%**，$T=40, B=1.15$，Section 5.1 Table 1，第 8 页）达 **80.0%**（掉 0.6%），估算能耗从 45.4 mJ 降至 **25.6 mJ**（节省 **42%**）；`GLUE` 基准转换定制 `MA-BERT`（$T=16\sim 24$，Table 3，第 9 页）平均仅掉 **0.3%**、节能 **58%**；45nm CMOS 综合显示 `WOSS` 模块仅增加 **9.88%** 芯片面积与 **12.35%** 功耗（Appendix Table 4，第 15 页）。
- 与本方向的关系：可借用的技术（一方面其 `WOSS` 赢家移位电路对图注意力 GAT（M2）的边 Softmax 脉冲化极具参考价值；另一方面提醒我们核查“免训练转换”论文时必须紧盯源模型是否偷偷把 LN/GELU 改成了 BN/ReLU——好在主流卷积式 ViG 教师恰好原生采用 `BatchNorm`！）
- 核对人与日期：Gemini，2026-11-16

### ECMT（2024）

- 标题：Towards High-performance Spiking Transformers from ANN to SNN Conversion
- 作者：Zihan Huang, Xinyu Shi, Zecheng Hao, Tong Bu, Jianhao Ding, Zhaofei Yu, Tiejun Huang
- 发表处：Proceedings of the 32nd ACM International Conference on Multimedia（ACM MM 2024，pp. 10688–10697，DOI: `10.1145/3664647.3680852`；预印本 `arXiv:2502.21193`）
- 链接：https://arxiv.org/abs/2502.21193 ；代码：https://github.com/h-z-h-cell/Transformer-to-SNN-ECMT
- 核对深度：全文
- 一句话：`DCGS` 的直接前身——用正负 $2n$ 级二次幂多阈值神经元（MT）降低单步量化误差，并为非线性算子（GELU/LN/Softmax）与双脉冲流矩阵乘设计按前 $t$ 步累积均值计算期望增量的补偿模块（ECM），首次在 $T=4$ 下免训练转换十亿参数级视觉大模型 EVA。
- 解决的难点：打破了此前免训练 Transformer 转换（如 `STA`）动辄需要 $T=32\sim 256$ 步的延迟瓶颈，在不修改源 ViT / EVA 任何结构与权重的条件下把步数压到 $T=2\sim 8$。
- 留下的局限：ECM 模块本质是在每个时间步保留浮点状态并重算 $T\cdot F(S(T)/T)-(T-1)\cdot F(S(T-1)/(T-1))$（公式 15），属于数模/浮整混合执行而非纯脉冲神经元动力学；且未做激活中心化与差分增量稀疏化（该缺点随后在 `DCGS` 中改进），亦未探讨 `MaxPool` 或图邻域算子。
- 是否需要训练或微调：否（仅需少量校准样本前向统计阈值，零重训、零微调）
- 涉及的算子：$2n$ 级正负二次幂多阈值神经元（MT，$\pm 2^{p-1}\theta_{1,2}$）、非线性期望补偿模块（ECM，覆盖 LayerNorm / Softmax / GELU）、矩阵乘期望补偿模块（Matrix-Product-EC）
- 关键数字：`ImageNet-1K` 免训练转换结果（Section 4.2 Table 1，第 7 页）：`EVA`（源 **89.62%**）在 $T=2 / 4$ 达 **87.38% / 88.60%**（能耗比 ER=**0.19 / 0.35**）；`ViT-L/16`（源 **84.88%**）在 $T=4 / 8$ 达 **83.20% / 84.48%**（ER=**0.27 / 0.56**）；`ViT-B/16`（源 **80.77%**）在 $T=4 / 8$ 达 **76.27% / 79.40%**（ER=0.33 / 0.66）。
- 与本方向的关系：可借用的技术（其“节点侧维护前缀均值重构 $S(t)/t$、算完非线性后再输出本步差分增量 $O(t)-O(t-1)$”的数学骨架，可直接迁移到视觉图的 `MRConv` 有符号相对 `max` 聚合中）
- 核对人与日期：Gemini，2026-11-16

### SpikeZIP-TF（2024）

- 标题：SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN
- 作者：Kang You, Zekai Xu, Chen Nie, Zhijie Deng, Xiang Wang, Qinghai Guo, Zhezhi He
- 发表处：Proceedings of the 41st International Conference on Machine Learning（ICML 2024，PMLR Vol. 235；预印本 `arXiv:2406.03470`）
- 链接：https://arxiv.org/abs/2406.03470 ；代码：https://github.com/Intelligent-Computing-Research-Group/SpikeZIP-TF
- 核对深度：全文
- 一句话：先将预训练 Transformer 的 GELU 换成 ReLU 微调并做 LSQ 量化感知蒸馏训练（QAT），再通过三值正负脉冲神经元 `ST-BIF+`、三项分解脉冲自注意力 `SESA` 与累积差分包装器等效转换为脉冲 Transformer。
- 解决的难点：证明了在相同时间步或更低功耗下，基于 QAT 转换得到的脉冲 ViT 在 ImageNet 精度上显著优于从头直接训练的 `Spikformer` / `Spike-driven Transformer` 系列。
- 留下的局限：（1）**绝非免训练转换**——Appendix A.4（第 15–17 页）明确记载需先做 50–100 轮 `GELU → ReLU` 替换微调，再做 50–300 轮带知识蒸馏的 LSQ 量化感知训练；（2）所谓 `Spike-Softmax` 与 `Spike-LayerNorm`（公式 5）实际上仍是对累积输入 $X_t$ 直接调用数字浮点 $\sigma(X_t)$ 后输出相邻步差值 $O_t-O_{t-1}$（Claude v1 草稿笼统称其“设计了脉冲化的 Softmax 和 LayerNorm”并不准确）；（3）推理仍需 $T=32\sim 128$ 步。
- 是否需要训练或微调：是（需两阶段重训微调：50–100 轮 GELU→ReLU 微调 + 50–300 轮 LSQ QAT 蒸馏训练）
- 涉及的算子：对称三值双向积分发放神经元（`ST-BIF+`，脉冲取值 $\{-1,0,+1\}$）、三项增量更新脉冲等效自注意力（`SESA`）、数字累积差分包装算子 `Spike-Softmax` / `Spike-LayerNorm`、LSQ 激活量化与 ReLU
- 关键数字：`ImageNet-1K` 上（Section 4.2 Table 5，第 7 页），`SViT-S-32Level` / `SViT-B-32Level` / `SViT-L-32Level` 在 $T=64$（对应 32 量化级×2）分别达 **81.45% / 82.71% / 83.82%**（源量化 ViT 为 81.59% / 82.85% / 83.86%）；NLP `RoBERTa` 在 `SST-2`（$T=128$，Table 6，第 8 页）达 **93.79%**、`MNLI` 达 **85.03%**。
- 与本方向的关系：可借用的技术（`ST-BIF+` 正负三值脉冲神经元解决负向过发放引起的残差偏置，对含负特征的图聚合有启发；同时也是定向问题 Q3 被引追踪的五篇种子论文之一）
- 核对人与日期：Gemini，2026-11-16

### FTBC（2024）

- 标题：FTBC: Forward Temporal Bias Correction for Optimizing ANN-SNN Conversion
- 作者：Xiaofeng Wu, Velibor Bojkovic, Bin Gu, Kun Suo, Kai Zou
- 发表处：European Conference on Computer Vision（ECCV 2024，LNCS Vol. 15105，pp. 155–173；预印本 `arXiv:2403.18388`）
- 链接：https://arxiv.org/abs/2403.18388 ；代码：https://github.com/shizukanaskytree/FTBC
- 核对深度：全文
- 一句话：无需反向传播，只用一个校准批次做纯前向统计，在**每一个时间步 $t$ 独立计算逐通道均值残差并注入随时间变化的动态偏置 $B^{(\ell)}(t)$**，可即插即用叠加在标准 ReLU 网络或 QCFS 网络之上。
- 解决的难点：传统偏置校准（如 `SNNC-AP` 的 BC）只在总时间步 $T$ 末端对齐总均值，忽略了早期时间步的瞬态偏差积累；按步校准 $B^{(\ell)}(t)$ 用几秒前向计算即可在低步下显著提分。
- 留下的局限：单独用于未经 QCFS 重训的标准 ReLU `ResNet-34` 时，在 $T\le 8$ 下精度依然很低（$T=16$ 仅 60.68%），若要压到 $T=4$ 仍需搭配重训过的 QCFS 源网络；且按空间高宽求通道均值的公式假定规则二维特征图，未处理 `MaxPool` 或动态图。
- 是否需要训练或微调：否（纯前向逐时间步均值误差统计，支持直接校准预训练 ReLU ANN，也可叠加于 QCFS 模型）
- 涉及的算子：逐层逐时间步通道动态偏置注入 $B^{(\ell)}(t)$（Algorithm 1，第 9 页）、百分位数/MMSE 阈值初始化、IF 软复位神经元
- 关键数字：`ImageNet` 上直接免训练转换标准 ReLU 模型（`Ours*`，Section 4.2 Table 1，第 10 页）：`VGG-16`（源 **75.36%**）在 $T=8 / 16 / 32$ 达 **64.20% / 71.19% / 73.89%**；`ResNet-34`（源 **75.66%**）在 $T=16 / 32 / 64$ 达 **60.68% / 70.88% / 73.79%**；若叠加在 QCFS 重训源模型上，则 `ResNet-34` 在 $T=4 / 8$ 达 **68.76% / 72.73%**，`VGG-16` 在 $T=4 / 8$ 达 **71.22% / 73.32%**（Table 1）。
- 与本方向的关系：可借用的技术（把二维特征图空间均值改成图节点集 $V$ 上的均值，即可零成本移植为图转换 baseline 的逐时间步节点偏置校准器）
- 核对人与日期：Gemini，2026-11-16

### STA（2024）

- 标题：Spiking Transformer: Spatial-Temporal Attention for Converting Any Pretrained Transformers into Spiking Neural Networks（OpenReview 正式收录标题为 *Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers*）
- 作者：Yizhou Jiang, Kunlin Hu, Tianren Zhang, Haichuan Gao, Yuqian Liu, Ying Fang, Feng Chen
- 发表处：The Twelfth International Conference on Learning Representations（ICLR 2024，OpenReview ID: `135eKq323G`）
- 链接：https://openreview.net/forum?id=135eKq323G ；代码：https://github.com/ViviaHu/STA
- 核对深度：全文
- 一句话：首个**完全不修改源架构、不微调**的预训练 Transformer 全脉冲转换方法：用离线在合成随机数据上预训练的小型脉冲子网（通用群算子 UGO）替代 GELU、LayerNorm、Softmax，并用时间步间布尔“与（AND）”加记忆修正逼近动态脉冲流乘法。
- 解决的难点：无需源训练数据、无需把 LN/GELU 改成 BN/ReLU，直接将 CLIP ViT 等现成预训练多模态视觉基座转换为纯二值脉冲网络完成零样本分类。
- 留下的局限：时间步开销极大（ImageNet 上需 $T=64\sim 256$ 步才能接近源精度）；且每个标量 GELU/指数/平方根运算都要展开成 $N=8\sim 32$ 个隐层 IF 神经元组成的 UGO 子网，神经元与突触规模显著膨胀，也未涉及离散选邻居。
- 是否需要训练或微调：否（UGO 仅在合成分布数据上离线拟合一次即可复用，对预训练 Transformer 完全免训练）
- 涉及的算子：通用群算子（UGO，$N\in\{8,16,32\}$ 个隐层 IF 神经元拟合 GELU / 指数 / 倒数 / 平方根）、时空注意力脉冲点积（TCSA，布尔 AND 估计—修正单元）、逐神经元 MMSE 阈值校准、突触噪声调制（SNM）与脉冲簇（$\Gamma=2$）
- 关键数字：零样本转换预训练 `CLIP ViT-B/32`（Section 4.2 Table 1，第 8 页）：`CIFAR-10`（源 **89.74%**）在 $T=32 / 64 / 256$ 达 **87.71% / 88.88% / 89.56%**；微调版 `ViT-B/32` 在 `ImageNet-1K`（源 **83.60%**，Appendix Table 5，第 21 页）在 $T=32 / 64 / 128 / 256$ 分别达 **78.72% / 82.33% / 82.61% / 82.79%**。
- 与本方向的关系：可借用的技术（确立了免训练全脉冲 Transformer 转换的可行性下限，同时也是定向问题 Q3 被引检索的五篇核心种子论文之一）
- 核对人与日期：Gemini，2026-11-16

### Fast-SNN（2023）

- 标题：Fast-SNN: Fast Spiking Neural Network by Converting Quantized ANN
- 作者：Yangfan Hu, Qian Zheng, Xudong Jiang, Gang Pan
- 发表处：IEEE Transactions on Pattern Analysis and Machine Intelligence（IEEE TPAMI 2023，Vol. 45, No. 12，pp. 14546–14562，DOI: `10.1109/TPAMI.2023.3275769`；预印本 `arXiv:2305.19868`）
- 链接：https://arxiv.org/abs/2305.19868 ；代码：https://github.com/yangfan-hu/Fast-SNN
- 核对深度：全文
- 一句话：严格证明 $b$-bit 无符号均匀量化激活等价于 $T=2^b-1$ 步脉冲率编码，通过引入带正负符号的积分发放神经元（SIF）消除发放顺序错位引起的负残余误差，并用逐层代理网络微调将分类、检测与分割统一压至 $T=3, 7, 15$ 步。
- 解决的难点：从数学上打通了“$b$-bit 量化网络”与“$2^b-1$ 步带符号脉冲网络”的等价性，解释了单极性 IF 神经元因先正后负脉冲到达顺序引起多发错脉冲（spurious spikes）的机理并用负向纠错脉冲抵消。
- 留下的局限：**重度依赖重训与改结构**——不仅要求对源网络做量化感知训练（QAT）和逐层最小二乘微调，而且明确要求**删掉源网络中的 `MaxPool` 层并改用步幅为 2 的卷积下采样重训**（Section 4.1）。
- 是否需要训练或微调：是（需先移除 `MaxPool` 做 $b$-bit QAT 训练，转换时再对每层做代理 ANN 梯度微调）
- 涉及的算子：带符号积分发放神经元（Signed IF / SIF，带负纠错脉冲抑制伪发放）、$b$-bit 均匀激活量化（对应 $T=2^b-1$）、移除 `MaxPool` 改步幅卷积、逐层时空误差代理微调
- 关键数字：`ImageNet` 上（Section 4.2 Table 5，第 9 页），`VGG-16` 在 $T=3 / 7$ 达 **71.31% / 72.95%**，`ResNet-34` 在 $T=3 / 7$ 达 **66.22% / 71.31%**；`PASCAL VOC 2007` 目标检测 `YOLOv2 (ResNet-34)`（源 76.08% mAP，Section 4.4 Table 9，第 12 页）在 $T=3 / 7 / 15$ 达 **62.51% / 73.43% / 75.77% mAP**；`PASCAL VOC 2012` 语义分割 `ResNet-34+ASPP`（源 72.1% mIoU，Table 10，第 14 页）在 $T=7 / 15$ 达 **66.7% / 69.7% mIoU**。
- 与本方向的关系：可借用的技术（SIF 用负脉冲撤回早期误发正脉冲的机制，与脉冲图 `max` 池化抑制相位高估同源；其 $T=2^b-1 \leftrightarrow b\text{-bit}$ 精确对应关系直接支撑 M4 同比特量化公平对照设计）
- 核对人与日期：Gemini，2026-11-16

### MST（2023）

- 标题：Masked Spiking Transformer
- 作者：Ziqing Wang, Yuetong Fang, Jiahang Cao, Qiang Zhang, Zhongrui Wang, Renjing Xu
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2023，pp. 1761–1771；预印本 `arXiv:2210.01208`）
- 链接：https://arxiv.org/abs/2210.01208 ；代码：未公开
- 核对深度：全文
- 一句话：早期将 Swin Transformer 转换为脉冲网络的尝试——把源模型的 `LayerNorm` 改成 `BatchNorm`、激活改成 `QCFS` 从头重训，再引入随机脉冲掩码（RSM）配合自蒸馏微调削减脉冲数量。
- 解决的难点：作为首批跑通层次化窗口视觉 Transformer（Swin-T）转换的工作，验证了随机掩蔽冗余脉冲在保持精度的同时可额外省电约四分之一。
- 留下的局限：（1）必须大改源架构并从头重训 + 蒸馏微调；（2）注意力里的 `Softmax` 依然保留原始模拟浮点运算未做脉冲化；（3）即便用了 QCFS，在 ImageNet 上仍需 $T=128\sim 512$ 步才能达到 77.88%–78.51%。
- 是否需要训练或微调：是（需把 `Swin-T` 的 LN 换 BN、GELU/ReLU 换 QCFS 从头训练，再为随机脉冲掩码做知识蒸馏微调）
- 涉及的算子：QCFS 量化阶梯激活、BatchNorm 替代 LayerNorm、随机脉冲掩码（RSM）、模拟浮点 Softmax 窗口注意力
- 关键数字：`ImageNet-1K` 上改装版 `Swin-T`（Section 4.2 Table 1，第 5 页）在 $T=128 / 256 / 512$ 分别达 **77.88% / 78.37% / 78.51%**（注：0% 掩码下 `Swin-T` 在 $T=128$ 为 78.51% 的接近值，开启 RSM 后在 `CIFAR-10` 上 75% 掩码率取得 **97.29%** 并降低 **26.8%** 功耗，Section 4.3 Table 3，第 8 页）。
- 与本方向的关系：背景（展示了早期视觉 Transformer 转换不得不“改结构重训 + 保留浮点 Softmax + 数百时间步”的历史局限）
- 核对人与日期：Gemini，2026-11-16

### COS（2023）

- 标题：Bridging the Gap Between ANNs and SNNs by Calibrating Offset Spikes
- 作者：Zecheng Hao, Jianhao Ding, Tong Bu, Tiejun Huang, Zhaofei Yu
- 发表处：The Eleventh International Conference on Learning Representations（ICLR 2023；预印本 `arXiv:2302.10685`）
- 链接：https://arxiv.org/abs/2302.10685 ；代码：https://github.com/hzc1208/ANN2SNN_COS
- 核对深度：全文
- 一句话：通过观察神经元在结束时刻的残余膜电位 $v^l(T)$ 是否落在异常区间（$<0$ 或 $>\theta^l$）来识别脉冲到达不均匀导致的“多发/少发偏移脉冲”，在推理期对每个测试样本先跑 $\rho$ 步预探测调整初始膜电位，再正式跑 $T$ 步。
- 解决的难点：首次专门针对 `QCFS` 遗留的第三类误差——“不均匀性误差（unevenness error）”给出逐神经元修正法则，使正式推理步 $T=1\sim 4$ 的精度显著跃升。
- 留下的局限：（1）源网络仍须经过 `QCFS`（含 `MaxPool → AvgPool` 替换）重训；（2）**推理总延迟容易被表观 $T$ 误导**——Appendix Algorithm 1（第 19 页）显示对每一张新测试图都必须串行执行两趟仿真（先跑 $\rho$ 步算残余电位偏移，重置后再跑 $T$ 步），故标称 $T=2, \rho=4$ 实际消耗 **6 个时间步**，$T=8, \rho=8$ 实际消耗 **16 个时间步**！
- 是否需要训练或微调：部分（偏移校准在推理期完成无需微调，但**源 ANN 必须先按 QCFS 替换为 `AvgPool` + 阶梯激活从头重训**）
- 涉及的算子：QCFS 阶梯激活、残余膜电位越界判据（$v^l(\rho)<0$ 或 $v^l(\rho)>\theta^l$）、每个测试样本两阶段推理（$\rho$ 步探测 + $T$ 步正式仿真）
- 关键数字：`ImageNet` `ResNet-34`（QCFS 源 **74.22%**，Section 4.2 Table 1 & Table 2，第 7–8 页）：在 $\rho=4$ 下名义 $T=1 / 2 / 4$（实际总步数 $5 / 6 / 8$）达 **61.84% / 67.12% / 71.86%**；在 $\rho=8, T=8$（实际总步数 16）达 **74.17%**；`VGG-16`（源 **74.19%**）在 $\rho=4, T=2$ 达 **70.59%**、$\rho=8, T=8$ 达 **74.19%**。
- 与本方向的关系：可借用的技术（提醒我们在做公平能耗与延迟对比时，必须把预探测/预热步数计入真实计算量；其利用越界膜电位诊断相位错位多发脉冲的判据可辅助误差归因）
- 核对人与日期：Gemini，2026-11-16

### SpiCalib 与 MLIPooling（2022）

- 标题：Spike Calibration: Fast and Accurate Conversion of Spiking Neural Network for Object Detection and Segmentation（PDF 第 1 页与 arXiv API；本条早先误写成另一篇论文的标题，复查时已更正；方法简称 SpiCalib）
- 作者：Yang Li, Xiang He, Yiting Dong, Qingqun Kong, Yi Zeng
- 发表处：arXiv（2022 年 7 月预印本，`arXiv:2207.02702`）
- 链接：https://arxiv.org/abs/2207.02702 ；代码：未公开
- 核对深度：全文
- 一句话：用发放间隔（ISI）驱动的双突触校正首脉冲延迟误差，并提出支持重叠窗口（$k\neq s$）和 `ceil_mode` 补零边界的修正侧抑制最大池化 `MLIPooling`，免重训转换了原封不动的 YOLO 检测器与 FCN 分割网。
- 解决的难点：补齐了 `LIPooling`（IJCAI 2022）在处理重叠步幅最大池化（如 ResNet 首层与目标检测主干中的 $3\times 3, s=2$ 或 $s=1$ `MaxPool`）时的边界与跨窗干扰缺陷，不改源模型一行结构即可做检测与分割转换。
- 留下的局限：属于纯单极性率编码路线，在目标检测与分割任务上仍需 $T=64\sim 512$ 个时间步才能收敛，远未达到个位数时间步；且 `MLIPooling` 同样假定非负脉冲计数，不支持负特征差取最大。
- 是否需要训练或微调：否（利用贝叶斯优化搜索 $p$-范数归一化因子，完全免重训、不修改源模型 `MaxPool` 层）
- 涉及的算子：基于脉冲发放间隔（ISI）的双突触在线偏差监测与校正（`SpiCalib`）、支持重叠核与 `ceil_mode` 的修正侧抑制最大池化（`MLIPooling`）、贝叶斯优化 $p$-Norm 权重归一化
- 关键数字：目标检测（Section 4.3 Table 2 & Table 3，第 8 页）：`PASCAL VOC` `YOLO (ResNet-50)`（源 **75.28% mAP**）在 $T=64 / 128 / 512$ 免训练转换达 **62.13% / 73.78% / 75.21% mAP**；`MS COCO`（源 **45.28%**）在 $T=128 / 512$ 达 **43.61% / 45.32%**；语义分割 `PASCAL VOC 2012` `FCN`（源 **60.20% mIoU**，Table 4 & Section 4.4，第 9 页）在 $T=64 / 256$ 达 **55.66% / 59.56% mIoU**（$T=64$ 时估算能耗为 ANN 的 **0.492×**）。
- 与本方向的关系：可借用的技术（目前少有的在真实重叠窗口 `MaxPool` 检测/分割主干上做纯脉冲免重训转换的完整对照方案）
- 核对人与日期：Gemini，2026-11-16

### Burst + LIPooling（2022）

- 标题：Efficient and Accurate Conversion of Spiking Neural Network with Burst Spikes
- 作者：Yuhang Li, Yi Zeng
- 发表处：Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence（IJCAI 2022，pp. 2485–2491；预印本 `arXiv:2204.13271`）
- 链接：https://arxiv.org/abs/2204.13271 ；代码：https://github.com/BrainCog-X/Brain-Cog
- 核对深度：全文
- 一句话：允许单个时间步内连续发放一簇最多 $\Gamma$ 个二值脉冲（Burst Spikes）以释放高激活截断误差，同时**从理论上证明朴素脉冲 `MaxPool` 输出必被高估至 $[\max, \text{sum}]$ 之间，并设计侧抑制池化 `LIPooling` 精确收敛于真最大值**。
- 解决的难点：（1）用仿生脉冲簇在 $T=32\sim 64$ 步免训练转换下压低截断误差；（2）论文第 4–5 页公式 (11)–(12) 给出了脉冲域逐时刻取 `max` 因胜者来回切换导致前期脉冲不可撤回、从而产生严重正向偏置的严格数学证明与互抑制消除机制。
- 留下的局限：`LIPooling` 的互抑制矩阵构造依赖非负脉冲计数（单极性率编码），且为了等各通道脉冲相位对齐仍需 $T=16\sim 32$ 步；在图网络中每个节点邻域大小 $k$ 可变且 `MRConv` 输入差值 $x_j-x_i$ 带负号，无法原样照搬。
- 是否需要训练或微调：否（直接转换带 BN 和原版 `MaxPool` 的预训练 CNN，无需重训或改为 `AvgPool`）
- 涉及的算子：脉冲簇神经元（Burst Spikes，单步最大发放数 $\Gamma=5$）、侧抑制最大池化神经元层（`LIPooling`）、BatchNorm 折叠
- 关键数字：在保留原版 `MaxPool` 的 `VGG-16` 上（Section 4.2 Table 1 & Table 2，第 5–6 页），`CIFAR-100`（$T=32$）单纯将 p-Norm 池化换成 `LIPooling` 即从 58.51% 跃升至 **65.63%**，再叠加 Burst（$\Gamma=5$）达 **71.41%**（$T=64$ 达 **74.98%**，源 ANN 76.31%）；`ImageNet` `VGG-16`（源 **74.49%**，Table 3，第 6 页）在 $T=32 / 64$ 达 **70.61% / 73.00%**，`ResNet-34`（源 **73.45%**）在 $T=32 / 64$ 达 **65.06% / 70.95%**。
- 与本方向的关系：直接相关（回答定向问题 Q10“取最大值对输入/脉冲误差敏感度”的核心文献：公式 (11) $\max(\mathbb{O})\le r_i^{(\ell)}\le\text{sum}(\mathbb{O})$ 精确刻画了脉冲流直接取 `max` 的单向高估定律）
- 核对人与日期：Gemini，2026-11-16

### Spiking MaxPooling 电路（2022）

- 标题：Spiking Approximations of the MaxPooling Operation in Deep SNNs
- 作者：Ramashish Gaurav, Bryan Tripp, Apurva Narayan
- 发表处：2022 International Joint Conference on Neural Networks（IJCNN 2022，DOI: `10.1109/IJCNN55064.2022.9892074`；预印本 `arXiv:2205.07076`）
- 链接：https://arxiv.org/abs/2205.07076 ；代码：未公开
- 核对深度：全文
- 一句话：面向 Intel Loihi 芯片设计并对比了两种完全由脉冲神经元构成的 `MaxPooling` 硬件近似电路：基于恒等式 $\max(a,b)=\frac{a+b}{2}+\frac{|a-b|}{2}$ 的绝对值群体电路 `AVAM` 与最大跳跃门控电路 `MJOP`。
- 解决的难点：在不支持硬件浮点比较器、只有标准 LIF 神经元核的第一代神经形态芯片（Loihi 1）上给出了纯脉冲电路实现 `MaxPool` 的构造性方案。
- 留下的局限：代价高且延迟长——`MJOP` 精度损失高达十余个百分点；较准的 `AVAM` 需要成群辅助神经元两两配对构建二叉树（邻域 $k=20$ 时树深与辅助神经元激增），且论文第 7 页 Section IV-B 明确承认**必须依赖突触低通滤波平滑电流，完全不支持未滤波直接脉冲或超低时间步编码**（实测需 $T=50\sim 120$ 步）。
- 是否需要训练或微调：否（基于 NengoDL / NengoLoihi 转换预训练 CNN，将 `MaxPool` 替换为预构建脉冲微电路）
- 涉及的算子：两两配对绝对值拟合脉冲最大池化（`AVAM`）、最大跳变输出池化（`MJOP`）、指数突触电流滤波、LIF 神经元
- 关键数字：在 `CIFAR-10` 转换实验中（$T=50\sim 120$ 步，Section III Table I，第 6 页），源 CNN 精度为 **65.3%**，`AVAM` 转换精度为 **64.68%**（Loihi 仿真掉点 <1%），而 `MJOP` 大幅掉点至 **51.80%**；`MNIST` 上 `AVAM` 达 **98.55%**（源 98.71%）。
- 与本方向的关系：可借用的技术（与 `LIPooling`、`Pre-Neuron MaxPool` 一起证明了“脉冲域做 `MaxPool` 绝非平凡无损”，在低时间步与大邻域 $k$ 下纯脉冲拟合电路并不划算）
- 核对人与日期：Gemini，2026-11-16

### QCFS（2022）

- 标题：Optimal ANN-SNN Conversion for High-Accuracy and Ultra-Low-Latency Spiking Neural Networks
- 作者：Tong Bu, Wei Fang, Jianhao Ding, PengLin Dai, Zhaofei Yu, Tiejun Huang
- 发表处：The Tenth International Conference on Learning Representations（ICLR 2022；预印本 `arXiv:2303.04347`）
- 链接：https://arxiv.org/abs/2303.04347 （OpenReview：https://openreview.net/forum?id=7B3IJMM1k_M ）；代码：https://github.com/putshua/SNN_conversion_QCFS
- 核对深度：全文
- 一句话：系统将转换误差拆分为“截断误差、量化（向下取整）误差、脉冲到达不均匀误差”三类，提出带 $1/2$ 平移项的量化剪裁取整激活函数（QCFS）替换源 ANN 的 ReLU 从头训练，首次将转换 SNN 的有效工作步数从数十步拉进 $T=4$。
- 解决的难点：严格证明当源网络采用步数为 $L$、平移 $\varphi=1/2$ 的 QCFS 激活且转换后初始膜电位设为 $\theta^l/2$ 时，对**任意**目标时间步 $T$（无需 $T=L$）转换误差的数学期望恒为零（Theorem 2，公式 16）。
- 留下的局限：（1）**必须修改并从头训练源 ANN 300 轮**——Appendix A.1 与 Algorithm 1（第 14、20 页）明确要求训练前先把所有 `MaxPooling` 替换为 `AvgPooling`、把 `ReLU` 替换为 `QCFS`；（2）在 `ImageNet` 大模型上单靠 QCFS 在 $T=4\sim 8$ 仍因不均匀误差明显掉点（`ResNet-34` 在 $T=16$ 仅 59.35%）。
- 是否需要训练或微调：是（需用 STE 直通估计器从头训练改用 `AvgPool + QCFS` 的源网络 300 轮）
- 涉及的算子：量化剪裁向下取整平移激活函数（QCFS，$\lambda^l\text{clip}(\frac{1}{L}\lfloor\frac{z^l L}{\lambda^l}+\frac{1}{2}\rfloor, 0, 1)$）、可学习阈值 $\lambda^l$、初始膜电位偏置 $v^l(0)=\theta^l/2$、软复位 IF 神经元、`AvgPool` 替代 `MaxPool`
- 关键数字：`CIFAR-10` 上（Section 6.2 Table 2，第 8 页），`VGG-16`（源 **95.52%**）在 $T=2 / 4 / 8$ 达 **91.18% / 93.96% / 94.95%**，`ResNet-18`（源 **96.04%**）在 $T=4 / 8$ 达 **90.43% / 94.82%**；`CIFAR-100` `VGG-16`（源 **76.28%**，Appendix Table S4，第 19 页）在 $T=4 / 8$ 达 **69.62% / 73.96%**；`ImageNet`（Table 3，第 9 页）`ResNet-34`（源 **74.32%**）在 $T=16 / 32 / 64$ 达 **59.35% / 69.37% / 72.35%**，`VGG-16`（源 **74.29%**）在 $T=16 / 32 / 64$ 达 **50.97% / 68.47% / 72.85%**。
- 与本方向的关系：可借用的技术（超低时间步转换理论的分水岭里程碑；其“初始膜电位预充 $\theta/2$ 消除向下取整半格系统偏置”的结论在免训练转换中也完全通用）
- 核对人与日期：Gemini，2026-11-16

### SNNC-AP / SNN Calibration（2021）

- 标题：A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration
- 作者：Yuhang Li, Shikuang Deng, Xin Dong, Ruihao Gong, Shi Gu
- 发表处：Proceedings of the 38th International Conference on Machine Learning（ICML 2021，PMLR Vol. 139，pp. 6316–6325；预印本 `arXiv:2106.06984`）
- 链接：https://arxiv.org/abs/2106.06984 ；代码：未公开（论文摘要写将发布于 GitHub，正文未附具体 URL）
- 核对深度：全文
- 一句话：将训练后量化（PTQ）中的逐层校准范式引入 ANN→SNN 转换，用网格搜索最小均方误差（MMSE）定阈值，并提供仅调偏置的轻量流水线（Light Pipeline）与联合调初始膜电位及权重的进阶流水线（Advanced Pipeline），首次成功转换带 BN 的 `MobileNet` 与 `RegNetX-4GF`。
- 解决的难点：攻克了早期转换方法面对带 `BatchNorm` 轻量化深度可分离卷积网络（如 MobileNet）在 $T\le 256$ 时精度近乎归零（0.11%）的崩溃问题，只需 128–1024 张无标签校准图即可恢复精度。
- 留下的局限：（1）**遇到 `MaxPool` 仍需改源网**——Appendix C.1（第 11 页）明确写明实验测试的 VGG/ResNet 均预先将 `MaxPool` 替换为 `AvgPool`（再转成固定权重深度可分离卷积）；（2）对不同目标时间步 $T$ 需分别校准并存储独立参数；（3）在 $T<16$ 下尚未可用（实验最低从 $T=32$ 起测）。
- 是否需要训练或微调：部分（若源模型本身仅含 `AvgPool`/步幅卷积如开源 `MobileNet`，则 Light Pipeline 完全免梯度训练，Advanced Pipeline 仅用 1024 张图逐层 SGD 校准权重数分钟；但含 `MaxPool` 的源网仍被预先改成 `AvgPool` 训练）
- 涉及的算子：逐通道 MMSE 网格搜索阈值、轻量偏置校准（BC，$b_c\leftarrow b_c+\mu_c(e)$）、初始膜电位校准（PC，$v(0)\leftarrow T\cdot e$）、基于 STE 的逐层权重校准（WC）、BatchNorm 折叠、`AvgPool` 转深度可分离卷积
- 关键数字：`ImageNet` 上（Section 5.3 Table 4，第 7 页），`ResNet-34`（带 BN，源 **75.66%**）在 $T=32 / 64$ 下 Light Pipeline 达 **50.21% / 63.66%**，Advanced Pipeline 达 **64.54% / 71.12%**（对比未校准基线在 $T=32$ 仅 **0.09%**）；`MobileNet`（源 **73.40%**）在 $T=64 / 128 / 256$ 下 Advanced Pipeline 达 **56.26% / 65.40% / 69.02%**（基线在 $T=256$ 仅 **0.964%**，提升超 68 个百分点）；`RegNetX-4GF`（源 **80.02%**）在 $T=32 / 64 / 128$ 达 **55.70% / 70.96% / 75.78%**。
- 与本方向的关系：可借用的技术（确立了“用百余张无标签样本逐层对齐 ANN 激活与 SNN 平均脉冲输出”的训练后校准范式，可直接作为视觉图转换的基础校准层）
- 核对人与日期：Gemini，2026-11-16

### RTS / Optimal Conversion（2021）

- 标题：Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks
- 作者：Shikuang Deng, Shi Gu
- 发表处：The Ninth International Conference on Learning Representations（ICLR 2021；预印本 `arXiv:2103.00476`）
- 链接：https://arxiv.org/abs/2103.00476 （OpenReview：https://openreview.net/forum?id=FZ1oTwcXchK ）；代码：https://github.com/Jackn0/snn_optimal_conversion_pipeline
- 核对深度：全文
- 一句话：通过二阶泰勒展开把全网转换损失分解为逐层激活均方误差之和，从理论上推导出在 SNN 每层偏置上加 $\frac{V_{th}}{2T}$ 最优平移项以抵消向下取整的系统性低估，并配合截断阈值 ReLU 将步数从上千步压至数十步。
- 解决的难点：首次严格指出 IF 软复位神经元的脉冲计数是输入电流的“向下取整（floor）”而非“四舍五入”，推导出 $\delta = \frac{V_{th}}{2T}$ 的半格偏置补偿公式与总误差 $\Delta\mathcal{L}_{\min}\approx \frac{L V_{th}^2}{4T}$（公式 15–16），把转换从经验调参推进到解析误差最小化。
- 留下的局限：未兼容 `BatchNorm` 与 `MaxPool`（论文第 9 页 Section 8 讨论明确将支持 BN 和正负双极激活列为未来工作，实验中全部去 BN 并使用 `AvgPool`，一旦直接套在带 BN 的 ImageNet ResNet-34 上在 $T=32$ 即崩至 0.09%）；且最优效果需用截断 `Threshold ReLU` 重训源 ANN。
- 是否需要训练或微调：部分（仅加偏置平移 `Work-S` 可免重训用于普通 ReLU 网络但需 $T\ge 256$；要在 $T=16\sim 32$ 收敛则须用 `Threshold ReLU` 重训源 ANN，且源网不含 BN 与 `MaxPool`）
- 涉及的算子：截断激活 `Threshold ReLU`（上限 $y_{th}$）、逐层偏置平移 $b_{\text{SNN}} = b_{\text{ANN}} + \frac{V_{th}}{2T}$、软复位 IF 神经元、`AvgPool`
- 关键数字：无 BN 架构下（Section 7.3 Table 1，第 8 页），`CIFAR-10` `VGG-16` 在 $T=16 / 32$ 达 **92.29% / 92.29%**（较 `RMP-SNN` 的 1536 步缩短近百倍），`ResNet-20` 在 $T=16 / 32$ 达 **92.41% / 93.30%**；`CIFAR-100` `VGG-16` 在 $T=16 / 32$ 达 **65.94% / 69.80%**；`ImageNet` `VGG-16` 在 $T=16 / 32 / 64$ 达 **55.80% / 67.73% / 70.97%**。
- 与本方向的关系：可借用的技术（半格偏置补偿 $\frac{V_{th}}{2T}$ 与二阶泰勒逐层误差分解是理解此后所有低步转换方法的理论基石）
- 核对人与日期：Gemini，2026-11-16

### RMP-SNN（2020）

- 标题：RMP-SNN: Residual Membrane Potential Neuron for Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network
- 作者：Bing Han, Gopalakrishnan Srinivasan, Kaushik Roy
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2020，pp. 13558–13567；预印本 `arXiv:2003.01811`）
- 链接：https://arxiv.org/abs/2003.01811 ；代码：未公开（论文第 6 页仅引用其派生自 Facebook `fb.resnet.torch` 基线仓库）
- 核对深度：全文
- 一句话：系统阐明硬复位（reset-to-zero）在发放瞬间抹掉超阈值电位的非线性台阶失真，结合软复位残余膜电位（RMP）神经元、无 BN/无偏置约束训练与阈值比例缩放（$\alpha \le 0.8$），在深层 VGG-16 与 ResNet-34 上实现近无损转换并提速 2–8 倍。
- 解决的难点：解决了 `Rueckauer 2017` 虽提出软复位但在带 BN/偏置的 ImageNet VGG-16 上仍掉点 14.28% 的遗留问题，确立了深层残差网络软复位转换的标准基线。
- 留下的局限：（1）**要求源网络训练时严禁使用偏置项和 `BatchNorm` 层**（Section 2，第 2 页），无法复用现成带 BN 的现代视觉预训练权重；（2）采用随机泊松率编码，在 ImageNet 上仍需 $T=512\sim 4096$ 个时间步。
- 是否需要训练或微调：部分（转换阶段本身只统计最大加权输入定阈值，但**源 ANN 必须在去掉全部 BN 层与偏置项、改用 Dropout 的严格约束下从头训练**）
- 涉及的算子：残余膜电位软复位神经元（RMP，$V_m(t)=V_m(t)-V_{th}$）、基于 SNN 脉冲统计的逐层最大输入阈值与比例缩放（$\alpha V_{\text{in}}^{\max}$）、泊松正负率编码、去 BN 无偏置 ReLU 卷积与残差块
- 关键数字：近无损高步精度（Section 5 Tables 1–3，第 7 页）：`CIFAR-10`（$T=2048$）`VGG-16` **93.63%**（源 93.63%，掉点 <0.01%）、`ResNet-20` **91.36%**（源 91.47%）；`CIFAR-100`（$T=2048$）`VGG-16` **70.93%**（源 71.22%）、`ResNet-20` **67.82%**（源 68.72%）；`ImageNet`（$T=4096$）`VGG-16` **73.09%**（源 73.49%，掉 0.4%）、`ResNet-34` **69.89%**（源 70.64%，掉 0.75%）；缩放阈值快速模式下 `ImageNet` `VGG-16` 在 $T=512$ 达 **68.93%**（Section 5，第 7–8 页）。
- 与本方向的关系：背景（软复位残余膜电位机制的经典代表作，解释了为何现代所有转换法一律采用减法软复位而非归零硬复位）
- 核对人与日期：Gemini，2026-11-16

### Spiking-YOLO（2020）

- 标题：Spiking-YOLO: Spiking Neural Network for Energy-Efficient Object Detection
- 作者：Seijoon Kim, Seongsik Park, Byunggook Na, Sungroh Yoon
- 发表处：Proceedings of the AAAI Conference on Artificial Intelligence（AAAI 2020，Vol. 34, No. 07，pp. 11270–11277，DOI: `10.1609/aaai.v34i07.6787`；预印本 `arXiv:1903.06530`）
- 链接：https://arxiv.org/abs/1903.06530 ；代码：未公开
- 核对深度：全文
- 一句话：首个将深层目标检测器免重训转换为 SNN 的里程碑工作，提出逐通道权重归一化（Channel-Norm）消除通道尺度悬殊导致的欠发放，并用带正负非对称阈值的双极脉冲神经元（IBT）精确转换 `LeakyReLU` 的负半轴斜率。
- 解决的难点：攻克了坐标边界框连续回归对细粒度数值极其敏感、传统整层归一化（Layer-Norm）压死小幅值特征通道，以及单极性 IF 无法表达 `LeakyReLU` 负斜率（如 $-0.1x$）的两大障碍。
- 留下的局限：为逼近连续检测框回归所需的精细分辨率，推理需要高达 **$T=3500\sim 8000$ 个时间步**，无法用于实时低延迟场景；且源模型 `Tiny YOLO` 中的步幅为 1 最大池化仍被替换为近似层处理。
- 是否需要训练或微调：否（直接对预训练 `Tiny YOLO` 执行逐通道最大激活归一化与双阈值映射，无需重训）
- 涉及的算子：逐通道激活归一化（Channel-Norm）、非对称正负双阈值带符号神经元（Imbalanced-Threshold / `IBT`，$V_{th}^+ = 1, V_{th}^- = -1/\alpha$ 映射 `LeakyReLU`）、软复位 IF 神经元
- 关键数字：在 `PASCAL VOC` 上转换 `Tiny YOLO`（源 **53.01% mAP**，Section 4 Table 1 & Table 2，第 6–7 页），传统整层归一化 + 标准 IF 仅得 **7.52% mAP**，引入 `Channel-Norm + IBT` 后在 $T=8000$ 达 **51.83% mAP**（$T=3500$ 收敛至约 51.4%）；在 `MS COCO`（源 **26.24% mAP**）上达 **25.66% mAP**；基于 TrueNorth 芯片功耗建模估算较 V100 GPU 节能约 **280×**（Section 4，第 7 页）。
- 与本方向的关系：可借用的技术（`Channel-Norm` 是此后几乎所有逐通道阈值校准的鼻祖；`IBT` 用非对称正负双阈值免训练表达 `LeakyReLU` / `GELU` 负半轴小斜率的做法可直接用于采用 `GELU` 或 `LeakyReLU` 的视觉图骨干如 ViG / DGCNN！）
- 核对人与日期：Gemini，2026-11-16

### Spike-Norm（2019）

- 标题：Going Deeper in Spiking Neural Networks: VGG and Residual Architectures
- 作者：Abhronil Sengupta, Yuting Ye, Robert Wang, Chiao Liu, Kaushik Roy
- 发表处：Frontiers in Neuroscience（2019 年，Vol. 13，Article 95，DOI: `10.3389/fnins.2019.00095`；预印本 `arXiv:1802.02627`）
- 链接：https://arxiv.org/abs/1802.02627 ；代码：未公开
- 核对深度：全文
- 一句话：提出在实际脉冲网络前向传播中逐层记录真实脉冲累积输入最大值来设定阈值的 `Spike-Norm` 算法，并定制无 BN、无偏置、全平均池化的残差块，首次把 ANN→SNN 转换成功推上 `ImageNet` 全集的深层 `VGG-16` 与 `ResNet-34`。
- 解决的难点：发现此前单纯基于 ANN 静态激活最大值定阈值会低估多步随机泊松脉冲叠加的瞬时峰值，通过“在环脉冲统计定阈值”把转换规模从 CIFAR 小网络扩展到深层 ImageNet 架构。
- 留下的局限：使用硬复位（hard-reset）IF 神经元且要求源网络去偏置、去 BN、把所有 `MaxPool` 换成 `AvgPool` 从头训练；在 `ImageNet` 上需要 **$T=2500$ 步**，且 `ResNet-34` 仍掉点逾 5 个百分点。
- 是否需要训练或微调：部分（阈值标定无需梯度训练，但必须预先按无偏置、无 BN、`AvgPool` 约束重训源 VGG/ResNet）
- 涉及的算子：SNN 在环逐层最大突触输入阈值平衡（`Spike-Norm`）、硬复位 IF 神经元、泊松率编码、无 BN 定制残差连接与 `AvgPool`
- 关键数字：在 $T=2500$ 下（Section 4 Tables 1–3，第 11–14 页），`CIFAR-10` `VGG-16` 达 **91.55%**（源 91.70%）；`ImageNet` 全集上 `VGG-16` 达 **69.96%**（源 **70.52%**，仅掉 0.56%），`ResNet-34` 达 **65.47%**（源 **70.69%**，掉 5.22%）。
- 与本方向的关系：背景（深层视觉主干转换的早期里程碑，直观展示了硬复位与朴素率编码在深层残差结构上的延迟代价）
- 核对人与日期：Gemini，2026-11-16

### SNN Toolbox（2017）

- 标题：Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification
- 作者：Bodo Rueckauer, Iulia-Alexandra Lungu, Yuhuang Hu, Michael Pfeiffer, Shih-Chii Liu
- 发表处：Frontiers in Neuroscience（2017 年，Vol. 11，Article 682，DOI: `10.3389/fnins.2017.00682`；早期理论与工具版见 NIPS 2016 Workshop 预印本 `arXiv:1612.04052`）
- 链接：https://doi.org/10.3389/fnins.2017.00682 （预印本：https://arxiv.org/abs/1612.04052 ）；代码：https://github.com/NeuromorphicProcessorProject/snn_toolbox
- 核对深度：全文（2016 预印本与 2017 期刊正式版全文均已核对）
- 一句话：现代无约束 CNN 转换的奠基之作——一次性提出了**减法软复位（reset-by-subtraction）、BatchNorm 参数折叠、99.9% 分位数鲁棒归一化、直接恒定电流首层输入、脉冲滑动滤波门控 `MaxPool` 以及脉冲对数比近似 `Softmax`**，让任意带偏置与 BN 的现成 CNN（含 Inception-V3）可直接免重训转换。
- 解决的难点：彻底解除了 `Cao 2015` 与 `Diehl 2015`“源网络不准加偏置、不准用 BN、不准用 MaxPool”的三大紧箍咒，首次给出转换误差随层传播的理论推导与开源通用转换工具箱。
- 留下的局限：受限于单步二值脉冲率编码与 `MaxPool` 门控所需的指数滑动滤波稳定时间，推理需要 **$T=300\sim 550$ 个时间步**；且在更深的 `ImageNet` `VGG-16` 子集上仍出现 **14.28%** 的较大精度损失（Table 2，第 8 页）。
- 是否需要训练或微调：否（直接转换带 BN、偏置、`MaxPool` 和 `Softmax` 的标准预训练网络）
- 涉及的算子：减法软复位 IF 神经元、BatchNorm 权重偏置折叠、99.9% 激活分位数鲁棒权重归一化（Robust Normalization）、基于 FIR/指数移动平均发放率估计门控的脉冲 `MaxPool`（Section 2.2.6，第 6 页）、深度可分离卷积与 `Inception-V3` 分支拼接、脉冲 `Softmax`
- 关键数字：`CIFAR-10` 上（Section 3.1 Table 1，第 8 页）带 BN 与脉冲 `MaxPool` 的 CNN 转换达 **90.85%**（源 **91.91%**，约 $T=400$ 步）；`ImageNet` 子集评测（Table 2，第 8 页）`Inception-V3`（1,382 张图，$T=550$）转换误差率 **25.04%**（即 Top-1 精度 **74.96%**，源精度 **76.12%**，仅掉 1.16%），但 `VGG-16`（2,570 张图，$T=400$）精度从 **76.11%** 掉至 **61.83%**（掉 14.28%）。
- 与本方向的关系：直接相关（现代免训练 ANN→SNN 转换算子工具箱的总源头，其中的 BN 折叠、软复位与分位数定阈值至今仍是所有转换算法的标准前置步骤）
- 核对人与日期：Gemini，2026-11-16

### Weight & Threshold Balancing（2015）

- 标题：Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing
- 作者：Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, Michael Pfeiffer
- 发表处：2015 International Joint Conference on Neural Networks（IJCNN 2015，pp. 1–8，DOI: `10.1109/IJCNN.2015.7280696`）
- 链接：https://doi.org/10.1109/IJCNN.2015.7280696 ；代码：https://github.com/dannyneil/spiking_relu_conversion
- 核对深度：仅见引用（已核实 Crossref 官方 DOI 元数据与开源代码库，并通过通读全文的 `Rueckauer 2017` Section 1–2、`RMP-SNN` Section 2/3 与 `QCFS` Section 5 多源交叉核实其方法公式与基准结论；因 IEEE Xplore 需订阅且苏黎世大学机构旧 PDF 链接已 404，严格标为仅见引用）
- 一句话：提出“基于模型权重范数”和“基于训练数据最大激活值”的逐层权重/阈值平衡（Data-based Normalization）法则，消除了早期手工设阈值导致的深层过饱和与脉冲熄灭问题，把 MNIST 转换脉冲 CNN 推至 99.1% 以上高精度。
- 解决的难点：证明将每层权重按该层最大激活比例缩放 $\mathbf{W}^\ell \leftarrow \mathbf{W}^\ell \frac{\lambda^{\ell-1}}{\lambda^\ell}$ 可保证任意神经元期望发放率不超过上限 1，奠定了此后所有阈值/权重缩放转换法的数学原型。
- 留下的局限：强制要求源网络**无偏置项、无 BatchNorm、使用平均池化替代最大池化**，且对单个极端离群激活极其敏感（导致阈值偏大、需数百时间步），采用硬复位神经元无法扩展到深层复杂数据集。
- 是否需要训练或微调：部分（归一化本身在训练后执行，但源 CNN 须在无偏置、平均池化约束下训练）
- 涉及的算子：基于数据的逐层最大激活权重归一化（Data-based Normalization）与基于模型的权重行和归一化（Model-based Normalization）、硬复位 IF 神经元、无偏置 ReLU 卷积与平均池化
- 关键数字：据 `Rueckauer 2017`（Section 1 & Table 1）与开源仓库记录核实，在 `MNIST` 上转换无偏置 3 层卷积网取得 **99.14%** 精度（当时脉冲网络最高纪录，但在更深 CIFAR/ImageNet 网络上因无 BN 与硬复位受限）。
- 与本方向的关系：背景（确立“激活尺度归一化 $\Leftrightarrow$ 发放阈值平衡”等价性的奠基文献）
- 核对人与日期：Gemini，2026-11-16

### Spiking Deep CNN 奠基（2015）

- 标题：Spiking Deep Convolutional Neural Networks for Energy-Efficient Object Recognition
- 作者：Yongqiang Cao, Yang Chen, Deepak Khosla
- 发表处：International Journal of Computer Vision（IJCV 2015，Vol. 113, Issue 1，pp. 54–66，DOI: `10.1007/s11263-014-0788-3`）
- 链接：https://doi.org/10.1007/s11263-014-0788-3 ；代码：未公开
- 核对深度：仅摘要（已直接打开 Springer 官方论文页核实完整摘要与元数据，并结合 `Rueckauer 2017`、`RMP-SNN 2020`、`QCFS 2022` 全文对其技术路线的引用交叉核实）
- 一句话：现代深度学习时代 ANN→SNN 转换路线的开创性工作——指出 ReLU 的非负分段线性响应与无漏电积分发放（IF）神经元的脉冲发放率天然同构，首次通过裁剪 CNN 结构（去偏置、平均池化）训练后直接把浮点权重复制给脉冲网络识别物体。
- 解决的难点：绕开了脉冲神经元不可微难以直接反向传播训练深层网络的死结，首次指明“先用成熟反向传播训练定制 ReLU CNN，再复制权重映射为事件驱动 SNN 上类脑芯片”的工程捷径。
- 留下的局限：缺少系统的层间权重/阈值归一化公式，必须手工定制无负输出、无偏置、无 `MaxPool` 的浅层 CNN，仅在 `CIFAR-10` 与 `Neovision2 Tower` 等中小规模目标识别上验证，且需要数百个时间步。
- 是否需要训练或微调：部分（转换只是直接复制权重，但必须先将源 CNN 裁剪为去偏置、非负平均池化的 SNN 兼容结构后训练）
- 涉及的算子：ReLU 到 IF 神经元发放率直接权重映射、零偏置卷积与全连接、空间平均池化替代最大池化
- 关键数字：据 Springer 官方摘要披露，在 `DARPA Neovision2 Tower` 与 `CIFAR-10` 数据集上取得与裁剪后源 CNN 相近的识别精度，且面向 DARPA SyNAPSE 类脑硬件的映射分析表明其能效比商用现成 FPGA 上的原 CNN 高出**两个数量级（约 100×）**（Abstract；具体逐表分类百分比因未获取付费正文记为未核实）。
- 与本方向的关系：背景（整条 ANN→SNN 转换研究脉络的起点）
- 核对人与日期：Gemini，2026-11-16

---

## 这条线是怎么演化的

纵观 2015–2026 年十一年间的演进，ANN→SNN 转换技术走过了四次清晰的方法论跃迁，每一步都在撬动“**时间步延迟 $T$**”、“**是否允许修改/重训源网络**”与“**可转换的非线性算子边界**”这三者的不可能三角：

1. **第一代：从“裁剪源网络迁就率编码”到“基础算子解绑”（2015–2020）**  
   `Cao 等`（IJCV 2015）率先洞察到 `ReLU` 与 `IF` 发放率的正比例关系，随后 `Diehl 等`（IJCNN 2015）建立逐层最大激活归一化防止脉冲饱和或熄灭，但代价是源网络不准有偏置、不准有 `BatchNorm`、不准用 `MaxPool`。`Rueckauer 等`（`SNN Toolbox`, 2017）用**减法软复位 + BN 折叠 + 99.9% 分位数阈值 + 滑动滤波门控 `MaxPool`** 第一次解除了对 CNN 结构的束缚；`Spike-Norm`（2019）与 `RMP-SNN`（CVPR 2020）把规模推上 `ImageNet` 深层残差网，`Spiking-YOLO`（AAAI 2020）则用逐通道归一化和正负双阈值神经元 `IBT` 首开目标检测先河。然而，这一代方法把数值完全寄托在长窗口二值脉冲频率上，动辄需要 **$T=300\sim 8000$ 个时间步**。

2. **第二代：误差解析分解与“量化感知重训（QAT）”把时间步打进个位数（2021–2023）**  
   研究者意识到低时间步崩盘的祸首是**向下取整系统性低估（flooring error）**、**大激活截断（clipping error）**与**脉冲先后顺序错位（unevenness error）**。`RTS`（ICLR 2021）推导出 $\frac{V_{th}}{2T}$ 半格偏置平移；`SNNC-AP`（ICML 2021）借鉴训练后量化（PTQ），仅用百余张样本逐层校准偏置、初始电位和权重；`QCFS`（ICLR 2022）干脆把源网络的 ReLU 换成带 $1/2$ 平移的阶梯剪裁量化函数从头重训，首次在 $T=4$ 打通超低步转换；随后的 `COS`（ICLR 2023，两阶段偏移校准）与 `Fast-SNN`（TPAMI 2023，带符号纠错脉冲 + $T=2^b-1$ 量化等价）进一步修补脉冲相位错位。但这一代高精超低步方案几乎全部退回了**“必须把 `MaxPool` 改成 `AvgPool`/步幅卷积并重新训练源网络”**的老路——正如脉络二中 `Lan 等`（ICCV 2023）转换 `PointNet++` 时所遭遇的困境。

3. **第三代：面向预训练大模型的“纯免重训校准”与“分级/差分/多基神经元”（2024–2026）**  
   随着 ViT、EVA、CLIP 等开源基础模型兴起，重训源模型变得不可接受，社区重新聚焦于**严格不重训源模型（Training-Free）**且兼顾复杂算子与超低步：
   - **针对复杂连续非线性（GELU / LayerNorm / Softmax / 注意力点积）**：分化出两条截然不同的路线——一条是以 `ECMT`（ACM MM 2024）和 `DCGS`（ICML 2025）为代表的**混合期望补偿路线**，配合正负 $2n$ 级二次幂阈值差分分级脉冲把 EVA/ViT 压到 **$T=2\sim 4$ 步近无损**，但非线性层内部仍依赖浮点前缀均值重算；另一条是以 `STA`（ICLR 2024）、`SpikedAttention`（NeurIPS 2024，仍需源网预改 BN/ReLU）和最新 `MBE`（AAAI 2026，多基指数衰减 + IEEE-754 拆分）为代表的**纯脉冲逼近路线**，可在 $T=16$ 全脉冲转换 ViT/GPT-2，但在 $T\le 8$ 时精度急剧跳水。
   - **针对 `MaxPool` 的免重训处理**：同样分化为两条路——纯脉冲侧抑制电路（`Burst+LIPooling` IJCAI 2022、`SpiCalib+MLIPooling` 2022、`AVAM` IJCNN 2022）证明朴素脉冲取 `max` 必陷入 $[\max, \text{sum}]$ 正向高估，虽可用互抑制消除但仅限非负率编码且需 $T\ge 16\sim 64$ 步；而 `Inference-Scale Conversion`（Bu 等，CVPR 2025）则利用 $\max_i\text{ReLU}(z_i)=\text{ReLU}(\max_i z_i)$ **把 `MaxPool` 搬到脉冲神经元之前用硬件比较器算膜前连续值**，93 秒完成分类/检测/分割免训练校准。

4. **对“视觉图神经网络免训练转换”的直接启示**  
   把整条脉络一拼合起来看，为什么视觉图骨干（如 ViG、DGCNN、原版 PointNet++）会成为现有转换方法共同的盲区？原因并非缺某一个孤立的线性层校准技巧，而是三重结构性错位同时叠加：（1）`DCGS` 等超低步差分分级神经元在官方代码里遇到通用 `MaxPool` 直接退回模拟浮点，更从未面对**特征空间逐层动态重算 $k$-NN 邻居集合**这一离散阶跃算子；（2）`Bu 等 2025` 的前置比较器 `MaxPool` 依赖紧跟在卷积与 ReLU 之间的层序交换且要求最大值大于零，而图网络的 `MRConv` / `EdgeConv` 是在激活之后对**可正可负的节点相对差 $x_j-x_i$** 跨可变邻域取最大；（3）`One-Spike SNN`（TETC 2025）唯一涉及的图转换仅停留在静态固定邻接矩阵乘的 2 层 ReLU Text GCN。因此，借用 `DCGS`/`ECMT` 的节点前缀重构差分思想配合硬件比较器与 `IBT`/`MBE` 的负半轴处理，在静态与流式视觉图上正面攻克“动态选邻居保真 + 有符号邻域 `max` 聚合 + 同比特量化公平能耗盈亏”，正是建立在脉络一十一年积淀之上的精准发力点。
