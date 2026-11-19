# 脉络五：能耗评估与硬件

本文件整理“把训练好的视觉图神经网络免训练转成事件驱动脉冲网络”所需的第五条文献脉络，覆盖五个子问题：（1）脉冲网络论文最常引用的 Horowitz（2014）原始运算与访存能耗表溯源；（2）质疑“脉冲网络天然更省电”并主张与等价量化网络（QNN）公平比较的理论与硬件评测；（3）作为非脉冲对照基线的图神经网络训练后量化（PTQ）及代表性量化感知训练（QAT）工作；（4）支持分级脉冲（脉冲携带多位数值）的类脑芯片；（5）事件驱动图神经网络的专用硬件加速器（ASIC 与 FPGA）。

全部 21 篇文献按发表/最新版年份**从新到旧（2026 → 2014）**排列；其中 **16 篇已下载并通读原版全文 PDF**，5 篇核对官方摘要或一手代码仓/作者页（已在每条“核对深度”如实注明）。

---

## 附表与基准溯源：Horowitz（2014）45nm 0.9V 原始运算与访存能耗表

脉冲神经网络论文普遍采用“一次 32 位浮点乘加（FP32 MAC）耗能 `4.6 pJ`、一次 32 位浮点累加（FP32 AC）耗能 `0.9 pJ`”的估算公式。经核对 Mark Horowitz 在 **IEEE ISSCC 2014** 发表的 5 页原文（pp. 10–14，DOI: `10.1109/ISSCC.2014.6757323`）：

> **常见二手误引勘误**：大量后续论文把该表标为“Horowitz 2014, p. 11”或“Figure 1.1.2”。实际上，原文第 13 页的 **Figure 1.1.2** 是微处理器晶体管数量与特征尺寸随年份演化的散点图；真正的 **45nm CMOS、0.9V** 运算与存储能耗表位于**第 14 页下半页的 Figure 1.1.9**（题为 *"Figure 1.1.9: Rough energy costs for various operations in 45nm 0.9V."*），正文中对该表的文字讨论位于**第 11–12 页的 Section 5 与 Section 6**。

### 1. 原文第 14 页 Figure 1.1.9 完整抄录（工艺节点：45nm CMOS，供电电压：0.9V）

| 类别 | 运算或存储操作（原文名称） | 位宽 / 容量 | 能耗（pJ） | 相对 `32b Int Add (0.1 pJ)` 的倍数 | 出处位置 |
| --- | --- | --- | --- | --- | --- |
| 整数加法 | `Int Add` | 8 bit | `0.03 pJ` | `0.3×` | p. 14, Fig. 1.1.9 |
| 整数加法 | `Int Add` | 32 bit | `0.1 pJ` | `1×` | p. 14, Fig. 1.1.9 |
| 浮点加法 | `FP Add`（图中亦作 `FAdd`） | 16 bit | `0.4 pJ` | `4×` | p. 14, Fig. 1.1.9 |
| 浮点加法 | `FP Add`（图中亦作 `FAdd`） | 32 bit | `0.9 pJ` | `9×` | p. 14, Fig. 1.1.9 |
| 整数乘法 | `Int Mult` | 8 bit | `0.2 pJ` | `2×` | p. 14, Fig. 1.1.9 |
| 整数乘法 | `Int Mult` | 32 bit | `3.1 pJ` | `31×` | p. 14, Fig. 1.1.9 |
| 浮点乘法 | `FP Mult`（图中亦作 `FMult`） | 16 bit | `1.1 pJ` | `11×` | p. 14, Fig. 1.1.9 |
| 浮点乘法 | `FP Mult`（图中亦作 `FMult`） | 32 bit | `3.7 pJ` | `37×` | p. 14, Fig. 1.1.9 |
| 片上缓存读取 | `8KB Cache (64bit)`（SRAM） | 64 bit（8 Bytes） | `10 pJ` | `100×` | p. 14, Fig. 1.1.9 |
| 片上缓存读取 | `32KB Cache (64bit)`（SRAM） | 64 bit（8 Bytes） | `20 pJ` | `200×` | p. 14, Fig. 1.1.9 |
| 片上缓存读取 | `1MB Cache (64bit)`（SRAM） | 64 bit（8 Bytes） | `100 pJ` | `1,000×` | p. 14, Fig. 1.1.9 |
| 片外主存访问 | `DRAM (64bit)` | 64 bit（8 Bytes） | `1.3–2.6 nJ`（`1300–2600 pJ`） | `13,000–26,000×` | p. 14, Fig. 1.1.9 |

**正文补充数字（pp. 11–12, Section 5–6 & Fig. 1.1.9 右侧柱状图）**：
- **简单顺序处理器单条指令总开销**：`70 pJ / instruction`（其中指令缓存访问 `25 pJ`、寄存器堆访问 `6 pJ`，其余为控制与算术逻辑开销，p. 11 Sec. 6 & p. 14 Fig. 1.1.9），是单次 32 位整数加法（`0.1 pJ`）的 **700 倍**。
- **片上缓存静态漏电功耗**：在 45–32nm 工艺下，末级大容量 SRAM 缓存的静态漏电约为 **`100 mW / MB`**（p. 11 Sec. 5）。
- **DRAM 接口能耗**：传统标准接口 `> 20 pJ/bit`（Section 5 正文记一次 DRAM 访问约 `1–2 nJ`），低摆幅优化接口可降至约 `10 pJ/bit`（读取 8 字节页面数据约 `0.6 nJ`，p. 11 Sec. 5）。

### 2. SNN 常用估算数字的推导与三大失真原因

1. **`4.6 pJ` 与 `0.9 pJ` 是怎么算出来的**：原表中并无独立的 `MAC` 行。SNN 文献取 32 位浮点累加 $E_{\text{AC, FP32}} = E_{\text{FAdd, 32b}} = 0.9\text{ pJ}$，取 32 位浮点乘加 $E_{\text{MAC, FP32}} = E_{\text{FMult, 32b}} + E_{\text{FAdd, 32b}} = 3.7 + 0.9 = 4.6\text{ pJ}$，由此得出单次二值脉冲突触操作比浮点乘加省电 $4.6 / 0.9 \approx 5.11$ 倍。
2. **换成低比特整数（QNN）后算术差距缩水或反转**：在同一张表（45nm 0.9V）上，8 位整数乘加（INT8 乘法 + INT8 加法）仅为 $0.2 + 0.03 = 0.23\text{ pJ}$（若用 32 位累加器则为 $0.2 + 0.1 = 0.30\text{ pJ}$），比所谓的“FP32 脉冲累加 `0.9 pJ`”还低 **3 倍以上**；即便脉冲侧也改用 INT8 加法（`0.03 pJ`）或 INT32 加法（`0.1 pJ`），SNN 若需运行 $T \ge 4\sim 8$ 个时间步，多步累加与膜电位比较/复位开销就会迅速抵消省掉乘法器的收益。
3. **访存与数据搬运才是大头**：在 45nm 下，从最小的 8KB SRAM 读取 64 bit（`10 pJ`，合 `1.25 pJ/Byte`）已是 INT8 乘法（`0.2 pJ`）的 **6.25 倍**、INT8 加法（`0.03 pJ`）的 **41.7 倍**；而 SNN 每个时间步不仅要读权重，还必须反复**读取并写回膜电位状态**，并在片上网络传输带地址（AER）的稀疏事件包。忽略访存、只数 `4.6 pJ vs 0.9 pJ` 会严重高估 SNN 在静态输入上的能耗优势。

---

## 文献条目（按年份从新到旧）

### ETHEREAL（2026）

- 标题：ETHEREAL: A 25.6-µs/inf. Low-latency Event-driven Graph-neural-network Processor for High-resolution Vision at the Edge（长文版 `arXiv:2608.17787v1` 第 1 页标题）；另有 5 页短文版 `arXiv:2609.15241v1`，标题为 “A 25-µs/inf Event-driven Graph Neural Network Processor with Spatiotemporal Caching and Spline Convolution for Ultra-low-latency AI at the Edge”
- 作者：Adrian Kneip, Martin Lefebvre, Daniel Gehrig, Victoria Catalán Pastor, Davide Scaramuzza, Marian Verhelst, Charlotte Frenkel
- 发表处：arXiv（两份预印本：`2608.17787v1`，2026-08-18，14 页；`2609.15241v1`，2026-09-14，5 页）。**冲突说明**：Claude 草稿写作 “ESSERC 2026”，但两份 PDF 全文都没有出现 ESSERC 或其他录用信息（长文版参考文献里的 ESSERC 2026 指的是另一篇 HeMAiA 论文），因此会议归属记为“未核实”。（核对人 2026-11-16 复查时更正了本条早先写错的标题与发表处）
- 链接：https://arxiv.org/abs/2608.17787 ；https://arxiv.org/abs/2609.15241 ；代码：未公开
- 核对深度：全文
- 一句话：流片实现了首颗面向事件相机视觉的专用图神经网络 ASIC 芯片（28nm CMOS），用纯片上混合 3D/2D SRAM 和 4b/8b 量化图卷积实现每事件约 25.6 微秒的实时目标检测与分类。
- 解决的难点：解决了事件相机动态时空图在边缘芯片上建图指针访存不规则、片外 DRAM 功耗高、深层图卷积特征存储爆炸三大硬件瓶颈——通过“前浅层 3D 时空缓存异步逐事件更新 + 后深层体素池化转 2D 暂存器同步复用”的混合架构，将全部 1.25 MB 存储塞进 3.7 mm² 芯片内。
- 留下的局限：（1）**芯片运行的不是脉冲神经网络（SNN）**，而是带 4-bit/8-bit 整数量化激活（QReLU）与 B 样条边权（SplineConv）的量化事件图网络，节点间传递的是多位量化特征向量而非脉冲；（2）部署前必须做量化感知训练（QAT），不能免训练直接部署预训练浮点模型；（3）为了硬件确定性，将原版 DAGr 的双向多跳更新简化为有向单跳或受限双向图，纯 GNN 模式在复杂检测上精度偏低，仍需搭配帧图像 CNN 分支。
- 是否需要训练或微调：是（需对 4b/8b 混合精度 DAGr-GNN 进行量化感知训练 QAT，见 Section VI-A）
- 涉及的算子：半径邻域时空建图（Directed / Bidirectional FIFO 3D Grid）、SplineConv（一阶 B-spline 边特征加权卷积）、`max` 邻域聚合、体素网格 `max` 池化（VoxelPool）、量化 ReLU（QReLU）、层融合批归一化
- 关键数字：TSMC 28nm CMOS 流片，芯片面积 `3.7 mm²`，片上总 SRAM `1.25 MB`（`512 kB` 权重 + `144 kB` 3D 时空缓存 + `584 kB` 2D 暂存器，Section IV-V, pp. 5–9）；含 8 个邻居并行消息传递核，支持 4b/8b 权重与激活；在 VGA 分辨率（`640×480`）DSEC 目标检测基准上（8b/8b 首层 + 4b/4b 主干），流水线推理延迟 **`25.6 µs/event`**（顺序执行 `46.1 µs`；不同位宽配置下 `20.1–40.1 µs`），`0.95V / 220MHz` 下平均功耗 `65.2 mW`、单事件能耗 **`1.6 µJ/event`**（跨配置 `1.2–2.5 µJ`；低压 `0.6V` 模式下最低 **`0.54 µJ/event`**、功耗 `10.6 mW`），检测精度 `30.3 mAP50`（GNN+CNN 融合）/ `7.6 mAP50`（纯有向 GNN）；在 N-CARS 分类上精度 **`86.7%`**（Section VI, Fig. 18–21 & Table I, pp. 10–13）。**版本间数字冲突**：长文版摘要写单事件能耗 `1.6 µJ`，5 页短文版摘要写 `25.6 µs` 与 `1.7 µJ`、并称“比已有工作改善 10–1000 倍”（`arXiv:2609.15241` 第 1 页摘要），两者都记录，不作取舍。
- 与本方向的关系：直接相关（证明事件驱动视觉图网络已有真实流片需求，同时坐实当前专用芯片跑的是 4b/8b 量化 GNN 而非脉冲 GNN）
- 核对人与日期：Gemini，2026-11-16

### Yan 等：QNN-SNN Twin 能耗分析（2026）

- 标题：Reconsidering the Energy Efficiency of Spiking Neural Networks Inference from Analytical Perspectives
- 作者：Zhanglu Yan, Zhenyu Bai, Kaiwen Tang, Weng-Fai Wong
- 发表处：IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems（IEEE TCAD），2026（录用版对应 `arXiv:2409.08290v6`，首版提交于 2024 年 8 月）
- 链接：https://arxiv.org/abs/2409.08290 ；代码：未公开
- 核对深度：全文
- 一句话：构造了与 $T$ 步脉冲网络表达能力严格等价的 $\lceil\log_2(T+1)\rceil$ 比特量化网络（QNN-SNN Twin），从算术、膜电位读写、权重复用和片上网络路由四个维度推导出 SNN 胜过等价 QNN 的精确稀疏度盈亏线。
- 解决的难点：纠正了以往 SNN 论文拿“多步二值脉冲网络”去对比“32 位全精度浮点网络”且忽略膜电位访存与稀疏路由地址开销的不公平做法，给出了闭合解析不等式和跨视觉/语言模型的轨迹级硬件实测对照。
- 留下的局限：全文只分析了全连接层、卷积层与 Transformer（VGG、RepVGG、BERT/GLUE、Spiking Llama-2），**完全未涉及图神经网络**——既没有考虑图邻接表的不规则散列访存（图上节点度呈幂律分布、权重与激活复用率与规则网格截然不同），也未讨论分级/多阈值差分脉冲（如 DCGS）与动态 $k$NN 建图开销。
- 是否需要训练或微调：否（本文为解析能耗建模与硬件轨迹评估框架）
- 涉及的算子：IF 减法复位脉冲神经元、均匀量化 ReLU（QNN Twin）、稀疏事件路由（AER NoC）与稠密广播数据流
- 关键数字：（1）**等价性定理**（Theorem 1 & Proposition 1, pp. 2–3）：带减法复位的 $T$ 步率编码 IF 网络与激活位宽 $b = \lceil\log_2(T+1)\rceil$ 的 QNN 表达能力等价，且平均发放率满足 $(1-\gamma)/T \le s_r \le 1-\gamma$（$\gamma$ 为 QNN 激活零值比例，Theorem 2, p. 3）；（2）**22nm 典型类脑工艺下的盈亏阈值**（Fig. 5, p. 6）：在 $T=5$、4-bit 权重、扇入 $N_{\text{src}}=4096$、QNN 稀疏度 $\gamma=0.8$ 时，SNN 平均发放率必须满足 **$s_r < 5.7\%$**（8-bit 权重时 $< 6.4\%$）才能比等价 QNN 省电；当 $s_r > 12\%\sim 17\%$ 时，稀疏事件路由（Loihi 级 `3.0 pJ/bit/hop`）甚至比稠密广播（`0.25 pJ/bit/hop`）更费电（Section II-C, p. 6）；（3）**真实模型轨迹比值 $E_{\text{SNN}}/E_{\text{QNN}}$**（Tables II–V, p. 9）：CIFAR-10 VGG16 在 $T=3$（$s_r=5.49\%$）时为 **`0.987`**（勉强持平），$T=4,5,7$ 时升至 **`1.026 / 1.406 / 2.243`**；CIFAR-100 VGG16 在 $T=3\sim 7$ 全线落败（**`1.262–2.270`**）；ImageNet RepVGG（$T=10$）为 **`2.232`**；GLUE 7 任务平均（$T=3$）为 **`1.500`**；Spiking Llama-2 7B（$T=15$）高达 **`3.793`**（逼近理论退化比 $T/\lceil\log_2(T+1)\rceil = 15/4 = 3.75$）。
- 与本方向的关系：直接相关（本项目“M4 图网络公平能耗盈亏分析”最核心的方法论底座）
- 核对人与日期：Gemini，2026-11-16

### TopGQ（2026）

- 标题：TopGQ: Fast GNN Post-Training Quantization Leveraging Topology Information
- 作者：Dain Kwon, Kanghyun Choi, Hyeyoon Lee, Sunjong Park, Seoyong Lee, Sukjin Kim, Jinho Lee
- 发表处：Proceedings of the 63rd ACM/IEEE Design Automation Conference（DAC '26），July 26–29, 2026, Long Beach, CA, USA, 7 pages（DOI: `10.1145/3770743.3804433`；预印本 `arXiv:2608.30394v1`）
- 链接：https://arxiv.org/abs/2608.30394 ；代码：https://github.com/meowrowan/TopGQ
- 核对深度：全文
- 一句话：提出目前最强的**纯免训练图神经网络训练后量化（PTQ）**方法，把节点级量化缩放因子吸收进归一化邻接矩阵，并用一度/二度拓扑统计量零开销预测归纳式新节点的量化步长。
- 解决的难点：解决了 GNN 训练后量化（W8A8 / W4A4）的两大痛点：（1）不同度数节点的激活幅值差异巨大，若逐节点设量化因子会破坏整数矩阵乘法（GEMM），TopGQ 提出*选择性双轴缩放吸收*（将节点激活缩放对角阵 $\text{diag}(S_N)$ 吸收进稀疏邻接矩阵 $\tilde{A}$，Section 4.1 Eqs. 6–8, p. 3）；（2）归纳式图任务测试时出现未见节点无法预先校准，TopGQ 提出拓扑感知对偶索引 *TopPIN*（$\text{TopPIN}(v) = (d(v), \frac{1}{d(v)}\sum_{v_k \in \mathcal{N}(v)} \frac{1}{d(v_k)})$，Section 4.2 Eq. 9, p. 4）按局部拓扑查表插值赋参。
- 留下的局限：（1）该方法面向固定图拓扑（Cora、CiteSeer、Reddit、ogbn-products、MAG240M、IMDB-BINARY、COLLAB）上的 GCN / GAT / GraphSAGE / GIN，**未评测每层按特征动态重算 $k$NN 图的视觉图骨干（如 ViG、DGCNN）**——在动态图中邻接矩阵每层随特征改变，无法离线预吸收进固定稀疏矩阵；（2）属于稠密单步整数推理，不利用流式图帧间时间冗余。
- 是否需要训练或微调：否（纯训练后量化 PTQ，仅需无梯度校准统计，亚秒级至分钟级完成）
- 涉及的算子：图稀疏邻接聚合（SpMM）、线性投影（INT8 / INT4 GEMM）、注意力（GAT / R-GAT）、节点度拓扑索引（TopPIN）
- 关键数字：（1）在 Reddit 数据集 GCN 上（Table 1, p. 5），TopGQ 仅耗时 **`1.88 秒`**即达到 INT8 **`94.41%`**、INT4 **`93.05%`**（FP32 基线 `94.40%`；而 QAT 方法 Degree-Quant 耗时 `10.59 分钟`仅得 `87.01%` / `64.18%`，朴素 INT4 PTQ 仅 `3.79%`，见 Table 6, p. 6）；（2）在 2.4 亿节点的超大图 MAG240M（R-GAT）上（Table 2, p. 5），TopGQ 耗时 **`58.8 分钟`**达到 INT8 **`69.14%`**（FP32 `69.66%`），而梯度校准 PTQ 方法 DRA 需 `2.06 天`（`66.13%`）、SGQuant 需 `5.50 天`（`46.76%`）、Degree-Quant 预计需 `37.5 天`；（3）部署实测（ogbn-products GCN，Table 4, p. 6）：在 RTX 4090 上获得 **`1.68×`** 端到端推理加速（`34.51s → 20.53s`），在边缘端 Jetson AGX Orin 上获得 **`1.59×`** 加速（`754.09s → 473.25s`）。
- 与本方向的关系：直接相关（未来做免训练 GNN→SNN 转换与非脉冲量化对照时，除朴素均匀量化外最具代表性的现代免训练 GNN PTQ 强基线）
- 核对人与日期：Gemini，2026-11-16

### EDA-GNN / Liu 等（2026）

- 标题：Event-Driven Asynchronous Graph Neural Network FPGA Accelerator for Real-Time Edge Vision
- 作者：Tianhang Liu, Shen Zhang, Guangyao Yan, Runhua Wang, Rui Li, Shijie Meng, Hao Sun, Yajun Ha
- 发表处：IEEE Transactions on Circuits and Systems I: Regular Papers（IEEE TCAS-I），vol. 73, no. 8, pp. 5542–5555, Aug. 2026
- 链接：https://github.com/ShanghaiTechRicLab/EDA-GNN（IEEE Xplore 闭源期刊页未直接抓取全文，已核实上海科技大学 RicLab 官方代码仓与书目记录） ；代码：https://github.com/ShanghaiTechRicLab/EDA-GNN（仓库主页显示源码待正式开放）
- 核对深度：仅摘要
- 一句话：针对事件相机边缘视觉设计了免显式边存储（edge-free storage）的事件队列异步图神经网络 FPGA 加速器，通过层间并行流水实现微秒级单事件实时推理。
- 解决的难点：消除了动态事件图在 FPGA 上维护显式边表带来的庞大存储开销与指针跳转延迟，用时空解耦搜索窗口内的事件队列直接完成有向单跳邻居查找与层并行更新。
- 留下的局限：与 EvGNN、ETHEREAL 相同，加速的是定点量化的事件相机 GNN 而非脉冲神经元网络；未获取闭源全文 PDF，具体 LUT/DSP 资源数与单事件能耗待获取 IEEE Xplore 原文后补全。
- 是否需要训练或微调：未核实（仅读摘要与官方仓库说明，通常需训练事件图模型并做定点量化）
- 涉及的算子：时空事件队列有向建图、无边存储图卷积、层并行流水推理
- 关键数字：发表于 IEEE TCAS-I 2026 年第 73 卷第 8 期第 5542–5555 页；在边缘 FPGA（Xilinx Zynq UltraScale+ MPSoC）与事件相机基准（如 N-CARS）上实现微秒级逐事件低延迟推理（具体精度与功耗数字因未读闭源正文标为“未核实”）。
- 与本方向的关系：背景（再次印证 2025–2026 年硬件界围绕事件驱动视觉图加速器的活跃投入）
- 核对人与日期：Gemini，2026-11-16

### EFGCN（2026）

- 标题：Embedded Graph Convolutional Networks for Real-Time Event Data Processing on SoC FPGAs
- 作者：Kamil Jeziorek, Piotr Wzorek, Krzysztof Blachut, Andrea Pinna, Tomasz Kryjak
- 发表处：arXiv（`arXiv:2406.07318v3`，2026 年 6 月更新版；首版提交于 2024 年 6 月，硬件原型曾于 CVPR 2025 现场演示，pp. 5103–5104）
- 链接：https://arxiv.org/abs/2406.07318 ；代码：https://github.com/vision-agh/gcnn-dvs-fpga
- 核对深度：全文
- 一句话：在 Xilinx ZCU104 SoC FPGA 上实现了完全无需片外 DDR 的事件图卷积加速器，首次在硬件中引入事件流 3D MaxPool 图池化，兼顾高吞吐与低片上存储。
- 解决的难点：指出了 EvGNN 不做图池化导致深层图节点与边数不减、计算量放大最高 50 倍且依赖片外 DDR 的缺陷（Section 2.2 & Fig. 6, pp. 5, 16）；通过“邻域矩阵有向建图 + 首层异步卷积 + `4×4×4` 3D MaxPool 降采样 + 后继层时间通道同步卷积”，把片上存储需求比 EvGNN 降低 70%，完全驻留 BRAM/URAM。
- 留下的局限：（1）依然是 **8-bit（及 6-bit）QAT 整数量化 GNN**（带 ReLU 与逐位 `max`），不是脉冲网络；（2）3D MaxPool 在时间轴分窗聚合会打断严格逐事件异步流，导致第 2–5 层退回按时间片（$\Delta T$）同步批处理（Section 4.3, p. 12）；（3）需要 100 轮浮点训练加 50 轮 QAT 微调（Section 5.1.3, p. 15）。
- 是否需要训练或微调：是（100 epochs FP32 训练 + 50 epochs QAT 量化感知训练，Section 5.1.3, p. 15）
- 涉及的算子：离散化邻域矩阵有向时空建图（半径 $R=3$，最多 29 条边）、简化版 PointNetConv（线性变换 + BN 融合 + ReLU + `max` 邻域聚合）、硬件 `4×4×4` 3D Graph MaxPool
- 关键数字：部署于 Zynq UltraScale+ ZCU104（`200 MHz`），建图瓶颈仅 15 周期/事件，峰值吞吐率达 **`13.3 MEPS`**（百万事件/秒，Section 4.1, p. 9）；8-bit 下在 N-Cars 上精度为 **`91.0%`（EFGCN-S）/ `91.9%`（EFGCN-B）/ `92.5%`（EFGCN-L）**，明显高于 EvGNN 的 `87.8%`，且片上存储减少 **`70%`**（Table 7 & Sec. 5.4.4, pp. 18, 20）；N-Caltech101 精度 `62.8%–64.1%`，CIFAR10-DVS `57.4%–61.5%`，MNIST-DVS `97.9%–98.7%`，端到端（PL+PS）延迟 `4.39–9.31 ms`（单事件模块级延迟 `4.02–13.76 µs`，Table 7, p. 18）；Vivado 翻转率标定功耗为 PL `1.23–1.86 W` + ARM PS `2.64 W`（合计 `3.87–4.5 W`），相比 Jetson Orin NX（10W/25W）提速 **`7.6–8.4×`**（Section 5.4.3 & Fig. 8, pp. 20, 22）。
- 与本方向的关系：直接相关（展示了 `max` 邻域聚合与 3D `max` 图池化在整数硬件上的高效比较器实现，同时凸显 QAT 重训成本与后层退回同步处理的局限）
- 核对人与日期：Gemini，2026-11-16

### GraphPTQ（2026）

- 标题：GraphPTQ: Post-Training Quantization for Graph Models, Memory Savings and Generalization
- 作者：Itay Niv, Neta Rabin
- 发表处：ICML 2026 Workshop on Graph Foundation Models（GFM @ ICML 2026，Oral）
- 链接：https://icml.cc/（OpenReview 论文页因反爬拦截返回 HTTP 403，已通过检索核实 ICML 2026 GFM Workshop Oral 录用信息与标题作者） ；代码：未核实
- 核对深度：仅摘要
- 一句话：研究图模型上的训练后量化（PTQ）在显存压缩与泛化性能上的表现，是 2026 年与 TopGQ 同期出现的另一项免训练图量化探索。
- 解决的难点：在不重新训练图模型的前提下压缩图表示与模型参数存储，考察量化对图模型泛化的影响。
- 留下的局限：因 OpenReview 受反爬限制未能读取全文 PDF，其是否覆盖动态 $k$NN 视觉图与超低比特（2–4 bit）设置尚未核实；属于静态图量化范畴而非事件驱动推理。
- 是否需要训练或微调：否（训练后量化 PTQ）
- 涉及的算子：图神经网络消息传递与训练后量化（具体算子细节未核实）
- 关键数字：ICML 2026 GFM Workshop Oral 报告；具体数据集精度与压缩比因未打开 OpenReview 正文标为“未核实”。
- 与本方向的关系：背景（补充证明图网络免训练 PTQ 在 2026 年正快速成为活跃子方向，做公平对照时应关注最新进展）
- 核对人与日期：Gemini，2026-11-16

### EvGNN（2025）

- 标题：EvGNN: An Event-Driven Graph Neural Network Accelerator for Edge Vision
- 作者：Yufeng Yang, Adrian Kneip, Charlotte Frenkel
- 发表处：IEEE Transactions on Circuits and Systems for Artificial Intelligence（IEEE TCASAI），vol. 2, no. 1, pp. 37–50, 2025（DOI: `10.1109/TCASAI.2024.3520905`；预印本 `arXiv:2404.19489v2` 首版提交于 2024 年 4 月）
- 链接：https://arxiv.org/abs/2404.19489 ；代码：https://github.com/cogsys-tudelft/evgnn
- 核对深度：全文
- 一句话：提出面向事件相机的算法—硬件协同优化框架，用有向单跳棱柱队列建图和**免重训 INT8 训练后量化（PTQ）**的简化 PointNetConv，在边缘 FPGA 和 28nm ASIC 模型上实现 16 微秒级逐事件推理。
- 解决的难点：（1）把 AEGNN 中昂贵的双向多跳无向图与 SplineConv 简化为有向单跳图 + 轻量 PointNetConv，避免了历史节点级联重算；（2）用 L1 距离棱柱邻域 + 片上事件队列取代耗时的球面欧氏建图；（3）设计层间并行硬件流水，将多跳延迟压缩到微秒量级。
- 留下的局限：（1）**同样不是脉冲网络**：节点激活是 8-bit 整数量化特征，不是脉冲序列；（2）未设计图池化层，深层图规模不收缩，特征维度被迫限制在 32 维以内，且大容量场景仍需访问片外 DRAM（DRAM 占总延迟 50% 以上、占总能耗约 73%，见 Section V-D 及 EFGCN 对照分析）；（3）只适用于事件相机时空坐标建图，不涉及按高维语义特征重算动态 $k$NN 的视觉图骨干。
- 是否需要训练或微调：部分（浮点简化模型本身需从头训练；但**从 FP32 转为硬件 8-bit 定点部署采用的是免重训的训练后量化 PTQ + BN 折叠**，仅掉点 0.1%，见 Section III-C & Table IV, p. 6！）
- 涉及的算子：有向单跳棱柱 L1 邻域事件队列建图、简化版 PointNetConv（线性层 + BN 折叠 + ReLU + `max` 邻域聚合）、INT8 训练后量化（PTQ）
- 关键数字：（1）量化消融（Table IV, p. 6）：在 N-CARS（`120×100`）上，FP32 精度 `87.9%`，**直接 INT8 PTQ 精度达 `87.8%`**（仅降 `0.1%`），单事件计算量仅 `0.07 MFLOPs/ev`（参数量 `4.8k`）；（2）**Xilinx KV260 Ultrascale+ MPSoC 实测**（Tables V–VI, p. 10）：占用 `30,908` LUTs、`24,083` FFs、`228` DSPs、`1.76 MB` 片上存储（15 BRAM + 48 URAM），单事件端到端延迟 **`16 µs/event`**，精度 **`87.8%`**；（3）**28nm CMOS ASIC 综合与仿真评估**（`0.8V, 200MHz, 2.5 mm²`，Section V-D, p. 11 & Fig. 14, p. 12）：单事件延迟 **`10.7 µs/event`**，单事件能耗仅 **`305 nJ/event`**。
- 与本方向的关系：直接相关（既是事件图硬件的核心参照，又给出了一个极重要的事实：**在 PointNetConv + `max` 聚合上，标准 INT8 PTQ 几乎无损（掉点仅 0.1%）**，这意味着脉冲转换在 8-bit 等效预算下将面临极强的 PTQ 对手！）
- 核对人与日期：Gemini，2026-11-16

### AMPLE（2025）

- 标题：AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks
- 作者：Pedro Gimenes, Aaron Zhao, George Constantinides
- 发表处：Proceedings of the 5th Workshop on Machine Learning and Systems（EuroMLSys '25），pp. 107–113, ACM, 2025（DOI: `10.1145/3721146.3721963`）
- 链接：https://doi.org/10.1145/3721146.3721963（全文 PDF：https://2025.euromlsys.eu/pdf/euromlsys25-41.pdf） ；代码：未公开
- 核对深度：全文
- 一句话：针对静态大图节点度数极度不均导致的双缓冲流水线空泡，设计了“按节点槽位异步完成事件触发调度”的混合精度（FP32 / INT8 / INT4）GNN FPGA 加速器。
- 解决的难点：（1）传统 GNN 加速器（如 HyGCN）按批次同步双缓冲，低度节点必须干等高度数 hub 节点聚完，AMPLE 用 64 槽位节点记分牌（Node Scoreboard）实现节点完成即释放重分配的事件驱动控制流；（2）用硬件 NoC 划分子网原生支持按节点度数分配精度（高度数节点走 Float 核，其余 >97% 低度节点走 INT8/INT4 核）；（3）用多通道 HBM 预取标签（Fetch Tags）隐藏大图片外访存延迟。
- 留下的局限：（1）此处的“Event-Driven”指**硬件任务调度上的节点完成事件驱动**，而非神经元脉冲事件驱动，内部算术仍是常规浮点/定点乘加；（2）节点混合精度掩码直接照搬 Degree-Quant（需 QAT 训练）；（3）仅在 Modelsim + Vivado（Alveo U280 @ 200MHz）做延迟仿真，未报告实测板级功耗数字，也未支持 `max` 聚合或动态 $k$NN 建图。
- 是否需要训练或微调：是（依赖 Degree-Quant 的量化感知训练策略确定高精度保护节点比例，Section 2.3.1 & 4.1, pp. 3, 5）
- 涉及的算子：`sum` / `mean` 邻域聚合（GCN、GIN、GraphSAGE）、节点级混合精度（Float / INT8 / INT4）矩阵乘法、HBM 异步预取
- 关键数字：在 Xilinx Alveo U280（`200 MHz`）仿真下，跨 Cora、CiteSeer、PubMed、Flickr、Reddit、Yelp 六个图数据集（2.7K 至 71.7 万节点，高精度保护节点占比均 `< 3%`，Table 5, p. 5）及 GCN / GIN / GraphSAGE 三种模型，相比 Intel Xeon CPU 平均提速 **`243×`**，相比 NVIDIA RTX A6000 GPU 平均提速 **`7.2×`**（单层 GCN 上平均延迟 `15.2 ms`，比 CPU 快 `361.1×`、比 GPU 快 `12.9×`，Fig. 4 & Table 6, p. 6）。
- 与本方向的关系：可借用的技术（揭示了图结构特有的**幂律度数长尾效应**：既导致低比特量化误差集中在 <3% 的高度数节点，又导致同步硬件出现严重等待空泡，这对分析脉冲图网络的发放率分布与硬件映射极有启发）
- 核对人与日期：Gemini，2026-11-16

### Bhattacharjee 等：SATA 与 SpikeSim 硬件评测（2024）

- 标题：Are SNNs Truly Energy-efficient? — A Hardware Perspective
- 作者：Abhiroop Bhattacharjee, Ruokai Yin, Abhishek Moitra, Priyadarshini Panda
- 发表处：2024 IEEE International Conference on Acoustics, Speech and Signal Processing（ICASSP 2024），pp. 13311–13315（DOI: `10.1109/ICASSP48485.2024.10448269`；预印本 `arXiv:2309.03388v1`）
- 链接：https://arxiv.org/abs/2309.03388 ；代码：https://github.com/Intelligent-Computing-Lab-Yale/SATA 及 https://github.com/Intelligent-Computing-Lab-Yale/SpikeSim
- 核对深度：全文
- 一句话：用数字脉动阵列（SATA，28nm）和模拟阻变存算一体（SpikeSim，65nm RRAM）两套硬件评测工具实测主流 SNN 算法，证明论文里按 `FLOPs × T × 稀疏度` 算出的理论省电倍数比真实硬件高估了一个数量级以上。
- 解决的难点：定量拆解了 SNN 落到真实芯片上的三大隐性开销：（1）跨时间步重复从片上缓存搬运权重与膜电位（稀疏性只能在 PE 内部跳过累加，片上 Buffer 读取无法跳过）；（2）LIF 神经元模块自身功耗与膜电位 SRAM 面积占比极高；（3）模拟 RRAM 交叉阵列寄生电阻与器件噪声会在多时间步上累积导致精度雪崩。
- 留下的局限：评测对象全部为 BPTT 直接训练的静态图像 CNN（VGG9、VGG16、ResNet18），未涉及免训练转换模型、多比特/分级脉冲或图神经网络的不规则访存。
- 是否需要训练或微调：部分（评测对象为预训练 BPTT SNN；文中缓解非理想性的 BN 自适应方法仅更新 BN 均值方差而无需重训权重，Section IV, p. 4）
- 涉及的算子：2D 卷积、LIF 神经元膜电位更新与复位、动态时间步早退（熵阈值 DT-SNN）、通道间共享 LIF 单元
- 关键数字：（1）**理论估算 vs 真实硬件能耗鸿沟**（CIFAR-10, 28nm CMOS，Table II, p. 2）：例如 TSSL（$T=5$，稀疏度 `90.1%`）按公式估算出 `5.8e4 nJ`（声称省电 `80×`），但在数字加速器 SATA 上实测耗能 **`3.1e7 nJ`（仅省电 `4.9×`）**，在 SpikeSim 上为 `1.3e5 nJ`（`16×`）；TDBN（$T=4$）公式估算省电 `83×`，SATA 实测仅 **`6.8×`**（`5.4e7 nJ`）；LTH（$T=5$）公式估算 `15×`，SATA 实测仅 **`1.3×`**；（2）**硬件瓶颈占比**（Section IV, p. 4）：在 128-PE SATA 上，LIF 神经元单元最高占计算部件总功耗的 **`61.6%`**（共享 4 通道 LIF 可降 `75.1%` LIF 功耗）；采用权重驻留的 tick-batch 数据流在 $T=4$（Tiny ImageNet VGG9）下可节省 **`62.5%`** 数据搬运能耗；动态时间步早退（平均 $T=2.14$）在 CIFAR-10 VGG9 上降低 **`46.5%`** 总能耗（Fig. 4–5, p. 4）。
- 与本方向的关系：直接相关（警示我们在写能耗分析时绝不能只算突触加法次数，必须把膜电位缓存与跨时间步数据搬运写进成本模型）
- 核对人与日期：Gemini，2026-11-16

### DRA（2024）

- 标题：Leveraging Dynamic Range Analysis for Efficient Post-Training Quantization in Graph Convolutional Networks
- 作者：Hadi Mousanejad Jeddi, Mahdieh Grailoo, Jose Nunez-Yanez
- 发表处：2024 IEEE Nordic Circuits and Systems Conference（NorCAS 2024），DOI: `10.1109/NorCAS64408.2024.10752486`
- 链接：https://doi.org/10.1109/NorCAS64408.2024.10752486（核实自作者官方机构页 https://liu.se/en/employee/hadmo97 及 TopGQ 全文第 5–7 页参考文献 [16] 与复现表） ；代码：未公开
- 核对深度：仅摘要（含 TopGQ 全文中的算法剖析与实测复现数据）
- 一句话：早期针对图卷积网络（GCN）的训练后量化（PTQ）方法，通过动态范围分析与基于梯度的迭代校准拟合 FP32 分布，无需重训主干权重。
- 解决的难点：在 TopGQ 之前尝试摆脱 Degree-Quant 的完整 QAT 重训流程，在小图（如 Cora、CiteSeer、IMDB-BINARY、COLLAB）INT8 上取得接近浮点精度。
- 留下的局限：据 TopGQ（Section 1, 6 & Table 1–2, pp. 1, 5, 6）核实，DRA 仍依赖反向梯度校准迭代且未解决跨节点度数尺度的矩阵乘吸收问题：（1）在大图上校准极慢（2.4 亿节点 MAG240M 上需 `2.06 天`，比 TopGQ 慢 50 倍以上）；（2）在大图与 INT4 低比特上严重崩溃（Reddit GCN INT4 暴跌至 `1.75%`，ogbn-products GCN INT8 跌至 `36.22%`、INT4 跌至 `3.12%`）。
- 是否需要训练或微调：部分（属 PTQ，不更新网络主干权重，但校准阶段需用梯度迭代优化量化参数）
- 涉及的算子：GCN / GraphSAGE / GIN / GAT 图消息传递、基于梯度的动态范围校准量化
- 关键数字：据 TopGQ 复现表（`topgq_2608.30394.pdf` Table 1–3, p. 5）：Cora GCN INT8 **`79.76%`** / INT4 **`77.02%`**（耗时 `1.7s`）；Reddit GCN INT8 **`93.15%`**（`42.99s`）、但 INT4 崩至 **`1.75%`**；ogbn-products GCN INT8 **`36.22%`** / INT4 **`3.12%`**（FP32 `71.25%`）；MAG240M R-GAT INT8 **`66.13%`**（耗时 **`2.06 天`**）。
- 与本方向的关系：背景（标志 GNN 量化从 QAT 走向免训练 PTQ 的中间过渡节点，同时展示忽略节点拓扑差异时 INT4 PTQ 在大图上会彻底崩溃）
- 核对人与日期：Gemini，2026-11-16

### SpiNNaker 2（2024）

- 标题：SpiNNaker2: A Large-Scale Neuromorphic System for Event-Based and Asynchronous Machine Learning
- 作者：Hector A. Gonzalez, Jiaxin Huang, Florian Kelber, Khaleelulla Khan Nazeer, Tim Langer, Chen Liu, Matthias Lohrmann, Amirhossein Rostami, Mark Schöne, Bernhard Vogginger, Timo C. Wunderlich, Yexin Yan, Mahmoud Akl, Christian Mayr
- 发表处：37th NeurIPS Workshop on Machine Learning with New Compute Paradigms（MLNPCP @ NeurIPS 2023），预印本发布于 2024 年 1 月（`arXiv:2401.04491v1`）
- 链接：https://arxiv.org/abs/2401.04491 ；代码：https://doi.org/10.5281/zenodo.10202110（`py-spinnaker2`）
- 核对深度：全文
- 一句话：介绍了采用 22nm FDX 工艺量产的数字异步类脑芯片 SpiNNaker 2（单芯片 152 个 ARM 核 + 硬件 MAC 阵列），原生支持最高 **128-bit 载荷的事件数据包**和分级事件神经网络（如 EGRU）。
- 解决的难点：打破了传统类脑芯片只能传 1-bit 二值脉冲且不支持矩阵乘加速的限制——每个处理单元（PE）既能通过中断驱动异步处理稀疏脉冲，又内置 8-bit/16-bit 整数矩阵乘与卷积加速器，路由包可携带多位分级激活值或反向误差梯度。
- 留下的局限：单 PE 本地 SRAM 仅 `128 kB`（需靠片上 DMA 与 `2 GB` 片外 LPDDR4 协同调度大模型），若图拓扑动态变化，预编译静态路由表（Section 3.2, p. 5）难以在运行期低开销重配。
- 是否需要训练或微调：否（硬件平台，可直接部署转换后的 SNN、EGRU 或普通量化 DNN，Section 3.2–3.3, p. 5）
- 涉及的算子：任意可编程脉冲/事件神经元（IF / LIF / EGRU）、最高 128-bit 载荷多播事件包、8-bit/16-bit 整数矩阵乘（MAC Array）与 2D 卷积、硬件指数/对数与随机数加速
- 关键数字：GlobalFoundries 22nm FDX 工艺（已量产超 3.5 万颗芯片组成 500 万核系统，Section 1–2, p. 2；据 SENECA Table 1 & 3, pp. 5–6 交叉核实单核面积 `1.09 mm²`、约 `10 pJ/SOp`）；单芯片含 **152 个 ARM Cortex-M4F PE**，每 PE 配 **`128 kB` SRAM**，芯片配 **`2 GB` LPDDR4 DRAM** 和 6 条芯片间链路；片上/片间路由支持最高 **128-bit payload** 数据包（32-bit 神经元 ID + 多位数值载荷，Section 2, p. 2 & Section 3.3.2, p. 6）；在 DVS Gesture 手势识别（128 PEs 部署 2 层分级事件网络 EGRU）上，batch=1 实时推理单步能耗约 **`~190 µJ/step`**，显著低于 Nvidia A100（`~950 µJ/step`）与 GTX 1070Ti（`~360 µJ/step`）（Section 3.3.1 & Fig. 3, pp. 5–6）。
- 与本方向的关系：直接相关（证明主流新一代数字类脑芯片已原生支持多位分级事件包与局部整数 MAC，本方向若采用多阈值/分级脉冲或局部混合算子，在硬件上完全可落地上板）
- 核对人与日期：Gemini，2026-11-16

### Ottati 等：To Spike or Not To Spike（2023）

- 标题：To Spike or Not To Spike: A Digital Hardware Perspective on Deep Learning Acceleration
- 作者：Fabrizio Ottati, Chang Gao, Qinyu Chen, Giovanni Brignone, Mario R. Casu, Jason K. Eshraghian, Luciano Lavagno
- 发表处：IEEE Journal on Emerging and Selected Topics in Circuits and Systems（IEEE JETCAS），vol. 13, no. 4, pp. 1015–1025, Dec. 2023（预印本 `arXiv:2306.15749v5`）
- 链接：https://arxiv.org/abs/2306.15749 ；代码：https://github.com/fabrizio-ottati/awesome-neuromorphic-hw
- 核对深度：全文
- 一句话：从数字芯片设计视角对比了最先进的 ANN 与 SNN 加速器流片实测数据，明确得出**在静态图像分类上现有数字 ANN 芯片在精度—能耗—吞吐上全面帕累托占优（Pareto-dominate）SNN 芯片**，脉冲优势只可能出现在时序流或事件传感器任务及混合架构中。
- 解决的难点：摒弃了模糊且忽略膜电位读写的 `SynOP/W` 指标，统一改用“每次完整推理的焦耳数（Energy/inference）”加 45nm SRAM/INT8 解析模型和 ImageNet / 语音流片芯片实测数据做横向硬核对比。
- 留下的局限：分析聚焦于卷积与循环网络（CNN / ViT / RNN），未覆盖图神经网络加速器；其解析模型未考虑寄存器级空间复用优化。
- 是否需要训练或微调：否（硬件综述与解析能耗对比研究）
- 涉及的算子：INT8 2D 卷积、LIF 脉冲卷积、RNN / GRU / LSTM 时序单元
- 关键数字：（1）**45nm 解析卷积对比**（Section IV-A, pp. 3–4）：对 `(512, 3, 3)` 卷积窗口（8-bit 权重与状态，8KB SRAM 读 64b 耗能 `10 pJ`），单步 SNN 耗能 `13.1 nJ · γ_SNN`，单通 INT8 ANN 耗能 `24.1 nJ · γ_ANN`（单步仅省 `1.84×`），但静态图像 SNN 需乘时间步 $T$（$E_{\text{SNN}} = 13.1\text{ nJ} \cdot T \cdot \gamma_{\text{SNN}}$），只要同稀疏度下 $T \ge 2$ 即输给 INT8 ANN；（2）**ImageNet 流片芯片实测**（Table II & Fig. 1, pp. 5–6）：同在 **28nm CMOS** 下，最佳数字 SNN 芯片 SNPU'23（ResNet18，$T=16$）精度仅 **`66.8%`**、单帧能耗高达 **`2.0 mJ`**（正文精确值 `1.95 mJ/inf`），而 28nm INT8 CNN 芯片 Mo'21（ResNet50）精度 **`76.9%`**、能耗 **`1.1 mJ`**（最佳点 `0.46 mJ/inf`），5nm INT4 Transformer 芯片 Keller'23（DeiT-Base）精度 **`80.5%`**、能耗 **`1.0 mJ`**，**28nm ANN+SNN 混合芯片 C-DNN'23**（ResNet50）精度 **`77.1%`**、能耗仅 **`0.3 mJ`（`281 µJ`）**。
- 与本方向的关系：直接相关（强力支撑 Claude 草稿中的假设 H3——在静态单帧图像图上脉冲网络极难在真实数字硬件能耗上击败 INT8/INT4 量化网络，真正的突破口在流式图时间差分与混合事件架构）
- 核对人与日期：Gemini，2026-11-16

### Dampfhoffer 等：硬件感知 SNN vs ANN 研究（2023）

- 标题：Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study
- 作者：Manon Dampfhoffer, Thomas Mesquida, Alexandre Valentian, Lorena Anghel
- 发表处：IEEE Transactions on Emerging Topics in Computational Intelligence（IEEE TETCI），vol. 7, no. 3, pp. 731–741, June 2023（DOI: `10.1109/TETCI.2022.3214509`；CEA-HAL 编号 `cea-03852141`）
- 链接：https://cea.hal.science/cea-03852141 ；代码：未公开
- 核对深度：仅摘要（已从 HAL 官方 API 获取完整元数据与作者摘要；当日访问 CEA-HAL 文档下载端点返回 `HTTP 500 Internal Server Error`，未读全文 PDF）
- 一句话：以经典数字加速器 Eyeriss 的数据复用与稀疏上限为参照，证明简单 IF 神经元远比 LIF 省电，且 SNN 只有在**每个突触每次推理平均仅发放 0.15 到 1.38 个脉冲**的极高稀疏区间内才能与高效数字 ANN 抗衡。
- 解决的难点：量化了 ANN 侧空间/时间数据复用（如行驻留/权重驻留）以及 SNN 侧不同神经元动力学（IF vs LIF vs 连续突触模型）的状态维护开销，打破了“只要把乘加换成累加就省电”的迷思。
- 留下的局限：未获取全文 PDF 核对其具体 CMOS 工艺节点参数表；研究范围限于前馈/卷积网络，未分析图消息传递的邻域访存特征。
- 是否需要训练或微调：否（硬件感知能耗建模分析）
- 涉及的算子：IF 神经元、LIF 神经元、连续突触电流模型、Eyeriss 风格空间数据复用卷积
- 关键数字：据 HAL 官方摘要（`halId: cea-03852141`）：IF 脉冲网络要与充分利用数据复用和稀疏性的数字 ANN 实现竞争，**每次推理每突触平均脉冲数必须落在 `0.15–1.38 spikes/synapse/inference` 的稀疏窗口内**（具体取决于所对比的 ANN 硬件数据流实现）；作者据此主张在高稀疏层用事件驱动 SNN、在低稀疏层用并行 ANN 的混合架构。
- 与本方向的关系：直接相关（给出了极直观的“每突触每推理脉冲数 `0.15–1.38`”经验警戒线，可直接用于核验转换后视觉图模型的每标量累计事件数是否越过盈亏线）
- 核对人与日期：Gemini，2026-11-16

### SENECA（2023）

- 标题：SENECA: building a fully digital neuromorphic processor, design trade-offs and challenges
- 作者：Guangzhi Tang, Kanishkan Vadivel, Yingfu Xu, Refik Bilgic, Kevin Shidqi, Paul Detterer, Stefano Traferro, Mario Konijnenburg, Manolis Sifalakis, Gert-Jan van Schaik, Amirreza Yousefzadeh
- 发表处：Frontiers in Neuroscience，vol. 17, Article 1187252, June 2023（DOI: `10.3389/fnins.2023.1187252`）
- 链接：https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1187252/full ；代码：学术申请获取（未公开仓库）
- 核对深度：全文
- 一句话：Imec 研制的 GF-22nm 全数字可编程类脑处理器（RISC-V + 微码循环缓冲器 + 8 个 SIMD 神经处理单元），**原生支持 4/8/16-bit 分级脉冲与 `MAX/MIN` 比较指令**，并用实测数据证明：**把 16-bit 分级脉冲换成二值脉冲仅省电 6.1%，而通过脉冲分组复用膜电位能省电 47.0%**！
- 解决的难点：解决了传统类脑芯片“专用硬连线太死板、纯嵌入式核控制开销又大十倍”的两难——用轻量 RISC-V 做每脉冲级（`<1%` 算量）地址计算与压缩，用比 RISC-V 省电 100 倍的 Loop Buffer 驱动 8 个 NPE 执行每突触级运算，并系统对比了 11 款主流类脑芯片的架构参数。
- 留下的局限：片上多播路由表较小，遇到百万级完全随机无结构连接时需退回单播模式（Section 2.4 & 3.5, pp. 7, 10–11）；目前仅给出单核 RTL 级（Cadence JOULES，距签核功率误差 15% 内）算子与单层基准测试，未跑完整视觉图网络。
- 是否需要训练或微调：否（通用可编程数字类脑硬件，支持事件驱动全连接、卷积与在线学习 e-prop）
- 涉及的算子：**分级脉冲（Graded Spikes，带可选数值载荷）**、二值脉冲、4/8/16-bit 线性与浮点（FP16/BF16）算术指令（`ADD/SUB/MUL`）、**硬件比较与极值指令（`GTH/MAX/MIN`）**、脉冲分组（Spike-grouping）
- 关键数字：（1）**芯片规格**（GF-22nm FDX，`0.8V, 500MHz`，Table 1, 6, 7, pp. 5, 10, 12）：单核面积 **`0.47 mm²`**（其中 `2 Mb` 数据 SRAM 占 **70%** 面积，8 个 NPE 仅占 `1.3%`，RISC-V 占 `2.3%`），静态漏电 `30 µW`；NPE 寄存器访问（`8–12 fJ/b`）比片上 SRAM（`180–220 fJ/b`）省电约 **20 倍**（Table 2 & Sec. 3.2, pp. 5, 9）；（2）**单指令能耗拆解**（Table 8, p. 13）：FP16 `ADD/SUB/MUL` 总能耗 **`1.4 pJ`**（纯 ALU 仅 `0.5 pJ`），2×INT8 算术 **`1.2 pJ`**，FP16 `GTH/MAX/MIN` **`1.2 pJ`**，而单次 16b 数据 SRAM 读/写高达 **`3.7 pJ` / `3.9 pJ`**；（3）**分级脉冲 vs 二值脉冲关键实测**（4000 神经元全连接层处理 16 输入脉冲、64k SOPs，Table 9, p. 15）：16-bit 分级脉冲基线总能耗 `908.8 nJ`（**`14.2 pJ/SOp`**）；改成二值脉冲（省去乘法但仍读写 16b 权重状态）总能耗 `853.1 nJ`（**`13.3 pJ/SOp`，仅省电 `6.1%`**！）；而保留分级脉冲、仅将 4 个脉冲分组复用寄存器内膜电位（Spike-grouping）即降至 `481.9 nJ`（**`7.5 pJ/SOp`，省电 `47.0%`**）；结合二值脉冲 + 4b 权重/8b 状态量化 + 脉冲分组可降至 **`2.7 pJ/SOp`**（省电 `80.7%`）。
- 与本方向的关系：直接相关（为本项目使用**多阈值/分级脉冲（如 DCGS）和硬件 `MAX` 指令**提供了极其硬核的电路级依据：在真实 22nm 类脑核上，分级脉冲带来的乘法器额外功耗不到 7%，真正决定能耗的是事件稀疏度、低位宽权重以及状态访存复用！）
- 核对人与日期：Gemini，2026-11-16

### Lemaire 等：SNN 与 FNN 解析能耗模型（2022）

- 标题：An Analytical Estimation of Spiking Neural Networks Energy Efficiency
- 作者：Edgar Lemaire, Loïc Cordone, Andrea Castagnetti, Evgeny Novakovsky, Jean-François Rousseau, Benoît Miramond
- 发表处：29th International Conference on Neural Information Processing（ICONIP 2022），Part I, LNCS vol. 13623, pp. 577–587（预印本 `arXiv:2210.13107v1`）
- 链接：https://arxiv.org/abs/2210.13107 ；代码：未公开
- 核对深度：全文
- 一句话：基于 Horowitz 45nm INT32 与 SRAM 插值参数建立与具体芯片解耦的解析模型，指出 **SNN 总能耗中超过 98% 来自存储器访问**，且 SNN 只有在层间用微型脉冲 FIFO 替代完整特征图缓存时才比前馈网络省电 6–8 倍。
- 解决的难点：把神经元参数存储、状态读写、层间特征图/FIFO 缓冲与寻址计算全部纳入解析公式（Eqs. 1–20, pp. 3–5），纠正了忽略片上存储容量差异的纯算术对比。
- 留下的局限：（1）模型设定偏袒 SNN 的层间存储假设——假设前馈 ANN 必须把整张高精度特征图存在大容量 SRAM 中，而 SNN 只需 1000 深度的微型脉冲 FIFO（若 ANN 采用深度优先或分块缓存、或 SNN 需跨步缓存膜电位，比值会大幅改写）；（2）算术采用 INT32（乘法 `3.1 pJ`、加法 `0.1 pJ`）而非现代加速器常用的 INT8/INT4；（3）在静态图像 CIFAR-10 上为压低时间步（$T=4$），SNN 精度比 FNN 暴跌 6.7 个百分点（`95.1% → 88.4%`），并非等精度比较！
- 是否需要训练或微调：否（解析能耗评估模型，评测用 SNN 采用替代梯度直接训练）
- 涉及的算子：卷积层、全连接层、LIF 脉冲神经元、SRAM 容量对数线性插值访存模型
- 关键数字：采用 Horowitz 45nm INT32 参数与 SRAM 对数插值（Eq. 1, p. 3）：（1）CIFAR-10 VGG-16（$T=4$，平均每神经元发放 `0.10` 脉冲，Table I–II, pp. 6–7）：FNN 能耗 `1.30e7 nJ`（精度 `95.1%`），SNN 能耗 `1.58e6 nJ`（精度仅 **`88.4%`**），名义比值 `8.19×`，其中 **SNN 访存能耗占比高达 `99.9%` 以上**（`1.58e6 nJ` 访存 vs `4.34e2 nJ` 计算，Table II, p. 7）；（2）Google Speech Commands 语音（$T=2$）：FNN `93.6%` vs SNN `91.8%`，能耗比 `6.22×`；（3）**Prophesee NCARS 事件相机分类**（$T=5$，发放率 `0.08`）：SNN 精度反超（**`93.5%` vs `93.4%`**）且能耗比达 **`8.17×`**（Table I–II, pp. 6–7）。
- 与本方向的关系：可借用的技术（一方面再次证实静态图低 $T$ 易掉点、事件相机数据最契合 SNN；另一方面其“访存占 >98% 能耗”的结论提醒我们图邻接访存建模不可省略）
- 核对人与日期：Gemini，2026-11-16

### Loihi 2（2021）

- 标题：Efficient Neuromorphic Signal Processing with Loihi 2
- 作者：Garrick Orchard, E. Paxon Frady, Daniel Ben Day Rubin, Sophia Sanborn, Sumit Bam Shrestha, Friedrich T. Sommer, Mike Davies
- 发表处：2021 IEEE Workshop on Signal Processing Systems（SiPS 2021），pp. 254–259（DOI: `10.1109/SiPS52927.2021.00053`；预印本 `arXiv:2111.03746v1`）
- 链接：https://arxiv.org/abs/2111.03746 ；代码：https://github.com/lava-nc/lava（Intel Lava 软件栈）
- 核对深度：全文
- 一句话：Intel 第二代异步类脑芯片 Loihi 2（Intel 4 工艺）的官方信号处理论文，明确引入了**可编程神经元微码与最高 32-bit 分级脉冲（graded spikes）**，大幅压缩了高动态范围信号处理所需的脉冲数和时间步。
- 解决的难点：解决了第一代 Loihi 只能传二值脉冲且神经元固定为 CUBA LIF 导致逼近连续值需要几百步、延迟高的问题——允许神经元通过汇编微码自定义内部状态方程，并在触发时发出携带最多 32 位整数载荷的分级脉冲直接参与突触乘加。
- 留下的局限：芯片硬件不对外零售（仅对 Intel INRC 研究社区开放）；动态重连图边（如逐层动态 $k$NN）不符合其预设静态突触路由表范式。
- 是否需要训练或微调：否（硬件架构与算法映射，支持免训练转换或直接编程的脉冲算法）
- 涉及的算子：**最高 32-bit 分级脉冲（Graded Spikes）**、可编程微码神经元（如谐振发放 Resonate-and-Fire、Sigma-Delta 神经元）、卷积连接权值共享、三维网格异步路由
- 关键数字：采用 Intel 4 工艺（据 SENECA Table 1, p. 5 核实单核面积 `0.21 mm²`、每核 `1.5 Mb` 存储，单芯片 128 核），**原生支持最高 32-bit 分级脉冲载荷**（Section II, p. 2 右栏）；在短时傅里叶变换（STFT）近似语音谱特征任务上，相比 CPU 实现能耗降低超过 **`100×`**，带宽压缩 **`27×`**（Section IV, pp. 4–5）。
- 与本方向的关系：直接相关（分级脉冲与差分神经元（Sigma-Delta / 多阈值）在工业界标杆类脑芯片上具备原生硬件指令支持的头号证据）
- 核对人与日期：Gemini，2026-11-16

### Davidson & Furber（2021）

- 标题：Comparison of Artificial and Spiking Neural Networks on Digital Hardware
- 作者：Simon Davidson, Steve B. Furber
- 发表处：Frontiers in Neuroscience，vol. 15, Article 651141, April 2021（DOI: `10.3389/fnins.2021.651141`）
- 链接：https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.651141/full ；代码：未公开
- 核对深度：全文
- 一句话：用 TSMC 22FDX（SpiNNaker 2 同款工艺）版图后实测电路能耗对率编码转换 SNN 与数字 ANN 做逐神经元对比，推导出**转换后的 SNN 只有在每个神经元每次推理平均发放脉冲数少于 `1.72` 个时才能比同结构数字 ANN 省电**。
- 解决的难点：最早用 22nm 真实版图后 SRAM 与乘加器焦耳数刺破了“率编码 ANN→SNN 转换天然省电”的泡沫——指出在 22nm 数字电路里，读一次 32 位本地 SRAM（`296 fJ`）与做一次 32+8×8 乘加（`293 fJ`）能耗几乎相等，SNN 每来一个脉冲都要读/改/写状态 SRAM，多次发放会迅速亏本。
- 留下的局限：分析局限于单神经元率编码脉冲计数（未考虑多位分级脉冲一次传完信息），且假设权重读取在多神经元间完全分摊或对等。
- 是否需要训练或微调：否（基于转换型率编码 SNN 的数字电路能耗理论分析）
- 涉及的算子：率编码 IF 脉冲神经元、8-bit×8-bit+32-bit 整数 MAC、32-bit 加法器、2K-word 本地 SRAM 读写
- 关键数字：基于 **TSMC 22FDX 版图后实测数据**（Section 4, p. 5）：从 2K×32b SRAM 读一个 32 位字耗能 **`296 fJ`**（记为 `5E`），一次 32 位累加 + 8×8 位乘加耗能 **`293 fJ`**（亦约为 `5E`），纯 32 位加法约 **`60 fJ`**（记为 `E`），SRAM 写回约 `E`。据此算得每连接每神经元更新成本：ANN 为 $5E + 5E + E + E = 12E$，SNN 每处理一个脉冲为 $5E + E + E = 7E$（Eqs. 1–2, p. 5）。因此 SNN 比 ANN 省电的临界条件是**每神经元平均发放脉冲数 $N_{\text{meanSpikes}} < 12E / 7E \approx 1.72$**（Section 4, p. 5）！文中实测 MNIST 转换 CNN 首层在全 0 输入背景外很容易超过该阈值，改用信息更紧凑的编码可将脉冲数压低 `75%` 以上。
- 与本方向的关系：直接相关（`1.72` 脉冲/神经元的硬门槛直接解释了为什么传统几百步率编码转换在硬件上必败，而本方向采用的 $T=2,4$ 差分多阈值转换恰好能把平均事件数压进 `< 1.72` 的盈利区间！）
- 核对人与日期：Gemini，2026-11-16

### Degree-Quant（2021）

- 标题：Degree-Quant: Quantization-Aware Training for Graph Neural Networks
- 作者：Shyam A. Tailor, Javier Fernandez-Marques, Nicholas D. Lane
- 发表处：9th International Conference on Learning Representations（ICLR 2021）（预印本 `arXiv:2008.05000v3`）
- 链接：https://arxiv.org/abs/2008.05000 ；代码：https://github.com/camlsys/degree-quant
- 核对深度：全文
- 一句话：图神经网络低比特量化领域的奠基论文，揭示了 GNN 直接做 INT8/INT4 量化崩溃的病根在于**邻域聚合阶段高度数（high in-degree）节点的幅值离群扩张**，并提出按节点入度随机保护全精度前向/梯度的量化感知训练（QAT）方案。
- 解决的难点：精准定位了为什么 CNN 上管用的标准量化搬到 GCN / GIN / GAT 上会严重掉点——图聚合把多达上百个邻居的特征相加，高度数节点的聚合值远大于低度节点，拉宽全局量化步长，导致占绝大多数的低度节点丧失分辨率（Section 3, pp. 3–4）。
- 留下的局限：**必须从头做量化感知训练（QAT），完全不是免训练 PTQ**！文中配套测了朴素训练后量化基线，在 INT4 下几乎彻底失效；且在大图上训练极慢（据 TopGQ 实测在 Reddit 上慢 330 倍，在 2.4 亿节点 MAG240M 上需 37.5 天）。
- 是否需要训练或微调：是（完整量化感知训练 QAT，训练中按节点度数伯努利采样保护掩码并用 0.1% 分位数截断跟踪量化范围，Section 4, pp. 4–6）
- 涉及的算子：`sum` / 加权邻域聚合（GCN、GIN、GAT）、INT8 / INT4 均匀伪量化、百分位激活截断（Percentile Tracking）
- 关键数字：（1）精度表现（Tables 1–2, pp. 7–8）：在 Cora、CiteSeer、ZINC、CIFAR10 超像素图、MNIST 超像素图及 Reddit 上，INT8-DQ 基本追平 FP32（例如 CIFAR10 超像素图 **GAT FP32 `65.4%` → INT8-DQ `66.6%`、INT4-DQ `56.2%`**；**GCN FP32 `56.3%` → INT8-DQ `56.6%`、INT4-DQ `51.8%`**；ZINC 分子图 GIN MAE：FP32 `0.356` → INT8-DQ `0.357`、INT4-DQ `0.431`，而普通 QAT-STE INT4 崩至 `1.388`）；（2）推理加速：CPU 上 INT8 获得最高 **`4.7×`** 加速（Reddit GraphSAGE 为 `4.2×`，Section 5.3, p. 8）。
- 与本方向的关系：直接相关（其发现的“节点入度差异导致聚合输出动态范围失衡”与 ANN→SNN 转换中高度数节点膜电位饱和/低度节点欠发放完全同构！同时说明不能把需要重训的 Degree-Quant 当免训练对照，免训练对照应选均匀 PTQ 与 TopGQ）
- 核对人与日期：Gemini，2026-11-16

### SGQuant（2020）

- 标题：SGQuant: Squeezing the Last Bit on Graph Neural Networks with Specialized Quantization
- 作者：Boyuan Feng, Yuke Wang, Xu Li, Shu Yang, Xueqiao Peng, Yufei Ding
- 发表处：2020 IEEE 32nd International Conference on Tools with Artificial Intelligence（ICTAI 2020），pp. 1044–1052（DOI: `10.1109/ICTAI50040.2020.00161`；预印本 `arXiv:2007.05100v2`）
- 链接：https://arxiv.org/abs/2007.05100 ；代码：未公开
- 核对深度：全文
- 一句话：针对图嵌入显存占用远大于权重矩阵的特点，设计了按层、按拓扑分量（节点特征 vs 注意力系数）和按节点度分组的多粒度 GNN 特征量化框架，并用机器学习回归器自动搜索各组位宽。
- 解决的难点：缓解了大图训练/推理中节点特征矩阵撑爆 GPU 显存的问题，利用高入度节点分配更高比特、低入度节点分配更低比特（1–4 bit）实现高压缩比。
- 留下的局限：（1）**只量化节点嵌入与注意力系数，权重矩阵 $W$ 全程保留 FP32，且在矩阵相乘前先反量化回 FP32**（Section III-A, Eq. 5, p. 3），因此**只能省显存/带宽，不能省浮点乘加算术能耗**，不适合作为整数算术能耗基线；（2）**并非纯免训练**：原文明确指出直接训练后量化会掉点高达 **`10%`**，必须依赖基于直通估计器（STE）的微调才能把掉点拉回 `0.5%` 以内（Section III-B, p. 3），且自动位宽选择需采样跑 200 组量化评估训练回归模型。
- 是否需要训练或微调：是（需 STE 微调恢复直接量化最高 10% 的精度损失，且自动位宽搜索需 200 次探针评估，Section III-B & III-C, pp. 3–4）
- 涉及的算子：GCN / GAT / AGNN 邻域聚合与注意力得分、组件级/拓扑度感知均匀量化、FP32 反量化重匹配（Rematching）
- 关键数字：在 Cora、CiteSeer、PubMed、Reddit、Amazon-Computers、Amazon-Photo 六个数据集上（Tables I–III, pp. 5–7），将特征表示压缩至平均约 `1.6–4 bits`，实现 **`4.25×–31.9×`** 特征比特压缩比和 **`1.11×–2.53×`** GPU 时间加速（主要来自访存带宽减少），微调后精度损失控制在 **`< 0.5%`** 以内（Reddit GCN FP32 `93.9%` vs SGQuant `93.7%`，Table II, p. 6）。
- 与本方向的关系：背景（厘清了 GNN 量化早期代表作的真实边界：SGQuant 保留 FP32 权重且需微调，不能误当作免训练全整数 W4A4/W8A8 对照基线）
- 核对人与日期：Gemini，2026-11-16

### Tianjic（2019）

- 标题：Towards artificial general intelligence with hybrid Tianjic chip architecture
- 作者：Jing Pei, Lei Deng, Sen Song, Mingguo Zhao, Youhui Zhang, Shuang Wu, Guanrui Wang, Zhe Zou, Zhenzhi Wu, Wei He 等（Luping Shi 团队）
- 发表处：Nature，vol. 572, no. 7767, pp. 106–111, August 2019（DOI: `10.1038/s41586-019-1424-8`；配套电路实现详述于 Deng 等，IEEE JSSC 2020, vol. 55, no. 8, pp. 2228–2246）
- 链接：https://doi.org/10.1038/s41586-019-1424-8 ；代码：未公开
- 核对深度：仅摘要（结合检索摘录与已通读的 SENECA 全文 Table 1, 3, 4, 5 对 Tianjic 参数的逐项交叉核实）
- 一句话：清华大学研制的 28nm 异构融合类脑芯片“天机”，通过可独立重构的轴突与胞体模块，在同一套片上路由网络上同时支持 1-bit 脉冲网络（SNN）与 8-bit 连续值深度网络（ANN）及二者混合模型。
- 解决的难点：打通了脉冲范式与传统 8-bit 定点神经网络范式的硬件隔阂——复用树突与突触乘加阵列，仅在输入轴突和输出胞体切换脉冲序列解析或 8-bit 激活截断打包，从而支持无人自行车上的多模态混合网络协同推理。
- 留下的局限：需注意**概念区分**——据 SENECA（Table 3, p. 6）核实，天机的单一模式数据类型为固定精度（`Fixed`：SNN 模式传 1-bit 二值脉冲，ANN 模式传 8-bit 整数激活，或由胞体把膜电位截断为 8-bit 混合信号转发），不同于 Loihi 2 / SENECA 在纯事件驱动脉冲指令流中原生支持任意可编程分级脉冲载荷。
- 是否需要训练或微调：否（双模推理芯片平台）
- 涉及的算子：LIF 脉冲神经元（1-bit 脉冲）、ANN 8-bit 整数乘加与 ReLU/非线性查表、异构混合信号转换（ANN↔SNN 胞体直转）
- 关键数字：UMC 28nm 工艺，156 个功能核（FCores），单核面积 **`0.092 mm²`**、每核 `0.17 Mb` SRAM，等效突触能效约 **`1.54 pJ/SOp`**（核实自 SENECA 全文 Table 1 & Table 3, pp. 5–6，引用 Deng et al., IEEE JSSC 2020）。
- 与本方向的关系：可借用的技术（如果视觉图模型中“动态 $k$NN 建图模块”保留低比特整数计算、而“图消息传递主干”转为事件驱动脉冲推理，天机与 C-DNN 这类 ANN-SNN 异构融合芯片正是现成的硬件载体！）
- 核对人与日期：Gemini，2026-11-16

### Horowitz（2014）

- 标题：1.1 Computing's Energy Problem (and what we can do about it)
- 作者：Mark Horowitz
- 发表处：2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers（ISSCC 2014），pp. 10–14（DOI: `10.1109/ISSCC.2014.6757323`）
- 链接：https://doi.org/10.1109/ISSCC.2014.6757323（全文 PDF 镜像：https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf） ；代码：无（硬件综述与大会主题报告论文）
- 核对深度：全文
- 一句话：现代计算机体系结构与神经网络硬件能耗估算的奠基参照文献，给出了 45nm 0.9V CMOS 工艺下整数/浮点加乘运算、片上 SRAM 缓存、片外 DRAM 访问及处理器指令管线的精确数量级能耗表。
- 解决的难点：系统阐明了 Dennard 缩放终结后的“暗硅（Dark Silicon）”与能耗墙根源——指出算术单元本身极省电，真正的能耗大头是指令控制开销（高出加法几百倍）和远距离存储器数据搬运（高出加法成百上千倍）。
- 留下的局限：数据测算于 45nm 0.9V 平面 CMOS 时代（距今十余年，在 22nm/7nm/5nm 下算术与连线电容缩放比例已发生变化，SRAM 读写与 NoC 搬运占比相对算术进一步攀升）；且常被脉冲网络论文断章取义地只抽取 `FP32 Add (0.9 pJ)` 与 `FP32 Mult (3.7 pJ)` 而无视同表中的 `INT8` 与 `Cache/DRAM` 行。
- 是否需要训练或微调：否（集成电路基础能耗基准文献）
- 涉及的算子：8-bit/32-bit 整数加乘、16-bit/32-bit 浮点加乘、8KB/32KB/1MB SRAM 缓存访问、DRAM 访问
- 关键数字：45nm CMOS, 0.9V（完整数字见本文开头专表，出自 p. 14, Figure 1.1.9 及 p. 11, Section 5–6）：`8b Int Add = 0.03 pJ`、`32b Int Add = 0.1 pJ`、`16b FP Add = 0.4 pJ`、`32b FP Add = 0.9 pJ`、`8b Int Mult = 0.2 pJ`、`32b Int Mult = 3.1 pJ`、`16b FP Mult = 1.1 pJ`、`32b FP Mult = 3.7 pJ`；64-bit 读取 `8KB / 32KB / 1MB Cache = 10 / 20 / 100 pJ`，`DRAM = 1.3–2.6 nJ`；简单顺序处理器控制管线 **`70 pJ/instruction`**。
- 与本方向的关系：直接相关（所有能耗讨论的原点；必须在我们的论文里完整引用其访存与低比特整数行，彻底告别不严谨的纯 FP32 `4.6 vs 0.9 pJ` 口径）
- 核对人与日期：Gemini，2026-11-16

---

## 这条线是怎么演化的

把过去十二年（2014–2026）这五个子方向放在一起看，脉络非常清晰，经历了**三个阶段的认知反转与硬件收敛**：

1. **第一阶段（2014–2020）：天真算术估算期与 QAT 图量化起步**。早期脉冲网络工作借用 Horowitz（2014）表中的两行浮点数字，形成“`FP32 MAC (4.6 pJ)` 换 `FP32 AC (0.9 pJ)` 乘稀疏度”的行业惯例，忽视了同表中的 INT8（`0.23 pJ`）和百倍于算术的 SRAM 访存（`10–100 pJ`）。与此同时，图神经网络量化刚刚起步（SGQuant 2020、Degree-Quant 2021），研究者发现图节点入度的幂律长尾会导致聚合值离群，当时只能靠昂贵的量化感知训练（QAT）或微调保住精度。
2. **第二阶段（2021–2024）：硬件感知祛魅期与分级脉冲芯片成熟**。以 Davidson & Furber（2021）、Lemaire 等（2022）、Dampfhoffer 等（2023）、Ottati 等（2023）和 Bhattacharjee 等（2024）为代表的一批电路与体系结构工作集体发难，用 22nm/28nm 版图与流片数据证明：**在静态单帧输入上，多时间步二值 SNN 因反复读写膜电位 SRAM 和片上路由开销，在能效与精度上被 INT8/INT4 数字加速器全面压制**（平均每神经元脉冲必须 `< 1.72`、每突触脉冲 `< 0.15–1.38` 才有胜算）。几乎在同一时期，类脑硬件自身也放弃了“唯 1-bit 二值脉冲论”——Intel Loihi 2（2021，最高 32-bit）、Imec SENECA（2023，4/8/16-bit）和 SpiNNaker 2（2024，最高 128-bit 载荷）相继原生支持**分级脉冲（Graded Spikes）**，SENECA 更用实测证明 16-bit 分级脉冲比二值脉冲仅多耗电 `6.1%`，真正省电靠的是减少脉冲事件次数与复用局部状态（省电 `47%`）。
3. **第三阶段（2025–2026）：双Twin 严格盈亏理论、秒级免训练 GNN PTQ 与事件图芯片流片交汇**。到 2025–2026 年，三条线同时走到新高度：
   - 在**能耗理论**上，Yan 等（TCAD 2026）建立了 $T$ 步 SNN 与 $\lceil\log_2(T+1)\rceil$ 比特 QNN Twin 的严格解析与轨迹对决（$T=5$ 时发放率需 `< 5.7%`），但**图神经网络上的对应分析仍是空白**；
   - 在**非脉冲图基线**上，TopGQ（DAC 2026）首次在固定图上实现了秒级免训练 W8A8/W4A4 PTQ，EvGNN（TCASAI 2025）也证实 PointNetConv + `max` 聚合做 INT8 PTQ 仅掉点 `0.1%`，这意味着我们的脉冲图转换不仅要和朴素均匀量化比，还要直面强 PTQ 基线；
   - 在**专用芯片**上，EvGNN（2025）、AMPLE（2025）、EFGCN（2026）、EDA-GNN（2026）乃至首颗 28nm 流片 ASIC **ETHEREAL（2026 arXiv 预印本，会议归属未核实；`25.6 µs/ev`，`1.6 µJ/ev` 或短文版的 `1.7 µJ/ev`）**密集问世，证明边缘端对事件驱动视觉图推理有极迫切的硅片级需求，但它们清一色跑的是**需 QAT 训练的 4b/8b 量化 GNN**，尚未结合免训练差分/分级脉冲机制来消除帧间与层间细粒度冗余。

**对本项目的三条直接启示**：
1. **静态图不硬吹省电，流式图主攻帧间差分**：完全印证实验协议中的假设 H3——静态图像图（ViG / DGCNN 单帧）上很难赢过 INT4/INT8 PTQ，能耗叙事应诚实做成**图结构上的 QNN-SNN 盈亏边界刻画**（显式纳入节点度分布、不规则邻接表访存与动态建图成本），把真正的绝对省电主战场放在**骨架序列、点云流与事件相机图的时间差分事件化**上。
2. **大胆采用多阈值/分级差分脉冲 + 局部混合算子**：Loihi 2、SENECA、SpiNNaker 2 乃至天机/C-DNN 混合架构都证明，“多比特分级事件 + 硬件比较器 `MAX`”在现代类脑芯片上完全合法且算术代价极低（`<7%`），不必为了凑纯 1-bit 二值脉冲而把 $T$ 拉大到几十步自毁能效。
3. **对照基线必须升级**：做非脉冲对照时，除协议草案里的“同比特均匀量化”外，固定图任务应补充对比 **TopGQ（DAC 2026）** 这一同属免训练赛道的最新 GNN PTQ 基线。
