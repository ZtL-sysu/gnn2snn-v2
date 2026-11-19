# 脉络三：视觉图骨干（教师模型选型，2017–2025）

本文件整理“视觉图神经网络免训练转换为脉冲网络”方向可供选作**预训练浮点教师模型（Teacher Backbones）**的四条主流视觉图谱系：（1）**2D 图像视觉图（ViG 家族）**；（2）**3D 点云图与集合网络（PointNet++ / DGCNN / PointNeXt / Point Transformer 家族）**；（3）**人体骨架时空图（ST-GCN → BlockGCN 家族）**；以及（4）**事件相机异步图与超像素图基准（RG-CNNs / AEGNN / DAGr / Benchmarking GNNs）**。

所有条目均已核对论文全文（PDF / Nature Open Access）并交叉核对官方开源仓库源码、预训练权重可用性与许可证条款；论文表述与官方源码实现不一致之处均已在表中和正文显式标出。

## 教师模型选型总览表（必填八列）

| 模型 | 任务与数据集 | 公开精度 | 建图方式（固定 / 每层按特征重算，k 是多少） | 聚合算子 | 激活与归一化 | 官方预训练权重链接 | 代码许可证 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **DVHGNN**（CVPR 2025） | 2D 分类 `ImageNet-1K`；检测 `COCO`；分割 `ADE20K` | `ImageNet` T/S/M/B：**79.8% / 83.1% / 83.8% / 84.2%**（Table 1）；`COCO` Mask R-CNN `DVHGNN-S`：**44.8** $\text{AP}^b$ / **40.2** $\text{AP}^m$（Table 2）；`ADE20K` UperNet：**46.8%** mIoU（Table 3） | **每层按特征重算超边**：先按余弦相似度聚为 $C=4$ 簇，再用多尺度空洞率 $r\in\{1,2,3\}$ 采样簇中心动态构建空洞超图（Section 3.2） | 超边均值聚合 + `Sigmoid` 门控逐元素调制 + 深度可分离卷积局部融合（`DHConv`） | `GELU` + `BatchNorm2d` | 未公开（官方仓库目前仅占位 `Coming soon...`） | `Apache-2.0` |
| **GreedyViG**（CVPR 2024） | 2D 分类 `ImageNet-1K`；检测 `COCO`；分割 `ADE20K` | `ImageNet` S/M/B：**81.1% / 82.9% / 83.9%**（Table 1）；`COCO` Mask R-CNN S/B：**43.2 / 46.3** $\text{AP}^{box}$（Table 2）；`ADE20K` S/B：**43.2% / 47.4%** mIoU（Table 2） | **四阶段半动态轴向掩码图（`DAGC`）**：沿行列固定步幅移位（四阶段步幅 $K=[8,4,2,1]$），按对角翻转特征距离估计均值与标准差，动态掩码保留 `dist < μ - σ` 的轴向邻居（**冲突**：论文写 $\ell_2$ 距离，源码 `greedyvig.py:122,130` 实为 $\ell_1$ 范数 `p=1`） | 轴向逐元素相对差最大值 $\max(0, m_{ij}(x_j-x_i))$（`MRConv` 变体）+ `CPE` $7\times 7$ 深度卷积 | `GELU` + `BatchNorm2d` | https://huggingface.co/SLDGroup/GreedyViG/tree/main | **UT Austin Research License（UTRL，仅限非商业学术研究，禁止再分发）** |
| **MobileViGv2**（CVPRW 2024） | 2D 分类 `ImageNet-1K`；检测 `COCO`；分割 `ADE20K` | `ImageNet` Ti/S/M/B：**77.7% / 79.8% / 81.7% / 83.4%**（Table 1）；`COCO` Mask R-CNN M/B：**42.5 / 43.0** $\text{AP}^{box}$；`ADE20K`：**42.9% / 44.3%** mIoU（Table 2） | **完全固定稀疏轴向十字图（`MGC`）**：第 2–4 阶段每个像素仅连自身与上下左右各偏移 $K=[8,4,2]$ 的 **4 个固定轴向邻居**（共 5 连接，零距离计算、零排序） | 4 个固定轴向邻居相对差 $x_j - x_i$ 逐通道拼接或分组卷积线性组合（**无 `max` 聚合**，源码用深度 $1\times 1$ 卷积混合拼接差值）+ `RepCPE` | `GELU` + `BatchNorm2d` | https://huggingface.co/SLDGroup/MobileViGv2/tree/main | `Apache-2.0` |
| **DAGr**（Nature 2024） | 事件相机目标检测：`Gen1`、`N-Caltech101`、`DSEC-Detection`（事件+RGB） | `Gen1` 纯事件 `DAGr-L`：**32.1** mAP（Table 2）；`N-Caltech101` `DAGr-S/L`：**70.2 / 73.2** mAP；`DSEC-Detection` `DAGr-S+ResNet-50`：**41.9** mAP（Table 1） | **异步流式固定时空有向半径图**：新事件到达时按像素与归一化时间连接 $\|n_p^i-n_p^j\|_\infty < R=0.01, t_i<t_j$ 的历史事件，每节点最多 **16** 个邻居（Methods） | 查表样条卷积 `LUT-SplineConv`（$d=1, k=5$，邻居**求和 `add`**）+ 前 3 级时空体素簇 **`max` 池化**（第 4 级 `mean` 池化，`dagr__net.py:78-98`） | 主干 `ELU` + `BatchNorm1d`；YOLOX 检测头 `SiLU` + `BatchNorm2d` | https://download.ifi.uzh.ch/rpg/dagr/data/dagr_s_50.pth | `GPL-3.0` |
| **Point Transformer V3（PTv3）**（CVPR 2024） | 3D 室内/室外点云语义分割、实例分割、检测：`ScanNet v2`、`ScanNet200`、`S3DIS`、`nuScenes`、`SemanticKITTI`、`Waymo` | 单数据集从头训 / 多数据集 PPT 联训：`ScanNet v2` Test **77.9% / 79.4%** mIoU；`S3DIS` Area5 **73.4% / 74.7%**、6-fold **77.7% / 80.8%** mIoU（Table 5）；`nuScenes` Test **82.7% / 83.0%** mIoU（Table 7） | **完全取消 $k$-NN，改用空间填充曲线序列化分块**：按 Z-order / Trans Z-order / Hilbert / Trans Hilbert 四种曲线排序点云并轮换打乱，每块固定切分 **1024** 个连续点（patch size = 1024，Table 15） | 块内多头**缩放点积自注意力（`Softmax`）** + 前置 3D 稀疏卷积 `xCPE`；下采样 `Grid Pooling` 仍用**体素内 `MaxPool`** | 主干 Pre-Norm `LayerNorm` + `GELU`；`Grid Pooling` 保留 `BatchNorm` + `ReLU`（Table 17） | https://huggingface.co/Pointcept/PointTransformerV3 | 代码：`CC BY-NC 4.0`（附学术许可）；`ptv3_scannet` 权重：`MIT` |
| **BlockGCN**（CVPR 2024） | 3D 骨架动作识别：`NTU-RGB+D 60`、`NTU-RGB+D 120`、`NW-UCLA` | 单流 Joint：`NTU-60` X-Sub/X-View **90.9% / 95.4%**，`NTU-120` X-Sub/X-Set **86.9% / 88.2%**；4 流集成：`NTU-60` **93.1% / 97.0%**，`NTU-120` **90.3% / 91.5%**，`NW-UCLA` **96.9%**（Table 2） | **静态最短路径拓扑查表 + 样本级持续同调动态编码**：可学习分组邻接 $A_k$（$K=8$ 组）叠加骨架最短路径跳数嵌入 $B_{ij}$，并按样本关节欧氏距离构造 Vietoris-Rips 滤子提取 0 维持续同调条形码映射为动态偏置 $C^{(l)}$（Section 3.2） | 分组块对角矩阵加权求和（`BlockGC`）+ 多尺度时序卷积 `MS-TC`（含时序 **`3×1 MaxPool`** 分支） | `ReLU` + `BatchNorm2d` | 未公开（官方仓库仅含训练代码） | `Apache-2.0` |
| **ViHGNN / Pyramid ViHGNN**（ICCV 2023） | 2D 分类 `ImageNet-1K`；检测 `COCO` | 各向同性 Ti/S/B：**74.3% / 81.5% / 82.9%**（Table 4）；金字塔 `PViHGNN` Ti/S/M/B：**78.9% / 82.5% / 83.4% / 83.9%**（Table 5）；`COCO` Mask R-CNN S：**43.1** $\text{AP}^b$（Table 11） | **每层按特征模糊 C 均值（FCM）动态聚类构建超边**（**冲突**：论文 Table 1–2 写超边数 $E=50$，官方源码 `pyramid_vihg.py:215` 设为 `k=25` 个簇） | 两阶段“节点→超边→节点”聚合（论文 Eq. 1 写度归一化加权求和，源码 `torch_vertex.py:88-133` 实为无权均值 `mean` + GIN 式 $(1+\epsilon)x$ 残差） | `GELU` + `BatchNorm2d` | 未公开（官方仓库仅提供训练与评测脚本） | `MIT` |
| **HD-GCN**（ICCV 2023） | 3D 骨架动作识别：`NTU-RGB+D 60`、`NTU-RGB+D 120`、`Kinetics-Skeleton`、`NW-UCLA` | 单流 Joint `NTU-120`：**85.7% / 87.3%**（Table 4）；6 路集成（无运动流）：`NTU-60` **93.4% / 97.2%**，`NTU-120` **90.1% / 91.6%**，`Kinetics` **40.9%**，`NW-UCLA` **97.2%**（Table 1） | **层次分解有根树图 + 每层按特征重算动态欧氏 $k$-NN**：3 个层次分解固定/可学习子集分支 + 1 个时序均值后按特征欧氏距离动态选邻居的 `S-EdgeConv` 分支 + 层次间 `H-EdgeConv`（Section 3.2–3.3） | 子集加权求和 + `EdgeConv` 边聚合 + `A-HA`（含时序 **`MaxPool_T`**、代表节点空间均值与 **`Sigmoid`** 层次门控）+ `MS-TC`（含 **`3×1 MaxPool`**） | `ReLU`、`Sigmoid` + `BatchNorm2d` | https://drive.google.com/drive/folders/1FB_IQdTMWE8cRvwE2KiyxC0P6LyqZku4?usp=sharing | `MIT` |
| **MobileViG**（CVPRW 2023） | 2D 分类 `ImageNet-1K`；检测/分割 `COCO` | `ImageNet` Ti/S/M/B：**75.7% / 78.2% / 80.6% / 82.6%**（Table 2）；`COCO` Mask R-CNN M/B：**41.3 / 42.0** $\text{AP}^b$、**38.1 / 38.9** $\text{AP}^m$（Table 4） | **完全固定稀疏轴向网格图（`SVGA`）**：前 3 阶段纯 `MBConv` 无图；仅第 4 阶段（$7\times 7$）将每个像素固定连同行、同列步幅 $K=2$ 的像素（移位 $\{2,4,6\}$ 共 **6 个轴向邻居 + 自身**，零距离、零 `topk`） | **逐元素取最大值 `max`**：`MRConv4d` 以自差 `0` 初始化，对所有轴向移位迭代取 $\max(x_j, x - \text{roll}(x, mK))$ 后拼接做 $1\times 1$ 卷积（Algorithm 1，`mobilevig.py:115-143`） | `GELU` + `BatchNorm2d` | https://huggingface.co/SLDGroup/MobileViG/tree/main | `Apache-2.0` |
| **ViG（Isotropic）**（NeurIPS 2022） | 2D 分类 `ImageNet-1K` | `ImageNet` `ViG-Ti/S/B`：**73.9% / 80.4% / 82.3%**（Table 4） | **每层按 $\ell_2$ 归一化特征重算空洞 $k$-NN**：$k$ 随层深从 **9 线性增至 18**，空洞率 $d=\min(\lfloor i/4\rfloor+1, 10)$，取前 $k\cdot d$ 近邻每隔 $d$ 取 1（`vig.py:106-116`） | **`MRConv2d` 相对差最大值** $m_i=\max_{j\in\mathcal{N}(i)}(\tilde{x}_j-\tilde{x}_i)$，交错拼接 $[\tilde{x}_i\|m_i]$ 接 4 组分组 $1\times 1$ 卷积（`torch_vertex.py:13-30`） | `GELU`（消融含 `ReLU`/`LeakyReLU`）+ `BatchNorm2d` | https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/vig | `Apache-2.0` |
| **Pyramid ViG**（NeurIPS 2022） | 2D 分类 `ImageNet-1K`；检测/分割 `COCO` | `ImageNet` Ti/S/M/B：论文 Table 5 报 **78.2% / 82.1% / 83.1% / 83.7%**（官方仓库 `readme.md` 发布权重报 `Ti` **78.5%**）；`COCO` Mask R-CNN S：**42.6** $\text{AP}^b$ / **39.4** $\text{AP}^m$（Table 10） | **每层按 $\ell_2$ 归一化特征 + 二维相对位置偏置重算空洞 $k$-NN**：全网恒定 **$k=9$**，前两阶段对源节点先做 $r\in[4,2,1,1]$ 空间 `AvgPool` 降采样，空洞率上限 $d_{\max}=5$（`pyramid_vig.py:121-138`） | 同上 **`MRConv2d` 相对差 `max` 聚合** + 4 组分组 $1\times 1$ 卷积 | `GELU` + `BatchNorm2d` | https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/pyramid-vig | `Apache-2.0` |
| **PointNeXt**（NeurIPS 2022） | 3D 点云分类 `ScanObjectNN`、`ModelNet40`；分割 `S3DIS`、`ScanNet V2`、`ShapeNetPart` | `ScanObjectNN` `PointNeXt-S`：**87.7±0.4%** OA；`ModelNet40`：**93.2%**（C=32）/ **94.0%**（C=64，Table 2）；`S3DIS` 6-fold `XL`：**74.9%** mIoU、`Area 5` **70.5%**（Table 1）；`ShapeNetPart`：**87.0%** ins. mIoU（Table 3） | **固定三维坐标空间球查询（Ball Query）**：FPS 降采样后按三维物理坐标查询半径 $r^l$ 内最多 **$k=32$** 个邻居，并对相对坐标除以半径归一化 $(\mathbf{p}_j^l-\mathbf{p}_i^l)/r^l$（Eq. 2） | 邻域单层 MLP 后接**通道逐元素 `MaxPool` 归约**，再接两层 $4\times$ 倒残差逐点分离 MLP（`InvResMLP`） | `ReLU` + `BatchNorm` | https://guochengqian.github.io/PointNeXt/modelzoo/ （托管于 Hugging Face Hub） | `MIT` |
| **InfoGCN**（CVPR 2022） | 3D 骨架动作识别：`NTU-RGB+D 60`、`NTU-RGB+D 120`、`NW-UCLA` | 6 模态集成：`NTU-60` **93.0% / 97.1%**，`NTU-120` **89.8% / 91.2%**（单流 Joint **85.1% / 86.3%**），`NW-UCLA` **97.0%**（Table 1） | **每层每帧按关节特征重算多头自注意力稠密图（`SA-GC`）**：$M=3$ 头缩放点积 **`Softmax`** 注意力矩阵与可学习共享拓扑 $\tilde{\mathbf{A}}_m$ 逐元素相乘（Eq. 13–15，$N\times N$ 全连接软边） | 多头自注意力加权求和 + `MS-TC`（含 **`3×1 MaxPool`** 分支） | `ReLU`、`Softmax` + 块尾 `LayerNorm`（内部卷积含 `BatchNorm`） | 未公开（官方仓库未附预训练权重） | 无许可证（保留全部权利） |
| **AEGNN**（CVPR 2022） | 事件相机分类 `N-Cars`、`N-Caltech101`；检测 `N-Caltech101`、`Gen1` | 分类：`N-Cars` **94.5%**（`0.47` MFLOP/ev）、`N-Caltech101` **66.8%**（`7.31` MFLOP/ev，Table 1）；检测：`N-Caltech101` **59.5%** mAP、`Gen1` **16.3%** mAP（`5.26` MFLOP/ev，Table 3） | **增量流式固定时空欧氏半径图**：均匀 1/10 抽样后在 $(x_i,y_i,\beta t_i)$ 空间按半径 $R$ 连边，最大度数 **$D_{\max}=32$**，新事件仅触发 $k$-hop 局部子图更新（Section 4.1–4.2） | 7 层三次 B 样条卷积 `SplineConv`（度归一化**均值/求和**）+ 第 5、7 层后 $12\times 16\times 16$ 体素簇 **`MaxPool`**（Section 4.3 & 8.1） | `ELU` + `BatchNorm1d` | 未公开（官方仓库仅含代码与异步转换器） | `MIT` |
| **Benchmarking GNNs（超像素图基准）**（JMLR 2022） | 2D 超像素图分类：`MNIST`（70k 图，40–75 节点）、`CIFAR10`（60k 图，85–150 节点）及 10 个通用图基准 | $L=4,\approx 100\text{K}$ 参数（Table 7）：`GatedGCN` **97.34% / 67.31%**；`GraphSAGE-Max` **97.31% / 65.77%**；`GAT` **95.54% / 64.22%**；`GIN` **96.49% / 55.26%**；`GCN` **90.12% / 54.14%** | **固定二维超像素坐标 $k$-NN 图**：SLIC 提取超像素后按 2D 坐标构建固定 **$k=8$** 近邻图，边权为高斯核 $\exp(-\|x_i-x_j\|^2/\sigma_x^2)$（Eq. 47） | 涵盖五类标准聚合：`Sigmoid` 软边门控加权和（`GatedGCN`）、邻域 **`Max`**（`GraphSAGE`）、`Softmax` 注意力（`GAT`）、求和（`GIN`）、对称度归一化均值（`GCN`） | `ReLU`（`GAT` 含 `LeakyReLU`/`ELU`）+ `BatchNorm` | 未单独发布 `.pkl` 权重（提供 4 随机种子一键复现配置与标准划分） | `MIT` |
| **Point Transformer（PTv1）**（ICCV 2021） | 3D 点云分割 `S3DIS`、`ShapeNetPart`；分类 `ModelNet40` | `S3DIS` Area5 **70.4%** mIoU、6-fold **73.5%** mIoU（Tables 1–2）；`ModelNet40` **93.7%** OA（Table 3）；`ShapeNetPart` **86.6%** ins. mIoU（Table 4） | **各阶段固定三维坐标 $k$-NN 图**：5 个降采样阶段内对当前点集三维坐标用堆排序构建固定 **$k=16$** 近邻图（Section 3.2 & Table 5） | **向量减法自注意力** $\sum_{j\in\mathcal{X}(i)}\text{softmax}(\gamma(\varphi(x_i)-\psi(x_j)+\delta))\odot(\alpha(x_j)+\delta)$；阶段间 `Transition Down` 使用 $k=16$ **局部 `MaxPool`**（Section 3.5） | `ReLU`、`Softmax` + `BatchNorm` | 未公开（作者复现实验仓与 `Pointcept` 均未附 PTv1 独立权重） | 无许可证（`POSTECH-CVLab` 仓未设 LICENSE；`Pointcept` 集成版为 `MIT`） |
| **CTR-GCN**（ICCV 2021） | 3D 骨架动作识别：`NTU-RGB+D 60`、`NTU-RGB+D 120`、`NW-UCLA` | 4 流集成：`NTU-60` X-Sub/X-View **92.4% / 96.8%**（Table 6）；`NTU-120` X-Sub/X-Set **88.9% / 90.6%**（单流 Joint **84.9%**，Tables 4–5）；`NW-UCLA` **96.5%**（Table 7） | **每层按样本特征动态精炼逐通道稠密图（`CTR-GC`）**：时序池化后算关节对特征差 $\tanh(\psi(x_i)-\phi(x_j))$ 升维生成样本级逐通道修正项 $\mathbf{Q}\in\mathbb{R}^{N\times N\times C'}$，叠加可学习共享拓扑 $\mathbf{A}$（$N\times N$ 全连接） | 逐通道精细化拓扑矩阵加权求和 + 四分支多尺度时序模块（含 **`3×1 MaxPool`** 分支，Figure 3a） | 主干 `ReLU` + `BatchNorm2d`；拓扑关联分支用 **`Tanh`**（Table 3） | https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing | 官方仓：`CC BY-NC 4.0` |
| **DGCNN**（ACM TOG 2019） | 3D 点云分类 `ModelNet40`；部件分割 `ShapeNetPart`；室内分割 `S3DIS` | `ModelNet40`：**92.9%** OA（1024 点，$k=20$）/ **93.5%** OA（2048 点，$k=40$，Table 2）；`ShapeNetPart`：**85.2%** mIoU（Table 6）；`S3DIS` 6-fold：**56.1%** mIoU / **84.1%** OA（Table 7） | **每层按特征空间欧氏距离重算动态 $k$-NN**：1024 点分类与分割每层取 **$k=20$**，2048 点分类取 **$k=40$**（分类共 4 层 `EdgeConv`、分割 3 层，Section 4.1 & 5.1） | **不对称边函数 + 邻域逐通道 `max` 聚合**：$x_i' = \max_{j:(i,j)\in\mathcal{E}} h_\Theta(x_i, x_j-x_i)$（Eq. 4，`dgcnn__pytorch_model.py:88-120`） | **`LeakyReLU(neg_slope=0.2)`** + `BatchNorm` | https://github.com/AnTao97/dgcnn.pytorch （官方推荐 PyTorch 仓内附 `pretrained/*.t7` 与 `.pth`） | `MIT` |
| **2s-AGCN**（CVPR 2019） | 3D 骨架动作识别：`NTU-RGB+D 60`、`Kinetics-Skeleton` | `NTU-60` X-Sub/X-View：**88.5% / 95.1%**（单流 Joint X-View **93.7%**，Tables 2–3）；`Kinetics-Skeleton` Top-1/Top-5：**36.1% / 58.7%**（Table 4） | **固定物理图 + 全局可学习图 + 每层按样本特征重算稠密图**：$\mathbf{A}_k + \mathbf{B}_k + \mathbf{C}_k$，其中 $\mathbf{C}_k=\text{softmax}(f_{in}^\top W_\theta^\top W_\phi f_{in})$ 为全关节 $N\times N$ 嵌入高斯相似度图（$K_v=3$ 子集，Eq. 3–5） | 三重邻接矩阵加权求和 + $K_t\times 1$ 时序卷积（关节流与骨骼向量流双流 Softmax 分数相加） | `ReLU`、`Softmax` + `BatchNorm2d` | 未公开（官方仓库未提供预训练 `.pt` 下载） | 官方仓：`CC BY-NC 4.0` |
| **RG-CNNs**（ICCV 2019） | 事件相机目标分类：`N-MNIST`、`MNIST-DVS`、`N-Caltech101`、`CIFAR10-DVS`、`N-CARS`、`ASL-DVS` | Top-1（Table 1）：`N-MNIST` **99.0%**、`MNIST-DVS` **98.6%**、`N-Caltech101` **65.7%**（`0.79` GFLOPs）、`CIFAR10-DVS` **54.0%**、`N-CARS` **91.4%**、`ASL-DVS` **90.1%** | **固定加权时空欧氏半径图**：非均匀网格采样（每格至多 8 事件）后按加权时空距离 $d_{i,j}\le R=3$ 连边，最大度数 **$D_{\max}=32$**（Section 3.1 & 5.1） | 一阶 B 样条空间图卷积 `SplineCNN`（$m=1, k=5$，度归一化**均值**）+ 每块后二维网格簇 **`MaxPool`（`MaxP`）**（Section 3.3 & 3.5） | `ReLU` + `BatchNorm` | 未公开（仓库仅公开 `ASL-DVS` 数据集与转换脚本） | 无许可证（`PIX2NVS/NVS2Graph` 未设 LICENSE） |
| **ST-GCN**（AAAI 2018） | 3D 骨架动作识别：`NTU-RGB+D 60`、`Kinetics-Skeleton` | `NTU-60` X-Sub/X-View：**81.5% / 88.3%**（Table 3）；`Kinetics-Skeleton` Top-1/Top-5：**30.7% / 52.8%**（Table 2） | **完全固定人体自然骨架时空图**：空间仅连 $D=1$ 跳物理骨骼邻居并按重心距离划分为 $K=3$ 个固定子集（自身、向心、离心），时间连相邻帧同一关节（$\Gamma=9$），配可学习边掩码 $\mathbf{M}$（Section 3.2–3.6） | 对称度归一化矩阵加权求和 $\sum_{j=0}^2 \Lambda_j^{-1/2}(A_j\otimes M)\Lambda_j^{-1/2} f_{in} W_j$ + $1\times 9$ 时序卷积（**全网无 `MaxPool`、无动态算子**） | `ReLU` + `BatchNorm2d` | https://github.com/yysijie/st-gcn/blob/master/OLD_README.md （附 Google Drive / 百度网盘预训练模型链接） | `BSD-2-Clause` |
| **PointNet++**（NeurIPS 2017） | 3D 点云分类 `ModelNet40`、`SHREC15`；分割 `ShapeNetPart`、`ScanNet` | `ModelNet40`：**90.7%**（1024 点纯坐标）/ **91.9%**（5000 点含法向，Table 2）；`ShapeNetPart`：**85.1%** mIoU（Supp. Table 4）；`ScanNet` 体素精度：**84.5%**（Figure 5） | **固定三维坐标空间多尺度球查询（SSG/MSG/MRG）**：FPS 选质心后按三维坐标半径 $r$ 查球邻域（分类半径 $0.1\sim 0.8$；消融 Table 5 亦测 $k=16,64$ $k$-NN）；分割上采样用 $k=3$ 反距离加权插值 | 共享局部 PointNet MLP 后接**对称邻域逐通道 `MAX` 池化**（Eq. 1 & Section 3.2） | `ReLU` + `BatchNorm` | 官方 TF 仓未附权重；高星社区 PyTorch 复现仓 https://github.com/yanx27/Pointnet_Pointnet2_pytorch 内置 `log/` 预训练权重 | 官方 TF 仓与 `yanx27` PyTorch 仓均为 `MIT` |
| **ResGatedGCN**（arXiv 2017） | 变长图节点匹配与半监督聚类（SBM）；后续成为 `MNIST/CIFAR10` 超像素图最强基线 | 10 簇 SBM 半监督聚类：**82%**（较非学习变分法高 36%，较无残差深层提升 10%，Section 4.2 & Abstract） | **输入给定静态图拓扑**（任意变长有向/无向图，每层保持固定边集 $j\to i$） | **各向异性 `Sigmoid` 软边门控加权求和**：$h_i^{\ell+1}=h_i^\ell+\text{ReLU}(\text{BN}(U^\ell h_i^\ell+\sum_{j\to i}\sigma(A^\ell h_i^\ell+B^\ell h_j^\ell)\odot V^\ell h_j^\ell))$（Eq. 9–12） | `ReLU`、`Sigmoid` + `BatchNorm` | 见 `benchmarking-gnns` 复现管线（独立原始仓无预训练权重） | `MIT` |

---

## 逐篇文献核查记录（按发表年份从新到旧排列）

### DVHGNN（2025）

- 标题：DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition
- 作者：Caoshuo Li, Tanzhe Li, Xiaobin Hu, Donghao Luo, Taisong Jin
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2025；预印本 `arXiv:2503.14867`）
- 链接：https://arxiv.org/abs/2503.14867 ；代码：https://github.com/lcs0215/DVHGNN （注：截至核查日仓库仅有 `Apache-2.0` 许可证与标题占位符 `Coming soon...`，模型源码与权重均未上传）
- 核对深度：全文
- 一句话：先按余弦相似度将图像特征粗聚为 4 个语义簇，再以多尺度空洞步幅从各簇中心采样构建动态空洞超图，并用 Sigmoid 门控超图卷积提升高阶多尺度视觉表征能力。
- 解决的难点：缓解了原版 `ViHGNN` 全局模糊 C 均值聚类在大特征图上计算重且超边感受野尺度单一的问题。
- 留下的局限：**作为本项目教师模型目前不可用**——官方代码与预训练权重均未发布；且其每层余弦相似度聚类、多尺度排序采样与 Sigmoid 门控均高度非脉冲友好。
- 是否需要训练或微调：是（从头在 ImageNet-1K 等数据集上端到端训练浮点骨干）
- 涉及的算子：余弦相似度动态分簇（$C=4$）、多尺度空洞超图构建（`DHGC`，$r=1,2,3$）、超边均值聚合、`Sigmoid` 门控逐元素乘法、深度可分离卷积、`BatchNorm`、`GELU`
- 关键数字：`ImageNet-1K` Top-1 分类精度（Table 1，第 6 页）：`DVHGNN-T/S/M/B` 分别达 **79.8% / 83.1% / 83.8% / 84.2%**；`MS COCO` 目标检测与实例分割（Table 2，第 7 页）：`DVHGNN-S` 配 RetinaNet 达 **43.3** AP，配 Mask R-CNN 达 **44.8** $\text{AP}^b$ / **40.2** $\text{AP}^m$；`ADE20K` 语义分割（Table 3，第 7 页）：`DVHGNN-S` 配 Semantic FPN 达 **43.8%** mIoU，配 UperNet 达 **46.8%** mIoU。
- 与本方向的关系：背景（代表 2D 视觉超图骨干的最新精度上限，但因无公开代码和预训练权重，不宜选作免训练转换实验教师）
- 核对人与日期：Gemini，2026-11-16

### GreedyViG（2024）

- 标题：GreedyViG: Dynamic Axial Graph Construction for Efficient Vision GNNs
- 作者：Mustafa Munir, William Avery, Md Mostafijur Rahman, Radu Marculescu
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2024，pp. 6118–6127；预印本 `arXiv:2405.06849`）
- 链接：https://arxiv.org/abs/2405.06849 ；代码：https://github.com/SLDGroup/GreedyViG （预训练权重：https://huggingface.co/SLDGroup/GreedyViG/tree/main ）
- 核对深度：全文（含官方 GitHub 仓库 `greedyvig.py` 源码与 `LICENSE.txt` 逐行核对）
- 一句话：把 `MobileViG` 的固定轴向图推进到全网四个阶段，用对角象限翻转快速估算特征距离均值与标准差，动态筛除距离大于 $\mu-\sigma$ 的轴向移位候选邻居后再做 `max` 相对差聚合。
- 解决的难点：消除了 `ViG` 全图 $O(N^2)$ 两两距离矩阵与 `topk` 排序瓶颈，同时避免 `MobileViG` 纯静态轴向连边把背景异类像素强行连入同一邻域。
- 留下的局限：（1）**核查发现两处关键冲突与合规风险**：一是论文 Section 3.1 声称用欧氏距离（$\ell_2$），但官方源码 `greedyvig.py` 第 122、130、141 行实际调用 `torch.norm(..., p=1)` 计算 **$\ell_1$ 曼哈顿距离**；二是不同于 `MobileViG` 的 `Apache-2.0`，`GreedyViG` 仓库采用 **UT Austin Research License（UTRL）**，严禁再分发衍生代码且要求将衍生修改回交许可方；（2）每层动态阈值硬掩码 $\mathbb{I}(\text{dist}<\mu-\sigma)$ 对脉冲量化噪声存在跳变风险。
- 是否需要训练或微调：是（浮点教师使用 `RegNetY-16GF` 知识蒸馏训练 300 轮）
- 涉及的算子：动态轴向图构建（`DAGC`，步幅 $K=[8,4,2,1]$，`torch.roll` 移位 + 翻转距离均值/标准差二值掩码）、相对差最大值聚合 $\max(0, \text{mask}\cdot(X_{\text{rolled}}-X))$、条件位置编码 `CPE`（$7\times 7$ 深度卷积）、倒残差 `MBConv`、`BatchNorm2d`、`GELU`
- 关键数字：`ImageNet-1K` Top-1 精度（Table 1，第 6 页）：`GreedyViG-S`（12.0M 参数，1.6 GMACs）达 **81.1%**，`GreedyViG-M`（21.9M，3.2 GMACs）达 **82.9%**，`GreedyViG-B`（30.9M，5.2 GMACs）达 **83.9%**；`MS COCO` Mask R-CNN（Table 2，第 7 页）：`GreedyViG-S/B` 达 **43.2 / 46.3** $\text{AP}^{box}$、**39.8 / 42.1** $\text{AP}^{mask}$；`ADE20K` Semantic FPN 达 **43.2% / 47.4%** mIoU。
- 与本方向的关系：直接相关（非常适合作为“半动态轴向掩码图 vs 全动态 $k$-NN 图 vs 纯静态轴向图”转换误差对比的教师模型，但引用其代码时须遵守 UTRL 限制、切勿直接拷贝源码分发）
- 核对人与日期：Gemini，2026-11-16

### MobileViGv2（2024）

- 标题：Scaling Graph Convolutions for Mobile Vision
- 作者：William Avery, Mustafa Munir, Radu Marculescu
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops（CVPRW 2024，Mobile AI Workshop；预印本 `arXiv:2406.05850`）
- 链接：https://arxiv.org/abs/2406.05850 ；代码：https://github.com/SLDGroup/MobileViGv2 （预训练权重：https://huggingface.co/SLDGroup/MobileViGv2/tree/main ）
- 核对深度：全文（含官方 GitHub 仓库源码与 `Apache-2.0` 许可证核对）
- 一句话：设计极简的移动端图卷积 `MGC`，在第 2–4 阶段只让每个像素固定连接自身与上下左右固定步幅的 4 个十字邻居，并用通道拼接加分组投影取代逐元素 `max` 聚合。
- 解决的难点：解决了初代 `MobileViG` 在高分辨率阶段轴向连接过多导致延迟飙升、只能在第 4 阶段用图层的缺陷。
- 留下的局限：图结构退化为固定空洞十字模板卷积，已不含动态建图或 `max` 聚合，因此在方法论上只能充当“无动态建图、无 `max` 聚合”的消融下界对照，不能单独代表一般视觉图转换的难点。
- 是否需要训练或微调：是（浮点教师使用 `RegNetY-16GF` 蒸馏训练）
- 涉及的算子：静态五点十字图卷积 `MGC`（步幅 $K=[0,8,4,2]$，上下左右 4 邻居相对差 $x_j-x_i$ 拼接线性映射）、重参数化条件位置编码 `RepCPE`、`MBConv`、`BatchNorm2d`、`GELU`
- 关键数字：`ImageNet-1K` Top-1 精度（Table 1，第 5 页）：`MobileViGv2-Ti/S/M/B` 分别达 **77.7% / 79.8% / 81.7% / 83.4%**（iPhone 13 Mini NPU 延迟 **0.79 / 1.00 / 1.49 / 2.38 ms**）；`MS COCO` Mask R-CNN `M/B` 达 **42.5 / 43.0** $\text{AP}^{box}$，`ADE20K` Semantic FPN 达 **42.9% / 44.3%** mIoU（Table 2，第 6 页）。
- 与本方向的关系：直接相关（`Apache-2.0` 协议宽松且 HuggingFace 权重齐全，是剥离“动态 $k$-NN 误差”与“`max` 聚合误差”的理想干净对照组教师）
- 核对人与日期：Gemini，2026-11-16

### DAGr（2024）

- 标题：Low-latency automotive vision with event cameras
- 作者：Daniel Gehrig, Davide Scaramuzza
- 发表处：Nature（2024，Vol. 629，pp. 1034–1040，DOI: `10.1038/s41586-024-07409-w`）
- 链接：https://www.nature.com/articles/s41586-024-07409-w ；代码：https://github.com/uzh-rpg/dagr （预训练权重：https://download.ifi.uzh.ch/rpg/dagr/data/dagr_s_50.pth ）
- 核对深度：全文（已核对 Nature Open Access 正文与 Methods 全文，以及官方 GitHub 仓库 `dagr__net.py` 等源码）
- 一句话：将事件相机微秒级异步事件流构建为有向时空半径图，结合查表样条图卷积、有向体素最大池化剪枝与低频 RGB 特征融合，在车载场景实现亚毫秒级超低延迟目标检测。
- 解决的难点：攻克了初代 `AEGNN` 无向图消息向过去事件回传导致 $k$-hop 子图随层数急剧膨胀的问题——通过只让旧事件指向新事件（$t_i<t_j$）并在体素 `max` 池化处截断未改变最大值的更新，削减了 **91%** 的异步图更新计算量。
- 留下的局限：整个网络仍是运行在 GPU 上的**连续浮点激活图网络**而非脉冲神经元网络；官方仓库采用传染性较强的 **`GPL-3.0`** 许可证，且公开预训练权重 `dagr_s_50.pth` 绑定了双流 `ResNet-50` RGB 分支与 `YOLOX` 检测头。
- 是否需要训练或微调：是（在批量同步事件图上端到端训练浮点权重，测试时等价部署为逐事件异步流式更新）
- 涉及的算子：有向时空半径建图（$\|n_p^i-n_p^j\|_\infty < 0.01, t_i<t_j$，上限 16 邻居）、查表一阶 B 样条卷积 `LUT-SplineConv`（邻居 `add` 求和）、有向时空体素簇 `max` 池化（前 3 级 `max`、第 4 级 `mean`）及更新截断、`ELU`、`SiLU`、`BatchNorm`
- 关键数字：`DSEC-Detection` 车载事件+图像检测（Table 1）：`DAGr-S + ResNet-50` 达 **41.9** mAP（单事件能耗 **5.42 µJ/event**），`DAGr-S + ResNet-18` 达 **37.6** mAP；纯事件 `Gen1` 检测（Table 2）：`DAGr-L` 达 **32.1** mAP，`DAGr-N` 仅需 **1.36 MFLOPs/event** 与 **1.93 µJ/event**；`N-Caltech101` 检测 `DAGr-S / L` 达 **70.2 / 73.2** mAP；早期体素 `max` 池化自然拦截了 **73%** 的冗余下游传播，有向建图+有向池化合计减少 **91%** 算力（Main Text & Fig. 4）。
- 与本方向的关系：直接相关（既是事件相机图检测标杆教师，又在硬件实测层面证明了“`max` 池化天然抑制小幅更新扩散”这一核心现象，同时可作为脉络四增量推理的重要先驱）
- 核对人与日期：Gemini，2026-11-16

### Point Transformer V3 / PTv3（2024）

- 标题：Point Transformer V3: Simpler, Faster, Stronger
- 作者：Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2024，pp. 4840–4851；预印本 `arXiv:2312.10035`）
- 链接：https://arxiv.org/abs/2312.10035 ；代码：https://github.com/Pointcept/PointTransformerV3 （预训练权重：https://huggingface.co/Pointcept/PointTransformerV3 ）
- 核对深度：全文（含官方仓库模型配置与许可证条款核对）
- 一句话：彻底抛弃占点云 Transformer 前向耗时 28% 的精确 $k$-NN 搜索与 26% 的相对位置编码，改用 Z 序与希尔伯特空间填充曲线把无序三维点云序列化为一维定长块做点积自注意力，换取 3.3 倍提速、10.2 倍省显存及 20 余项三维任务全面刷榜。
- 解决的难点：打破了点云骨干因执着于严格置换不变性与精确 $k$-NN 检索而无法扩大感受野（从 16 点扩至 1024 点）和模型规模的效率瓶颈。
- 留下的局限：实质上已从“图消息传递网络”转向“空间填充曲线分块窗口 Transformer + 3D 稀疏卷积 `xCPE` + 网格 `MaxPool`”，因此若对其做脉冲转换，主要检验的是脉冲 `Softmax` 点积注意力与 `LayerNorm`（而非动态图转换）；且源码许可证为非商业 `CC BY-NC 4.0`。
- 是否需要训练或微调：是（单数据集从头训练或多数据集 `PPT` 协同预训练）
- 涉及的算子：Z-order / Hilbert 空间填充曲线位编码排序与轮换打乱（`Shift/Shuffle Order`）、1024 点非重叠块多头点积注意力（`Softmax`）、前置 3D 稀疏卷积 `xCPE`、`LayerNorm`、`GELU`、`Grid Pooling`（含 `BatchNorm` 与 `MaxPool`）
- 关键数字：效率（Table 1，第 3 页，`nuScenes` 单卡 RTX 4090）：46.2M 参数下推理延迟 **44ms**、显存 **1.2G**（较 `PTv2` 提速 **3.3×**、显存降 **10.2×**）；室内语义分割（Table 5，第 7 页，从头训 / PPT 联训）：`ScanNet v2` Test **77.9% / 79.4%** mIoU，`ScanNet200` Test **37.8% / 39.3%** mIoU，`S3DIS` Area5 **73.4% / 74.7%**、6-fold **77.7% / 80.8%** mIoU；室外分割（Table 7）：`nuScenes` Test **82.7% / 83.0%** mIoU，`SemanticKITTI` Test **74.2% / 75.5%** mIoU。
- 与本方向的关系：可借用的技术（其“用空间填充曲线一维定长分块彻底消除动态 $k$-NN 搜索”的设计，为规避脉冲网络在线排序选邻居提供了极有启发性的结构替代思路）
- 核对人与日期：Gemini，2026-11-16

### BlockGCN（2024）

- 标题：BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition
- 作者：Yuxuan Zhou, Xudong Yan, Zhi-Qi Cheng, Yan Yan, Qi Dai, Xian-Sheng Hua
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2024，pp. 2049–2058）
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.html ；代码：https://github.com/ZhouYuxuanYX/BlockGCN
- 核对深度：全文（含官方仓库 `Apache-2.0` 许可证与目录核对）
- 一句话：指出可学习骨架邻接矩阵在训练中会发生“物理骨骼拓扑灾难性遗忘”，通过最短路径跳数静态编码 + 持续同调条形码动态编码保住拓扑先验，并用块对角分组权重矩阵砍掉 40% 以上冗余参数。
- 解决的难点：在不依赖多头自注意力的前提下，以更少参数同时保留静态物理骨骼约束与样本级全局几何动态。
- 留下的局限：官方仓库未公开预训练模型权重；且其动态拓扑分支需要对每个输入序列构建 Vietoris-Rips 单纯复形滤子并计算 0 维持续同调生灭条形码，无法直接用脉冲神经元实现（只能作为首层离线输入特征预处理）。
- 是否需要训练或微调：是（端到端浮点训练 140 轮）
- 涉及的算子：最短路径距离查表嵌入（`Static Topological Encoding`）、关节欧氏距离 Vietoris-Rips 持续同调条形码向量化（`Dynamic Topological Encoding`）、块对角分组图卷积 `BlockGC`（$K=8$ 组）、多尺度时序卷积（含 `3×1 MaxPool` 分支）、`BatchNorm`、`ReLU`
- 关键数字：单关节流 `BlockGCN`（1.3M 参数，1.63G FLOPs，较基线减参 **43%**，Table 2 & 4，第 7–8 页）：`NTU-RGB+D 60` X-Sub / X-View 达 **90.9% / 95.4%**，`NTU-RGB+D 120` X-Sub / X-Set 达 **86.9% / 88.2%**；4 流集成（Table 2，第 7 页）：`NTU-60` 达 **93.1% / 97.0%**，`NTU-120` 达 **90.3% / 91.5%**，`NW-UCLA` 达 **96.9%**。
- 与本方向的关系：背景（骨架图近两年的代表性强基线；因无公开预训练权重且含持续同调拓扑预处理，转换实验优先选公开权重的 `ST-GCN`、`CTR-GCN` 或 `HD-GCN`）
- 核对人与日期：Gemini，2026-11-16

### ViHGNN 与 Pyramid ViHGNN（2023）

- 标题：Vision HGNN: An Image is More than a Graph of Nodes
- 作者：Yan Han, Peihao Wang, Souvik Kundu, Ying Ding, Zhangyang Wang
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2023，pp. 19878–19888）
- 链接：https://openaccess.thecvf.com/content/ICCV2023/html/Han_Vision_HGNN_An_Image_is_More_than_a_Graph_of_ICCV_2023_paper.html ；代码：https://github.com/VITA-Group/ViHGNN
- 核对深度：全文（含官方 GitHub 仓库 `pyramid_vihg.py` 与 `torch_vertex.py` 源码逐行核对）
- 一句话：把 `ViG` 的成对边图推广为允许一条超边同时连接多个图像块的视觉超图网络（ViHGNN），用模糊 C 均值聚类动态生成超边并执行“节点→超边→节点”两步高阶消息传递。
- 解决的难点：突破了普通图每条边只能刻画两个图像块二元关系的限制，用高阶超边建模多区域语义共现并缓解深层过平滑。
- 留下的局限：（1）**官方仓库未发布任何预训练权重**；（2）**核查发现论文与官方代码存在显著不一致**：论文 Tables 1–2（第 5 页）写明超边数 $E=50$ 且 Eq. 1 采用度矩阵归一化与可学习超边对角权重 $\mathbf{W}$，但官方仓库 `pyramid_vihg.py:215` 实际设置聚类簇数 `k=25`，且 `torch_vertex.py:88-133` 用的是无权 `torch.mean` 聚合加 $(1+\epsilon)x$ 残差；（3）每层在线迭代模糊 C 均值（FCM）聚类对免训练脉冲转换极不友好。
- 是否需要训练或微调：是（ImageNet-1K 上训练 300 轮）
- 涉及的算子：动态模糊 C 均值（FCM）超图构建、节点到超边与超边到节点两步均值/加权聚合、`ConvFFN`、`BatchNorm2d`、`GELU`
- 关键数字：`ImageNet-1K` Top-1 精度：各向同性 `ViHGNN-Ti/S/B` 达 **74.3% / 81.5% / 82.9%**（Table 4，第 6 页）；金字塔 `Pyramid ViHGNN-Ti/S/M/B` 达 **78.9% / 82.5% / 83.4% / 83.9%**（较同规模 `Pyramid ViG` 分别高 +0.7% / +0.4% / +0.3% / +0.2%，Table 5，第 6 页）；`MS COCO` Mask R-CNN `PViHGNN-S` 达 **43.1** $\text{AP}^b$ / **39.8** $\text{AP}^m$（Table 11，第 8 页）。
- 与本方向的关系：背景（ViG 家族超图分支代表作；由于无官方预训练权重且代码与论文超参数冲突，不宜选作免训练转换的首发教师）
- 核对人与日期：Gemini，2026-11-16

### HD-GCN（2023）

- 标题：Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition
- 作者：Jungho Lee, Minhyeok Lee, Dogyoon Lee, Sangyoun Lee
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2023，pp. 10444–10453；预印本 `arXiv:2208.10741`）
- 链接：https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Hierarchically_Decomposed_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_ICCV_2023_paper.html ；代码：https://github.com/Jho-Yonsei/HD-GCN （预训练权重：https://drive.google.com/drive/folders/1FB_IQdTMWE8cRvwE2KiyxC0P6LyqZku4?usp=sharing ）
- 核对深度：全文（含官方仓库 `MIT` 许可证与 Google Drive 权重链接核实）
- 一句话：以身体质心为根将人体骨架分解为多层有根树，把相邻层级间的远距语义对称关节（如左右手、左右脚）显式全连接，并融合特征空间动态 $k$-NN `EdgeConv` 与层次注意力聚合。
- 解决的难点：解决了传统骨架图无法直接沟通身体两侧对称末端关节、而纯注意力全连接图又丢失物理层次结构的矛盾，且无需使用噪声较大的关节/骨骼运动差分流即可登顶四大基准。
- 留下的局限：单块内同时包含固定/可学习子集图卷积、特征欧氏距离动态 $k$-NN（`S-EdgeConv` 与 `H-EdgeConv`）、时序 `MaxPool`、`Sigmoid` 门控以及多尺度时序 `3×1 MaxPool`，算子异构度高，直接全网免训练转换需同时解决动态 $k$-NN、`MaxPool` 与 `Sigmoid` 三道关卡。
- 是否需要训练或微调：是（浮点教师端到端训练 90 轮）
- 涉及的算子：有根树层次分解全连接图（`HD-Graph`）、特征欧氏距离动态 $k$-NN 边卷积（`S-EdgeConv` / `H-EdgeConv`）、注意力引导层次聚合 `A-HA`（含时序 `MaxPool_T`、代表节点均值池化 `RSAP` 与 `Sigmoid` 门控）、多尺度时序卷积（含 `3×1 MaxPool`）、`BatchNorm2d`、`ReLU`
- 关键数字：单流 Joint 模型在 `NTU-RGB+D 120` X-Sub / X-Set 达 **85.7% / 87.3%**（仅 1.60 GFLOPs、1.68M 参数，Table 4，第 8 页）；6 路集成（3 种质心树 × 关节/骨骼双流，Table 1，第 7 页）：`NTU-RGB+D 60` X-Sub / X-View **93.4% / 97.2%**，`NTU-RGB+D 120` X-Sub / X-Set **90.1% / 91.6%**，`Kinetics-Skeleton` Top-1 / Top-5 **40.9% / 63.5%**，`NW-UCLA` **97.2%**。
- 与本方向的关系：直接相关（拥有 `MIT` 协议与官方公开预训练权重的强骨架教师；非常适合用来测试含动态 `EdgeConv` 的现代混合骨架 GNN 在免训练转换下的行为）
- 核对人与日期：Gemini，2026-11-16

### MobileViG（2023）

- 标题：MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications
- 作者：Mustafa Munir, William Avery, Radu Marculescu
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops（CVPRW 2023，Mobile AI Workshop，pp. 2211–2219；预印本 `arXiv:2307.00395`）
- 链接：https://arxiv.org/abs/2307.00395 ；代码：https://github.com/SLDGroup/MobileViG （预训练权重：https://huggingface.co/SLDGroup/MobileViG/tree/main ）
- 核对深度：全文（已逐行核对论文正文、Algorithm 1 以及官方仓库 `mobilevig.py` 源码与 `Apache-2.0` 许可证；对应定向问题 Q7）
- 一句话：在前三阶段使用移动端倒残差卷积，仅在第四阶段（$7\times 7$ 特征图）用**完全固定的稀疏轴向网格连边（SVGA）**替代动态 $k$-NN 搜索，直接通过张量循环移位和逐元素 `max` 完成图聚合，首次让视觉图骨干在手机 NPU 上跑进 1 毫秒级。
- 解决的难点：彻底消除了原版 `ViG` 中移动端 NPU 极不擅长的两两特征距离矩阵计算、动态 `topk` 排序以及频繁的 4D↔3D 张量重排开销。
- 留下的局限：（1）只在最末 $7\times 7$ 低分辨率阶段包含图层，前三阶段本质仍是纯 CNN（`MBConv`）；（2）**其 `MRConv4d` 依然保留了逐元素取最大值 `max` 聚合与 `GELU` 激活**——源码初始化 `x_j = x - x = 0`，将负相对差全部截断为 0 后对 6 个轴向移位差值取 `max`，在低时间步二值脉冲下仍会遭遇转换偏差。
- 是否需要训练或微调：是（浮点教师在 ImageNet-1K 上配合 `RegNetY-16GF` 蒸馏训练）
- 涉及的算子：静态稀疏轴向图构建 `SVGA`（步幅 $K=2$，`torch.roll` 同行同列移位 $\{2,4,6\}$ 共 6 个固定轴向邻居）、轴向相对差**逐元素 `max` 聚合 `MRConv4d`**（$\max_{mK\in\{2,4,6\}}(0, x - \text{roll}(x, mK))$ 后拼接做 $1\times 1$ 卷积）、`MBConv`、`BatchNorm2d`、`GELU`
- 关键数字：`ImageNet-1K` Top-1 精度与 iPhone 13 Mini NPU 实测延迟（Tables 2–3，第 5–6 页）：`MobileViG-Ti`（5.2M 参数）**75.7%** / **0.78 ms**，`MobileViG-S`（7.2M）**78.2%** / **0.99 ms**，`MobileViG-M`（14.0M）**80.6%** / **1.38 ms**，`MobileViG-B`（26.7M）**82.6%** / **2.30 ms**；`MS COCO` Mask R-CNN（Table 4，第 7 页）：`MobileViG-M / B` 达 **41.3 / 42.0** $\text{AP}^b$、**38.1 / 38.9** $\text{AP}^m$。
- 与本方向的关系：直接相关（**本项目极佳的首批免训练转换教师模型之一**：`Apache-2.0` 协议、HuggingFace 权重齐全，且天然固定了图拓扑而保留了 `MRConv` 的 `max` 聚合与 `GELU`，能与动态建图的 `ViG` 形成完美的单变量对照！）
- 核对人与日期：Gemini，2026-11-16

### ViG 与 Pyramid ViG（2022）

- 标题：Vision GNN: An Image is Worth Graph of Nodes
- 作者：Kai Han, Yunhe Wang, Jianyuan Guo, Yehui Tang, Enhua Wu
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2022，Vol. 35，pp. 8291–8303；预印本 `arXiv:2206.00272`）
- 链接：https://arxiv.org/abs/2206.00272 ；代码：https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch （预训练权重：https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/vig 与 https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/pyramid-vig ）
- 核对深度：全文（已逐行核对论文正文、全部消融表以及官方仓库 `vig.py`、`pyramid_vig.py`、`torch_edge.py`、`torch_vertex.py`、`torch_nn.py` 全部源码；对应定向问题 Q6）
- 一句话：开创性地不再把图像看作规则像素网格或全连接序列，而是把图像块当作图节点，在**每一层动态按特征重算空洞 $k$ 近邻图**并用相对差最大值 `MRConv` 传递消息，证明纯视觉图神经网络在 ImageNet 与 COCO 上可匹敌主流 CNN 与 ViT。
- 解决的难点：打破了 CNN 固定局部方形卷积核与 ViT 稠密全局自注意力的二元对立，能够灵活连接不规则物体轮廓与语义同源区域，并通过 `Grapher` 双线性投影 + `FFN` 抑制了深层 GNN 的过平滑崩溃。
- 留下的局限：**集中暴露了视觉图免训练转脉冲的全部三大核心障碍**：（1）每一层 `Grapher` 都对 $\ell_2$ 归一化后的特征重算两两距离并用 `topk` 选邻居，低时间步脉冲量化噪声会直接扰动边集合（拓扑漂移）；（2）默认聚合算子 `MRConv2d` 对带符号特征差 $\tilde{x}_j-\tilde{x}_i$ 取逐元素 `max`，在脉冲流上存在系统性高估与正负号错位；（3）全网采用 `GELU` 激活（虽全用 `BatchNorm2d` 可折叠进 $1\times 1$ 卷积，但 `GELU` 负半轴非单调）。此外核查发现**论文与官方仓库两处数字冲突**：论文 Table 5 报 `Pyramid ViG-Ti` 为 **78.2%**、`Pyramid ViG-B` 参数量为 **92.6M**，而官方 `vig_pytorch/readme.md` 发布的 `pvig_ti` 实测权重精度为 **78.5%**、`pvig_b` 参数量为 **82.6M**。
- 是否需要训练或微调：是（ImageNet-1K 上用 DeiT 配方端到端训练 300 轮）
- 涉及的算子：通道 $\ell_2$ 归一化特征成对欧氏距离平方（等价余弦距离）+ 二维相对位置偏置（金字塔版）、动态空洞 $k$-NN 搜索（各向同性 $k=9\to 18$，金字塔恒定 $k=9$，空洞率 $\lceil l/4\rceil$）、源节点空间 `AvgPool` 降采样（金字塔版前两阶段 $r=[4,2,1,1]$）、相对差最大值图卷积 `MRConv2d`（拼接 $[\tilde{x}_i\|\max_{j\in\mathcal{N}(i)}(\tilde{x}_j-\tilde{x}_i)]$ 接 4 组分组 $1\times 1$ 卷积）、`BatchNorm2d`、`GELU`、`FFN`
- 关键数字：`ImageNet-1K` Top-1 精度：各向同性 `ViG-Ti/S/B` 达 **73.9% / 80.4% / 82.3%**（Table 4，第 6 页）；金字塔 `Pyramid ViG-Ti/S/M/B` 论文 Table 5（第 7 页）报 **78.2% / 82.1% / 83.1% / 83.7%**（仓库发布权重 `Ti` 为 **78.5%**）；算子消融（Tables 1–3，第 5–6 页，`ViG-Ti`）：`Max-Relative` **73.9%** 优于 `EdgeConv`（72.3%）、`GIN`（72.8%）、`GraphSAGE`（73.3%），`GELU` **73.9%** 显著优于 `ReLU`（**71.4%**，低 2.5%）与 `LeakyReLU`（72.5%）；`MS COCO` `Pyramid ViG-S` 配 RetinaNet 达 **41.8** mAP，配 Mask R-CNN 达 **42.6** $\text{AP}^b$ / **39.4** $\text{AP}^m$（Table 10，第 9 页）。
- 与本方向的关系：直接相关（**本项目最核心的首要教师模型**；`Apache-2.0` 许可证、官方 PyTorch 代码精炼、GitHub Release 全系列预训练权重可直接下载）
- 核对人与日期：Gemini，2026-11-16

### PointNeXt（2022）

- 标题：PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
- 作者：Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2022，Vol. 35，pp. 23192–23204；预印本 `arXiv:2206.04670`）
- 链接：https://arxiv.org/abs/2206.04670 ；代码：https://github.com/guochengqian/PointNeXt （预训练权重：https://guochengqian.github.io/PointNeXt/modelzoo/ ）
- 核对深度：全文（含官方仓库 `openpoints` 源码与 `MIT` 许可证核对）
- 一句话：系统证明经典 `PointNet++` 落后于新型点云网络很大程度源于训练配方老旧与缺乏残差缩放；通过引入现代数据增强/优化器、相对坐标半径归一化与分离式倒残差 `InvResMLP` 模块，在不改球查询+`MaxPool` 骨架的前提下重夺分类与分割双料第一且推理快 10 倍。
- 解决的难点：解决了原始 `PointNet++` 加深加宽时梯度消失、过拟合以及邻域三层 MLP 计算过重难以扩展的瓶颈。
- 留下的局限：虽然建图完全基于**输入固定的三维物理坐标球查询**（彻底无特征空间动态建图漂移！），但每一块的核心归约算子仍然是 **$k=32$ 个球内邻居的通道逐元素 `MaxPool`**；若像以往脉冲点云工作那样强行改成 `AvgPool` 就必须重新训练，而免训练直接用二值脉冲过 `MaxPool` 则会产生阶数偏差。
- 是否需要训练或微调：是（浮点教师使用 `AdamW` + 标签平滑 + 点重采样/丢色增强训练）
- 涉及的算子：最远点采样（FPS）、三维坐标静态球查询（$k=32$）、相对坐标除以查询半径归一化 $(\mathbf{p}_j^l-\mathbf{p}_i^l)/r^l$、邻域单层 MLP + **通道 `MaxPool` 归约** + 两层 $4\times$ 倒残差逐点 MLP（`InvResMLP`）、`BatchNorm`、`ReLU`
- 关键数字：`ScanObjectNN (PB_T50_RS)` 真实扫描分类（Table 2，第 7 页）：仅靠改进训练策略就把原封不动的 `PointNet++` 从 77.9% 提至 **86.1±0.7%**，`PointNeXt-S`（1.4M 参数）进一步达 **87.7±0.4%** OA（快于 `PointMLP` **10×**）；`ModelNet40` `PointNeXt-S` 达 **93.2±0.1%**（C=32）/ **94.0%**（C=64）；`S3DIS` 6-fold 语义分割 `PointNeXt-XL` 达 **74.9%** mIoU（Area 5 **70.5±0.3%**，Table 1，第 6 页）；`ScanNet V2` Test 达 **71.2%** mIoU；`ShapeNetPart` 达 **87.0±0.1%** ins. mIoU（Table 3，第 7 页）。
- 与本方向的关系：直接相关（**3D 点云静态坐标图方向首选教师模型**：`MIT` 许可证、HuggingFace 官方模型库齐全、纯 `BatchNorm + ReLU` 无复杂超越函数，唯一非线性聚合瓶颈就是纯粹的邻域 `MaxPool`，与 `DGCNN` 的动态特征 $k$-NN 形成绝佳对照）
- 核对人与日期：Gemini，2026-11-16

### InfoGCN（2022）

- 标题：InfoGCN: Representation Learning for Human Skeleton-based Action Recognition
- 作者：Hyung-gun Chi, Myoung Hoon Ha, Seunggeun Chi, Sang Wan Lee, Qixing Huang, Karthik Ramani
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2022，pp. 20186–20196）
- 链接：https://openaccess.thecvf.com/content/CVPR2022/html/Chi_InfoGCN_Representation_Learning_for_Human_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html ；代码：https://github.com/stnoah1/infogcn
- 核对深度：全文（含官方仓库核对）
- 一句话：结合最大均值差异（MMD）变分信息瓶颈损失、每帧多头自注意力动态内在拓扑图卷积（SA-GC）与多跳相对骨骼多模态集成，提升人体骨架动作表征的紧凑性与判别力。
- 解决的难点：缓解了仅靠外在物理骨骼连边传递远端关节消息时随最短路径指数衰减的稀释效应，并压缩与动作类别无关的姿态冗余噪声。
- 留下的局限：官方仓库**既无开源许可证也未发布预训练权重**；且每层每帧均需计算多头缩放点积 `Softmax` 注意力矩阵（`SA-GC`）与块尾 `LayerNorm`，不适合作为免训练脉冲转换的首选骨架教师。
- 是否需要训练或微调：是（含变分重参数化噪声采样与 MMD 信息瓶颈损失端到端训练）
- 涉及的算子：多头缩放点积自注意力内在拓扑 `SA-GC`（$M=3$ 头，`Softmax` 乘可学习拓扑 $\tilde{\mathbf{A}}_m$）、多尺度时序卷积 `MS-TC`（含 `3×1 MaxPool`）、变分重参数化、`LayerNorm`、`BatchNorm`、`ReLU`
- 关键数字：6 模态集成精度（Table 1，第 6 页）：`NTU-RGB+D 60` X-Sub / X-View **93.0% / 97.1%**，`NTU-RGB+D 120` X-Sub / X-Set **89.8% / 91.2%**（单流 Joint 为 **85.1% / 86.3%**，4 流集成为 **89.4% / 90.7%**），`NW-UCLA` **97.0%**。
- 与本方向的关系：背景（代表“自注意力内在拓扑 + 信息瓶颈”骨架图路线，因无公开权重和许可证，转换选型中不作主教师）
- 核对人与日期：Gemini，2026-11-16

### AEGNN（2022）

- 标题：AEGNN: Asynchronous Event-based Graph Neural Networks
- 作者：Simon Schaefer, Daniel Gehrig, Davide Scaramuzza
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2022，pp. 12371–12381；预印本 `arXiv:2203.17149`）
- 链接：https://arxiv.org/abs/2203.17149 ；代码：https://github.com/uzh-rpg/aegnn
- 核对深度：全文（已核对论文正文与附录 FLOPs 推导，以及官方仓库 `MIT` 许可证与异步转换器源码）
- 一句话：把事件相机输出建模为随时间演化的三维时空半径图，推导出在同步批量图上训练、在测试期只沿新事件的 $k$-hop 局部子图做逐事件异步增量更新的通用规则，把单事件计算量降低最多 11–20 倍、延迟降低 8 倍。
- 解决的难点：解决了此前事件图网络（如 `RG-CNNs`、`EvS-S`）每来一个新事件都必须整图重算全部节点激活的严重算力浪费，同时支持多层层次化样条图卷积与体素最大池化。
- 留下的局限：（1）**无任何幅度变化门控（$\Delta$-threshold）**：每个新事件无条件精确传播至第 $k$ 层的完整 $k$-hop 邻域，7 层深网在稠密事件区子图仍会迅速扩散（后续被 `DAGr` 的有向图与 `max` 池化截断改进）；（2）仍是浮点连续值网络（`SplineConv + ELU + BN + MaxPool`），且官方仓库未公开预训练模型文件（需自行在 `N-Cars` / `N-Caltech101` 上训练）。
- 是否需要训练或微调：是（同步批量训练浮点模型；测试期通过 `make_model_asynchronous()` 免重训转为等价异步局部子图推理）
- 涉及的算子：均匀事件抽样（$K=10$）、时空欧氏半径建图（最大度数 $D_{\max}=32$）、$k$-hop 局部子图增量消息传递、三次 B 样条卷积 `SplineConv`（$m=3$，度归一化求和）、$12\times 16\times 16$ 时空体素网格 `MaxPool`、`ELU`、`BatchNorm`
- 关键数字：事件分类（Table 1，第 6 页）：`N-Cars` 精度 **94.5%**、仅耗 **0.47 MFLOP/ev**（较次优异步图方法 `NVS-S` 省算力 **11×**，较 `EvS-S` 省 **13×**；25k 事件平均仅 `30±4.8 kFLOPs/ev`，第 7 页），`N-Caltech101` 精度 **66.8%**、**7.31 MFLOP/ev**（25k 事件异步更新 **129ms** vs 稠密 GNN **1014ms**，加速 **8×**，第 8 页）；事件检测（Table 3，第 8 页）：`N-Caltech101` **59.5%** mAP（`7.41 MFLOP/ev`），`Gen1` **16.3%** mAP（表格列为 **5.26 MFLOP/ev**，正文第 8 页一处笔误写为 `0.39 MFLOPs/ev`，特此注记冲突）。
- 与本方向的关系：直接相关（兼具脉络三事件相机图教师与脉络四 $k$-hop 异步图推理奠基工作的双重地位；`MIT` 协议且代码模块化极好，其 `make_model_asynchronous` 可直接作为事件驱动脉冲图推理的工程参照）
- 核对人与日期：Gemini，2026-11-16

### Benchmarking GNNs 超像素视觉图基准（2022）

- 标题：Benchmarking Graph Neural Networks
- 作者：Vijay Prakash Dwivedi, Chaitanya K. Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson
- 发表处：Journal of Machine Learning Research（JMLR 2022，Vol. 23，No. 329，pp. 1–48；首版预印本 `arXiv:2003.00982`，2020）
- 链接：http://jmlr.org/papers/v23/22-0567.html （预印本：https://arxiv.org/abs/2003.00982 ）；代码：https://github.com/graphdeeplearning/benchmarking-gnns
- 核对深度：全文（含官方仓库 `MIT` 许可证及 `GatedGCN`、`GraphSAGE`、`GAT`、`GIN`、`GCN` 层源码核对）
- 一句话：建立固定 100K / 500K 参数预算的公平图神经网络评测框架，其中将 `MNIST` 与 `CIFAR-10` 制成 SLIC 超像素 $k=8$ 近邻图，系统量化了同构求和/均值聚合与各向异性门控/`Max` 聚合在视觉图分类上的巨大性能鸿沟。
- 解决的难点：终结了早期 GNN 仅在极小引文图（Cora/Citeseer）或高方差 TU 数据集上无法区分架构优劣的混乱局面，给出了固定静态视觉图上五大主流消息传递算子的标准同预算基线。
- 留下的局限：超像素图拓扑由 SLIC 2D 坐标静态预计算（不含特征空间动态建图），且在 `CIFAR10` 超像素（仅 85–150 节点）上绝对精度上限约 67%–70%，需要与 `ImageNet` 级的 `ViG` / `MobileViG` 配合才能覆盖完整视觉规模。
- 是否需要训练或微调：是（提供 4 随机种子统一训练脚本，单卡数小时内精确复现全部教师模型）
- 涉及的算子：SLIC 超像素 2D 坐标 $k=8$ 高斯边权建图、`GatedGCN`（`Sigmoid` 软边门控 + 显式边特征更新）、`GraphSAGE`（邻域 **`Max` 池化**聚合 Eq. 8）、`GAT`（多头 `LeakyReLU + Softmax` 注意力 + `ELU`）、`GIN`（$(1+\epsilon)$ 求和）、`GCN`（对称度归一化均值）、`BatchNorm`、`ReLU`、拉普拉斯特征向量位置编码（`LapPE`）
- 关键数字：视觉超像素图分类（$L=4, \approx 104\text{K}$ 参数，Table 7，第 30 页，`MNIST` / `CIFAR10` 测试精度）：各向异性 `GatedGCN` 最高达 **97.340±0.143% / 67.312±0.311%**，带邻域 `Max` 聚合的 `GraphSage` 紧随其后达 **97.312±0.097% / 65.767±0.308%**，`GAT` 达 **95.535±0.205% / 64.223±0.455%**，而纯求和/均值的 `GIN`（**96.485% / 55.255%**）与对称度归一化 `GCN`（**90.120% / 54.142%**）在 `CIFAR10` 上落后 `GatedGCN` 与 `GraphSage-Max` 逾 **11–13 个百分点**！
- 与本方向的关系：直接相关（这组实验数字是论证本项目核心动机的铁证——在静态视觉图上，只要为了容易转脉冲而退回同构 `GCN`/`GIN` 求和均值，精度就会暴跌 11%–13%；必须攻克 `Max` 聚合与各向异性边门控的免训练脉冲化！且实验仓 `ZtL-sysu/gnn2snn` 与 `DeNA-SGN` 均使用该超像素基准）
- 核对人与日期：Gemini，2026-11-16

### Point Transformer / PTv1（2021）

- 标题：Point Transformer
- 作者：Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip H.S. Torr, Vladlen Koltun
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2021，pp. 16259–16268；预印本 `arXiv:2012.09164`）
- 链接：https://arxiv.org/abs/2012.09164 ；代码：https://github.com/POSTECH-CVLab/point-transformer （亦集成于 https://github.com/Pointcept/Pointcept ）
- 核对深度：全文
- 一句话：在点云局部 $k=16$ 近邻内引入基于特征向量减法与可学习三维相对位置编码的逐通道向量自注意力（Vector Attention），在室内大场景分割 `S3DIS Area 5` 上首次突破 70% mIoU 大关。
- 解决的难点：证明了逐通道调制的减法向量注意力比传统标量点积注意力在三维不规则点集上高出 5.8 个百分点 mIoU（Table 7），并验证了双支路相对坐标编码的必要性。
- 留下的局限：同时依赖局部向量 `Softmax` 注意力与降采样层的 $k=16$ 局部 `MaxPool`，且第一作者提供的复现实验仓未附预训练权重与开源许可证（注：参数量存在跨文献口径差异——原论文 Section 4.1 第 5 页报 `4.9M`，而 `PointNeXt` Table 1 第 6 页复测完整分割网报 `7.8M`）。
- 是否需要训练或微调：是（SGD 端到端训练）
- 涉及的算子：最远点采样（FPS）、三维坐标堆排序 $k$-NN（$k=16$）、向量减法自注意力（MLP + `ReLU` + `Softmax` 逐通道权重调制）、可训练相对位置编码 $\delta=\theta(p_i-p_j)$、`Transition Down` 局部 `MaxPool`、三线性插值上采样、`BatchNorm`、`ReLU`
- 关键数字：`S3DIS` 室内语义分割：`Area 5` 达 **70.4%** mIoU / **76.5%** mAcc / **90.8%** OA（Table 1，第 6 页），6 折交叉验证达 **73.5%** mIoU / **90.2%** OA（Table 2，第 6 页）；`ModelNet40` 分类达 **93.7%** OA / **90.6%** mAcc（Table 3，第 6 页）；`ShapeNetPart` 部件分割达 **86.6%** ins. mIoU / **83.7%** cat. mIoU（Table 4，第 7 页）；邻居数消融（Table 5，第 7 页）确认 $k=16$（70.4%）显著优于 $k=4$（59.6%）、$k=8$（67.7%）与 $k=32$（68.3%）。
- 与本方向的关系：背景（三维点云向量注意力里程碑，也正是脉络二中直接训练脉冲点云工作 `PPT` [CVPR 2024] 所脉冲化的原型架构）
- 核对人与日期：Gemini，2026-11-16

### CTR-GCN（2021）

- 标题：Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition
- 作者：Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng, Weiming Hu
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2021，pp. 13359–13368；预印本 `arXiv:2107.12213`）
- 链接：https://arxiv.org/abs/2107.12213 ；代码：https://github.com/Uason-Chen/CTR-GCN （预训练权重：https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing ）
- 核对深度：全文（含官方仓库源码、Google Drive 预训练权重链接与 `CC BY-NC 4.0` 许可证核实）
- 一句话：用“全局可学习共享邻接矩阵 + 按样本关节特征差动态推断的逐通道拓扑修正项”替代所有通道共用一张图的传统做法，成为近五年人体骨架动作识别使用最广的黄金基准（亦是脉冲骨架网络 `MK-SGN` 的蒸馏教师）。
- 解决的难点：在不显著增加参数量（仅 1.46M）的前提下，解除不同特征通道必须按同一张拓扑聚合的强约束，大幅提升诸如“剪指甲”“打字”等精细手指交互动作的识别率。
- 留下的局限：（1）逐通道拓扑关联分支使用双极性 **`Tanh`** 激活（Table 3 显示改成非负 `ReLU`/`Sigmoid` 会掉点）并与特征做样本级动态张量乘；（2）多尺度时序模块含 **`3×1 MaxPool`** 分支；（3）官方仓库许可证为非商业 `CC BY-NC 4.0`。
- 是否需要训练或微调：是（SGD 训练 65 轮）
- 涉及的算子：逐通道拓扑精炼图卷积 `CTR-GC`（时序池化 + 关节特征差 $\tanh(\psi(x_i)-\phi(x_j))$ + 逐通道广播修正 $\mathbf{R}=\mathbf{A}+\alpha\mathbf{Q}$ + 通道图矩阵乘）、四分支多尺度时序卷积（含空洞卷积与 **`3×1 MaxPool`**）、`BatchNorm2d`、`ReLU`、`Tanh`
- 关键数字：单流模型（1.46M 参数，1.97G FLOPs）在 `NTU-RGB+D 120` X-Sub 上达 **84.9%**（Joint）/ **85.7%**（Bone，Tables 4–5，第 7–8 页）；4 流集成（Joint + Bone + JointMotion + BoneMotion）：`NTU-RGB+D 60` X-Sub / X-View **92.4% / 96.8%**（Table 6，第 8 页），`NTU-RGB+D 120` X-Sub / X-Set **88.9% / 90.6%**（Table 5，第 8 页），`NW-UCLA` **96.5%**（Table 7，第 8 页）。
- 与本方向的关系：直接相关（**骨架动作识别方向动态拓扑首选教师模型**：官方公开了 Google Drive 预训练权重，且脉络二中的 `MK-SGN` [2026] 正是用 `CTR-GCN` 当教师从头蒸馏训练脉冲网络——若我们能免训练直接转换 `CTR-GCN`，即可与重训蒸馏路线正面同台比较！）
- 核对人与日期：Gemini，2026-11-16

### DGCNN（2019）

- 标题：Dynamic Graph CNN for Learning on Point Clouds
- 作者：Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon
- 发表处：ACM Transactions on Graphics（ACM TOG 2019，Vol. 38，No. 5，Article 146，pp. 1–12；预印本 `arXiv:1801.07829`）
- 链接：https://arxiv.org/abs/1801.07829 ；代码：https://github.com/WangYueFt/dgcnn （官方推荐 PyTorch 版及预训练权重：https://github.com/AnTao97/dgcnn.pytorch ）
- 核对深度：全文（已核对 ACM TOG 全文及 `WangYueFt/dgcnn` 与 `AnTao97/dgcnn.pytorch` 模型源码与 `MIT` 许可证）
- 一句话：提出边卷积（`EdgeConv`），在**每一层特征空间实时重算欧氏距离 $k$ 近邻图**，对中心点特征与邻居相对特征差做不对称非线性映射后再取邻域逐通道最大值 `max`，奠定了动态图点云学习的经典范式。
- 解决的难点：克服了 `PointNet++` 只按输入三维物理坐标静态分组、深层无法按语义相似性跨空间聚合远处同类部件（如飞机两侧机翼）的局限。
- 留下的局限：**与 `ViG` 完全同构的三大免训练脉冲转换难点**——（1）每层在当前隐层特征空间重算精确两两欧氏距离并取 $k=20$（或 $40$）近邻；（2）每层对 $k$ 条边特征执行逐通道 `max` 聚合；（3）全网激活函数采用带负斜率 **0.2** 的 **`LeakyReLU`**（标准单向正阈值 IF 神经元会直接抹掉负半轴信息，需双极性带符号脉冲神经元支持）。
- 是否需要训练或微调：是（端到端训练浮点教师；`AnTao97/dgcnn.pytorch` 直接提供分类、部件分割与室内分割全套预训练权重）
- 涉及的算子：每层特征空间欧氏距离动态 $k$-NN（1024 点 $k=20$，2048 点 $k=40$）、不对称边卷积 `EdgeConv`（拼接 $[x_j-x_i \| x_i]$ 接 $1\times 1$ 卷积 + `BatchNorm` + `LeakyReLU(0.2)` + **邻域 `max` 池化**）、多尺度层间拼接 + 全局 `MaxPool` / `AvgPool`
- 关键数字：`ModelNet40` 分类（Tables 2 & 4，第 7–8 页）：1024 点（$k=20$）达 **92.9%** OA / **90.2%** mAcc，2048 点（$k=40$）达 **93.5%** OA / **90.7%** mAcc；固定首层静态 $k$-NN 消融掉至 **91.7%** OA（证实动态重算图带来 **+1.2%** 净增益，Table 4）；`ShapeNetPart` 部件分割达 **85.2%** mIoU（Table 6，第 10 页）；`S3DIS` 6 折室内分割达 **56.1%** mIoU / **84.1%** OA（Table 7，第 10 页）；在后续 `ScanObjectNN` 复测中改进训练可达 **86.0%** OA（见 `PointNeXt` Table 6）。
- 与本方向的关系：直接相关（**3D 点云动态图方向最核心的必选教师模型**：架构极其干净（仅 4 层 `EdgeConv`、1.8M 参数）、`MIT` 许可证、全套 PyTorch 预训练权重开箱即用，且脉络二中的 `NI-SGCN` [AAAI 2025] 因无法转换动态 `max` 被迫改成均值重训——免训练攻克原版 `DGCNN` 极具说服力！）
- 核对人与日期：Gemini，2026-11-16

### 2s-AGCN（2019）

- 标题：Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition
- 作者：Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu
- 发表处：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR 2019，pp. 12026–12035；预印本 `arXiv:1805.07694`）
- 链接：https://arxiv.org/abs/1805.07694 ；代码：https://github.com/lshiwjx/2s-AGCN
- 核对深度：全文（含官方仓库 `CC BY-NC 4.0` 许可证核对）
- 一句话：首次把骨架图邻接矩阵拆成“物理先验图 + 全局可学习参数图 + 按样本特征计算的嵌入高斯 Softmax 注意力图”三项之和，并提出关节坐标与骨骼方向向量双流融合范式。
- 解决的难点：解决了 `ST-GCN` 全层全样本共用固定人体物理连边、无法直接建立“拍手”动作中左右手强关联的缺陷，并开创了此后所有骨架 GNN 标配的关节+骨骼多流评测体系。
- 留下的局限：官方仓库未公开预训练权重且采用 `CC BY-NC 4.0` 许可证；样本自适应项 $\mathbf{C}_k$ 在每层引入了全关节时空展平特征矩阵乘法与 `Softmax`。
- 是否需要训练或微调：是（SGD 训练 50–65 轮）
- 涉及的算子：三重自适应邻接矩阵求和（物理归一化图 $\mathbf{A}_k$ + 无约束可学习矩阵 $\mathbf{B}_k$ + 嵌入高斯点积 `Softmax` 相似度图 $\mathbf{C}_k$，$K_v=3$）、$K_t\times 1$ 时序卷积、`BatchNorm2d`、`ReLU`、关节/骨骼双流融合
- 关键数字：`NTU-RGB+D 60` X-Sub / X-View 达 **88.5% / 95.1%**（其中单关节流 `Js-AGCN` 在 X-View 达 **93.7%**、单骨骼流 `Bs-AGCN` 达 **93.2%**，Tables 2–3，第 8 页）；`Kinetics-Skeleton` Top-1 / Top-5 达 **36.1% / 58.7%**（较 `ST-GCN` 提升 +5.4% Top-1，Table 4，第 8 页）。
- 与本方向的关系：背景（承上启下连接固定骨架图 `ST-GCN` 与逐通道动态图 `CTR-GCN` 的经典里程碑；实际实验选教师时，`CTR-GCN` 与 `ST-GCN` 因公开预训练权重而更优）
- 核对人与日期：Gemini，2026-11-16

### RG-CNNs（2019）

- 标题：Graph-Based Object Classification for Neuromorphic Vision Sensing
- 作者：Yin Bi, Aaron Chadha, Alhabib Abbas, Eirina Bourtsoulatze, Yiannis Andreopoulos
- 发表处：Proceedings of the IEEE/CVF International Conference on Computer Vision（ICCV 2019，pp. 491–501；预印本 `arXiv:1908.06648`；后续扩展版发表于 IEEE TIP 2020）
- 链接：https://arxiv.org/abs/1908.06648 ；代码：https://github.com/PIX2NVS/NVS2Graph
- 核对深度：全文
- 一句话：首次将事件相机输出的异步事件流经非均匀网格降采样后直接构建为加权时空半径图，用带残差跳连的 B 样条空间图卷积（`RG-CNNs`）和二维网格簇最大池化完成分类，并发布了 10 万样本的真实手语事件数据集 `ASL-DVS`。
- 解决的难点：避免了把稀疏事件流强行压成稠密二维图像帧带来的内存与卷积冗余，仅用 `ResNet-50` 五分之一的计算量即在六个事件视觉分类集上超越帧式 CNN。
- 留下的局限：图按整段 30ms 窗口批量重算（尚不支持逐事件异步局部更新，后被 `AEGNN` 解决），每层后接二维网格簇 `MaxPool`，且仓库仅公开 `ASL-DVS` 数据集与建图脚本，未公开模型预训练权重与许可证。
- 是否需要训练或微调：是（Adam 训练 150 轮）
- 涉及的算子：三维时空网格随机单点采样（$k=8$）、加权时空欧氏半径建图（$R=3, D_{\max}=32$）、一阶 B 样条空间卷积 `SplineCNN`（$m=1, k=5$，度归一化均值）、残差图块（`K=5` 主干 + `K=1` 捷径）、二维空间簇 **`MaxPool`（`MaxP`）**、`BatchNorm`、`ReLU`
- 关键数字：六个事件分类基准 Top-1 精度（Table 1，第 7 页）：`N-MNIST` **99.0%**、`MNIST-DVS` **98.6%**、`N-Caltech101` **65.7%**、`CIFAR10-DVS` **54.0%**、`N-CARS` **91.4%**、`ASL-DVS` **90.1%**；在 `N-Caltech101` 上仅需 **0.79 GFLOPs** 与 **19.46 MB** 模型体积，优于帧式 `ResNet-50` 的 63.7% / 3.87 GFLOPs（Tables 2–3，第 8 页）。
- 与本方向的关系：可借用的技术（事件相机点图表示的奠基之作，全网仅用 `SplineConv + BN + ReLU + MaxPool`，也是脉络二中诸多 DVS 事件分类脉冲网络的重要浮点对照基准）
- 核对人与日期：Gemini，2026-11-16

### ST-GCN（2018）

- 标题：Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
- 作者：Sijie Yan, Yuanjun Xiong, Dahua Lin
- 发表处：Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence（AAAI 2018，pp. 7444–7452；预印本 `arXiv:1801.07455`）
- 链接：https://arxiv.org/abs/1801.07455 ；代码：https://github.com/yysijie/st-gcn
- 核对深度：全文（含官方仓库源码、`BSD-2-Clause` 许可证与 `OLD_README.md` 预训练模型链接核对）
- 一句话：首次把人体动作序列建模为“帧内自然骨骼边 + 帧间同关节轨迹边”的固定时空图，通过按骨骼重心距离划分自身/向心/离心三个固定子集做度归一化图卷积与时间一维卷积，开创了图神经网络骨架动作识别领域。
- 解决的难点：摆脱了早期手工划分身体部位规则或用 RNN/伪图像丢失骨骼拓扑的局限。
- 留下的局限：图拓扑完全静态固定（可学习边掩码 $\mathbf{M}$ 只是对已有物理边乘标量权重，无法在原本不相连的左右手之间生成新边），因此绝对精度明显低于后续自适应动态图（如 `CTR-GCN`、`HD-GCN`）；但**正因其全网仅含固定稀疏矩阵乘、$1\times 9$ 卷积、`BatchNorm`、`ReLU` 与全局 `AvgPool`（零 `MaxPool`、零 `Softmax` 注意力、零动态 `topk`）**，它是检验基础图转换保真度的理想无干扰基线！
- 是否需要训练或微调：是（SGD 训练；官方仓库及 `MMSkeleton` 公开了 `NTU-RGB+D` 与 `Kinetics` 预训练权重）
- 涉及的算子：固定人体骨骼空间构型三子集划分（自身、向心、离心，$K=3, D=1$）、可学习固定边重要性掩码 $\mathbf{A}_j\otimes\mathbf{M}$、对称度归一化线性矩阵乘加权和、$1\times 9$ 时序卷积（第 4、7 层步幅 2 降采样）、残差跳连、输入层与卷积层 `BatchNorm`、`ReLU`、全局 `AvgPool`
- 关键数字：`Kinetics-Skeleton` 验证集 Top-1 / Top-5 达 **30.7% / 52.8%**（Table 2，第 8 页）；`NTU-RGB+D 60` X-Sub / X-View 达 **81.5% / 88.3%**（Table 3，第 8 页；注：`2s-AGCN` Section 5.3 第 7 页指出仅改进学习率调度与预处理即可将原版 `ST-GCN` 在 X-View 提至 **92.7%**；另注 `InfoGCN` Table 1 误引 `ST-GCN` X-View 为 `88.0%`，以 `ST-GCN` 原文 Table 3 **88.3%** 为准）。
- 与本方向的关系：直接相关（**最纯净的固定图+线性求和+ReLU 视觉教师基线**：许可证宽松 [`BSD-2-Clause`]、有官方预训练权重，且尚未有任何工作发表过针对 `ST-GCN` 的免训练转换评测，见定向问题 Q2）
- 核对人与日期：Gemini，2026-11-16

### PointNet++（2017）

- 标题：PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- 作者：Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas
- 发表处：Advances in Neural Information Processing Systems（NeurIPS 2017，Vol. 30，pp. 5099–5108；预印本 `arXiv:1706.02413`）
- 链接：https://arxiv.org/abs/1706.02413 ；代码：https://github.com/charlesq34/pointnet2 （高星社区 PyTorch 版及预训练权重：https://github.com/yanx27/Pointnet_Pointnet2_pytorch ）
- 核对深度：全文（含补充材料架构细节与两仓库 `MIT` 许可证核对）
- 一句话：通过迭代最远点采样（FPS）、度量空间多尺度球邻域分组（MSG/MRG）以及局部微型 PointNet（共享 MLP + 对称 `MAX` 池化）递归提取三维点云的多尺度局部几何结构。
- 解决的难点：解决了初代 `PointNet` 只有单次全局最大池化、完全缺失局部邻域几何上下文且对非均匀采样密度脆弱的根本缺陷。
- 留下的局限：每个集合抽象层（`SA`）都完全依赖局部邻域上的**逐通道 `MAX` 池化**保证点集置换不变性——脉络一、二中已有两篇转换论文（ICCV 2023 `Lan 等` 与 AAAI 2025 `Adaptive Calibration`）试图转换 `PointNet++`，**但它们无一例外都因为处理不了脉冲 `MAX` 池化，被迫先把源网的 `MAX` 改成均值池化（`AvgPool`）并从头重新训练浮点源网络**，真正的原版 `MAX` 池化 `PointNet++` 免训练转换至今仍是空白！
- 是否需要训练或微调：是（Adam 训练；`yanx27` PyTorch 仓库内置 `ModelNet40` 与 `ShapeNetPart` 预训练模型）
- 涉及的算子：最远点采样（FPS）、固定三维坐标球查询（SSG / 多尺度 MSG / 跨层 MRG）、局部相对坐标平移 $x_i - \hat{x}$、共享逐点 MLP（`Conv1x1 + BatchNorm + ReLU`）、**邻域逐通道对称 `MAX` 池化**、分割解码 $k=3$ 反平方距离加权插值
- 关键数字：`ModelNet40` 分类（Table 2，第 6 页）：1024 点纯坐标达 **90.7%** OA，5000 点加面法向达 **91.9%** OA（邻域消融 Supp. Table 5 第 13 页：球查询 $r=0.2$ **90.7%** 略优于 $k=64$ $k$-NN 的 **90.3%** 与 $k=16$ 的 **89.3%**）；`ShapeNetPart` 部件分割达 **85.1%** mIoU（Supp. Table 4，第 13 页）；`ScanNet` 室内体素标注精度 `MSG+DP` 达 **84.5%**（非均匀扫描下 **80.4%**，Figure 5，第 6 页）；`SHREC15` 非刚体测地距离分类达 **96.09%**（Table 3，第 8 页）。
- 与本方向的关系：直接相关（检验“不改源网、保留原始对称 `MAX` 集合归约直接免训练转脉冲”能否超越“改 `AvgPool` 重训源网”路线的经典试金石，直接对应定向问题 Q2）
- 核对人与日期：Gemini，2026-11-16

### ResGatedGCN（2017）

- 标题：Residual Gated Graph ConvNets
- 作者：Xavier Bresson, Thomas Laurent
- 发表处：arXiv preprint `arXiv:1711.07553`（2017 首版，2018 v2 修订）
- 链接：https://arxiv.org/abs/1711.07553 ；代码：https://github.com/graphdeeplearning/benchmarking-gnns
- 核对深度：全文
- 一句话：将边缘 Sigmoid 软门控机制、批归一化（BN）与跨层恒等残差连接融合成通用深层空间图卷积架构，使消息传递图网络在叠加至 10 层以上时依然单调提点且快于图循环网络。
- 解决的难点：解决了早期同构图卷积无法自动抑制异类邻居噪声边（社区边界边），以及超过 3–4 层后性能剧烈退化的问题。
- 留下的局限：每一条边 $(i,j)$ 在每一层都需计算 `Sigmoid` 门控值 $\eta_{ij}=\sigma(A^\ell h_i^\ell + B^\ell h_j^\ell)$（在后续 `Benchmarking GNNs` 版中还需做邻域门控归一化 $\frac{\sigma(\hat{e}_{ij})}{\sum_{j'}\sigma(\hat{e}_{ij'})+\varepsilon}$ 并维护边特征），将二值或分级脉冲免训练映射到连续乘法边门控是静态异构边权图转换的典型难点。
- 是否需要训练或微调：是（Adam 端到端训练）
- 涉及的算子：有向/无向边 `Sigmoid` 门控 $\sigma(A^\ell h_i^\ell + B^\ell h_j^\ell)$、Hadamard 逐元素门控邻居加权和、中心节点独立变换 $U^\ell h_i^\ell$、`BatchNorm`、`ReLU`、恒等残差连接
- 关键数字：在变长随机块模型（SBM）子图匹配与 10 簇半监督图聚类任务上（Section 4，第 6–9 页），较图 RNN 精度高 **3%–17%**、速度快 **1.5×–4×**，较非学习变分 Dirichlet 方法精度高 **36%**（达 **82%**），叠加超过 6 层时残差跳连带来 **10%** 精度增益（Abstract & Section 5）。
- 与本方向的关系：直接相关（作为 `MNIST`/`CIFAR10` 超像素图上表现最佳的各向异性边门控教师架构，代表了除 `Max` 与 `Softmax` 之外第三类关键非线性图聚合——`Sigmoid` 边门控）
- 核对人与日期：Gemini，2026-11-16

---

## 这条线是怎么演化的

视觉图神经网络（教师模型）从 2017 年至今的演化，清晰地围绕**“怎么建图（静态先验 → 特征空间全动态重算 → 硬件友好的受限/静态稀疏近似）”**与**“怎么聚合（同构求和/均值 → 各向异性 `Max` / 门控 / 注意力）”**这两根主轴展开：

1. **第一阶段：固定物理/几何邻域上的消息传递与对称 `Max` 归约（2017–2018）**。三维点云线由 `PointNet++`（2017）奠基，依靠固定三维物理坐标球查询加上局部 `MAX` 池化提取多尺度几何特征；骨架线由 `ST-GCN`（2018）奠基，依靠固定人体自然骨骼图上的度归一化矩阵求和与时序卷积提取动作模式；通用变长图线则由 `ResGatedGCN`（2017）证明“残差 + BN + `Sigmoid` 边门控”是把视觉超像素图网络做深并拉开同构 `GCN` 10 个百分点以上差距的关键（后经 `Benchmarking GNNs` 2022 系统确立）。
2. **第二阶段：每层按隐层特征重算动态拓扑的全盛期（2019–2023）**。研究者很快发现，只按输入坐标或物理骨骼静态连边，深层节点无法跨空间聚合语义同类的远端区域。于是点云上的 `DGCNN`（2019）率先在每一层特征空间实时重算欧氏 $k$-NN 并做相对差 `max` 聚合（`EdgeConv`）；骨架上的 `2s-AGCN`（2019）、`CTR-GCN`（2021）、`InfoGCN`（2022）与 `HD-GCN`（2023）相继把固定骨骼图升级为按样本特征实时推断的全局注意力图、逐通道 `Tanh` 精炼图或特征 $k$-NN 混合图；而 2D 图像上的 `ViG / Pyramid ViG`（2022）及超图版 `ViHGNN`（2023）、`DVHGNN`（2025）更是直接把图像切块，每层在 $\ell_2$ 归一化特征上重算空洞 $k$-NN（或超边聚类）并配合 `MRConv`（相对差 `max`）与 `GELU`，在 `ImageNet` 与 `COCO` 上追平乃至超越同规模 CNN 和 ViT。
3. **第三阶段：面向硬件延迟与规模化的“去重型动态 $k$-NN”回潮与异步图剪枝（2022–2025）**。全图两两距离与 `topk` 排序在端侧 NPU 与大场景点云上代价高昂，催生了三条极具启发性的折中路线：（a）在 2D 视觉图上，`MobileViG`（2023）用固定轴向移位图 `SVGA` 保留 `max` 聚合但砍掉距离排序，`GreedyViG`（2024）用轴向距离阈值掩码做轻量半动态建图，`MobileViGv2`（2024）进一步简化为固定五点十字图；（b）在 3D 点云上，`PointNeXt`（2022）证明静态坐标球查询 + `MaxPool` 配倒残差缩放即可重回 SOTA，`PTv3`（2024）更是用空间填充曲线一维序列化彻底取代 $k$-NN；（c）在事件相机视觉上，`RG-CNNs`（2019）→ `AEGNN`（2022）→ `DAGr`（2024）完成了从“窗口批量重算”到“无阈值 $k$-hop 异步更新”再到“有向图 + 体素 `max` 池化天然拦截 73% 冗余传播”的跨越。

**对本项目教师模型选型的直接结论**：若按“官方代码与预训练权重完整公开、许可证宽松合规（`Apache-2.0` / `MIT` / `BSD`）、能成对隔离核心科学难点”三个硬指标筛选，**最理想的主力教师矩阵**是：
- **2D 图像分类与检测**：`Pyramid ViG` / `ViG`（`Apache-2.0`，动态特征 $k$-NN + `MRConv max` + `GELU`）对照 `MobileViG`（`Apache-2.0`，固定轴向图 + `MRConv max` + `GELU`）与 `MobileViGv2`（`Apache-2.0`，固定十字图 + 无 `max`）；
- **3D 点云分类与分割**：`DGCNN`（`MIT`，动态特征 $k$-NN + `EdgeConv max` + `LeakyReLU`）对照 `PointNeXt` / `PointNet++`（`MIT`，固定三维坐标球查询 + 邻域 `MaxPool` + `ReLU`）；
- **骨架动作与超像素基准**：`ST-GCN`（`BSD-2-Clause`，固定图 + 求和 + `ReLU` 零非线性聚合下界）对照 `CTR-GCN` / `HD-GCN`（动态拓扑 + `MaxPool`/`Tanh`/`Sigmoid`）以及 `Benchmarking GNNs`（`MIT`，同图同预算下直比 `GCN` / `GIN` / `GraphSAGE-Max` / `GatedGCN` / `GAT`）。
