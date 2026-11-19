# 项目进度（交接记录）

> 每个 AI 结束工作前在最上方追加一条记录，新的在上面。

## 当前状态
任务 01 已完成并合并（PR #1）。任务 02（Notion 讲解）可以开始。文献树 v1 的若干判断已被任务 01 推翻，待任务 03 修订为 v2。

## 待办
- [x] 任务 01：文献收集与核查（Gemini，PR #1 已合并）
- [ ] 任务 02：Notion 通俗讲解（Gemini，等 01 合并）
- [ ] 任务 03：Claude 审查并把文献树修订为 v2

## 已知问题
- 文献树 v1 没有通读任何论文全文；"未见图网络免训练转换"等结论只基于有限检索。
- 文献树 v1 的路线图在在线版里是一张图，仓库版改成了表格，内容相同。

---

## 工作记录

### 2026-11-17 · claude · main（审查 PR #1）
- **完成**：审查并合并 PR #1。独立核实了 7 篇论文的出处页面和 3 个官方代码仓库里的具体行（DCGS、Spiking PointCNN、MobileViG），全部与记录吻合；在 PR 分支上直接修了三处小问题（提交 413222a）。详见 PR #1 的审查评论。
- **未完成**：任务 03（把文献树修订为 v2）还没做。
- **注意 / 坑**：
  - 120 条里只抽查了一部分，其余“全文”核对按 Gemini 的记录接受，未独立复现。
  - PMSM 的 GCN 实验、四篇论文的发表处（见 PR 评论）未能独立确认，引用前要复核。
  - 做任务 02 时，文献条目里带倾向的措辞要放进“判断”标注块，见 PR 评论末尾的三点提醒。

### 2026-11-16 · gemini · gemini/literature-01
- **完成**：任务 01 全部八个交付物，放在 `literature/`：
  - `01-ann2snn-conversion.md`（27 条）、`02-spiking-gnn.md`（29 条）、`03-vision-graph-backbones.md`（22 个模型的八列教师表 + 逐篇条目）、`04-event-driven-inference.md`（21 条，含 DynaGraph、Ingress 的核实）、`05-energy-and-hardware.md`（含 Horowitz 2014 第 14 页原表）。
  - `questions.md`：定向问题 1–11 逐条回答。
  - `audit-of-claude-v1.md`：草稿八张表共 56 行逐行判定，外加表格外的论断和漏掉的工作。
  - `search-log.md`：SL-01～SL-77。
- **主要发现**（详见 audit 文件开头）：草稿“没有图网络免训练转换”被 One-Spike SNN（IEEE TETC 2025）推翻；Spiking PointCNN 不是转换；`arXiv:2409.03368` 被误读为“max 池化改成神经元层”；“图上多跳扩散未解”说过头；AEGNN 200 倍与 11 倍两个版本数字冲突。
- **未完成 / 只核到摘要**：Adaptive Fission（OpenReview 反爬拦截）、Dampfhoffer 等（HAL 返回 500）、GraphPTQ（OpenReview 403）、Tianjic、DRA、EDA-GNN、Beyer 等 1999 只核对到摘要或定理陈述；STA 的被引列表是用 OpenAlex 全文检索近似得到的，完整性不确定；MSG 相关工作节那句话未逐句核对。
- **注意 / 坑**：
  - 早期下载的若干 arXiv 编号与论文不符（`1608.08404`、`2501.06640`、`2312.17499`、`2508.05743`、`2010.00175`），都已排除；复查时还更正了 01、02、05 中五个写错的标题（DCGS、Wang 等 AAAI 2026、并行转换、SpiCalib、SpikeGCL）和 ETHEREAL 的标题与发表处。之后补条目时请用 arXiv API 或 PDF 首页核对标题，不要凭检索摘要填写。
  - ETHEREAL 两份预印本都没有写 ESSERC 录用信息，目前只能记为 arXiv。
  - OpenAlex 的被引覆盖不全：One-Spike SNN、PMSM 都不在五篇被查论文的被引列表里，是另行检索到的。
  - DBLP 当天不可用，Crossref 偶发限流。
### 2026-11-16 · claude · main（初始提交）
- **完成**：初始化仓库和协作文档；放入文献树 v1；写好任务 01、02 的说明。
- **未完成**：任务 01、02、03 都还没开始。
- **注意 / 坑**：仓库创建时是空的，没有可供开 PR 的基线，所以这一次直接提交到了 `main`。之后所有改动都走分支和 PR。
