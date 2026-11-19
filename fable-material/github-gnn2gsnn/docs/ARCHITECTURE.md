# 结构说明

本仓库只有文档。信息按下面的方向流动：

1. `docs/literature-tree/literature-tree-v1.md`：Claude 的初步判断（只读）。
2. `literature/`：Codex 和 Gemini 找全文献、逐条核实，并核查第 1 步的判断。
3. `notion/`：Gemini 把第 2 步的内容做成通俗讲解，并导出到这里。
4. `docs/literature-tree/literature-tree-v2.md`：Claude 审查第 2、3 步后修订的判断（尚未创建）。

各目录的文件清单见对应的任务说明：`tasks/01-literature-collection.md`、`tasks/02-notion-presentation.md`。
