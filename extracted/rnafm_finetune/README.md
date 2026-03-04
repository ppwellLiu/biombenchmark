# RNA-FM 微调相关代码提取

此目录提取了仓库中与 **RNA-FM 下游微调/训练** 直接相关的代码副本，包含：

- 任务入口脚本（按 `--method RNAFM` 走 RNA-FM 分支）
- RNA-FM 专用 evaluator（加载 RNA-FM 预训练权重并构建微调头）
- 微调包装模型（`freeze_base` 等参数控制是否冻结 RNA-FM 主干）
- RNA-FM 词表
- 微调运行脚本（含 freeze / nofreeze 配置）

提取范围见本目录文件树。

> 说明：
> 1. 该目录为“提取副本”，原始代码仍保留在原路径；
> 2. `scripts/mrl/mrl_run_batch.sh` 中引用了 `model/configs/RNAFM.json`，但该文件在当前仓库中不存在。
