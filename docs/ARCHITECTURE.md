# 代码结构

统一入口为 `scripts/run_all_problems.py`；它负责配置、预算、官方评估、检查点与最佳合法 Plan 的保留。`src/algorithms/routes.py` 根据路线和场景调用候选构造与反馈选择。三个算法子目录互不导入，算法共用的基础能力位于 `src/common/`。

`routes.py` 保留 OJOmacro 的官方反馈候选筛选分支。该分支衔接统一入口的场景分发、评估预算和候选去重；仅为目录隔离而迁移它需要改写已通过回归的分支条件。各路线的候选构造与搜索仍在各自算法目录，路由中的筛选行为保持不变。

## 三条算法

| 目录 / 模块 | 职责 |
| --- | --- |
| `algorithms/v2plus/resource_partition.py` | V2plus 的资源感知 Task 分区、拓扑回退与代理打分基础。 |
| `algorithms/v2plus/algorithm.py` | V2plus 的多粒度候选、内存风险排序、计划选择与候选构造接口。 |
| `algorithms/rampplus/dag.py` | 多级模块粗化、分区细化和 RAMP DAG 搜索。 |
| `algorithms/rampplus/algorithm.py` | RAMPplus 候选排序、split/merge/move/reorder 反馈及初始、反馈、Block 候选接口。 |
| `algorithms/rampplus/block.py` | 可选的局部 Block 重优化邻域。 |
| `algorithms/ojomacro/candidates.py` | OJOmacro 构造、macro 组合选择与官方反馈候选接口。 |
| `algorithms/ojomacro/search.py` | 原始 Op 粒度的多尺度搜索、修复、排程邻域及可选 CP-SAT。 |
| `algorithms/ojomacro/macro.py` | 结构性 macro 分区候选。 |
| `algorithms/routes.py` | 三路线的统一分发、场景选择、反馈候选去重、安全初始 Plan 与跨路线候选排名；不实现各路线的搜索器。 |

## 共享模块

| 模块 | 职责 |
| --- | --- |
| `common/graph.py` | 解析官方 JSON 图，建立 Op、Tensor 和依赖视图。 |
| `common/partition.py` | 检查 Op 唯一归属及 Task 商图无环；RAMPplus 和 OJOmacro 都调用商图构建。 |
| `common/module_seed.py` | V2plus 与 RAMPplus 共用的自然模块生成器。 |
| `common/schedule.py` | 通用 Task 多核排程及 Plan 构造。 |
| `common/memory.py` | L1/UB 活跃内存风险及官方 Step1 顺序估计。 |
| `common/scene_cost.py` | 三个硬件场景的候选代理成本。 |
| `common/evaluation.py` | 解析 `--official-root` / 环境变量，定位本地附件，并调用官方评估器、生成 Plan 摘要与原子写入。 |
| `common/feedback.py` | 官方反馈阶段的通用预算与调用编排。 |

`scripts/export_verified_plan.py` 导出已验证的最佳 Plan；`scripts/smoke_test.py` 用合成图检查三条路线、统一分发和超时检查点。官方附件始终放在仓库外。
