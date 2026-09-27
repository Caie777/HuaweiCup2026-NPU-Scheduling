# 公开实验数据与统计口径

## 来源与保留边界

- `old_budget_units.csv`：从团队原始 `队友三算法全量实验汇总_20260926.csv`（SHA-256 `1891eb602a664adce18aa576b67d5f7059d5fec60908c8fc37ccddfca23a5f32`）筛出非隐私字段，1200 配置 × 3 算法 = 3600 行。原始性能字段文本数值未改动；私人路径、用户名相关列、Plan/结果路径与备注标签未公开。
- `singlecore_baseline_cycles`：取最终论文附录 `appendix_results.tex` 的问题一单核官方基准，100 例；源文件 SHA-256 `c1a6b2b1be7eb528b289c923a5d0dcc909726ed996f479ddf0076182bd2c4c92`。
- `ojo_7min_summary.csv`：照录最终归档 ZIP 中 `example.tex` 的 `tab:all-speedups`，源文件 SHA-256 `681357a495dd95e33b0244a01159467ba3bc1d34375ad5ee569fdb3ca9df8c9f`。论文数据及原始文件均不在此仓库。
- `old_budget_large_graph_pairs.csv`：只公开规模分组汇总，不公开官方计算图。计算 Op 数来自本地官方 `data/case_*.json` 的 `ops` 列表，排除 `COPY_IN`、`COPY_OUT`；按 Op 数降序取最大的 25 张图，最小为 7360 Op。每张图在 P1/P2/P3 × 2–5 核分别与 OJOmacro 配对；双方都有效时，官方 Makespan 较低者获胜。

## 旧预算的证据与限制

原 CSV 的每算法 `total_seconds` 最高约 300.4 秒，多数超时记录在 300 秒附近。与该 CSV 字段相符的原比赛运行器 `solution/scripts/run_all_problems.py`（SHA-256 `a871baf6f33686c6cef4fe5c37305ce86548904a7a40a5a25d11159e761ab892`）默认 `--per-case-seconds 300`；`overnight` 预设 `max_candidates=8`。`solution/scripts/start_overnight.sh` 以 `--profile overnight` 调用此入口，可继续透传显式覆盖参数。原 CSV 的 P1/P2 最多 8 次官方调用，P3 最多 9 次；P3 可额外调用一次 P2 同方案参照评估，与源码上限吻合。无合法方案时的安全保底评估也可能额外占一席。

**队友当时实际使用的完整启动命令与日志未保留在本地**，因此不能仅凭 CSV 确证是否覆盖过单次评估超时、搜索秒数或其他开关。旧预算的 300 秒与常规 8 次额度有源码及调用次数分布交叉支持；更细的配置仍待原运行者确认。`official_attempts` 是实际调用数，不等同于额度，也不等同于算法内部候选或迭代数。

## 计算方法

`old_budget_summary.csv` 按场景、核数、算法分为 36 组，每组请求 100 例。`valid_n` 仅含 `PASS` 与 `PARTIAL_PASS_BUDGET` 中具有正数官方 Makespan 的记录。`TIMEOUT` 与缺失结果均不补零、不纳入均值；分别保留状态数和分母。每例加速比 = 该例单核基准 / 算法官方 Makespan；组平均是逐例加速比的算术平均，不是两个平均 Makespan 的比值。平均额外 DDR 搬运量另用 `copy_bytes_n` 标记分母。

规模配对只在双方同一用例、场景、核数且均有有效 Makespan 时比较；`paired_valid_n = ojo_lower_makespan_n + tie_n + opponent_lower_makespan_n`。大图样本由图本身的非 COPY 计算 Op 数定义，不使用 case 编号作为规模代理。

最终 TeX 记载每个“用例 × 问题 × 核数”配置为 420 秒墙钟预算，覆盖构造、搜索和官方评估，不设候选或真实评估调用总次数上限；其表格每点为 100 例的逐例加速比均值。`Cache / 无 L2` 是逐例 P2 Makespan / P3 Makespan 的均值。该结果与旧预算不作同预算排名。

