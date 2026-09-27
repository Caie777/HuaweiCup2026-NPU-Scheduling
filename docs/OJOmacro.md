# OJOmacro：原始 Op 搜索与结构性 Macro 候选

实现入口为 `src/algorithms/ojomacro/search.py` 中的 `OJOLNS`，结构候选见 `src/algorithms/ojomacro/macro.py`；三个问题均通过 `scripts/run_all_problems.py` 的 `OJOmacro` 路线调用。代码从 `GraphModel` 的原始 Op、Tensor 和依赖边生成 Task 分区，检查 Op 覆盖、Task 商图无环等约束，再为候选生成多核 Plan。

初始阶段结合拓扑顺序、计算链与 Tensor 关系构造不同粒度的原始 Op 分区，并加入结构性 macro 候选。问题一通过多尺度搜索生成候选；问题二和三先构造候选，再按各自场景代理排序。代理用于节省官方评估预算，不代替合法性和 Makespan 判定。

当运行器取得一个经官方评估的有效方案后，可以以它为当前方案继续搜索 Op 级邻域。问题一还可搜索固定分区下的排程；问题二、三使用场景相关的反馈和评分。默认调用显式设置 `cpsat=False`，不要求 OR-Tools。所有送出结果最终仍须由使用者单独提供的官方评估器验证。

公开运行器对 OJOmacro 默认提供每配置 420 秒、无官方评估调用总次数上限；单次评估超时、单轮候选数与局部搜索迭代仍各自受控。论文的完整搜索实现和环境不能仅凭这些默认参数逐项复现，公开成绩按最终提交 TeX 的记录展示。无需官方附件的基础检查见 `scripts/smoke_test.py`。

