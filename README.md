# moving_gc — 移动式 / 代际增量垃圾回收内核（可复用教学模块）

仅依赖 Python 标准库的两个垃圾回收模拟内核，共用同一套槽级对象模型、
整数 payload、显式根、强/弱边与基于 epoch 的外部 handle：

1. **`Heap`**（原有，契约不变）：半空间（semi-space）移动式回收器。
2. **`GenHeap`**（本次新增）：**年轻代半空间复制 + 老年代标记清扫**的
   两代回收器，支持：
   - 每个回收调用带**明确扫描边数预算**，调用方可在有限回收步之间继续
     分配、改强边；
   - **增量更新（incremental-update）写屏障**：黑（或灰对象已扫字段）
     指向白对象时立即补灰；
   - **跨代写屏障 + 记忆集**：老年代 → 年轻代强边逐条记录，小回收不扫
     整堆、不漏边；
   - **ephemeron 表**：表对象与键都强可达时，值才进入闭包，反复求到
     最小不动点后才处理普通弱引用；
   - **显式终结器事件**：大回收时按固定地址序最多执行一次，回调可把
     对象重新挂回根（复活），并可在事件中再次分配。

本模块是有限契约的教学/实验模型，**不宣称**与 JVM、.NET、V8 等任何
行业标准运行时的内存管理兼容。

## Python 版本与依赖

- Python **3.14.7**（Linux 原生验证；3.10+ 语法即可运行，目标 3.14.7）
- **仅标准库**，无需 `pip install`；完全离线，不访问网络、账号、密钥、
  数据库、中间件、云 API、Docker 或硬件。
- 所有示例数据由仓库内确定性代码生成（固定图 + 固定随机种子）。

## 运行命令

```bash
# 全部必要测试（unittest 标准发现；旧 28 例 + 新 29 例）
python -m unittest discover -s tests -v

# 只跑快速回归（原有半空间内核，亚秒级）
python -m unittest tests.test_moving_gc -v

# 扩展验证（代际/增量/ephemeron/终结器/穷举/规模）
python -m unittest tests.test_generational -v

# 固定输入演示（实测 < 0.01 秒；无 sleep、无预录输出）
python demo.py
```

预算口径：每个 `collect_step(scan_budget)` 最多做 `scan_budget` 次
“检查”（一次根、一条边槽、一个 ephemeron 条目、一次老对象发现/回收、
一次存活对象撤离），`StepResult.scanned` 精确报告、`overscan` 恒为 0。
物理槽位随所属对象的撤离单位一起移动（单对象固定布局、有界），不属于
“扫整堆”。唯一不计预算的是周期提交时的指针/表翻转与地址序维护，
它不检查任何边。

---

# 一、原有半空间内核 `Heap`（契约保持）

**接入方式（题面要求的"独立版本/显式选项"）**：新能力以**独立的新类
`GenHeap` 与新异常类型**提供，不改动也不重载 `Heap` 的任何方法签名、
异常类型或结果格式；两套内核共用同一数据模型（`Handle`、槽布局、
整数校验）但互不共享内部状态。因此旧调用方 `from moving_gc import
Heap` 的行为逐字节保持，不需要任何迁移；要用新能力须显式改用
`GenHeap`。不存在"静默改变旧回收语义"的开关。

接口与旧版完全一致：`Heap(capacity)`、`allocate / add_root / clear_root /
remove_root / root_handle / roots / get_strong / set_strong / get_weak /
set_weak / payload / set_payload / object_shape / handle / collect /
reachable_addresses`，以及 `Handle(address, epoch)`、
`CollectionResult`。完整契约见 git 历史；要点：

- 容量以槽计，`1 <= capacity <= 4096`（`MAX_CAPACITY`）；对象占
  `2+n_strong+n_weak` 槽。
- `collect()` 从显式根只沿强边做 Cheney 半空间复制；弱边不保持存活，
  指向死亡目标的弱边清空为 `None`，存活目标改址。
- 每次成功回收 epoch+1，旧 handle 立即失效（地址恰好相同也拒绝）。
- 分配空间不足时**先自动回收恰好一次**再判定，仍不足抛
  `HeapCapacityError`，不留半对象。

**这部分行为、异常类型、确定性结果全部保留；旧 28 个测试原样通过，
未删除或放松任何断言。**

---

# 二、两代增量内核 `GenHeap`

```python
from moving_gc import (
    GenHeap, StepResult, YOUNG, OLD,
    CollectionInProgressError, FinalizerError, SealedForReclamationError,
)
```

## 2.1 地址与容量

- `GenHeap(young_capacity, old_capacity, *, promote_age=2)`。
- `young_capacity >= 1`、`old_capacity >= 1`，且
  `young_capacity + old_capacity <= 4096`。
- 年轻代是**两个**半空间（复制回收，角色每次小回收翻转）；老年代是
  **一个**标记清扫空间（地址永不移动，回收产生空洞，首次适配复用并
  合并相邻空洞）。
- 逻辑地址：年轻对象地址为 `[0, young_capacity)` 的原始槽号；老对象
  地址为 `young_capacity + 原始老槽号`。一个 `Handle` 因此唯一命名一个
  对象。
- 新对象总分配在年轻代。对象在小回收中存活后年龄 +1；年龄达到
  `promote_age`（默认 2：出生 0、熬过一次为 1、下一次提升）或被
  `request_promotion` 强制时，在随后的小回收中提升到老年代。

## 2.2 公开方法

| 方法 | 语义 |
|---|---|
| `allocate(payload=0, *, n_strong=0, n_weak=0) -> Handle` | 在年轻代分配；空闲不足且无挂起周期时自动跑**一次**小回收；挂起周期中年轻代满则抛 `CollectionInProgressError` |
| `add_root / clear_root / remove_root / root_handle / roots` | 同 `Heap`，先校验 handle 再改根 |
| `get_strong/set_strong/get_weak/set_weak`、`payload/set_payload`、`object_shape`、`handle` | 同槽模型；写强边穿过写屏障（记忆集 + IU），撤离期对未复制存活目标即时复制 |
| `add_ephemeron(table, key, value) -> eid` | 注册 `(table,key,value)`；`value` 可为 `None`；不保持 table/key 存活 |
| `remove_ephemeron(eid)` / `ephemeron(eid)` / `ephemeron_ids()` | ephemeron 表管理 |
| `set_finalizer(handle, cb) -> token` | 注册 `cb(heap, handle)`；每对象最多一次；挂起周期中注册抛 `CollectionInProgressError`，已执行后再注册抛 `FinalizerError` |
| `finalizer_ran(handle)` | 该对象的终结器是否已执行 |
| `request_promotion(handle)` | 强制在下次小回收提升（仅存活者生效） |
| `collect_step(scan_budget, kind=None) -> StepResult` | 推进一个回收周期；首步 `kind="minor"/"major"`，后续步省略或相同 |
| `run_minor(scan_budget=256) / run_major(scan_budget=256)` | 内部连续泵步直到周期完成（自动应急回收也走这里） |
| `objects(generation=None)`、`generation_of`、`age_of`、`used/young_used/old_used/free_slots`、`capacity/young_capacity/old_capacity/epoch/phase` | 只读内省 |
| `remembered_set()` | 当前强记忆集条目 `(holder_addr, field_idx)`，有序 |
| `gray_pending()`、`marked_addresses()`、`pending_finalizers()`、`promotion_pending()` | 周期中间态检查（题面要求“所有中间态可检查”） |
| `reachable_addresses()` | 只读独立强可达遍历（诊断/测试） |

## 2.3 `StepResult` 字段

`kind, phase, completed, budget, scanned, overscan, gray_remaining,
steps_in_cycle, barrier_records`；周期完成步另带
`relocated`（年轻→新年轻地址）、`promoted`（年轻→老地址）、
`live_young_slots / live_old_slots`、`cleared_weak_edges`、
`cleared_ephemerons`、`finalized_now / finalized`
（每项 `(generation, address, token, resurrected)`）。

中间 `phase` 序列：

- 小回收：`mark → ephemeron → plan → evac → done`。
- 大回收：`mark → fdiscover → finalize → ephemeron → weakclear → sweep
  → done`。
- `plan`（提升落位预算化预演）与 `evac`（复制与各类边修正）是小回收
  专有；`fdiscover/finalize/weakclear/sweep` 是大回收专有。

## 2.4 屏障与记忆集（确定规则）

- **跨代写屏障**：老对象强边写入年轻目标 → 该老 holder 进入强记忆集，
  作为下次小回收的附加根；老→小弱边只进弱记忆集（仅用于撤离后改址/
  清空，绝不当根）。小回收不遍历整个老年代。
- **增量更新屏障（IU）**：开放阶段（mark/fdiscover/ephemeron/plan/
  finalize）写强边时——
  - 小回收中老→小写：无条件把年轻目标补进小闭包并加入种子；
  - 否则当持有者为**黑**（其强字段已全部扫过），或持有者为**灰但该
    字段已扫过**，而目标为白时，立即把目标补灰入队。字段逐槽增量扫描，
    所以“灰对象的已扫字段被改写”也必须补记（标准按对象原子扫描只查
    黑的简化在此不成立）。
- 补灰次数累计在 `barrier_records`。

## 2.5 ephemeron 闭包（最小不动点）

- 大回收：值仅在 **table 与 key 都已在强闭包**时才被追踪；每激活一个
  原本为白的值，就必须对全表再扫一遍，直到一整遍没有新增长为止
  （终止性：每遍至少新增一个对象）。
- 小回收：老对象一律视为存活（小回收不回收老年代）；仅追踪指向年轻
  代的值。
- ephemeron 不动点在**普通弱边清理之前**完成。大回收清扫时删除
  table/key 已死的条目，value 已死则置空；小回收时改写存活条目的
  地址、删除 table/key 死亡的条目。
- 特例：键与值互相强引用但二者无其他可达路径时，ephemeron **不能**让
  它们凭空存活（有专门穷举/差分用例）。

## 2.6 终结器与复活（固定顺序）

大回收顺序固定为：

1. 强闭包标记到不动点（含 IU 屏障吸收的交错写）；
2. `fdiscover`：按老对象地址序，预算化地快照“不可达且带未执行终结器”
   的候选；发现途中若屏障又染灰，则排空后整轮重来，快照永不在非不动点
   上拍板；
3. `finalize`：按候选地址序，**逐个**触发显式事件；每对象先标记
   `fin_ran`（保证最多一次），调用回调；回调可改根、改边、分配；
4. 每个回调后从全部根重新闭合强闭包并再跑 ephemeron 不动点，然后才
   判定该对象是否复活（`resurrected`）。被更早的复活链/ephemeron 级联
   带到存活的候选不触发事件、保留资格到下次；
5. 闭包就此封死，进入 `weakclear`：清空指向死亡老目标的普通弱边，
   修剪 ephemeron；
6. `sweep`：回收死亡老对象，归还空洞。

终结期新分配的对象出生即灰（参与开放闭包）。已执行但本周期未复活的
对象在清扫前若又被屏障挂活，则存活但**不会再次终结**（标准
“finalize 后被保持即存活、不再二次执行”）。

## 2.7 移动后外部 handle 的 epoch 契约

- 每次**成功完成**的小回收/大回收 epoch+1。年轻代复制会改址，旧 handle
  一律失效；老年代标记清扫不改址，但 epoch 仍 +1（与 `Heap` 一致：以
  epoch 为代际代号）。
- 撤离进行到一半时，旧年轻地址会经转发表解析到新位置；调用方在步骤
- 之间读到的始终是一致的已发布或已转址状态。
- 大回收进入 `weakclear/sweep` 后，死亡集合已封死：此时把将回收的老
  对象挂到存活对象会抛 `SealedForReclamationError`，失败写不留悬空边。
- 小回收撤离期引用“非存活年轻对象”抛 `SealedForReclamationError`；
  引用存活但尚未物理复制的对象会即时复制（有界、计入该次写）。

## 2.8 输入与错误语义

- 所有数值形参为纯整数：`bool` → `TypeError`；`NaN/±Infinity` →
  `ValueError`；其他浮点/非整数 → `TypeError`。无浮点结果，无误差口径。
- 容量越界、负边数、`scan_budget<1`、非法 `kind` → `ValueError`；
  字段越界 → `IndexError`；未知根/ephemeron → `KeyError`。
- 错误层级：`GCError` 为基类，派生 `StaleHandleError`、
  `HeapCapacityError`、`CollectionInProgressError`、`FinalizerError`、
  `SealedForReclamationError`。
- 有状态操作原子：非法 handle 的根替换/字段写在修改前失败；提升落位
  在移动任何字节前用预演自由表证明装得下，装不下则本次小回收整体不
  发生（`HeapCapacityError`，堆仍从原状态提供服务，epoch 不增）。
- **预算耗尽 vs 无解**：`collect_step` 的预算耗尽是正常的“未完成”
  （`completed=False`，可继续），不是异常；提升集/对象确实装不下才是
  确定的 `HeapCapacityError`。本模型不存在“数值未决/不可辨识”：小参考
  域（5 对象穷举）与差分用例都给确定答案，没有用 unknown 充当实现。

## 2.9 规模

- 接口边界：两代容量之和 ≤ 4096 槽（`MAX_CAPACITY`）。
- 支持范围：测试中实际构造并验证 **≥ 2000 次对象分配**、两代容量可配、
  **10000 次交错更新**（`TestScale`：40 个永久扎根的老年代骨干 + 12 波
  各 200 个年轻对象，累计 2440 次分配、约 48 次小回收、3 次大回收，
  记忆集/提升全程参与，亚秒级完成）。
- 小堆穷举固定使用 **5 个对象**（240 个确定种子，强弱边矩阵 + 增量
  屏障交错，全部提升到老年代以对稳定地址做完整核对）。

---

# 三、独立参考与验收方式

- `tests/reference.py`：原半空间内核的独立图模型（不导入被测包）。
- `tests/reference_gen.py`：两代内核的**两个**独立模型，均不导入
  `moving_gc`：
  - `RefGen`：批处理规格预言机（minor/major、ephemeron 最小不动点、
    终结器复活 LFP、提升落位、弱边清理、空洞合并）；终结器回调直接对
    参考世界执行同一回调（复活/分配在参考世界真实发生），**不调用被测
    求解器**；
  - `MarkLedger`：一单位一步的增量标记镜像，与被测逐锁步，应用同一批
    交错写，逐步核对灰队列、标记集与屏障计数。
- 期望值只来自上述独立模型或测试自带的命名图 BFS；“读回被测结果”只作
  补充断言。穷举域对每个种子用独立 BFS 完整核对最终老对象集、所有强边
  无悬空、弱边改址或清空。
- 对认证/最优性/误差界不信任被测自报布尔：存活集、可达性、边一致性都
  由独立遍历/参考世界核算；`overscan==0`、`scanned<=budget` 也是逐断言
  而非只看标志。

## 测试覆盖摘要（新 29 例）

- 批处理差分对照（vs `RefGen`）：基本压缩与年龄、年龄/强制提升边界、
  老→新记忆集、老年代回收与空洞复用、弱边清理。
- ephemeron：不动点链、不保持 key 存活、键值互引不复活、值成为另一条
  目键的级联。
- 终结器：复活且仅一次、不复活回收后清弱边、终结期再分配。
- 增量屏障账本：30 个种子逐步锁步核对；老→小写补灰专项。
- 5 对象 × 240 种子穷举：强弱边 + 黑/灰已扫字段补记交错，独立 BFS
  完整核对。
- 预算：`scanned<=budget`、`overscan==0`、预算 1 必推进、同输入同结果、
  周期 kind 冲突拒绝、非法预算。
- 句柄/密封/校验：epoch 失效、悬空弱边返回 None、密封期拒绝附加、
  重复终结注册、容量边界、对象大于年轻代、挂起周期注册终结器。
- 规模：≥2000 分配 / 10000 交错更新的非退化例子。

---

# 四、设计取舍与已知限制

- 槽级扁平模型，固定布局对象，无数组/变长类型；payload 为任意 Python
  整数（无界），地址/槽数为精确整数，无浮点、无舍入。
- 老年代用空闲链表首次适配 + 相邻空洞合并；清扫后高端连续空洞回收为
  高水位。
- 记忆集是 holder 粒度的保守集合，在小回收的边修正阶段按实际边逐条
  剪枝；卡表精度不是本教学模型的目标。
- 终结器只在大回收投递；带未执行终结器的年轻对象被当作小回收附加根
  （先熬过提升），这是确定的固定取舍。
- 非并发、非并行，单线程；“增量”指回收工作可被预算切成多步并与 mutator
  交错，不等于多线程并发。
- 终结期回调在同一线程同步执行；回调内禁止嵌套调用 `collect_step`
  （抛 `CollectionInProgressError`）。
- 不保证 Python 进程内存与槽数线性对应；4096 槽是逻辑接口边界。

## 仓库结构

```
moving_gc/
  __init__.py        # 公开接口导出（Heap 与 GenHeap 两套）
  core.py            # 原有 Heap / Handle / CollectionResult + Cheney（未改语义）
  generational.py    # GenHeap / StepResult：两代增量回收、屏障、记忆集、
                     #   ephemeron、终结器
  errors.py          # GCError 及五个派生异常
tests/
  reference.py       # 原半空间独立参考
  reference_gen.py   # 两代独立批处理预言机 + 增量标记账本
  test_moving_gc.py  # 原有 28 例（原样保留）
  test_generational.py  # 新 29 例（差分/穷举/账本/预算/规模）
demo.py              # 固定输入演示：旧内核 + 新三能力 + 两个真实拒绝
README.md
```
