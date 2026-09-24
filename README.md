# moving_gc — 分代增量垃圾回收内核（年轻代复制 / 老年代标记清扫）

仅依赖标准库的 GC 教学/实验内核，包含两个并存版本：

- **v1（旧版，行为完全保留）**：`moving_gc.Heap` —— 单代半空间（semi-space）
  移动式回收，强边/弱边、整数 payload、显式根、epoch handle。旧测试一个未删、
  断言一条未放松。
- **v2（本次新增，显式入口）**：`moving_gc.GenerationalHeap` —— 年轻代
  半空间复制 + 老年代标记清扫的**分代**组合，支持：
  - 调用方可在每个**有限回收步**之间继续分配与改强边；
  - **增量更新（IU）写屏障**：黑对象写入白对象时立即补记；
  - **精确记忆集（remembered set）**：记录持有任何老→年轻槽的老对象；
  - **ephemeron 表**：表与键同时可达时值才参与闭包，迭代到不动点后才处理
    普通弱引用；
  - **显式终结器事件**：每个对象身份最多终结一次，允许在终结期间把对象重新
    挂回根（复活），允许终结期间分配；复活→重新标记→ephemeron 不动点→弱引用
    清理的顺序固定；
  - 每次回收调用带**明确的边扫描预算**，超出预算至多一条边，绝不单次扫完整堆。

它是有限契约的教学/实验模型，**不宣称**与 JVM、.NET、V8 等任何行业标准
运行时的内存管理兼容。

## Python 版本与依赖

- Python **3.14.7**（Linux 原生验证；3.10+ 语法即可运行，目标版本 3.14.7）
- **仅标准库**，无需 `pip install`；完全离线（无网络、账号、密钥、数据库、
  Docker、硬件或第三方 API）；示例数据全部由仓库内确定性代码生成。

## 运行命令

```bash
# 1) 快速回归（保留旧入口；包含 28 个旧用例 + 32 个 v2 快速用例）
python -m unittest discover -s tests -v

# 2) 扩展验证（与快速回归分开；5 对象穷举 + 主规模 + 随机交错）
python verification/extended_verification.py             # 全部套件
python verification/extended_verification.py strong5     # 单个套件
# 套件: strong5 weak5 ephemeron finalizer barrier scale fuzz5

# 3) 90 秒以内的固定输入演示（无 sleep、无预录答案）
python demo.py
```

实测时间（本机）：快速回归 < 0.3 秒；扩展验证全套约 29 秒（其中
`strong5` 124,416 个双引擎锁步周期约 26 秒，其余合计约 3 秒）；demo < 1 秒。

## v2 公开数据结构与接口

```python
from moving_gc import (
    GenerationalHeap, GenCollectionResult, StepReport, CycleInfo,
    FinalizationEvent, Relocation, WeakClear, EphemeronClear,
    OLD, YOUNG,                     # 0 / 1
    GCError, StaleHandleError, HeapCapacityError,
)
```

### 容量与对象布局

`GenerationalHeap(young_capacity=512, old_capacity=512, *, promotion_age=1)`

- 两代容量各自以**槽**计，`1 <= *_capacity <= 4096`，独立可配；
- 对象占 `2 + n_strong + n_weak + 2*n_ephemeron` 槽
  （`[头, payload, 强边…, 弱边…, (键,值)…]`，ephemeron 每项占 2 槽）；
- `promotion_age >= 1`：年轻对象每经历一次小回收年龄 +1，达到阈值即晋升老年代；
  默认 `1`（小回收存活即晋升，最确定的缺省）。若年轻存活者放不下年轻半空间，
  溢出者按撤离顺序**提前晋升**（确定性规则）。

### Handle 与 epoch（新旧契约）

v2 的 `Handle(address, epoch, generation)` 增加 `generation` 字段
（`OLD=0` / `YOUNG=1`）。v1 的 `Handle` 仍可只按 `(address, epoch)` 构造，
`generation` 缺省为 -1 且 v1 从不检查。**每次成功提交回收（小/大）epoch +1，
旧 handle 一律拒绝**，包括对象晋升或移动后恰好落在同槽号的情况；晋升不改变
这一契约——外部持有的年轻代 handle 在小回收后同样失效，必须经根或
`heap.handle(addr, gen)` 重新签发。

### 方法

| 方法 | 语义 |
|---|---|
| `allocate(payload=0, *, n_strong=0, n_weak=0, n_ephemeron=0, generation=YOUNG)` | 在指定代分配（默认年轻代）。年轻代装不下时**先恰好一次小回收**，仍装不下再一次大回收；老年代装不下时一次大回收；之后仍失败抛 `HeapCapacityError`，失败不留半对象、epoch 不前移 |
| `add_root / clear_root / remove_root / root_handle / roots` | 显式具名根（语义与 v1 相同） |
| `get_strong/set_strong`, `get_weak/set_weak` | 强/弱边读写（`None` 为空边）；**所有写都穿过屏障** |
| `get_ephemeron(h,i) -> (key, value)` / `set_ephemeron(h,i,key,value)` / `clear_ephemeron(h,i)` | ephemeron 槽读写；键为 `None` 表示空项 |
| `set_finalizer(h, event_id)` / `finalizer_event_id(h)` / `finalized_event_ids()` / `pending_finalizers()` | 显式终结器事件注册/查询；终结过的对象身份禁止再注册（抛 `GCError`） |
| `begin_collection(kind)` | 开始 `"minor"`/`"major"` 增量周期 |
| `mark_step(scan_budget) -> StepReport` | 标记+ephemeron 闭包推进一步，至多扫 `scan_budget` 条边 |
| `finalize_step(handler)` | 恰好执行**一个**终结器事件；`handler(event_id, handle)` 可挂根/改边/分配；handler 抛异常则事件不被消费 |
| `finish_collection(scan_budget) -> StepReport` | 推进终结后的重新标记（含 ephemeron 不动点）、弱引用/ephemeron 清理与提交；`complete=True` 时带 `GenCollectionResult` |
| `run_collection(kind, scan_budget, *, finalizer=None)` / 别名 `collect(...)` | 用固定步预算驱动完整周期的便利入口（内部仍是同样的有限步） |
| `cycle_phase()` / `cycle_info()` | 当前阶段（`idle/marking/finalizing/remark/clearing`）与可检查中间态（marked、gray、待决 ephemeron、待终结事件、已扫边数） |
| `remembered_set() -> frozenset[int]` | 当前记忆集中的老年代地址（精确，非位图近似） |
| `reachable()` / `ephemeron_closure()` | 只读独立账本：纯强可达；以及 ephemeron 不动点闭包 |
| `used_slots(gen)` / `free_slots(gen)` / `objects(gen=None)` / `payload` / `set_payload` / `object_shape` / `handle(addr, gen)` / `generation_of` | 只读与字段访问 |

### GenCollectionResult 字段

| 字段 | 含义 |
|---|---|
| `kind` | `"minor"` / `"major"` |
| `relocated: tuple[Relocation,...]` | 每个移动对象的 `(from_gen, from_addr, to_gen, to_addr)`，按固定序排序；老年代存活对象原地不动，不出现在其中 |
| `promoted` | 本次年轻→老年晋升对象数 |
| `live_old_slots / live_young_slots` | 提交后两代占用槽数 |
| `roots_preserved` | 非空根数 |
| `cleared_weak_edges: tuple[WeakClear,...]` | 目标死亡被清空的普通弱边（提交后坐标 `(gen, holder, index)`） |
| `cleared_ephemerons: tuple[EphemeronClear,...]` | 键死亡（或空键且值死亡）被清空的 ephemeron 项 |
| `resolved_ephemerons` | 键可达、其值被要求进入闭包的项 |
| `rescued_values` | `resolved` 的严格子集：值原本不可达、被该 ephemeron 拉入闭包 |
| `finalized` | 本次实际执行了终结器体的事件 id（固定序） |
| `reclaimed_objects` | 本次实际回收对象数（小回收只计年轻代） |
| `edges_scanned` | 整个周期累计边扫描数（与逐步账本一致） |

## 分代引擎契约（固定语义）

### 小回收（minor）

1. 根集 = 显式根 + **记忆集**：被记录老对象的年轻强边目标是小回收根，
   保证老→年轻的边绝不漏掉年轻目标；
2. 只扫描年轻对象；被根触及的老对象作为**原子黑根**，不沿其强边递归；
3. 年轻存活者复制到年轻空闲半空间（未达晋升年龄）或晋升老年代（达到年龄/
   溢出），随后翻转年轻半空间；
4. 记忆集对**已死老持有者保守**：只被不可达老对象引用的年轻对象可能活到下一次
   大回收（标准 RS 保守性，README 明确记录而非隐含）；
5. 指向老年代的普通弱边、键在老年代的 ephemeron 项**不**被小回收清理
   （过早清理不可逆）；键在老年代的 ephemeron 在表可达时保守激活；两者都由
   大回收精确裁决。小回收仍会清理：不可达年轻持有者、年轻键死亡的 ephemeron，
   并对不可达老持有者指向已回收年轻对象的悬空槽做转发/清空（见下）。
6. 小回收不扫、不回收老年代；但年轻地址翻转后立即复用，因此提交时会重写
   **不可达老对象**的槽：移动的年轻目标转发，已回收的年轻目标清空，避免悬空
   编码地址（这些对象若靠残留 handle 复活，也不可能救回已不可达而回收的目标）。

### 大回收（major）

1. 从显式根沿强边标记全堆（两代都扫）；
2. ephemeron 表项在灰度排空后逐遍求解，直到不动点：**表与键都在标记集**时
   把值标记（值原本未标记则记入 `rescued_values`）；链式激活跨多遍完成。
   逐遍求解采用**公平轮转游标**（游标跨步存活、预算检查先于游标推进），
   保证任意小步预算下每个待决项都会在有限步内被检查，不存在预算饥饿；
   空键项的值若被回收，其值槽一并清空（防悬空），并计入
   `cleared_ephemerons`；
3. 不动点达成后才**暂存终结器事件**（不可达且带未执行终结器的对象），随后：
   对每个事件执行一次终结器体（可复活/改边/分配）→ 全部事件执行完后，
   **从根重新标记到新的不动点**（终结期间的删除生效）；
4. 不动点后才清理普通弱边与死亡键 ephemeron；
5. 老年代原地标记清扫（首适配空闲表 + 清扫后合并），年轻存活者全部晋升老年代。

### 增量更新屏障（IU）

- 每个对象在扫描其强边时对该对象当前强边取**逐对象快照**（snapshot-at-scan）；
  扫描后删除的旧边由快照保护到本轮，终结后的重标从当前堆重新裁决；
- 标记/灰/黑对象写入指向白对象的强边时，白对象立即补记为灰（IU 屏障）；
- 黑表在周期中新写入的 ephemeron 项显式登记进待决集；
- 周期中新生对象出生即标记（年轻新生为灰，其边会被扫到）。
- 错相位调用（如在 `marking` 阶段调 `finish_collection`）抛 `GCError`，
  不静默空转。

### 预算

- `mark_step(b)` 与 `finish_collection(b)` 的 `b >= 1`（否则 `ValueError`）；
- 一步内每扫一条强边或裁决一个 ephemeron 项计一次边扫描；
- **超出预算至多一条边**（计满即停，正在处理的那一条会完成）；
- 预算耗尽不是错误：返回 `StepReport(budget_exhausted=True, complete=False)`，
  调用方可继续分配/改边后再调下一步；整个周期可由任意多步组成，
  `run_collection` 只是便利驱动器。
- 容量不足且回收后仍装不下是**确定无解**（`HeapCapacityError`），与预算耗尽
  严格分开；提交前做晋升容量预检（模拟清扫后的首适配），失败时堆原样保留。

### 终结器顺序（固定）

强标记+ephemeron 不动点 → 暂存事件（gen、地址、事件 id 固定序）→
逐个执行终结器体（每个对象身份最多一次；可复活、可分配）→
从根重新标记（含 ephemeron 再求不动点）→ 弱引用/ephemeron 清理 → 提交。
复活对象在重标中不可达则本轮即回收（且不会再终结）。

## 输入与错误语义

- 全部数值形参为普通整数：`bool→TypeError`；`NaN/±Inf→ValueError`；
  其他浮点/非整数→`TypeError`。索引越界 `IndexError`；未知根名 `KeyError`；
  非法 handle（旧 epoch、未知代、空地址）`StaleHandleError`；
  容量无解 `HeapCapacityError`；错相位、终结器重注册等协议错误 `GCError`。
- 错误均带形参名与非法值；有状态操作失败不留部分变更（分配无半对象；
  回收预检失败不推进 epoch）。
- 不存在浮点计算，故无误差口径；地址、槽数、payload、扫描计数都是精确整数。
- 输出全部为固定序（排序规则见字段说明），**同一输入序列+相同步预算得到
  同一结果**；改变步预算只改变调度，不改变提交后的堆。

## v1 → v2 变更逐条说明与迁移

v1 引擎 (`Heap`) 一行未改，旧调用无需迁移。新增能力采用**独立类型
`GenerationalHeap`**（显式选项），而不是改旧类语义。若要迁移到分代模型：

1. `Heap(capacity)` → `GenerationalHeap(young_capacity, old_capacity)`；
2. `collect()` → `collect("minor"|"major", scan_budget, finalizer=...)`，
   返回类型由 `CollectionResult` 变为 `GenCollectionResult`；
3. Handle 增加 `generation` 字段；旧代码若自行解包两元组，改为访问属性；
4. 弱边语义保持；新增 ephemeron 与终结器为独立 API，不用即不生效；
5. v1 的“分配失败恰好一次全堆回收”在 v2 对应“年轻代失败先一次小回收、
   必要时再一次大回收；老年代失败一次大回收”。

## 独立账本与验证（不信任求解器自报标志）

- `tests/gen_reference.py` 是**第二套独立实现**（对象以身份 id 挂引用，
  不导入被测包，编码风格刻意不同）；`tests/gen_harness.py` 让被测引擎与
  独立实现执行**同一条操作序列**，在**每一个增量步之后**核对 marked/gray/
  ephemeron 待决集/终结器队列/扫描计数，在**每次提交后**逐对象逐槽核对存活集、
  搬迁、弱边清空、ephemeron 裁决、记忆集、根与全部边。
- 另用独立的纯强可达 BFS（`strong_reachable`，不含任何 GC 语义）做健全性
  单向检查：**任何强可达对象不得被回收**；ephemeron 闭包单独求不动点对照。
- 检查器不读取求解器的布尔标志作为正确性证据：`StepReport.complete` 只伴随
  真实提交结果出现，且一切结论由与独立实现的比对得出。
- 期望值从不通过回调被测求解器得到。

### 快速回归（`python -m unittest discover -s tests -v`）

28 个旧 v1 用例原样保留；v2 快速用例覆盖：晋升年龄、老年代清扫与空闲表复用、
记忆集（老→新强边不漏、覆写移出、老→新弱边无悬空槽）、ephemeron（值获救、
键死清空、链式不动点、键值互引、不动点先于弱清理）、终结器（复活、不重复、
终结期间分配、事件中间态可检查、重注册拒绝）、IU 屏障黑→白补记与删除边、
预算逐边不超标/零预算拒绝、旧 handle/epoch、容量边界与失败原子性、
确定性（不同预算结果相同），以及 3 对象强图全穷举。

### 扩展验证（`verification/extended_verification.py`，与快速回归分开）

- **strong5**：5 个对象、每条强边 6 种取值（空/指向 A–E）共 6⁵=7776 个图 ×
  8 种根集 × 步预算 {1,4} = **124,416** 个完整双引擎锁步大回收周期；
- **weak5**：5 个典型强图 × 5 种根集 × 每对象弱边 3 种取值（空/A/D）共
  3⁵ = 38,880 周期；
- **ephemeron**：两张表、键/值各 4 种取值（含互相指向另一张表）× 7 根集 ×
  2 预算 = 3,584 周期，覆盖键值互引与链式表；
- **finalizer**：终结器掩码 × 根集 × 复活安排 × 是否终结期间分配；
- **barrier**：4 对象 5⁴ 强图 × 根集 × 在 6 种扫描深度插入黑→白写；
- **scale**：**2,000 个对象、10,000+ 次交错更新**，小/大回收交替、
  7/64/256 的有限步预算、周期内改边、周期性删根加对象（非退化主规模例子）；
- **fuzz5**：400 个固定种子随机 5 对象场景，强弱边/ephemeron/终结器/
  周期内改边全部混合（步预算 {1,3,9}；预算 1 的交错穷举由 strong5/
  barrier 套件保证）。

## demo（90 秒内真实计算）

`python demo.py` 保留旧版三段演示（半空间搬迁+弱边清空、旧 handle 拒绝、
满堆应急回收后容量错误），并新增一段端到端 v2 例子，**同时**使用三组新能力：
年轻代复制/晋升 + 记忆集老→新边 + ephemeron（键值互引、链式激活）+
终结器复活（终结期间再分配），全部用显式小步预算推进并打印实际触发的
`HeapCapacityError`（老年代无法吸收晋升）。无 sleep、无预打印答案。

## 已知限制

- 教学模型：固定布局对象，无数组/变长对象；stop-the-world 风格的增量步
  （单线程内协作调度，无真正并发）；非线程安全。
- 记忆集对已死老持有者保守（见小回收契约第 4 条）；小回收不清理指向老年代的
  弱边/老键 ephemeron，精确裁决留给大回收。
- 终结器事件 id 为调用方给定整数；模型不做跨堆的 id 唯一性管理。
- 老年代首适配存在外部碎片（有合并空闲表，不压缩搬移老对象，符合标记清扫
  定义）。
- 不保证 Python 进程内存与“槽数”线性对应；4096 槽是逻辑接口边界。

## 仓库结构

```
moving_gc/
  __init__.py        # v1+v2 公开接口导出
  core.py            # v1：Heap/Handle/CollectionResult + Cheney（未改语义）
  gen.py             # v2：GenerationalHeap 分代增量引擎（新增）
  errors.py          # GCError/StaleHandleError/HeapCapacityError
                     #      /UnsupportedOperationError
tests/
  reference.py       # v1 独立参考（原样保留）
  gen_reference.py   # v2 独立参考实现（不导入 moving_gc，新增）
  gen_harness.py     # v2 双引擎锁步逐步账本（新增）
  test_moving_gc.py  # 28 个旧用例（原样保留）
  test_generational_gc.py  # 32 个 v2 快速用例（新增）
verification/
  extended_verification.py # 穷举/主规模/随机扩展验证（新增）
demo.py              # 旧三段 + v2 端到端段
README.md  TASK.md(旧任务留档)
```
