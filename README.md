# moving_gc — 带弱引用与弱键关联的移动式垃圾回收内核（可复用教学模块）

一个从零实现、仅依赖标准库的**半空间（semi-space）移动式垃圾回收器**模拟内核，
支持强边 / 弱边 / **弱键条目（ephemeron 式键弱关联）**、整数 payload、显式根、
基于 epoch 的外部 handle 失效检测。
它是一个有限契约的教学/实验模型，**不宣称**与任何行业标准运行时（JVM、.NET、
V8 等）的内存管理兼容。

## Python 版本与依赖

- Python **3.14.7**（Linux 原生验证；3.10+ 语法即可运行，目标版本为 3.14.7）
- **仅标准库**，无需 `pip install` 任何第三方包
- 完全离线：不访问网络、账号、密钥、数据库、中间件、云 API、Docker 或硬件
- 示例数据全部由仓库内确定性代码生成（固定图 + 固定随机种子的差分测试）

## 运行命令

```bash
# 必要测试（unittest 标准发现）
python -m unittest discover -s tests -v

# 固定输入演示（实测约 0.03 秒，远小于 8 秒；无 sleep、无预录输出）
python demo.py
```

## 公开数据结构与接口

```python
from moving_gc import (
    Heap, Handle, CollectionResult,
    GCError, StaleHandleError, HeapCapacityError,
    MAX_CAPACITY,   # 4096
)
```

### `Handle(address: int, epoch: int)`（frozen dataclass）

外部对象引用。`address` 是对象头所在槽号（0 基），`epoch` 是签发时堆的
epoch。handle 不可变；每次成功 `collect()` 后堆 epoch +1，**所有旧 handle
立即失效**，即使对象搬迁后恰好落在同一槽号（地址相同也拒绝）。

### `CollectionResult`（frozen dataclass）

| 字段 | 含义 |
|---|---|
| `relocated: dict[int,int]` | 搬迁映射 `旧地址 -> 新地址`，每个存活对象一条 |
| `live_slots: int` | 回收后活动半空间中存活对象占用的槽数（= 新 bump 指针） |
| `cleared_weak_edges: tuple[(holder_addr, field_idx), ...]` | 目标已死、被清空为 `None` 的弱边；holder 地址为**回收后**地址，按持有者地址、字段号排序 |
| `cleared_entries: tuple[(table_addr, entry_idx), ...]` | 键在不动点处仍不可达、被删除（键值两槽均置 `None`）的弱键条目；表地址为**回收后**地址，按表地址、条目号排序；死表的条目不出现（随表一起消失） |
| `roots_preserved: int` | 非空（被更新）的显式根数量 |

### `Heap(capacity: int)`

半空间容量以**槽**计，`1 <= capacity <= 4096`（`MAX_CAPACITY`）。

| 方法 | 语义 |
|---|---|
| `allocate(payload=0, *, n_strong=0, n_weak=0, n_entries=0) -> Handle` | 分配对象；对象占 `2+n_strong+n_weak+2*n_entries` 槽（头、payload、强边槽、弱边槽、每条目键值两槽）。只要当前装不下就**先自动收集恰好一次**，再检查；仍装不下（含对象本身大于整个半空间）抛 `HeapCapacityError`。失败绝不留下半对象 |
| `add_root(name, handle)` | 注册/替换具名显式根（先校验 handle，再替换，非法 handle 不会覆盖旧根） |
| `clear_root(name)` / `remove_root(name)` | 根置空（保留名字）/ 彻底注销；未知名字抛 `KeyError` |
| `root_handle(name) -> Handle | None` | 取根的当前 handle，空根返回 `None` |
| `roots() -> dict[str, Handle|None]` | 全部根快照 |
| `get_strong/set_strong(h, i, ref|None)` | 强边读写；`None` 表示空边 |
| `get_weak/set_weak(h, i, ref|None)` | 弱边读写。**弱边不保持目标存活**；回收后：存活目标被改址，死亡目标变 `None`（显式空边与被清空的弱边不可区分，符合标准弱引用语义） |
| `set_entry(h, i, key|None, value|None)` | 写入弱键条目 `i` 的键值对。键为**弱持有**（条目永不使键存活）；值为**条件强持有**（回收时仅当表与键均已存活才作为强边复制）。`set_entry(h, i, None)` 置空条目；`key=None` 搭配非空 `value` 抛 `ValueError`（该条目永不可能激活） |
| `get_entry_key(h, i)` / `get_entry_value(h, i)` | 读条目键/值；空条目、无值条目与被回收删除的条目都返回 `None`（三者不可区分，与弱引用语义一致） |
| `entry_count(h) -> int` | 对象的弱键条目数 |
| `payload(h)` / `set_payload(h, v)` | 整数 payload 读写 |
| `object_shape(h) -> (n_strong, n_weak)` | 对象边数（不含条目数，条目数用 `entry_count`） |
| `handle(address) -> Handle` | 为当前存活于该地址的对象签发 handle；无对象抛 `StaleHandleError` |
| `collect() -> CollectionResult` | 执行一次回收（见下） |
| `capacity / epoch / used_slots / free_slots` | 只读属性 |
| `objects() -> tuple[int,...]` | 当前全部对象头地址 |
| `reachable_addresses() -> frozenset[int]` | 只读的独立强可达遍历（诊断/测试用，不复制） |

## 回收算法（契约）

1. **只从显式根出发、只沿强边**计算初始可达闭包；弱边与弱键条目在此阶段
   均不追踪。
2. **弱键条目不动点**：反复执行直到不再增长——对每个**已存活表**的每个
   条目，若其**键已存活**则激活该条目，把值当作强边复制并继续 Cheney 扫描。
   值可能让其他条目的键（以及其他表）存活，因此单遍扫描不够，必须迭代到
   最小不动点。键自身从不因条目而被标记，所以**只被自己的值反向引用的键
   不能凭空存活**（键值环无外部根时整体回收）。
3. **死键条目删除**：不动点结束后，存活表中键仍不可达的条目被删除
   （键、值两槽均置 `None`，记入 `cleared_entries`）；死表的条目随表消失，
   不单独报告。
4. **Cheney 半空间复制**，顺序确定：
   - 根阶段：按根注册顺序，把每个非空根对象复制进空闲半空间；
   - 扫描阶段：按复制队列（目标地址升序）扫描存活对象，强边按**字段顺序**
     复制未迁移目标（共享子图只复制一次）；
   - 条目激活阶段：按表复制顺序、条目号顺序激活，值插入复制队尾。
5. **转发放置**：根、全部强边、已激活条目的键与值、以及指向存活目标的
   弱边改写到新地址；指向死亡目标的弱边清空为 `None`。
6. 成功后翻转活动半空间、压缩存活对象到低端、epoch +1。
7. 无 finalizer。

复制期间使用独立的转发表字典（对象头槽在本模型中保留但不存转发指针）。
回收是**全有或全无**的：若存活集在复制途中确认装不进半空间（在源空间本身
合法的前提下不会发生，仍做防御），清空已部分写入的目标空间并抛
`HeapCapacityError`，堆继续从原半空间提供服务，可观察状态无部分变更。

## 输入与错误语义（先固定的契约）

- 所有数值形参都是**整数**：
  - `bool`（Python 中是 `int` 子类）→ `TypeError`；
  - 浮点 `NaN` / `+Infinity` / `-Infinity` → `ValueError`（显式拒绝）；
  - 有限浮点（如 `1.5`）及其余非整数类型 → `TypeError`。
  - 本模块**不存在浮点计算结果**，因此没有误差/容差口径：所有输出
    （地址、槽数、payload、搬迁映射）均为精确整数，不存在舍入。
- 范围/索引错误：容量越界、负边数 → `ValueError`；字段索引越界 →
  `IndexError`；未知根名 → `KeyError`。错误信息带形参名与非法值，可定位，
  绝不静默纠正。
- 有状态操作的原子性：非法 handle 的字段写、根替换在修改前即失败，
  不留部分变更；分配失败无半对象；回收失败不留半空间残骸。
- 错误类型层级：`GCError` 为基类，派生 `StaleHandleError`、
  `HeapCapacityError`。
- “预算耗尽 vs 无解”在本题的对应物：分配时**先收集一次再判定**——
  回收后仍装不下是**确定无解**（`HeapCapacityError`，消息给出所需与剩余
  槽数），不存在可重试的搜索预算概念。

## 输入上限

- `capacity ∈ [1, 4096]` 槽（每个半空间）；`n_strong, n_weak, n_entries ≥ 0` 且
  `2+n_strong+n_weak+2*n_entries ≤ capacity`；payload 为任意 Python 整数（无界）。
- 根名字为非空 `str`；根/对象数量受容量隐含限制。
- 4096 是**接口边界**；测试中的满堆用例会触达该边界，短演示不穷举上限规模。

## 设计取舍

- **槽级扁平堆模型**：两个长度 `capacity` 的列表即两个半空间，槽值为
  整数（payload/地址）或 `None`（空槽/空边）；对象布局固定，无变长对象。
- **旁路转发表**而非头槽内转发指针：槽格式统一、教学直观，代价是回收期
  额外 O(存活对象) 字典内存。
- **epoch 作代际代号**：每次成功回收 +1。旧 epoch handle 一律拒绝，
  天然覆盖“地址巧合相同”的陈旧引用；无单独的 remembered set / 卡表。
- **弱边在强闭包与条目不动点完成后一次性修正**，顺序确定（持有者新地址升序、
  字段号升序），因此清空列表可直接作为输出回归对照。
- **弱键条目用"激活集合 + 迭代到不动点"实现**：已激活条目记入旁路集合，
  每轮按表复制顺序扫描存活表、激活键已存活的条目并排空 Cheney 队列，
  直到一轮无任何激活。键槽只在激活时改写一次，因此旧/新地址不会混淆；
  该不动点是**最小**不动点（只从已存活对象出发扩张），键值环无法自举。
- **自动应急回收仅一次**：`allocate` 空间不足时触发恰好一次 `collect()`，
  不循环重试，避免“分配触发多次 GC”的不可预测停顿。
- **独立参考实现**（`tests/reference.py`）不导入被测包，用自带的命名图 +
  BFS 可达性 + 独立 Cheney 模拟产出期望值；往返一致只作补充断言。

## 已知限制

- 固定布局对象，无数组/变长对象、无对象头类型信息（GC 不依赖类型，布局
  由旁路 `_layout` 字典维护）。
- 单“代”、stop-the-world、非增量、非并发、非并行；无写屏障、read barrier、
   remembered set；内部槽访问在真实实现中应是裸内存读写，本模型为
  Python 列表，访问是 O(1) 但有解释器开销。
- 弱边清空后无法区分”原本为空”与”被 GC 清空”；条目同理（空条目、无值条目、
  被删除条目读后均为 `None`）；无弱引用通知队列、无条目删除通知、
  无 finalizer、无软/虚引用、无弱根。
- 条目是定长槽对而非开放寻址哈希表：无按键查找、无去重、无自动扩容；
  同一键可出现在多条目中，语义各自独立。
- `reachable_addresses()` 仍是**仅强边**的只读遍历，不含条目激活闭包。
- 句柄需由调用者在每次回收后主动重新获取（如经根），不提供自动更新的
  间接句柄表。
- 非线程安全（单线程教学模型）。
- 不保证 Python 进程内存与“槽数”线性对应；4096 槽是模拟堆的逻辑边界。

## 仓库结构

```
moving_gc/
  __init__.py        # 公开接口导出
  core.py            # Heap / Handle / CollectionResult + Cheney 回收
  errors.py          # GCError / StaleHandleError / HeapCapacityError
tests/
  reference.py       # 独立参考模型（不导入 moving_gc）：命名图 + 强边 BFS
                     # + 条目最小不动点 + 独立 Cheney 模拟
  test_moving_gc.py  # 37 个用例：场景对照、弱键条目、边界、输入校验、种子化差分 fuzz
demo.py              # 固定输入演示：搬迁 + 弱边清空 + 弱键条目不动点 + 两类实际触发的错误
README.md
```

## 测试覆盖摘要

- 与独立参考实现对照：自环、无根环（全死）、仅弱引用（清空）、多弱边同死
  目标、共享子图（只复制一次）、弱边指向存活者（改址）、根/字段复制顺序、
  强弱混合菱形。
- 弱键条目：两级激活链（值让下一级的键存活，单遍扫描会漏）、键值环无外部根
  （整体回收、条目删除）、表本身死亡（条目不激活、值随表回收）、键死值死、
  键仅被其他条目的值救活、条目不占强边身份、满堆下条目保活导致容量不足、
  条目 API 契约与陈旧 handle 拒绝。
- 边界：4096 槽刚好满堆回收后再分配失败、小堆满后失败无半对象、单对象超容、
  分配失败触发回收后成功、容量构造边界。
- 旧 handle：epoch 不符即拒绝（含地址恰好相同、伪造当前 epoch 空地址）、
  旧 handle 在全部读写入口被拒、失败写不改边。
- 输入校验：`bool`/NaN/Infinity/浮点/字符串、负边数、索引越界、根名非法、
  原子性（失败根替换保留旧根、空根/注销）。
- 12 组固定种子随机图的两轮回收差分测试（含回收间删根、新分配、改边，
  图内随机混入弱键条目，与参考模型的不动点结果逐字段对照）。
