# moving_gc — 带弱引用与弱键关联的移动式垃圾回收内核（可复用教学模块）

一个从零实现、仅依赖标准库的**半空间（semi-space）移动式垃圾回收器**模拟内核，
支持强边 / 弱边、**弱键关联（weak-key association）**、整数 payload、显式根、
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

# 固定输入演示（实测约 0.04 秒，远小于 90 秒；无 sleep、无预录输出）
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
| `roots_preserved: int` | 非空（被更新）的显式根数量 |
| `surviving_associations: tuple[(table_addr, key_addr, value_addr), ...]` | 回收后仍存活的弱键条目，全部用**回收后**地址，按 `(表地址, 键地址)` 排序 |
| `cleared_associations: tuple[(table_addr, key_addr), ...]` | 因键死亡而被删除的条目，用**回收前**地址（死键没有回收后地址，整对停留在同一地址空间，表地址可经 `relocated` 换算），按 `(表地址, 键地址)` 排序；**表本身死亡**的条目不报入，随表一起消失 |

### `Heap(capacity: int)`

半空间容量以**槽**计，`1 <= capacity <= 4096`（`MAX_CAPACITY`）。

| 方法 | 语义 |
|---|---|
| `allocate(payload=0, *, n_strong=0, n_weak=0) -> Handle` | 分配对象；对象占 `2+n_strong+n_weak` 槽（头、payload、强边槽、弱边槽）。只要当前装不下就**先自动收集恰好一次**，再检查；仍装不下（含对象本身大于整个半空间）抛 `HeapCapacityError`。失败绝不留下半对象 |
| `add_root(name, handle)` | 注册/替换具名显式根（先校验 handle，再替换，非法 handle 不会覆盖旧根） |
| `clear_root(name)` / `remove_root(name)` | 根置空（保留名字）/ 彻底注销；未知名字抛 `KeyError` |
| `root_handle(name) -> Handle | None` | 取根的当前 handle，空根返回 `None` |
| `roots() -> dict[str, Handle|None]` | 全部根快照 |
| `get_strong/set_strong(h, i, ref|None)` | 强边读写；`None` 表示空边 |
| `get_weak/set_weak(h, i, ref|None)` | 弱边读写。**弱边不保持目标存活**；回收后：存活目标被改址，死亡目标变 `None`（显式空边与被清空的弱边不可区分，符合标准弱引用语义） |
| `association_put(table, key, value|None)` | 写入/替换一条**弱键关联**（旁路元数据，不是对象槽位，因此表既不强引用也不弱引用键、值）；`value=None` 删除该键条目。三个 handle 先全部校验再变更，非法 handle 不会产生半条目 |
| `association_get(table, key) -> Handle|None` | 读取当前值；无条目（从未写入、被显式删除、或因表/键死亡被回收删除）返回 `None` |
| `association_remove(table, key)` | 删除一条；键不存在为 no-op |
| `association_entries(table) -> tuple[(key_h, value_h), ...]` | 该表现存活条目的快照，按键地址升序，handle 均为当前 epoch |
| `payload(h)` / `set_payload(h, v)` | 整数 payload 读写 |
| `object_shape(h) -> (n_strong, n_weak)` | 对象边数 |
| `handle(address) -> Handle` | 为当前存活于该地址的对象签发 handle；无对象抛 `StaleHandleError` |
| `collect() -> CollectionResult` | 执行一次回收（见下） |
| `capacity / epoch / used_slots / free_slots` | 只读属性 |
| `objects() -> tuple[int,...]` | 当前全部对象头地址 |
| `reachable_addresses() -> frozenset[int]` | 只读的独立强可达遍历（诊断/测试用，不复制） |

## 回收算法（契约）

### 级别 1：普通强边闭包

1. **只从显式根出发、只沿强边**计算可达性；弱边永不追踪。
2. **Cheney 半空间复制**，顺序确定：
   - 根阶段：按根注册顺序，把每个非空根对象复制进空闲半空间；
   - 扫描阶段：按复制队列（目标地址升序）扫描存活对象，强边按**字段顺序**
     复制未迁移目标（共享子图只复制一次）。

### 级别 2：弱键关联激活（最小不动点）

3. 仅**表对象已存活**（已被复制进目标空间）的条目参与；死表的所有条目
   随表消失，不参与、也不报入清空列表。
4. 反复执行激活趟，直到某一趟不再增长：
   - 参与表按其**回收后地址升序**访问，表内条目按**回收前键地址升序**；
   - 对尚未激活的条目 `table[key]=value`，**先判定 `key` 是否已被复制**：
     是则激活——把 `value` 当作一条额外强根复制，并立即排空 Cheney 扫描队列
     （因此新复制的值能沿强边救活其他对象，包括其他条目的键乃至别的表）；
   - 一趟结束后若有任何新激活，再来一趟。
5. **激活判定先于值复制**，所以“键只被自己条目的值反向引用”的环无法凭空
   自举：判定时该值还未复制、键不在已复制集合中，条目不激活，键值一起回收，
   死键条目删除。这正是对象 ID 图上的最小不动点：
   `X0 = 强闭包(根)`；
   `X_{n+1} = 强闭包(X_n ∪ { value | table,key ∈ X_n，table[key]=value })`。
   值又能使别的键存活，因此**一次扫描不够**，必须迭代到不动点。

### 转发与清理

6. **转发放置**：根、全部强边、存活的弱键条目、以及指向存活目标的普通弱边
   改写到新地址；指向死亡目标的普通弱边清空为 `None`；死表/死键的关联删除。
   被激活值“顺带救活”的普通弱边目标也按存活转发。
7. 成功后翻转活动半空间、压缩存活对象到低端、epoch +1。
8. 无 finalizer。

复制期间使用独立的转发表字典（对象头槽在本模型中保留但不存转发指针）。
回收是**全有或全无**的：若存活集在复制途中确认装不进半空间（在源空间本身
合法的前提下不会发生，仍做防御），清空已部分写入的目标空间并抛
`HeapCapacityError`，堆继续从原半空间提供服务，可观察状态无部分变更
（关联元数据、根、epoch 均不变）。

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

- `capacity ∈ [1, 4096]` 槽（每个半空间）；`n_strong, n_weak ≥ 0` 且
  `2+n_strong+n_weak ≤ capacity`；payload 为任意 Python 整数（无界）。
- 根名字为非空 `str`；根/对象数量受容量隐含限制。
- 4096 是**接口边界**；测试中的满堆用例会触达该边界，短演示不穷举上限规模。

## 设计取舍

- **槽级扁平堆模型**：两个长度 `capacity` 的列表即两个半空间，槽值为
  整数（payload/地址）或 `None`（空槽/空边）；对象布局固定，无变长对象。
- **旁路转发表**而非头槽内转发指针：槽格式统一、教学直观，代价是回收期
  额外 O(存活对象) 字典内存。
- **epoch 作代际代号**：每次成功回收 +1。旧 epoch handle 一律拒绝，
  天然覆盖“地址巧合相同”的陈旧引用；无单独的 remembered set / 卡表。
- **弱边在强闭包复制完成后一次性修正**，顺序确定（持有者新地址升序、
  字段号升序），因此清空列表可直接作为输出回归对照。
- **弱键关联存在旁路元数据中**（`表地址 -> {键地址: 值地址}`），不是对象槽
  位，因此不会被误当成强边或普通弱边；表对象死亡即整组丢弃。其活性规则
  （表与键都独立存活时值才强可达）用“每趟重扫 + 激活即复制值并排空 Cheney”
  实现为最小不动点；参与顺序确定（表按新地址、条目按旧键地址），故存活/清空
  条目列表同样可作回归对照。
- **关联以对象身份（地址）为键**：无哈希、无等价比较，一个键对象对应至多一个
  值；重复写入替换旧值；`value=None` 即删除。它对应运行时教材中的
  ephemeron / weak-key table 概念，但只实现本题所需的最小语义子集。
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
- 弱边清空后无法区分“原本为空”与“被 GC 清空”；无弱引用通知队列、
  无 finalizer、无软/虚引用、无弱根。
- 弱键关联只实现**表+键都存活时值才强**这一条规则：无哈希/等值语义、
  无键枚举通知、无移除回调、无弱值（weak-value）模式、不模拟 Python
  `WeakValueDictionary` 的完整行为；表对象只是普通堆对象，关联容量受
  逻辑堆可容纳对象数隐含限制。
- 不动点激活在最坏情况下每趟至少激活一个条目，趟数与“值→键”链条长度同阶；
  本模型无增量/并发安排（教学模型，4096 槽逻辑边界）。
- 句柄需由调用者在每次回收后主动重新获取（如经根），不提供自动更新的
  间接句柄表。
- 非线程安全（单线程教学模型）。
- 不保证 Python 进程内存与“槽数”线性对应；4096 槽是模拟堆的逻辑边界。

## 仓库结构

```
moving_gc/
  __init__.py        # 公开接口导出
  core.py            # Heap / Handle / CollectionResult + Cheney 回收 + 弱键不动点激活
  errors.py          # GCError / StaleHandleError / HeapCapacityError
tests/
  reference.py       # 独立参考模型（不导入 moving_gc）：对象 ID 图上的最小不动点
  test_moving_gc.py  # 44 个用例：场景对照、弱键关联、边界、输入校验、种子化差分 fuzz
demo.py              # 固定输入演示：搬迁 + 弱边清空 + 两级激活链/键值环 + 两类错误边界
README.md
```

## 测试覆盖摘要

- 与独立参考实现对照：自环、无根环（全死）、仅弱引用（清空）、多弱边同死
  目标、共享子图（只复制一次）、弱边指向存活者（改址）、根/字段复制顺序、
  强弱混合菱形。
- **弱键关联**：表+键都存活值才强、死键条目删除且值回收、死表条目不参与、
  两级激活链（值强达下一键，一趟不够）、激活值救活其他表、键值环无外部根
  不能自举、互指条目的全死/级联两情形、两条目共享值只复制一次、普通弱边
  指向被激活值时转发、表自身作键、键值同体、条目跨两次/三次回收转发、
  两轮回收间键根被撤后条目删除、分配压力触发的应急回收自动激活、激活后
  容量不足的分配失败无半对象、关联 put/get/remove/校验与陈旧 handle 拒绝。
- 边界：4096 槽刚好满堆回收后再分配失败、小堆满后失败无半对象、单对象超容、
  分配失败触发回收后成功、容量构造边界。
- 旧 handle：epoch 不符即拒绝（含地址恰好相同、伪造当前 epoch 空地址）、
  旧 handle 在全部读写入口（含关联 API）被拒、失败写不改边。
- 输入校验：`bool`/NaN/Infinity/浮点/字符串、负边数、索引越界、根名非法、
  原子性（失败根替换保留旧根、空根/注销）。
- 12 组固定种子随机图的两轮回收差分测试（含随机弱键条目、回收间删根、
  新分配、改边）；参考模型在每次预测时用对象 ID 图上的直接最小不动点
  自检其复制式存活集。
