# moving_gc — 带弱引用的移动式垃圾回收内核（可复用教学模块）

一个从零实现、仅依赖标准库的**半空间（semi-space）移动式垃圾回收器**模拟内核，
支持强边 / 弱边、整数 payload、显式根、基于 epoch 的外部 handle 失效检测。
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
    WeakClearNotification,
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

### `WeakClearNotification`（frozen dataclass）

| 字段 | 含义 |
|---|---|
| `holder: Handle` | 被清空弱字段所属对象在**成功回收后**的新 handle |
| `index: int` | 被清空的弱字段编号 |

通知只记录持有者和字段，不记录死亡目标地址，因此通知自身不可能通过该记录
重新保活目标；`holder` 也只是普通 handle，不会注册为根。若持有者之后死亡，
已领取通知中的 handle 会按正常 epoch 规则变陈旧。

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
| `payload(h)` / `set_payload(h, v)` | 整数 payload 读写 |
| `object_shape(h) -> (n_strong, n_weak)` | 对象边数 |
| `handle(address) -> Handle` | 为当前存活于该地址的对象签发 handle；无对象抛 `StaleHandleError` |
| `collect() -> CollectionResult` | 执行一次回收（见下） |
| `claim_weak_clear_notifications() -> tuple[WeakClearNotification, ...]` | 一次性领取并清空全部待领弱引用清空通知；无待领记录时返回空元组 |
| `capacity / epoch / used_slots / free_slots` | 只读属性 |
| `pending_weak_clear_notifications: int` | 当前尚未领取的弱引用清空通知数量 |
| `objects() -> tuple[int,...]` | 当前全部对象头地址 |
| `reachable_addresses() -> frozenset[int]` | 只读的独立强可达遍历（诊断/测试用，不复制） |

## 回收算法（契约）

1. **只从显式根出发、只沿强边**计算可达性；弱边永不追踪。
2. **Cheney 半空间复制**，顺序确定：
   - 根阶段：按根注册顺序，把每个非空根对象复制进空闲半空间；
   - 扫描阶段：按复制队列（目标地址升序）扫描存活对象，强边按**字段顺序**
     复制未迁移目标（共享子图只复制一次）。
3. **转发放置**：根、全部强边、以及指向存活目标的弱边改写到新地址；
   指向死亡目标的弱边清空为 `None`。
4. 成功后翻转活动半空间、压缩存活对象到低端、epoch +1。
5. 每次真正清空弱边时发布一条可领取通知；显式空边、指向存活者并被转发的
   弱边不发布。通知按成功回收批次、持有者新地址、字段号组成 FIFO 顺序，
   领取后移除；字段之后被重新设置时，可在下一次真正清空时再次通知。
6. 通知只在回收成功提交后发布；回收失败不会发布部分通知。
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
- **通知队列独立于堆图**：队列只保存持有者的不可变 handle/字段编号，不把
  任何通知对象加入根集合，也不保存死亡目标；因此领取前保存通知不会影响
  GC 可达性。
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
- 直接读取弱边仍只返回 `Handle | None`，不会因为存在通知而改变弱边值；
  历史“原本为空还是被清空”的区分需调用者及时领取通知。无 finalizer、
  无软/虚引用、无弱根。
- 句柄需由调用者在每次回收后主动重新获取（如经根），不提供自动更新的
  间接句柄表。
- 非线程安全（单线程教学模型）。
- 不保证 Python 进程内存与“槽数”线性对应；4096 槽是模拟堆的逻辑边界。

## 仓库结构

```
moving_gc/
  __init__.py        # 公开接口导出
  core.py            # Heap / Handle / CollectionResult + 弱引用清空通知
  errors.py          # GCError / StaleHandleError / HeapCapacityError
tests/
  reference.py       # 独立参考模型（不导入 moving_gc）
  test_moving_gc.py  # 34 个用例：场景对照、通知、边界、输入校验、fuzz
demo.py              # 固定输入演示：搬迁 + 弱边清空 + 两类实际触发的错误
README.md
```

## 测试覆盖摘要

- 与独立参考实现对照：自环、无根环（全死）、仅弱引用（清空）、多弱边同死
  目标、共享子图（只复制一次）、弱边指向存活者（改址）、根/字段复制顺序、
  强弱混合菱形。
- 弱引用通知：领取即移除、空边和存活转发不通知、重新设置后再次通知、
  FIFO 跨回收顺序、通知不保活对象、失败回收不发布。
- 边界：4096 槽刚好满堆回收后再分配失败、小堆满后失败无半对象、单对象超容、
  分配失败触发回收后成功、容量构造边界。
- 旧 handle：epoch 不符即拒绝（含地址恰好相同、伪造当前 epoch 空地址）、
  旧 handle 在全部读写入口被拒、失败写不改边。
- 输入校验：`bool`/NaN/Infinity/浮点/字符串、负边数、索引越界、根名非法、
  原子性（失败根替换保留旧根、空根/注销）。
- 12 组固定种子随机图的两轮回收差分测试（含回收间删根、新分配、改边）。
