# gc_kernel — 带弱引用的移动式垃圾回收内核

离线运行时系统工程可复用模块：模拟半空间（semi-space）复制式垃圾回收，
对象带固定数量强边、弱边与整数 payload，外部引用为 (地址, epoch) 句柄。

- **Python 版本**：3.14.7（仅标准库，无第三方依赖，无需安装任何包）
- **平台**：Linux 原生；核心、测试、演示全部离线
- **输入上限**：堆容量 ≤ `MAX_SLOTS = 4096` 槽，每槽一个对象

## 运行

```bash
python -m unittest discover -s tests -v   # 必要测试（含独立可达性对照）
python demo.py                            # 固定输入演示，约 0.1 秒内完成
```

## 接口（`gc_kernel.py`）

### 数据结构

- `Handle(address, epoch)`：冻结 dataclass，外部对象引用。
- `CollectionResult(moved, live_slots, freed_slots, roots, epoch)`：
  `moved` 为搬迁映射 {旧地址: 新地址}（仅含存活对象），`roots` 为搬迁后的
  已注册根（按注册顺序），`epoch` 为翻转后的新纪元。

### `Heap(capacity, strong_arity=1, weak_arity=1)`

- `alloc(payload=0, strong=None, weak=None) -> Handle`
  分配一个对象。`strong`/`weak` 为长度恰等于固定元数的 `Handle|None` 序列。
  空间不足时**先执行一次回收**（已注册根 + 待装入强边目标作为根；弱边目标
  **不**续命），仍不足则抛 `HeapFullError`，不留下半对象。
- `collect() -> CollectionResult`
  仅从已注册根沿强边追踪；按根顺序、再按字段顺序广度优先复制（Cheney）；
  用转发表修正根、强边及目标存活的弱边；目标已死的弱边清空为 `None`。
- 根管理：`add_root(handle)`、`clear_roots()`、`roots()`。
- 字段访问：`read_payload` / `write_payload` / `read_strong` / `write_strong`
  / `read_weak` / `write_weak` / `describe(handle)`。
- 属性：`capacity`、`strong_arity`、`weak_arity`、`epoch`、`live_slots`。

### 错误语义（均可定位，不静默纠正）

- `GCError`：所有错误的基类。
- `InvalidArgumentError(GCError, ValueError)`：非法配置/payload/字段下标。
  整数参数拒绝 `bool`；非整数值（含 `float('nan')`、`float('inf')`）一律拒绝。
- `InvalidHandleError(GCError, ValueError)`：句柄类型错误、字段非法、地址越界。
- `StaleHandleError(InvalidHandleError)`：epoch 与当前堆纪元不一致的旧句柄
  （含伪造的未来 epoch），所有读写/注册/分配路径均拒绝。
- `HeapFullError(GCError)`：一次完整回收后容量仍不足；失败分配不产生对象。

## 设计取舍

- **每槽一对象**：容量以槽计，对象定长（固定元数 + 一个整数 payload），
  地址即槽下标；epoch 区分翻转前后的同名地址。
- **Cheney 复制**：先按注册顺序疏散根，再按扫描指针顺序扫描强字段，
  保证“按根和字段顺序复制”的确定性布局；共享子图因转发表去重只复制一次。
- **弱边不续命**：追踪阶段完全忽略弱边；追踪结束后统一修正或清空。
- **分配触发回收时的待装入边**：强边目标临时作为额外根（否则会产生悬垂
  强边），弱边目标不续命、死亡即清空——与常规 GC 语义一致。
- **失败语义**：`HeapFullError` 前的那次回收是已说明的状态变更（纪元 +1、
  已注册根被搬迁、可通过 `roots()` 取回），但不产生任何新对象。
- **无 finalizer**，按题面契约不实现。

## 已知限制

- 单线程、单堆；无并发、无分代、无增量回收。
- payload 仅支持 Python 整数（拒绝 bool 与一切非整数值）；无浮点结果，
  因此不涉及浮点误差口径——所有输出（地址、槽数、映射）均为精确整数。
- 句柄可被用户自行构造；模块能检测类型/越界/纪元错误，但无法防御
  “同一 epoch 内恰好指向他对象的伪造句柄”（与真实运行时一致，句柄应由
  运行时签发而非用户伪造）。
- 上限 4096 槽为接口边界；测试与演示使用小规模确定性数据，未穷举上限规模。

## 测试说明

`tests/test_gc_kernel.py` 内置**独立参考实现** `reference_collect`：纯 Python
的可达性/复制顺序模拟，只作用于测试自建的图描述，不调用被测核心得到期望值。
覆盖：自环、仅弱引用（死亡并清空）、弱边指向存活对象（重映射）、共享子图
（只复制一次）、复制顺序、刚好满堆（拒绝且无半对象）、分配触发回收、
旧句柄拒绝、非法输入（bool/NaN/Infinity/越界/畸形句柄）、以及 60 对象
确定性伪随机图的全字段对照。
