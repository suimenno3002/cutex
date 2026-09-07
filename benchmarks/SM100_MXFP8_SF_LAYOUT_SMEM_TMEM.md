# Blackwell MXFP8 scale factor (SFA / SFB) 的 SMEM 与 TMEM 布局

对应 kernel：`cutex/kernels/dense_gemm_v11.py`（v9/v10 结构相同，stage 数不同）。
范围：`tcgen05.mma.kind::mxf8f6f4.block_scale`，`SF_VECTOR_SIZE = 32`。

本文回答两个问题：

1. 为什么 SMEM layout 里出现了 `(32, 4)`？
2. 这个 SMEM 布局和 TMEM 上 16/32 列的布局是什么关系？

---

## 0. 先给结论

**SF atom 是"一个 scale 覆盖 32 个 K 元素"的最小单元。** 它的形状是 `32 行 x 32 K`，但物理上只占
**32 行 x 1 字节**——因为 K 方向 32 个元素共享同一个 scale（stride 0 广播）。`(32, 4)` 里的 `32` 是
**atom 的行数**（32 行 M/ N 被同一个原子读走），`4` 是**把 128 行分成 4 个这样的原子组**。

```latex
atom 形状 = ((32, 4), (32, 1)) : ((16, 4), (0, 1))
             ^ M 方向    ^ K 方向      ^ M stride  ^ K stride
       M: 32 行 x 4 组            K: 32 元素 x 1 个广播
```

所以 `(32,4)` 的含义是 **"32 行一组，共 4 组"，不是"4 行一组共 32 组"**。方向反了会导致把 16 B 行宽和
4 B 组距搞反。

---

## 1. SF atom 定义

Blackwell 硬件对 SF（scale factor）操作数只接受一种 canonical 排布，称为 `32x4x4`：

```text
sfa_atom = make_layout(
    ((32, 4), (SF_VECTOR_SIZE, 4)),          # shape
    ((16, 4), (0, 1)),                       # stride
)
```

四个模式的物理含义：

| 模式 | shape | stride | 含义 |
|---|---|---|---|
| 内层 M | 32 | 16 | 一行 M，占用 16 B（每行 4 字节，4 个 k-block） |
| 外层 M | 4 | 4 | 4 个行组，组间 4 B |
| 内层 K | 32 | **0** | 32 个 K 元素共享**同一个** scale（广播，不占字节） |
| 外层 K | 4 | 1 | 4 个 K-block 在行内连续 |

一个 atom 的物理占用是 `32 行 x 1 字节 = 32 B`，但逻辑上代表 `32 行 x 32 K 元素`（512 个 scale 的
信息被压缩到这个 32 B 里）。key 是内层 K 的 `stride=0`：**它让"32 个 K 元素"变成一个字节**。

---

## 2. SMEM 布局

`make_smem_layout_sfa(mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES)` 把 atom 平铺到一个 stage 的
`M x K = 128 x 128` tile 上（`K = MMA_TILE[2] / SF_VECTOR_SIZE`，即 `128/32 = 4` 个 K-block）。

每 stage 结果（shape/stride 已简化）：

```text
s_layout_sfa(单 stage) = (((32,4), (32,1)), 1, 4) : (((16,4), (0,0)), 0, 1)
                        ^                  ^               ^
                        M 方向 (32行 x 4组)  K 方向          K-block 连续
                        stride (16, 4)      (广播, 1)       stride 1
```

展开成字节。设 `m = M 行号 (0..127)`，`kb = K-block (0..3)`，每个 stage 有 128 行 x 4 kb = 512 B：

```text
byte(m, kb) = (m % 32) * 16 + (m // 32) * 4 + kb
```

地址位域分解：`[8:4] = m%32`（行内 0..31），`[3:2] = m//32`（行组 0..3），`[1:0] = kb`（0..3）。
这个公式的每一项都来自 atom 的一个 stride：

| 项 | 来自 | 说明 |
|---|---|---|
| `(m%32)*16` | 内层 M stride 16 | 同一行组里不同行的位置 |
| `(m//32)*4` | 外层 M stride 4 | 不同行组的位置 |
| `kb` | 外层 K stride 1 | 行内 4 个 k-block 连续排 |

**字节顺序的一个容易混淆的点**：k-block 是在**行内**连续排列的，而不是整块 k=0 放前面、k=1 放后面。
这种分布正是硬件方便随机访问的去冲突设计。让我们画一张表。**SFA（128 行 x 4 k-block，512 B）**：

| 字节地址 | byte0..3 | byte4..7 | byte8..11 | byte12..15 |
|---|---|---|---|---|
| **m=0**（行组 0） | kb0 | kb1 | kb2 | kb3 |
| **m=1** | kb0 | kb1 | kb2 | kb3 |
| ... | | | | |
| **m=31** | kb0 | kb1 | kb2 | kb3 |
| **m=32**（行组 1） | kb0 | kb1 | kb2 | kb3 |
| ... | | | | |
| **m=127** | kb0 | kb1 | kb2 | kb3 |

- 每行 4 字节，从一个 `m` 到下一个 `m`，地址跳 16 B。
- 从 `m=31` 到 `m=32`，地址跳 4 B（新行组开始）。

**SFB（256 行 N 方向，1 KiB）** 是同样结构，只是 N=256 拆成两个 128 行半块：

```text
s_layout_sfb(单 stage) = (((32,4), (32,1)), 2, 4) : (((16,4), (0,0)), 512, 1)
                                                    ^ 第二半块偏移 512 B
```

- N=0..127：字节 0..511。
- N=128..255：字节 512..1023，即 `byte += (n // 128) * 512`。

**为什么 SMEM 每 stage 是 512B / 1KiB？**

```text
SFA: 128 行 x 4 bytes = 512 B
SFB: 256 行 x 4 bytes = 1024 B
```

---

## 3. TMEM 布局

TMEM 是 `128 lane x 512 column x 32 bit`。一个 lane = 一个 M 行（本 CTA 的 128 行半块）。

`make_tmem_layout_sfa/make_tmem_layout_sfb` 生成**单份、复用**的 scale 存储：整个 K-tile 循环里，每个
k_tile 的 s2t 都会把**当前 stage** 的 scale 覆写进这同一块 TMEM。块大小：

```text
SFA: 16 列  = 4 (行组) x 4 (k-block)
SFB: 32 列  = 2 (N 半块) x 4 (行组) x 4 (k-block)
```

`16 列 x 512 B = 8192 B`，`32 列 x 512 B = 16384 B`。**有效信息量只有 512 B（SFA）/ 1 KiB（SFB），
其余 15/16 是格式占位。**

**TMEM 列图**（每个 CTA 的 512 列分配）：

```text
column:  0                        255 256  271  272   303 304    511
        +-------------------------+--------+--------+---------+
lane 0  |                          |  SFA   |  SFB   |         |
...     |  acc  (FP32, 256列)      | 16列   | 32列   | 未使用  |
lane127 |                          | 8KiB   | 16KiB  |         |
        +-------------------------+--------+--------+---------+
           256 列                   16 列     32 列    208 列
```

---

## 4. 核心问题：为什么有 `(32, 4)`，和 TMEM 的关联

**`(32, 4)` 是硬件对 M/N 方向的最小访问单位。** 数据流两个方向：

1. **TMA 从 GMEM 写 SMEM 时**，SF 也按这个 atom 的 stride-0 广播写入（VECTOR_SIZE=32 意味着每 32 个
   K 元素共享一个 scale，所以 scale 是"每行一个"，而不是"每 K 一个"）。

2. **SMEM 到 TMEM 的 s2t（`tcgen05.cp`）**，以及**MMA 从 TMEM 读 scale**，都按这个 canon 展开。
   每个 (行组, k-block) 组合在 TMEM 里占一列的某个位置。

**关键结论**：SMEM 的 `M x K = 128 x 128` 被**分成 4 个行组、4 个 K-block**，形成 4x4 的网格。
每格 = `32 行 x 32 K`，正好是 atom 的大小。所以：

```text
SMEM 的 4x4 网格
K-block →    kb0       kb1       kb2       kb3
行组 ↓
组0 (m 0..31)  atom      atom      atom      atom      -> 4 格
组1 (m 32..63) atom      atom      atom      atom      -> 4 格
组2            ...       ...       ...       ...       -> 4 格
组3            ...       ...       ...       ...       -> 4 格
             --------------------------------
             总共 4 x 4 = 16 个 atom
```

这 16 个 atom 就是 TMEM 的 **16 列**。所以：

```text
SMEM atom 网格 (4行组 x 4kblock = 16个) ──对应──> TMEM SFA 16 列
```

- 每个 TMEM 列 = 一个 (行组, k-block) 的 scale 值。
- 一列 128 lane，其中该行组的 32 个有效 scale 落在连续 32 个 lane 上（其余 lane 空/复制）。
- `mma.set(Field.SFA, t_sfa[(None,None,kblock)].iterator)`（v11:693-695）就是让 4 条 MMA 指令各自
  指向这 16 列中的一片。

**一个简单的举例：**

```text
假设 m=5, kb=2 的 scale 值 = 0x7F。

SMEM: 地址 = (5%32)*16 + (5//32)*4 + 2 = 5*16 + 0 + 2 = 82
      在 SFA 的字节 82 处。

TMEM: 这一格对应 (行组 0, kb 2)，占 TMEM 一列（第 2 列，假设 kb 是内层）。
      后续 mma 对 (m=5, k=64..95) 的操作都从这一列读这个 scale。
```

---

## 5. `(32,4)` 和 K=128 的关系

`SF_VECTOR_SIZE = 32` 让每个 scale 覆盖 32 个 K 元素，而 `MMA_TILE[2] = 128` 让一个 stage 包含
**4 个 K-block**：

```text
K-block 数 = MMA_TILE[2] / SF_VECTOR_SIZE = 128 / 32 = 4
```

- 若增大 `MMA_TILE[2]`（比如到 256），则行内 4 个 k-block 变 8 个，SMEM 每 stage 变成 1024B。
- 若缩小 `MMA_TILE[2]`（比如到 64），则只有 2 个 k-block，此时 atom 的 `(…, 4)` 就变成 `(…, 2)`。

---

## 6. 自检

单个 stage 的 TMEM 列数公式：

```text
列数 = (行数 / 32) * 4     // 4 = K-block 数，来自 SF_VECTOR_SIZE=32
SFA: (128/32) * 4 = 16 列
SFB: (256/32) * 4 = 32 列
```

与我们之前画的列图一致：`256 (acc) + 16 (SFA) + 32 (SFB) = 304 列`，与模块 docstring 的
`layout footprint = 256 accumulator + 16 SFA + 32 SFB = 304 columns` 一致。

---

## 7. 结论

1. **`(32, 4)` 代表 "32 行一组，共 4 组"。** `32` 是 atom 的行数（16 B 行宽 x 每行 4 B），`4` 是
   128 行被分成了 4 个组。
2. **atom 的 K 方向 32 个元素共享一个 scale（stride=0）**，所以单个 atom 只占 32 B 的物理空间，
   却代表 32 行 x 32 K。
3. **SMEM 和 TMEM 通过 `(行组, k-block)` 一一对应**。SMEM 是持续块（每 stage 512B / 1KiB），
   TMEM 是**固定的、被复用的** 16/32 列。

---

*注：以上 SMEM shape/stride 直接从 `dense_gemm_v11.py` 的注释（847-851 / 863-865 / 1000-1006）推导。
本机无 cutlass，无法 `print(s_layout_sfa)` 或 `print(t_sfa_layout)` 验证 TMEM 的真实列排布。建议在
Modal 上用 `cute.printf("{}", t_sfa.layout)` 核对，重点确认 `find_tmem_tensor_col_offset(t_sfa)` 返回
的列偏移是否为 16。*
