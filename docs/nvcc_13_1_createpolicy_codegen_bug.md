# nvcc 13.1 ptxas codegen bug: createpolicy.fractional dropped on sm_90

## 概要

CUDA Toolkit 13.1（`nvcc V13.1.80`，build `cuda_13.1.r13.1/compiler.36836380_0`）的 ptxas 在编译特定的 sm_90 kernel 时，会**丢弃 `createpolicy.fractional.L2::evict_first.b64` 这条 PTX 指令**，但保留依赖它结果的 `cp.async.cg.shared.global.L2::cache_hint`。生成的 SASS 里，`LDGSTS` 用一个**未被赋值的 uniform 寄存器**作为 L2 cache descriptor，硬件解码 desc 时 trap 成 `cudaErrorIllegalInstruction` (CUDA error 716)。

## 影响

- **CUDA 版本**：CUDA 13.1（其他 13.x / 12.x 未实测）
- **Driver**：实测 550.127.08 复现，未做版本边界验证
- **GPU**：H20（sm_90）实测；其他 Hopper SKU（H100/H200）原理上同样受影响，未独立验证
- **触发**：使用 `cp.async.cg.shared.global.L2::cache_hint` 的 inline PTX，且 kernel 结构复杂到一定程度（多 stage 软件流水 + 多路独立 cp.async）。**仅在 cp.async 的 cache_hint 变体上触发；普通 cp.async (`cp_async4_pred`) 不受影响**。

## 症状

复现路径：[`marlin/marlin_cuda_kernel.cu`](../marlin/marlin_cuda_kernel.cu) 中 `Marlin<256,1,8,8,4,8>` 这个特化。

```text
cudaErrorIllegalInstruction (error 716) at kernel launch
========= Illegal instruction
=========     at void Marlin<256,1,8,8,4,8>(...)+0x1180
=========     by thread (131,0,0) in block (45,0,0)
========= ERROR SUMMARY: 2371 errors
```

关键字眼：是 **`Illegal instruction`**，不是 `Misaligned address` 也不是 `Illegal address`。

## 根因（确认链）

### 第一步：PTX 阶段是对的

把 `marlin_cuda_kernel.cu` 用 nvcc 13.1 单独编 PTX：

```bash
nvcc -arch=sm_90 -O3 -ptx marlin_cuda_kernel.cu -o marlin.ptx
grep -c createpolicy marlin.ptx
# → 239
```

`createpolicy.fractional.L2::evict_first.b64` 在 PTX 中出现 239 次（每个使用 `cp_async4_stream` 的位置都展开了一次，因为是 `__device__ inline`）。这说明 **C++ → PTX 阶段没问题**。

### 第二步：SASS 阶段 createpolicy 被丢

编出 cubin 后反汇编：

```bash
cuobjdump --dump-sass --function _Z6MarlinILi256ELi1ELi8ELi8ELi4ELi8EEvPK4int4S2_PS0_S2_iiiPi \
    marlin_cuda.so > /tmp/fail.sass
```

LDGSTS 里 desc 操作数指向的寄存器分布：

```text
14 desc[UR12]   ← cp_async4_pred (A buffer), 没有 cache_hint
30 desc[UR1]    ← cp_async4_stream (B + scales), 带 cache_hint
```

`UR1` 在 kernel 全文出现 **30 次**，全都是被 LDGSTS 当 desc **读取**：

```bash
grep -nE 'UR1[^0-9]' /tmp/fail.sass | wc -l         # 30
grep -nE 'UR1[^0-9]' /tmp/fail.sass | grep -v LDGSTS | wc -l   # 0
```

**0 处赋值**。换句话说 ptxas 把 30 条 `createpolicy → 物化到 UR1` 的 SASS 全省略了，但保留了 30 条 `LDGSTS [..], desc[UR1][..]`，UR1 是垃圾值。

### 第三步：对照同模板的 sibling 特化

同模板换一个参数 `Marlin<256,1,8,8,4,-1>`（小 batch + per-column scales 路径），同样使用 `cp_async4_stream`，SASS 里：

```text
14 desc[UR36]     ← cp_async4_pred (A)
 6 desc[UR8]      ← cp_async4_stream (B), each properly initialized
 2 desc[UR22]
 2 desc[UR24]
 1 desc[UR26]
 2 desc[UR28]
 2 desc[UR30]
 2 desc[UR32]
 2 desc[UR34]
 2 desc[UR38]
```

每个 desc 寄存器都能在 SASS 里找到对应的 `ULDC.64 URn, c[0x0][0x...]` 加载。**说明 ptxas 不是对 `cp_async4_stream` 这个 helper 的所有 instantiation 都漏发，而是仅在 `Marlin<256,1,8,8,4,8>` 的某种内部状态下漏发**。

### 第四步：触发条件

不知道 ptxas 内部确切的判定逻辑。我做了一次自包含 standalone 实验：

```cpp
// 4-stage pipeline + 4 chunks per stage + cp_async4_stream
__global__ __launch_bounds__(256) void repro_kernel(...) { ... }
```

跑出来 `desc[UR4]` 被正确赋值（`ULDC` 6 次定义、13 处使用），kernel 没有 trap。说明仅"4-stage cp.async 流水"还不足以触发。

`Marlin<256,1,8,8,4,8>` 比 `Marlin<256,1,8,8,4,-1>` 多出来的、和该 bug 高度相关的差异：

1. 主循环里有 **两个独立的 `cp_async4_stream` 调用点**——B 加载 (`marlin_cuda_kernel.cu:401`) 和 scales 加载 (`marlin_cuda_kernel.cu:408`)；gs=-1 路径在主循环内只有 B 一处。
2. 模板参数让 group_blocks=8 这条分支被 `#pragma unroll` 完整展开，导致同一个 kernel 里 createpolicy/LDGSTS 实例数量是 sibling 的近 1.6 倍（30 vs 6+2*8=22 甚至更少）。

合理推测：ptxas 的 dead code elimination / register coalescing 在大量同形 createpolicy + 高寄存器压力下走偏了，把所有 createpolicy 当成"结果寄存器没有跨基本块用户"做掉，但 LDGSTS 的 desc 操作数显然是用户。这是 ptxas 内部 SSA 模型对 `desc[]` 操作数语义的处理没正确感知。

## 最小复现

### 方法 A：通过 Marlin（最稳定，6 行 Python）

```bash
git clone https://github.com/IST-DASLab/marlin
cd marlin
git checkout <commit-before-the-fix>     # 或 revert 61cf87b
TORCH_CUDA_ARCH_LIST=9.0 python setup.py install

python - <<'PY'
import torch, marlin
torch.cuda.set_device(0)
DEV = 'cuda:0'
m, k, n, gs = 1, 4096, 4096, 128
A = torch.randn((m, k), dtype=torch.half, device=DEV)
B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), dtype=torch.int, device=DEV)
C = torch.zeros((m, n), dtype=torch.half, device=DEV)
s = torch.zeros((k // gs, n), dtype=torch.half, device=DEV)
workspace = torch.zeros(n // 128 * 16, dtype=torch.int, device=DEV)
marlin.mul(A, B, C, s, workspace, -1, -1, -1)
torch.cuda.synchronize()
PY
# → torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

### 方法 B：直接看 SASS（不需要运行）

```bash
cuobjdump --dump-sass \
    --function _Z6MarlinILi256ELi1ELi8ELi8ELi4ELi8EEvPK4int4S2_PS0_S2_iiiPi \
    marlin_cuda*.so | grep 'desc\[UR1\]' | wc -l
# → 30   （30 处使用未赋值的 UR1 → 一定 trap）
```

判定条件：如果 SASS 里 LDGSTS 引用的某个 desc 寄存器在该 kernel 内**从未被 ULDC/UMOV/createpolicy 物化**，就是 hit 这个 bug。

### 方法 C（未成功）：standalone .cu

我尝试过用 `cp_async4_stream` 写一个不依赖 PyTorch 的最小 kernel，跑了 4-stage × 4-chunk × 256-thread 的流水，**没有复现**——standalone kernel 里 desc 寄存器都被正确赋值。说明触发需要更复杂的结构（多 cp.async 路径 + 高寄存器压力 + 大量内联展开）。要做出真正"最小"的 repro 需要进一步 bisect Marlin 自己，工作量较大，没继续追。

## 规避方案

### 推荐：去掉 `L2::cache_hint`，仅在 sm_90 上生效

```cpp
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
#else
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
#endif
}
```

代价：Hopper 上 B 加载不再带 evict_first hint，会进 L2。Marlin 一次 GEMM 期间 B 是 streaming 一次，所以 hint 的实际收益本来就有限；实测在 H20 上去掉 hint 后 batch=1/16 的 TFLOP/s 与 GB/s 没明显回退。

### 备选：换工具链

- 降到 CUDA 12.x（如 12.4 / 12.6），实测能跑。
- 等 NVIDIA 修 13.x 后续版本。

### 不推荐：禁用相关 ptxas 优化

`-Xptxas -O0` 或 `-Xptxas --opt-level=0` 也能让 createpolicy 不被丢，但代价是整个 kernel 性能崩塌，不可用作生产规避。

## 提报建议

如果走 NVIDIA 工单：

- **Title**: ptxas 13.1 drops `createpolicy.fractional.L2::evict_first.b64` for sm_90, leaves dependent LDGSTS reading uninitialized uniform register
- **Repro**: Marlin commit `<sha-before-fix>`, build with `TORCH_CUDA_ARCH_LIST=9.0`, `nvcc 13.1.80`, run `marlin.mul` with `m=1, k=4096, n=4096, groupsize=128`
- **Evidence**: PTX contains 239× `createpolicy`; SASS for `Marlin<256,1,8,8,4,8>` references `desc[UR1]` 30× with zero defs of UR1
- **Severity**: silent miscompile causing runtime trap; sibling specializations OK so easy to miss in CI

## 参考

- 完整排查记录：[`h20_illegal_instruction_investigation.md`](./h20_illegal_instruction_investigation.md)
- 修复 commit：`61cf87b Fix cudaErrorIllegalInstruction on H20 in groupsize-aware small-batch path`
- 受影响代码：[`marlin/marlin_cuda_kernel.cu:69-79`](../marlin/marlin_cuda_kernel.cu#L69)
- 命中的 dispatch：[`marlin/marlin_cuda_kernel.cu:802`](../marlin/marlin_cuda_kernel.cu#L802) `CALL_IF(1, 8, 8, 8)`
