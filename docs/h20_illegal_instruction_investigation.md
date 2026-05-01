# H20 上 Marlin small-batch + groupsize=128 路径 illegal instruction 排查记录

## 0. TL;DR

- **现象**：`marlin.mul` 在 H20 上跑 `m ≤ 16 + groupsize=128` 时挂，错误码 `cudaErrorIllegalInstruction`。
- **真因**：CUDA 13.1 (`nvcc V13.1.80`, build `cuda_13.1.r13.1/compiler.36836380_0`) 在编译 `Marlin<256,1,8,8,4,8>` 这一个特化时，把 `cp_async4_stream` 内联 PTX 里的 `createpolicy.fractional.L2::evict_first.b64` 漏发了，生成的 SASS 里 30 条 `LDGSTS` 用 `desc[UR1]` 作为 L2 cache descriptor，但 UR1 全程未被赋值，硬件解码 desc 时 trap。
- **不是**：H20 不支持 LDGSTS，也不是 cp.async/Hopper 兼容性问题。
- **修复**：在 sm_90 上把 `cp_async4_stream` 退回到无 cache_hint 的普通 `cp.async.cg.shared.global`，commit 见 `61cf87b`。

环境信息：

| 项 | 值 |
|---|---|
| 运行环境 | 内部 H20 测试环境 |
| GPU | NVIDIA H20 (8 卡，sm_90, 78 SMs) |
| Driver | 550.127.08 |
| CUDA Runtime | 13.1 |
| nvcc | 13.1.80 (Nov 2025 build) |
| 代码路径 | Marlin checkout |

## 1. 拿到的"诊断"和我的判断

最初的诊断结论是：

> Marlin 在 H20 上跑 groupsize=128 的 small-batch 路径时，自动选中了一个有兼容性问题的 CUDA kernel 特化 `Marlin<256,1,8,8,4,8>`。这个特化里的 `cp.async` 被编译成 LDGSTS 后，在 H20 执行时 trap。

> 更精确地说，不是"H20 不支持 LDGSTS"，而是"H20 上这个 `Marlin<256,1,8,8,4,8>` 特化生成的 LDGSTS 序列不可执行"。

读完这段我的第一反应：

1. **dispatch 路径推断对**：看 `marlin_cuda_kernel.cu:755-758`，small-batch 路径 `thread_k = thread_n = 128`，和 `groupsize=128` 一组合，确实命中 `marlin_cuda_kernel.cu:802` 的 `CALL_IF(1, 8, 8, 8)`，也就是 `Marlin<256,1,8,8,4,8>`。
2. **"H20 不支持 LDGSTS"是错的**：H20 是 Hopper (sm_90)，`cp.async`/LDGSTS 是 Ampere (sm_80) 引入的，Hopper 整代都支持，不存在型号粒度的"不支持"。
3. **"特化生成的 LDGSTS 序列不可执行"这个表述也站不住**：LDGSTS 的语义是固定的，不存在"某个 kernel 特化整体生成不可执行 LDGSTS"这种事。要么是地址错（misaligned/OOB），要么是某条具体指令的某个操作数有问题。

但关键是 **trap 现象本身真实存在**，所以诊断是观察对、归因错。我的目标是把"哪条指令、为什么 trap"落实到 SASS 级别。

## 2. 排查路线

```
看到诊断
  ├─ 验 dispatch：源码确认 (1, 8, 8, 8) 路径 ✓
  ├─ 否定"架构不支持"：H20 = sm_90，LDGSTS 显然支持
  └─ 真因待查
        ↓
在测试环境复现
        ↓
矩阵化定位（独立子进程，避免 CUDA context 污染）
        ↓
compute-sanitizer 落到指令偏移
        ↓
反汇编 SASS，看挂的指令是什么、操作数从哪来
        ↓
反证实验：去掉嫌疑指令，看是否修复
        ↓
对照同名特化 (gs=-1) 的 SASS，找编译差异
        ↓
确认 nvcc codegen bug
```

## 3. 执行步骤

### 3.1 确认测试环境

```bash
nvidia-smi -L
nvcc --version
```

确认：8 张 H20，CUDA 13.1，已经 build 好的 `.so` 在 `build/lib.linux-x86_64-cpython-312/marlin_cuda.cpython-312-x86_64-linux-gnu.so`。

### 3.2 确认 .so 编译目标

```bash
cuobjdump --list-elf build/lib.linux-x86_64-cpython-312/marlin_cuda.cpython-312-x86_64-linux-gnu.so
# → marlin_cuda.cpython-312-x86_64-linux-gnu.1.sm_90.cubin
```

只编了 sm_90 native，没走 PTX JIT。排除"binary 是 sm_80 跑在 sm_90 上 JIT 出问题"的可能。

### 3.3 最小复现

```python
# /tmp/probe.py
import sys, torch, marlin
m = int(sys.argv[1]); gs = int(sys.argv[2])
torch.cuda.set_device(0)
DEV = 'cuda:0'
n = k = 4096
gs_eff = k if gs == -1 else gs
A = torch.randn((m, k), dtype=torch.half, device=DEV)
B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), dtype=torch.int, device=DEV)
C = torch.zeros((m, n), dtype=torch.half, device=DEV)
s = torch.zeros((k // gs_eff, n), dtype=torch.half, device=DEV)
workspace = torch.zeros(n // 128 * 16, dtype=torch.int, device=DEV)
marlin.mul(A, B, C, s, workspace, -1, -1, -1)
torch.cuda.synchronize()
print('OK')
```

每个 case 起独立进程跑（一旦 illegal instruction 触发，整个 CUDA context 都污染了，后续 launch 全都会假性失败）：

```bash
for m in 1 16 32 64; do
  for gs in -1 128; do
    out=$(python /tmp/probe.py $m $gs 2>&1 | tail -1)
    echo "m=$m gs=$gs  -> $out"
  done
done
```

矩阵结果：

| m | gs=-1 | gs=128 |
|---|-------|--------|
| 1 | OK | **FAIL** |
| 16 | OK | **FAIL** |
| 32 | OK | OK |
| 64 | OK | OK |

清晰命中 small-batch (m ≤ 16) + groupsize=128 这一格。`m=32` 走的是 `prob_m > 16` 的另一条 thread tile 配置（`thread_k=64, thread_n=256`），所以不挂。

### 3.4 compute-sanitizer 定位指令偏移

```bash
compute-sanitizer --tool memcheck python /tmp/probe.py 1 128
```

关键输出：

```
========= Illegal instruction
=========     at void Marlin<(int)256, (int)1, (int)8, (int)8, (int)4, (int)8>+0x1180
=========     by thread (131,0,0) in block (45,0,0)
========= ERROR SUMMARY: 2371 errors
```

注意：是 `Illegal instruction`，**不是** `Misaligned address` / `Invalid global read`。这意味着 SM 解码到了一条无法合法执行的指令，而不是地址越界。

### 3.5 反汇编看 +0x1180 是什么

```bash
cuobjdump --dump-sass \
  --function _Z6MarlinILi256ELi1ELi8ELi8ELi4ELi8EEvPK4int4S2_PS0_S2_iiiPi \
  build/lib.linux-x86_64-cpython-312/marlin_cuda.cpython-312-x86_64-linux-gnu.so \
  > /tmp/fail.sass
```

挂的那条：

```asm
/*1180*/  LDGSTS.E.BYPASS.LTC128B.128 [R9+UR0+0x5000], desc[UR1][R92.64]
```

是 `LDGSTS`（即 `cp.async`）的 L2 cache hint 形式，对应 `marlin_cuda_kernel.cu:69-79` 的 `cp_async4_stream`：

```cpp
asm volatile(
  "{\n"
  "   .reg .b64 p;\n"
  "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
  "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
  "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
);
```

### 3.6 对比同名特化的 SASS（gs=-1，能跑通）

`Marlin<256,1,8,8,4,-1>` 的 SASS 里也大量用 `cp_async4_stream`（B 加载），但它能跑通。把两个 dump 都拿出来对比 LDGSTS 的 descriptor 寄存器：

```bash
# FAIL: gs=128
grep 'LDGSTS' /tmp/fail.sass | grep -oE 'desc\[U?R[0-9]+\]' | sort | uniq -c
#  14 desc[UR12]
#  30 desc[UR1]

# PASS: gs=-1
grep 'LDGSTS' /tmp/pass.sass | grep -oE 'desc\[U?R[0-9]+\]' | sort | uniq -c
#   2 desc[UR22]
#   2 desc[UR24]
#   1 desc[UR26]
#   2 desc[UR28]
#   2 desc[UR30]
#   2 desc[UR32]
#   2 desc[UR34]
#  14 desc[UR36]
#   2 desc[UR38]
#   6 desc[UR8]
```

PASS kernel 用了一堆不同 descriptor reg；FAIL kernel 只用 `UR12`（A 的普通 cp.async）和 `UR1`（B + scales 的 cache_hint cp.async）。

### 3.7 关键证据 — UR1 在 FAIL kernel 中从未被赋值

```bash
grep -nE 'UR1[^0-9]' /tmp/fail.sass | wc -l
# 30
grep -nE 'UR1[^0-9]' /tmp/fail.sass | grep -v LDGSTS | wc -l
# 0
```

FAIL kernel 全文 30 处提到 UR1，**全部是被 LDGSTS 当 desc 操作数读取**，没有一处是写入/赋值。换句话说 `createpolicy.fractional.L2::evict_first.b64` 这条 PTX 应该被 ptxas 编译成一条物化 UR1 的 SASS 指令，但 ptxas 把它整段丢了。

PASS kernel 里对照（`UR8 = ULDC.64 c[0x0][0x240]`）能找到 desc 寄存器从 constant memory 加载的明确 SASS。

### 3.8 反证实验

把 `cp_async4_stream` 改成不带 cache_hint 的普通 cp.async：

```cpp
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}
```

重新编译并跑全矩阵：

```bash
TORCH_CUDA_ARCH_LIST='9.0' python setup.py build_ext --inplace --force
for m in 1 16 32 64 128; do
  for gs in -1 128; do
    out=$(python /tmp/probe.py $m $gs 2>&1 | tail -1)
    echo "m=$m gs=$gs  -> $out"
  done
done
```

10 个 case 全 OK，`test.py` 6 个单测通过，bench 性能正常（Llama7B batch=1 ~3.5 TFLOP/s, batch=16 ~55 TFLOP/s）。

## 4. 修复

按"保留 sm_80 上原行为，仅在 sm_90 上规避"提交：

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

Commit：`61cf87b Fix cudaErrorIllegalInstruction on H20 in groupsize-aware small-batch path`

为什么这样 gate：

- A100 / RTX 3090 等 sm_80/sm_86 上原来的 cache_hint 实测能用，是 evict_first 那条 hint 的合理使用，丢掉会污染 L2，所以保留。
- Marlin 自己的 README 也写明"not optimized for Hopper"，Hopper 上去掉 hint 没有性能负担。
- 不动 PASS kernel 路径下的 `cp_async4_stream`，所以 gs=-1 的 SASS 形态完全和上游一致。

## 5. 收尾验证

```bash
python test.py             # 6/6 unit tests pass
python bench.py --models Llama7B --batch-sizes 1,16 --groupsizes 128 --iters 30
# device=cuda:0, name=NVIDIA H20, capability=9.0, sms=78
# Llama7B
# batch=0001: TFLOP/s=003.520, GB/s=0908.564, speedup=1.19
# batch=0016: TFLOP/s=055.597, GB/s=0913.990, speedup=1.24
```

## 6. 经验教训

1. **trap 不一定是架构兼容性问题**。H20/H100 这一代经常被默认挂上"不支持 X 指令"的帽子，实际上 Hopper 是 sm_80 的超集，绝大多数 LDGSTS/cp.async 现象都是 toolchain 或地址问题。
2. **诊断观察对、归因错很常见**。"小 batch + gs=128 挂"的现象抓得很准，但跳到"H20 不支持 LDGSTS 序列"就把后面的人引偏了。要把现象和归因分开。
3. **CUDA error 的语义要分清**：`cudaErrorIllegalInstruction` ≠ `cudaErrorMisalignedAddress` ≠ `cudaErrorIllegalAddress`。第一个明确指向"指令本身"，应该直接走 SASS 路线。
4. **同模板不同参数的 SASS 对比** 是定位 codegen bug 的第一手段。两个特化共用所有 `__device__` helper，差异点可控。
5. **CUDA context 一旦 trap 就废了**，做矩阵化探针一定要用独立子进程，否则会得到一片虚假的 FAIL。

## 7. 衍生文档

详细的 nvcc bug 说明和最小复现见 [`nvcc_13_1_createpolicy_codegen_bug.md`](./nvcc_13_1_createpolicy_codegen_bug.md)。
