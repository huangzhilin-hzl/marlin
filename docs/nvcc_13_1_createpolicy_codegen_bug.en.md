# nvcc 13.1 ptxas codegen bug: createpolicy.fractional dropped on sm_90

## Summary

CUDA Toolkit 13.1 (`nvcc V13.1.80`, build `cuda_13.1.r13.1/compiler.36836380_0`) ptxas, when compiling a specific sm_90 kernel, **silently drops the `createpolicy.fractional.L2::evict_first.b64` PTX instruction** while keeping the dependent `cp.async.cg.shared.global.L2::cache_hint`. In the resulting SASS, `LDGSTS` uses an **uninitialized uniform register** as its L2 cache descriptor; when the hardware decodes the descriptor it traps with `cudaErrorIllegalInstruction` (CUDA error 716).

## Impact

- **CUDA version**: CUDA 13.1 (other 13.x / 12.x not tested)
- **Driver**: reproduced on 550.127.08; version boundaries not characterized
- **GPU**: confirmed on H20 (sm_90); other Hopper SKUs (H100/H200) likely affected by the same mechanism but not independently verified
- **Trigger**: inline-PTX use of `cp.async.cg.shared.global.L2::cache_hint` inside a sufficiently complex kernel (multi-stage software pipeline with multiple independent cp.async paths). **Only the cache_hint variant of cp.async is affected; plain cp.async (`cp_async4_pred`) is fine.**

## Symptom

Reproducer site: the `Marlin<256,1,8,8,4,8>` specialization in [`marlin/marlin_cuda_kernel.cu`](../marlin/marlin_cuda_kernel.cu).

```text
cudaErrorIllegalInstruction (error 716) at kernel launch
========= Illegal instruction
=========     at void Marlin<256,1,8,8,4,8>(...)+0x1180
=========     by thread (131,0,0) in block (45,0,0)
========= ERROR SUMMARY: 2371 errors
```

Note the precise error: it is **`Illegal instruction`**, not `Misaligned address` and not `Illegal address`.

## Root cause (evidence chain)

### Step 1: PTX is correct

Compile `marlin_cuda_kernel.cu` to PTX with nvcc 13.1:

```bash
nvcc -arch=sm_90 -O3 -ptx marlin_cuda_kernel.cu -o marlin.ptx
grep -c createpolicy marlin.ptx
# → 239
```

`createpolicy.fractional.L2::evict_first.b64` shows up 239 times in PTX (one per inlined `cp_async4_stream` call site, since the helper is `__device__ inline`). The **C++ → PTX stage is fine.**

### Step 2: ptxas drops createpolicy in SASS

Disassemble the cubin:

```bash
cuobjdump --dump-sass --function _Z6MarlinILi256ELi1ELi8ELi8ELi4ELi8EEvPK4int4S2_PS0_S2_iiiPi \
    marlin_cuda.so > /tmp/fail.sass
```

Distribution of descriptor registers used by LDGSTS:

```text
14 desc[UR12]   ← cp_async4_pred (A buffer), no cache_hint
30 desc[UR1]    ← cp_async4_stream (B + scales), with cache_hint
```

`UR1` appears **30 times** throughout the kernel, **all of them as a read by LDGSTS as the desc operand**:

```bash
grep -nE 'UR1[^0-9]' /tmp/fail.sass | wc -l         # 30
grep -nE 'UR1[^0-9]' /tmp/fail.sass | grep -v LDGSTS | wc -l   # 0
```

**Zero definitions.** ptxas elided all 30 `createpolicy → materialize into UR1` SASS instructions but kept the 30 dependent `LDGSTS [..], desc[UR1][..]` instructions; UR1 holds garbage at runtime.

### Step 3: Compare against the sibling specialization

A sibling instantiation of the same template, `Marlin<256,1,8,8,4,-1>` (small-batch + per-column scales path), uses the same `cp_async4_stream` helper. Its SASS:

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

Every desc register has a corresponding `ULDC.64 URn, c[0x0][0x...]` definition in the SASS. **So ptxas does not drop createpolicy on every instantiation of `cp_async4_stream`; it drops it only under some internal state specific to `Marlin<256,1,8,8,4,8>`.**

### Step 4: Trigger conditions

The exact ptxas heuristic is unknown. I attempted a self-contained standalone reproducer:

```cpp
// 4-stage pipeline, 4 chunks per stage, cp_async4_stream
__global__ __launch_bounds__(256) void repro_kernel(...) { ... }
```

In that kernel `desc[UR4]` was correctly materialized (6 ULDC defs, 13 uses) and the kernel ran cleanly. So a "4-stage cp.async pipeline" alone is not sufficient to trigger the bug.

The differences between `Marlin<256,1,8,8,4,8>` (FAIL) and `Marlin<256,1,8,8,4,-1>` (PASS) that look most relevant:

1. The main loop has **two independent `cp_async4_stream` call sites**: B loads (`marlin_cuda_kernel.cu:401`) and scales loads (`marlin_cuda_kernel.cu:408`). The gs=-1 path only has B inside the loop.
2. The `group_blocks=8` template parameter causes the unrolled scales-fetch branch to be fully expanded by `#pragma unroll`, raising the per-kernel count of createpolicy / LDGSTS instances by ~1.6× compared to the sibling.

Working hypothesis: ptxas's dead-code elimination / register coalescing, when faced with many isomorphic createpolicy instances under high register pressure, treats every createpolicy as having no cross-block use and drops them all — even though the LDGSTS desc operands are obvious users. The underlying issue is that ptxas's internal SSA model does not correctly track `desc[]`-operand dependencies.

## Minimal reproduction

### Path A: via Marlin (most reliable, ~6 lines of Python)

```bash
git clone https://github.com/IST-DASLab/marlin
cd marlin
git checkout <commit-before-the-fix>     # or revert 61cf87b
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

### Path B: SASS inspection only (no execution required)

```bash
cuobjdump --dump-sass \
    --function _Z6MarlinILi256ELi1ELi8ELi8ELi4ELi8EEvPK4int4S2_PS0_S2_iiiPi \
    marlin_cuda*.so | grep 'desc\[UR1\]' | wc -l
# → 30   (30 reads of an undefined UR1 → guaranteed trap at runtime)
```

Diagnostic rule: if SASS shows an LDGSTS desc-operand register that is **never materialized by ULDC/UMOV/createpolicy anywhere in the same kernel**, you are hitting this bug.

### Path C (unsuccessful): standalone .cu

I tried writing a minimal kernel with `cp_async4_stream` outside of PyTorch — a 4-stage × 4-chunk × 256-thread pipeline — and **could not reproduce**. In the standalone kernel, the desc register was always materialized correctly. The bug seems to require additional structural complexity (multiple cp.async paths + high register pressure + heavy inlining). Producing a truly minimal standalone reproducer would require bisecting the Marlin kernel itself, which I did not pursue.

## Workarounds

### Recommended: drop the `L2::cache_hint`, gated to sm_90

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

Cost: on Hopper, B loads no longer carry the evict_first hint and will populate L2. Since Marlin streams B exactly once per GEMM, the practical benefit of the hint was already small; on H20 we measured no meaningful regression in TFLOP/s or GB/s for batch=1 / 16 after dropping the hint.

### Alternative: change toolchain

- Downgrade to CUDA 12.x (e.g. 12.4 / 12.6) — verified to compile and run correctly.
- Wait for a fix in a later 13.x release.

### Not recommended: disable the affected ptxas optimization

`-Xptxas -O0` or `-Xptxas --opt-level=0` also keeps createpolicy alive, but tanks overall kernel performance and is not a viable production workaround.

## Reporting to NVIDIA

Suggested ticket content:

- **Title**: ptxas 13.1 drops `createpolicy.fractional.L2::evict_first.b64` for sm_90, leaves dependent LDGSTS reading uninitialized uniform register
- **Repro**: Marlin commit `<sha-before-fix>`, build with `TORCH_CUDA_ARCH_LIST=9.0`, `nvcc 13.1.80`, run `marlin.mul` with `m=1, k=4096, n=4096, groupsize=128`
- **Evidence**: PTX contains 239× `createpolicy`; SASS for `Marlin<256,1,8,8,4,8>` references `desc[UR1]` 30× with zero defs of UR1
- **Severity**: silent miscompile causing runtime trap; sibling specializations are unaffected, making this easy to miss in CI

Channels (in order of preference):

1. **NVIDIA Developer Forum, CUDA category** — https://forums.developer.nvidia.com/c/accelerated-computing/cuda/ (public, indexed, often picked up by NVIDIA compiler-team engineers)
2. **NVIDIA Bug Reporting (NVBug)** — https://developer.nvidia.com/nvidia_bug/login (requires NVIDIA Developer Program account; private formal tracker)
3. **Enterprise / DevTech contact** if available — fastest path to a hotfix nvcc or vetted workaround for production

`nvcc` / `ptxas` are closed-source and have **no public GitHub issue tracker**. Adjacent GitHub repos (`NVIDIA/cccl`, `NVIDIA/cuda-samples`, `llvm/llvm-project`'s NVPTX backend) are not the right places to report this bug — they will redirect you to the channels above.

## References

- Full investigation log: [`h20_illegal_instruction_investigation.md`](./h20_illegal_instruction_investigation.md)
- Fix commit: `61cf87b Fix cudaErrorIllegalInstruction on H20 in groupsize-aware small-batch path`
- Affected source: [`marlin/marlin_cuda_kernel.cu:69-79`](../marlin/marlin_cuda_kernel.cu#L69)
- Triggering dispatch: [`marlin/marlin_cuda_kernel.cu:802`](../marlin/marlin_cuda_kernel.cu#L802) `CALL_IF(1, 8, 8, 8)`
