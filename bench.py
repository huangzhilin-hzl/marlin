import argparse
import time

import torch


DEV = torch.device('cuda:0')
MARLIN = None


def benchmark(f, warmup=1, iters=10, cooldown=1.0):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    tick = time.time()
    for _ in range(iters):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarking as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
    torch.cuda.synchronize()
    res = (time.time() - tick) / iters
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    if cooldown > 0:
        time.sleep(cooldown)
    return res


def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    A = torch.randn((m, k), dtype=torch.half, device=DEV)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), dtype=torch.int, device=DEV)
    B_ref = torch.randn((k, n), dtype=torch.half, device=DEV)
    C = torch.zeros((m, n), dtype=torch.half, device=DEV)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=DEV)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s


def benchmark_dense(A, B, C, warmup, iters, cooldown):
    res = benchmark(lambda: torch.matmul(A, B, out=C), warmup, iters, cooldown)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }


def benchmark_quant(A, B, C, s, thread_k, thread_n, sms, warmup, iters, cooldown):
    workspace = torch.zeros(C.shape[1] // 128 * 16, dtype=torch.int, device=DEV)
    res = benchmark(lambda: MARLIN.mul(A, B, C, s, workspace, thread_k, thread_n, sms), warmup, iters, cooldown)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }


def get_models(sms):
    return {
        'ideal': [
            (4 * 256 * sms, 256 * sms)
        ],
        'Llama7B': [
            (4096, 3 * 4096),
            (4096, 4096),
            (4096, 2 * 10752),
            (10752, 4096)
        ],
        'Llama13B': [
            (5120, 3 * 5120),
            (5120, 5120),
            (5120, 2 * 13568),
            (13568, 5120)
        ],
        'Llama33B': [
            (6656, 3 * 6656),
            (6656, 6656),
            (6656, 2 * 17664),
            (17664, 6656)
        ],
        'Llama65B': [
            (8192, 3 * 8192),
            (8192, 8192),
            (8192, 2 * 21760),
            (21760, 8192)
        ],
        'Falcon180B': [
            # Note that parallel attention and FC allows layer fusions
            (14848, 14848 * 5 + 1024),
            (14848 * 5, 14848)
        ]
    }


def parse_int_list(value):
    return [int(item) for item in value.split(',') if item]


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Marlin FP16xINT4 kernels.')
    parser.add_argument('--device', default='cuda:0', help='CUDA device to benchmark, for example cuda:0.')
    parser.add_argument('--sms', type=int, default=None, help='Override SM count. Defaults to CUDA device properties.')
    parser.add_argument('--models', default=None, help='Comma-separated model names to run. Defaults to all models.')
    parser.add_argument('--batch-sizes', type=parse_int_list, default=None, help='Comma-separated batch sizes.')
    parser.add_argument('--groupsizes', type=parse_int_list, default=None, help='Comma-separated groupsizes. Use -1 for per-column scales.')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup iterations per shape.')
    parser.add_argument('--iters', type=int, default=10, help='Measured iterations per shape.')
    parser.add_argument('--cooldown', type=float, default=1.0, help='Seconds to sleep between measured shapes.')
    parser.add_argument('--all', action='store_true', help='Run the complete sweep instead of the README-sized sweep.')
    return parser.parse_args()


def main():
    global DEV, MARLIN
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run Marlin benchmarks.')
    if args.warmup < 0:
        raise ValueError('Warmup iterations must be non-negative.')
    if args.iters <= 0:
        raise ValueError('Measured iterations must be positive.')

    DEV = torch.device(args.device)
    torch.cuda.set_device(DEV)
    import marlin
    MARLIN = marlin

    props = torch.cuda.get_device_properties(DEV)
    sms = args.sms if args.sms is not None else props.multi_processor_count
    if sms <= 0:
        raise ValueError('SM count must be positive; got %d.' % sms)

    capability = '%d.%d' % (props.major, props.minor)
    print('device=%s, name=%s, capability=%s, sms=%d' % (DEV, props.name, capability, sms))
    if props.major >= 9:
        print('warning: Marlin README says this kernel is not optimized for Hopper; H20 results are for compatibility/perf exploration.')
    print()

    models = get_models(sms)
    if args.models:
        selected_models = [model for model in args.models.split(',') if model]
        unknown = sorted(set(selected_models) - set(models))
        if unknown:
            raise ValueError('Unknown model(s): %s. Available: %s' % (','.join(unknown), ','.join(models)))
    else:
        selected_models = list(models)

    if args.groupsizes is not None:
        groupsizes = args.groupsizes
    else:
        groupsizes = [-1, 128] if args.all else [128]

    if args.batch_sizes is not None:
        default_batchsizes = args.batch_sizes
    elif args.all:
        default_batchsizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    else:
        default_batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for groupsize in groupsizes:
        print('groupsize=%d' % groupsize)
        print()
        for model in selected_models:
            layers = models[model]
            print(model)
            for batch in default_batchsizes:
                if not args.all and args.batch_sizes is None and model != 'ideal' and batch != 16:
                    continue
                tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0}
                for layer in layers:
                    A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], groupsize)
                    res_d = benchmark_dense(A, B_ref, C, args.warmup, args.iters, args.cooldown)
                    if model == 'ideal' and batch == 16:
                        # This is a special case constructed to be optimal for a thread-shape different than the default one
                        res_q = benchmark_quant(A, B, C, s, 64, 256, sms, args.warmup, args.iters, args.cooldown)
                    else:
                        res_q = benchmark_quant(A, B, C, s, -1, -1, sms, args.warmup, args.iters, args.cooldown)
                    res_q['speedup'] = res_d['s'] / res_q['s']
                    tot_q['s'] += res_q['s']
                    for k in tot_q:
                        if k != 's':
                            tot_q[k] += res_q[k] * res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] /= tot_q['s']
                print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                    batch,
                    tot_q['s'],
                    tot_q['TFLOP/s'],
                    tot_q['GB/s'],
                    tot_q['speedup']
                ))
            print()


if __name__ == '__main__':
    main()
