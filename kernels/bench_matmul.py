import torch
import time
import my_kernels  # your compiled CUDA extension


# ---------------------------------------------------------------------
# Helper: CUDA time measurement using cuda events (very accurate)
# ---------------------------------------------------------------------
def measure_time(func, A, B, warmup=10, iters=50):
    # Warm-up
    for _ in range(warmup):
        func(A, B)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        func(A, B)
    end.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


# ---------------------------------------------------------------------
# Register your kernels here.
# IMPORTANT: Put exactly the names exported in binding.cpp
# ---------------------------------------------------------------------
kernel_functions = {
    "my matmul 3": my_kernels.matmul_my_3,
    "my matmul 4": my_kernels.matmul_my_4,
    "my matmul 6": my_kernels.matmul_my_6,
    "cublas": my_kernels.matmul_cublas,
    # "optimized": my_kernels.matmul_optimized,
    # "wmma": my_kernels.matmul_wmma,
}


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------
def main():    
    M = 512 * 1024
    K = N = 1024

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    print(f"\nMatrix size: A[{M},{K}], B[{K},{N}]")
    print("Running benchmarks...\n")

    for name, func in kernel_functions.items():
        if func is None:
            continue   # skip missing kernels (e.g., tensorcore on older GPUs)

        try:
            t = measure_time(func, A, B)
            print(f"{name:15s}: {t:.4f} ms")
        except Exception as e:
            print(f"{name:15s}: FAILED ({e})")


if __name__ == "__main__":
    main()
