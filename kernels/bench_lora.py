import torch
import time
import my_kernels  # your compiled CUDA extension


# ---------------------------------------------------------------------
# Helper: CUDA time measurement using cuda events (very accurate)
# ---------------------------------------------------------------------
def measure_time(func, warmup=10, iters=50):
    # Warm-up
    for _ in range(warmup):
        func()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        func()
    end.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


# ---------------------------------------------------------------------
# Define some functions
# ---------------------------------------------------------------------
def naive_lora(X, A, B_down_proj, B_up_proj):
    res = my_kernels.matmul_my_6(X, A)
    h_ = my_kernels.matmul_my_6(X, B_down_proj)
    lora = my_kernels.matmul_my_6(h_, B_up_proj)
    res = res + lora
    return res

def correct_lora(X, A, B_down_proj, B_up_proj):
    res = X@A
    h_ = X @ B_down_proj
    lora = h_ @ B_up_proj
    res = res + lora
    return res
    
# ---------------------------------------------------------------------
# Wrap functions
# ---------------------------------------------------------------------
def wrap_matmul(A, B):
    return lambda: my_kernels.matmul_my_6(A, B)

def wrap_naive_lora(X, A, B_down_proj, B_up_proj):
    return lambda: naive_lora(X, A, B_down_proj, B_up_proj)

def wrap_fused_lora(X, A, B_down_proj, B_up_proj):
    return lambda: my_kernels.lora_fuse(X, A, B_down_proj, B_up_proj)

def wrap_matmul_cublas(A, B):
    return lambda: A @ B

def wrap_matadd_cublas(A, B):
    return lambda: A + B - B

# def wrap_tc(A, B):
#     return lambda: my_kernels.matmul_tc(A, B)

# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------
def main():    
    # --------------------
    # Prepare test tensors
    # --------------------
    M = 256 * 2048
    K = N = 2048
    R = 16

    X = torch.randn(M, K, device="cuda", dtype=torch.float32)
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    B = torch.randn(M, N, device="cuda", dtype=torch.float32)
    B_down_proj = torch.randn(K, R, device="cuda", dtype=torch.float32)
    B_up_proj = torch.randn(R, N, device="cuda", dtype=torch.float32)

    print(f"\nMatrix size: X[{M},{K}], A[{K},{N}], B_down_proj[{K}, {R}], B_up_proj[{R}, {N}]")    
    
    
    
    # --------------------
    # Kernel wrapper layer
    # --------------------
    kernel_wrappers = {}

    # kernel_wrappers["residual"] = wrap_matmul(X, A)
    # kernel_wrappers["residual+lora(naive)"] = wrap_naive_lora(X, A, B_down_proj, B_up_proj)
    # kernel_wrappers["residual+lora(fused)"] = wrap_fused_lora(X, A, B_down_proj, B_up_proj)
    # kernel_wrappers["cublas matmul"] = wrap_matmul_cublas(A, B)
    kernel_wrappers["cublas matadd"] = wrap_matadd_cublas(A, B)
    # kernel_wrappers["residual+lora(fused)"] = wrap_fused_lora(X, A, B_down_proj, B_up_proj)

    
    # --------------------
    # Run benchmark
    # --------------------
    print("Running benchmarks...\n")    
    for name, func in kernel_wrappers.items():
        try:
            t = measure_time(func)
            print(f"{name:15s}: {t:.4f} ms")
        except Exception as e:
            print(f"{name:15s}: FAILED ({e})")
            
    # print("RES fuse lora: ", my_kernels.lora_fuse(X, A, B_down_proj, B_up_proj)[0][:10])
    # # print("RES-matmul-my", my_kernels.matmul_my_6(X, B_down_proj)[0][:10])
    # # print("RES-matmul-correct", (X @ B_down_proj)[0][:10])
    # print("RES-lora-correct", correct_lora(X, A, B_down_proj, B_up_proj)[0][:10])
    # print("RES-naive", naive_lora(X, A, B_down_proj, B_up_proj)[0][:10])


if __name__ == "__main__":
    main()
