#include <torch/extension.h>

void launch_matmul_my_3(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void launch_matmul_my_4(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void launch_matmul_my_6(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void launch_matmul_cublas(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void launch_lora_fuse(torch::Tensor X, torch::Tensor A, 
                    torch::Tensor B_down_proj, torch::Tensor B_up_proj, 
                    torch::Tensor C);


torch::Tensor matmul_my_3(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only FP32 supported");

    auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
    launch_matmul_my_3(A, B, C);
    return C;
}

torch::Tensor matmul_my_4(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only FP32 supported");

    auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
    launch_matmul_my_4(A, B, C);
    return C;
}

torch::Tensor matmul_my_6(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only FP32 supported");

    auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
    launch_matmul_my_6(A, B, C);
    return C;
}

torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only FP32 supported");

    auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
    launch_matmul_cublas(A, B, C);
    return C;
}

torch::Tensor lora_fuse(torch::Tensor X, torch::Tensor A, 
                        torch::Tensor B_down_proj, torch::Tensor B_up_proj) {
    TORCH_CHECK(X.is_cuda() && A.is_cuda() && B_down_proj.is_cuda() && B_up_proj.is_cuda(), "Input must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only FP32 supported");

    auto C = torch::zeros({X.size(0), A.size(1)}, A.options());
    launch_lora_fuse(X, A, B_down_proj, B_up_proj, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("matmul_my_3", &matmul_my_3, "My CUDA matmul 3");
    // m.def("matmul_my_4", &matmul_my_4, "My CUDA matmul 4");
    m.def("matmul_my_6", &matmul_my_6, "My CUDA matmul 6");
    // m.def("matmul_cublas", &matmul_cublas, "cublas matmul");
    m.def("lora_fuse", &lora_fuse, "lora fuse");
}