#include <torch/extension.h>

torch::Tensor fwd_cuda(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& g_K);

std::vector<torch::Tensor> bwd_cuda(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor g_K, torch::Tensor DQK);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwd_cuda, "GLA compute A semiring (CUDA)");
  m.def("backward", &bwd_cuda, "GLA compute A semiring (CUDA)");
}
