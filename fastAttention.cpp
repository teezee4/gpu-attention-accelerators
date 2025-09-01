#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <chrono>

void matTrans(torch::Tensor AT, torch::Tensor A);

torch::Tensor transpose(torch::Tensor A) {
  torch::Tensor AT = torch::zeros_like(A, torch::TensorOptions().device(A.device())); 
  matTrans(AT, A);
  return AT;
}

// Attention function declarations
torch::Tensor naiveAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor fusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor tcFusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor blockSparseAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                                bool causal = true, int block_size = 64);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // example:
  m.def("naive_transpose", &transpose, "naive transpose");
  // below are the functions you need to implement and compare
  m.def("naive_attention", &naiveAttention, "naive attention");
  m.def("fused_attention", &fusedAttention, "fused attention");
  m.def("tc_fused_attention", &tcFusedAttention, "fused attention with tensor cores");
  m.def("block_sparse_attention", &blockSparseAttention, "block sparse attention", 
      py::arg("Q"), py::arg("K"), py::arg("V"), 
      py::arg("causal") = true, py::arg("block_size") = 64); 
 // m.def("sparse_tc_fused_attention", &sparseTcFusedAttention, "sparse fused attention with tensor cores");
  // add more here if you have more variants to test
}
