# Fast Attention Kernels 
**Custom CUDA + PyTorch kernels for Transformer attention mechanisms with Tensor Core acceleration**  

This project implements and benchmarks multiple attention variants as a PyTorch C++/CUDA extension, achieving **up to 50× speedup** on NVIDIA V100 GPUs while analyzing accuracy, throughput, and hardware efficiency.  

---

## Features
- **Multiple Attention Implementations**
  - Naive (baseline)
  - Fused (with shared memory, masking attempts)
  - Tensor Core fused (leveraging NVIDIA Tensor Cores)
  - Block-Sparse (causal and local variants)

- **PyTorch Integration**
  - Built as a C++/CUDA extension
  - Python test drivers for flexible benchmarking

- **Performance Benchmarks**
  - Sequence lengths: 1024 – 4096
  - Embedding dimensions: 128 – 4096
  - Tested on **NVIDIA V100 (CUDA 12.2)**

---

## Results

### Tensor Core Fused Attention
- Achieved **up to 50× speedup** vs. naive attention  
  *(e.g., 8.7 ms vs 444 ms at seq_len=1024, dim=1024)*  
- Maintained correctness with relative error < **1e−6**  

### Block-Sparse Attention (Local)
- **9–10× speedup** for long sequences  
- Accuracy issues (relative error 2–5), highlighting sparsity–accuracy tradeoffs  

### Block-Sparse Attention (Causal)
- Slower than naive; relative errors > 1  

### Fused Attention
- Slower than naive due to unresolved memory issues  

---

## Example Benchmark (V100, seq_len=1024, dim=1024)

| Implementation        | Time (ms) | Relative Error | Speedup (vs naive) |
|-----------------------|-----------|----------------|---------------------|
| Naive                 | 444.26    | 1.3e−6         | 1.0×               |
| Fused                 | 973.28    | 1.3e−6         | 0.46×              |
| Tensor Core Fused     | 8.71      | 1.3e−6         | **51.0×**          |
| Block-Sparse (causal) | 890.62    | 1.5            | 0.50×              |
| Block-Sparse (local)  | 185.33    | 2.1            | 2.40×              |

---

## Analysis Highlights
- **Throughput**: Decreases with embedding dim; Tensor Core fused sustains best performance
- **Memory Bandwidth**: Tensor Core fused reduces global memory traffic with shared memory
- **Kernel Occupancy**: Sparse methods gain occupancy but lose arithmetic intensity
- **Sparsity Tradeoff**: Block-sparse promising for long sequences, but accuracy must improve

---

## Requirements
- CUDA 12.2
- PyTorch (C++ extension support)
- Python 3.10+
- NVIDIA V100 GPU (tested)
