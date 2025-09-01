import torch
import numpy as np
import fastAttention
import argparse
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def pytorch_vanilla_attention(Q, K, V):
  warmup = 10
  niters = 20
  
  Q = Q.unsqueeze(0).unsqueeze(1)
  K = K.unsqueeze(0).unsqueeze(1)
  V = V.unsqueeze(0).unsqueeze(1)
  
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  
  with sdpa_kernel(backends=[SDPBackend.MATH]):
    for _ in range(warmup):
      ref_attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, scale=1.0/np.sqrt(Q.size(-1)))
    
    start.record()
    for _ in range(niters):
      ref_attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, scale=1.0/np.sqrt(Q.size(-1)))
    end.record()
    end.synchronize()
  
  print(f"Vanilla attention time: {start.elapsed_time(end)/niters} ms")
  
  return ref_attn

def test_implementation(name, function, Q, K, V, ref_attn):
  warmup = 10
  niters = 20
  
  print(f"\n===== Testing {name} =====")

  # Warmup
  for _ in range(warmup):
    my_attn = function(Q, K, V)
  
  # Timing
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  
  start.record()
  for _ in range(niters):
    my_attn = function(Q, K, V)
  end.record()
  end.synchronize()
  
  time_ms = start.elapsed_time(end)/niters
  print(f"{name} time: {time_ms} ms")
  
  # Check correctness
  rel_error = torch.norm(ref_attn - my_attn) / torch.norm(ref_attn)
  print(f"Relative error: {rel_error}")

  if rel_error > 1e-4:
    print(f"{name} is incorrect")
    print(f"Relative error {rel_error} is more than 1e-4")
  else:
    print(f"{name} implementation is correct!")
  
  return time_ms, my_attn

def main(args):
  embed_dim = args.embed_dim
  seq_len = args.seq_len
  
  print(f"\nRunning with embed_dim={embed_dim}, seq_len={seq_len}")
  
  Q = torch.randn(args.seq_len, args.embed_dim,
                 dtype=torch.float32, device="cuda")
  K = torch.randn(args.seq_len, args.embed_dim,
                 dtype=torch.float32, device="cuda")
  V = torch.randn(args.seq_len, args.embed_dim,
                 dtype=torch.float32, device="cuda")
  
# Debugging: Validate tensor properties
  assert Q.dtype == torch.float32 and Q.device.type == "cuda", "Q must be float32 and on CUDA"
  assert K.dtype == torch.float32 and K.device.type == "cuda", "K must be float32 and on CUDA"
  assert V.dtype == torch.float32 and V.device.type == "cuda", "V must be float32 and on CUDA" 

  ref_attn = pytorch_vanilla_attention(Q, K, V).squeeze(0).squeeze(0)
  
  # Test implementations
  naive_time, _ = test_implementation("Naive attention", fastAttention.naive_attention, Q, K, V, ref_attn)
  fused_time, _ = test_implementation("Fused attention", fastAttention.fused_attention, Q, K, V, ref_attn)
  tc_time, _ = test_implementation("Tensor Core fused attention", fastAttention.tc_fused_attention, Q, K, V, ref_attn)
  # Test block-sparse attention with causal masking
  sparse_causal_time, _ = test_implementation("Block-sparse attention (causal)", lambda Q, K, V: fastAttention.block_sparse_attention(Q, K, V, True, 64), Q, K, V, ref_attn)

  # Test block-sparse attention with local window
  sparse_local_time, _ = test_implementation("Block-sparse attention (local)",  lambda Q, K, V: fastAttention.block_sparse_attention(Q, K, V, False, 64),  Q, K, V, ref_attn)
  
  # Report speedups
  print(f"\n===== Performance Comparison =====")
  print(f"Naive attention: {naive_time:.2f} ms")
  print(f"Fused attention: {fused_time:.2f} ms (Speedup vs naive: {naive_time/fused_time:.2f}x)")
  print(f"TC Fused attention: {tc_time:.2f} ms (Speedup vs naive: {naive_time/tc_time:.2f}x, vs fused: {fused_time/tc_time:.2f}x)")
  print(f"Block-sparse attention (causal): {sparse_causal_time:.2f} ms (Speedup vs naive: {naive_time/sparse_causal_time:.2f}x)")
  print(f"Block-sparse attention (local): {sparse_local_time:.2f} ms (Speedup vs naive: {naive_time/sparse_local_time:.2f}x)")


if __name__ == "__main__":
  # test our transpose kernel
  # only for demonstration purposes
  Q = torch.randn(50, 50, device="cuda")
  QT = fastAttention.naive_transpose(Q)
  QT_ref = Q.T
  assert torch.allclose(QT, QT_ref), "Transpose kernel is incorrect"
  print("Transpose kernel is working correctly")
 
  parser = argparse.ArgumentParser()
  parser.add_argument("--embed_dim","-e", type=int, default=128)
  parser.add_argument("--seq_len","-s", type=int, default=1024)
  args = parser.parse_args()
  
  main(args)
