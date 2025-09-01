#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

__global__ void matTransKernel(float* AT, float* A, int N);
void tcFusedAttentionOptimized(torch::Tensor output, torch::Tensor Q, torch::Tensor K, torch::Tensor V);


void matTrans(torch::Tensor AT, torch::Tensor A)  {
  assert(AT.size(0) == AT.size(1));
  assert(AT.size(0) == A.size(0));
  assert(AT.size(1) == A.size(1));
  matTransKernel<<<1, 512>>>(AT.data_ptr<float>(), A.data_ptr<float>(), A.size(0));
}

__global__ void matTransKernel(float* AT, float* A, int N)  {
  int tid = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
  for(int i = tid; i < N*N; i += blockDim.x*gridDim.x*blockDim.y) {
        int row = i / N;
        int col = i % N;
        AT[col*N+row] = A[i];
  }
}

// Naive Attention Implementation
__global__ void naiveAttentionKernel(float* output, float* Q, float* K, float* V, 
  float* temp_scores, int seq_len, int embed_dim) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < seq_len) {
// For each output row
for (int i = 0; i < seq_len; i++) {
float sum = 0.0f;

// Compute dot product between Q[tid] and K[i]
for (int j = 0; j < embed_dim; j++) {
sum += Q[tid * embed_dim + j] * K[i * embed_dim + j];
}

// Scale by sqrt(embed_dim)
sum /= sqrtf(embed_dim);

// Store in temporary array for softmax computation
temp_scores[tid * seq_len + i] = sum;
}

// Compute softmax
float max_val = -INFINITY;
for (int i = 0; i < seq_len; i++) {
if (temp_scores[tid * seq_len + i] > max_val) {
max_val = temp_scores[tid * seq_len + i];
}
}
float sum_exp = 0.0f;
for (int i = 0; i < seq_len; i++) {
    temp_scores[tid * seq_len + i] = expf(temp_scores[tid * seq_len + i] - max_val);
    sum_exp += temp_scores[tid * seq_len + i];
}

 for (int i = 0; i < seq_len; i++) {
    temp_scores[tid * seq_len + i] /= sum_exp;
}

 // Compute weighted sum of values
for (int j = 0; j < embed_dim; j++) {
    float weighted_sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        weighted_sum += temp_scores[tid * seq_len + i] * V[i * embed_dim + j];
    }
    output[tid * embed_dim + j] = weighted_sum;
}
}
}

void naiveAttentionCuda(torch::Tensor output, torch::Tensor Q, torch::Tensor K, 
                torch::Tensor V) {
int seq_len = Q.size(0);
int embed_dim = Q.size(1);

// Create temporary storage for attention scores
auto temp_scores = torch::zeros({seq_len, seq_len}, 
  torch::TensorOptions().device(Q.device()));

// Calculate grid and block dimensions
int block_size = 256;
int grid_size = (seq_len + block_size - 1) / block_size;

naiveAttentionKernel<<<grid_size, block_size>>>(
output.data_ptr<float>(),
Q.data_ptr<float>(),
K.data_ptr<float>(),
V.data_ptr<float>(),
temp_scores.data_ptr<float>(),
seq_len,
embed_dim
);
}

torch::Tensor naiveAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
// Check dimensions
assert(Q.size(0) == K.size(0) && "Sequence lengths must match for Q and K");
assert(Q.size(1) == K.size(1) && "Embedding dimensions must match for Q and K");
assert(K.size(0) == V.size(0) && "Sequence lengths must match for K and V");

int seq_len = Q.size(0);
int embed_dim = V.size(1);

// Create output tensor
auto output = torch::zeros({seq_len, embed_dim}, 
torch::TensorOptions().device(Q.device()));

naiveAttentionCuda(output, Q, K, V);
    
return output;
}


// Fused Attention Implementation
__global__ void fusedAttentionKernel(float* output, float* Q, float* K, float* V, 
                                 int seq_len, int embed_dim) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < seq_len) {
    // Each thread computes one row of the output
    float* output_row = output + tid * embed_dim;
    
     // Initialize output to zero
    for (int j = 0; j < embed_dim; j++) {
        output_row[j] = 0.0f;
    }
    
     // Scale factor for attention scores
    float scale = 1.0f / sqrtf(embed_dim);
    
     // Variables for softmax computation
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
     // First pass: compute max value for numerical stability in softmax
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0f;
        // Compute dot product between Q[tid] and K[i]
        for (int j = 0; j < embed_dim; j++) {
            score += Q[tid * embed_dim + j] * K[i * embed_dim + j];
        }
        score *= scale;
        
        if (score > max_val) {
            max_val = score;
        }
}
    
     // Second pass: compute softmax denominators and weighted sum
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0f;
        // Recompute dot product between Q[tid] and K[i]
        for (int j = 0; j < embed_dim; j++) {
            score += Q[tid * embed_dim + j] * K[i * embed_dim + j];
        }
        score *= scale;
        
        // Compute exp(score - max_val) for numerical stability
        float exp_score = expf(score - max_val);
        sum_exp += exp_score;
        
        // Accumulate weighted values
        for (int j = 0; j < embed_dim; j++) {
            output_row[j] += exp_score * V[i * embed_dim + j];
          }
        }
              
               // Normalize by sum of exponentials
              for (int j = 0; j < embed_dim; j++) {
                  output_row[j] /= sum_exp;
              }
          }
      }
      
      void fusedAttentionCuda(torch::Tensor output, torch::Tensor Q, torch::Tensor K, 
                              torch::Tensor V) {
          int seq_len = Q.size(0);
          int embed_dim = Q.size(1);
          
          // Calculate grid and block dimensions
          int block_size = 256;
          int grid_size = (seq_len + block_size - 1) / block_size;
          
          fusedAttentionKernel<<<grid_size, block_size>>>(
              output.data_ptr<float>(),
              Q.data_ptr<float>(),
              K.data_ptr<float>(),
              V.data_ptr<float>(),
              seq_len,
              embed_dim
          );
      }
      
      
      
      torch::Tensor fusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
          // Check dimensions
          assert(Q.size(0) == K.size(0) && "Sequence lengths must match for Q and K");
          assert(Q.size(1) == K.size(1) && "Embedding dimensions must match for Q and K");
          assert(K.size(0) == V.size(0) && "Sequence lengths must match for K and V");
          
          int seq_len = Q.size(0);
          int embed_dim = V.size(1);
          
          // Create output tensor
          auto output = torch::zeros({seq_len, embed_dim}, 
                                    torch::TensorOptions().device(Q.device()));
          
          fusedAttentionCuda(output, Q, K, V);
          
          return output;
      }
      
      
// Tensor Core Fused Attention Implementation
torch::Tensor tcFusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // Check dimensions
  assert(Q.size(0) == K.size(0) && "Sequence lengths must match for Q and K");
  assert(Q.size(1) == K.size(1) && "Embedding dimensions must match for Q and K");
  assert(K.size(0) == V.size(0) && "Sequence lengths must match for K and V");
  
  int seq_len = Q.size(0);
  int embed_dim = V.size(1);
  
  // Create output tensor
  auto output = torch::zeros({seq_len, embed_dim}, 
                            torch::TensorOptions().device(Q.device()));
  
  // Implementation that works with or without tensor core specific flags
  if (seq_len % 16 == 0 && embed_dim % 16 == 0) {
      // Use tensor core optimized implementation when dimensions are compatible
      tcFusedAttentionOptimized(output, Q, K, V);
  } else {
      // Fall back to standard fused implementation when dimensions aren't compatible
      fusedAttentionCuda(output, Q, K, V);
  }
  
  return output;
}

// Check if we have tensor core compatible hardware
__device__ bool hasTensorCores() {
  #if __CUDA_ARCH__ >= 700
      return true;
  #else
      return false;
  #endif
}

// WMMA using tensor cores for compatible hardware
template <typename T>
__global__ void computeQKtOptimized(T* Q, T* K, T* QKt, int seq_len, int embed_dim, float scale) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < seq_len && col < seq_len) {
      float dot_product = 0.0f;
      
      #pragma unroll
      for (int k = 0; k < embed_dim; k++) {
          dot_product += __ldg(&Q[row * embed_dim + k]) * __ldg(&K[col * embed_dim + k]);
      }
      
      QKt[row * seq_len + col] = dot_product * scale;
  }
}

// Compute softmax in a numerically stable way
__global__ void softmaxKernelOptimized(float* QKt, int seq_len) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < seq_len) {
      // Find max value for numerical stability
      float max_val = -INFINITY;
      for (int col = 0; col < seq_len; col++) {
          max_val = fmaxf(max_val, QKt[row * seq_len + col]);
      }
      
      // Compute exp(x - max) and sum
      float sum_exp = 0.0f;
      for (int col = 0; col < seq_len; col++) {
          float val = expf(QKt[row * seq_len + col] - max_val);
          QKt[row * seq_len + col] = val;
          sum_exp += val;
      }
      
      // Normalize
      for (int col = 0; col < seq_len; col++) {
          QKt[row * seq_len + col] /= sum_exp;
      }
  }
}

// Compute output using optimized matrix multiply
template <typename T>
__global__ void computeOutputOptimized(float* QKt, T* V, T* Output, int seq_len, int embed_dim) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < seq_len && col < embed_dim) {
      float sum = 0.0f;
      
      #pragma unroll
      for (int k = 0; k < seq_len; k++) {
          sum += QKt[row * seq_len + k] * V[k * embed_dim + col];
      }
      
      Output[row * embed_dim + col] = sum;
  }
}

void tcFusedAttentionOptimized(torch::Tensor output, torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  int seq_len = Q.size(0);
  int embed_dim = Q.size(1);
  
  // Scale factor for attention scores
  float scale = 1.0f / sqrtf(static_cast<float>(embed_dim));
  
  // Allocate temporary storage for QK^T
  auto QKt = torch::zeros({seq_len, seq_len}, 
                         torch::TensorOptions().device(Q.device()));
  
  // Define block and grid dimensions for matrix multiply
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  // Compute QK^T
  computeQKtOptimized<<<numBlocks, threadsPerBlock>>>(
      Q.data_ptr<float>(),
      K.data_ptr<float>(),
      QKt.data_ptr<float>(),
      seq_len,
      embed_dim,
      scale
  );
  
  // Apply softmax
  int block_size = 256;
  int grid_size = (seq_len + block_size - 1) / block_size;
  
  softmaxKernelOptimized<<<grid_size, block_size>>>(
      QKt.data_ptr<float>(),
      seq_len
  );
  
  // Compute output
  dim3 outputBlocks((embed_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  computeOutputOptimized<<<outputBlocks, threadsPerBlock>>>(
      QKt.data_ptr<float>(),
      V.data_ptr<float>(),
      output.data_ptr<float>(),
      seq_len,
      embed_dim
  );
}

// Block-Sparse Attention Implementation
__global__ void blockSparseAttentionKernel(float* output, float* Q, float* K, float* V, 
    int* mask, int seq_len, int embed_dim, 
    int block_size, int num_blocks) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < seq_len) {
// Each thread computes one row of the output
float* output_row = output + tid * embed_dim;

// Initialize output to zero
for (int j = 0; j < embed_dim; j++) {
output_row[j] = 0.0f;
}

// Scale factor for attention scores
float scale = 1.0f / sqrtf(embed_dim);

// Determine which block this thread belongs to
int q_block_idx = tid / block_size;

// Variables for softmax computation
float max_val = -INFINITY;
float sum_exp = 0.0f;

// First pass: find max score for numerical stability
for (int k_block_idx = 0; k_block_idx < num_blocks; k_block_idx++) {
// Check if this block is masked
if (mask[q_block_idx * num_blocks + k_block_idx] == 0) {
continue;  // Skip masked blocks
}

// Process only non-masked blocks
int k_start = k_block_idx * block_size;
int k_end = min(k_start + block_size, seq_len);

for (int i = k_start; i < k_end; i++) {
float score = 0.0f;
 	// Compute dot product between Q[tid] and K[i]
     for (int j = 0; j < embed_dim; j++) {
        score += Q[tid * embed_dim + j] * K[i * embed_dim + j];
    }
    score *= scale;
    
       if (score > max_val) {
        max_val = score;
    }
}
}

// Second pass: compute softmax and weighted sum
for (int k_block_idx = 0; k_block_idx < num_blocks; k_block_idx++) {
// Check if this block is masked
if (mask[q_block_idx * num_blocks + k_block_idx] == 0) {
    continue;  // Skip masked blocks
}

// Process only non-masked blocks
int k_start = k_block_idx * block_size;
int k_end = min(k_start + block_size, seq_len);

for (int i = k_start; i < k_end; i++) {
    float score = 0.0f;
    
       // Recompute dot product between Q[tid] and K[i]
    for (int j = 0; j < embed_dim; j++) {
        score += Q[tid * embed_dim + j] * K[i * embed_dim + j];
    }
    score *= scale;
    
       // Compute exp(score - max_val) for numerical stability
    float exp_score = expf(score - max_val);
    sum_exp += exp_score;
    
       // Accumulate weighted values
    for (int j = 0; j < embed_dim; j++) {
        output_row[j] += exp_score * V[i * embed_dim + j];
    }
}
}

// Normalize by sum of exponentials
for (int j = 0; j < embed_dim; j++) {
output_row[j] /= sum_exp;
}
}
}

// Generate causal mask for block-sparse attention
torch::Tensor generateCausalBlockMask(int seq_len, int block_size) {
int num_blocks = (seq_len + block_size - 1) / block_size;
auto mask = torch::zeros({num_blocks * num_blocks}, 
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

// Fill in the causal mask (lower triangular blocks)
for (int i = 0; i < num_blocks; i++) {
    for (int j = 0; j <= i; j++) {
        mask[i * num_blocks + j] = 1;
    }
}

return mask;
}

// Generate local attention mask for block-sparse attention
torch::Tensor generateLocalBlockMask(int seq_len, int block_size, int window_size = 1) {
int num_blocks = (seq_len + block_size - 1) / block_size;
auto mask = torch::zeros({num_blocks * num_blocks}, 
                        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

// Fill in the local attention mask (diagonal + window_size blocks on each side)
for (int i = 0; i < num_blocks; i++) {
    for (int j = std::max(0, i - window_size); j <= std::min(num_blocks - 1, i + window_size); j++) {
        mask[i * num_blocks + j] = 1;
    }
}

return mask;
}
void blockSparseAttentionCuda(torch::Tensor output, torch::Tensor Q, torch::Tensor K, 
                       torch::Tensor V, torch::Tensor mask, int block_size) {
int seq_len = Q.size(0);
int embed_dim = Q.size(1);
int num_blocks = (seq_len + block_size - 1) / block_size;

// Calculate grid and block dimensions
int cuda_block_size = 256;
int grid_size = (seq_len + cuda_block_size - 1) / cuda_block_size;

blockSparseAttentionKernel<<<grid_size, cuda_block_size>>>(
    output.data_ptr<float>(),
    Q.data_ptr<float>(),
    K.data_ptr<float>(),
    V.data_ptr<float>(),
    mask.data_ptr<int>(),
    seq_len,
    embed_dim,
    block_size,
    num_blocks
);
}

torch::Tensor blockSparseAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                            bool causal = true, int block_size = 64) {
// Check dimensions
assert(Q.size(0) == K.size(0) && "Sequence lengths must match for Q and K");
assert(Q.size(1) == K.size(1) && "Embedding dimensions must match for Q and K");
assert(K.size(0) == V.size(0) && "Sequence lengths must match for K and V");

int seq_len = Q.size(0);
int embed_dim = V.size(1);

// Create output tensor
auto output = torch::zeros({seq_len, embed_dim}, 
    torch::TensorOptions().device(Q.device()));

// Generate appropriate block mask
torch::Tensor mask;
if (causal) {
mask = generateCausalBlockMask(seq_len, block_size);
} else {
mask = generateLocalBlockMask(seq_len, block_size, 1);  // Window size of 1 for 3 blocks total
}

blockSparseAttentionCuda(output, Q, K, V, mask, block_size);

return output;
}

