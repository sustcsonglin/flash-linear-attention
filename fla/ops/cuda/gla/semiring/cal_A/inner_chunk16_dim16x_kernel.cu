#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#include "ATen/ATen.h"

typedef at::BFloat16 bf16;

template <typename scalar_t>
__global__ void fwd_inner_chunk16_dim16x(int batchSize, int M, int N_K,
                                         scalar_t* Q, scalar_t* K, float* G_K,
                                         scalar_t* QK) {
  // Batch index
  const int batchIdx = blockIdx.x;
  // allocate buffer for current block in fast shared mem
  __shared__ float Q_tile[16][16];
  __shared__ float K_tile[16][16];
  __shared__ float G_tile[16][16];
  __shared__ float G_tile_trans[16][16];

  const uint threadCol = threadIdx.x % 16;
  const uint threadRow = threadIdx.x / 16;

  int K_Stride = M * N_K;

  // Adjust the pointer for batch and matrix size
  Q += batchIdx * K_Stride;
  K += batchIdx * K_Stride;
  G_K += batchIdx * K_Stride;
  QK += batchIdx * M * M;

  float tmp = 0.0;
  // printf("Hello world");
  // printf("%d, %d, %d \n", threadRow, threadCol, N_K);
  for (int bkIdx = 0; bkIdx < N_K; bkIdx += 16) {
    Q_tile[threadRow][threadCol] = (float)Q[threadRow * N_K + threadCol];
    K_tile[threadRow][threadCol] = (float)K[threadRow * N_K + threadCol];
    float tmp_gk = (float)G_K[threadRow * N_K + threadCol];
    G_tile[threadRow][threadCol] = (float)tmp_gk;
    G_tile_trans[threadCol][threadRow] = (float)tmp_gk;

    __syncthreads();

    Q += 16;
    K += 16;
    G_K += 16;

    if (threadCol <= threadRow) {
      for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {
        // avoid bank conflict?
        float exp_term =
            expf(G_tile[threadRow][dotIdx] - G_tile_trans[dotIdx][threadCol]);
        tmp += Q_tile[threadRow][dotIdx] * K_tile[threadCol][dotIdx] * exp_term;
      }
    }
    __syncthreads();
  }

  if (threadCol <= threadRow) {
    QK[threadRow * M + threadCol] = (scalar_t)tmp;
  } else {
    QK[threadRow * M + threadCol] = (scalar_t)0.0;
  }
}

template <typename scalar_t>
__global__ void bwd_inner_chunk16_dim16x(int batchSize, int M, int N_K,
                                         scalar_t* Q, scalar_t* K, float* G,
                                         scalar_t* DQK, scalar_t* DQ,
                                         scalar_t* DK, float* DG) {
  // Batch index
  const uint batchIdx = blockIdx.x;

  // allocate buffer for current block in fast shared mem
  __shared__ float Q_tile[16][16];
  __shared__ float QK_tile[16][16];
  __shared__ float K_tile[16][16];
  __shared__ float G_tile[16][16];
  __shared__ float G_tile_trans[16][16];

  const uint threadCol = threadIdx.x % 16;
  const uint threadRow = threadIdx.x / 16;

  int K_Stride = M * N_K;

  Q += batchIdx * K_Stride;
  DQ += batchIdx * K_Stride;
  K += batchIdx * K_Stride;
  DK += batchIdx * K_Stride;
  G += batchIdx * K_Stride;
  DG += batchIdx * K_Stride;

  DQK += batchIdx * M * M;
  QK_tile[threadRow][threadCol] =
      (threadCol <= threadRow) ? (float)DQK[threadRow * M + threadCol] : 0.0;
  __syncthreads();

  for (int bkIdx = 0; bkIdx < N_K; bkIdx += 16) {
    Q_tile[threadRow][threadCol] = (float)Q[threadRow * N_K + threadCol];
    K_tile[threadRow][threadCol] = (float)K[threadRow * N_K + threadCol];
    float tmp_gk = (float)G[threadRow * N_K + threadCol];
    G_tile[threadRow][threadCol] = tmp_gk;
    // G_tile_trans[threadCol][threadRow] = tmp_gk;

    __syncthreads();

    float threadResults_dK = 0;
    float threadResults_dQ = 0;

    for (uint dotIdx = threadRow; dotIdx < 16; dotIdx += 1) {
      float tmp =
          QK_tile[dotIdx][threadRow] *
          expf(G_tile[dotIdx][threadCol] - G_tile[threadRow][threadCol]) *
          Q_tile[dotIdx][threadCol];
      threadResults_dK += tmp;
    }

    for (uint dotIdx = 0; dotIdx <= threadRow; dotIdx += 1) {
      float tmp =
          QK_tile[threadRow][dotIdx] *
          expf(G_tile[threadRow][threadCol] - G_tile[dotIdx][threadCol]) *
          K_tile[dotIdx][threadCol];
      threadResults_dQ += dotIdx <= threadRow ? tmp : 0;
    }

    __syncthreads();
    DQ[threadRow * N_K + threadCol] = (scalar_t)threadResults_dQ;
    DK[threadRow * N_K + threadCol] = (scalar_t)threadResults_dK;
    DG[threadRow * N_K + threadCol] =
        (threadResults_dQ * Q_tile[threadRow][threadCol] -
         threadResults_dK * K_tile[threadRow][threadCol]);
    Q += 16;
    K += 16;
    G += 16;
    DQ += 16;
    DK += 16;
    DG += 16;
    __syncthreads();
  }
}

std::vector<torch::Tensor> bwd_cuda(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor g_K, torch::Tensor DQK) {
  auto DQ = torch::empty_like(Q);
  auto DK = torch::empty_like(K);
  auto Dg_K = torch::empty_like(g_K);

  int B_size = Q.size(0);     // This is the batch size dimension.
  int H_size = Q.size(1);     // This is the head dimension
  int num_chunk = Q.size(2);  // This is the chunk dimension.
  int M = Q.size(-2);
  int N_K = Q.size(-1);

  dim3 gridDim(B_size * H_size * num_chunk);
  dim3 blockDim(256);

  switch (Q.type().scalarType()) {
    case torch::ScalarType::BFloat16:
      bwd_inner_chunk16_dim16x<<<gridDim, blockDim>>>(
          B_size * H_size * num_chunk, M, N_K, Q.data_ptr<bf16>(),
          K.data_ptr<bf16>(), g_K.data_ptr<float>(), DQK.data_ptr<bf16>(),
          DQ.data_ptr<bf16>(), DK.data_ptr<bf16>(), Dg_K.data_ptr<float>());
      break;
    default:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          Q.scalar_type(), "bwd_inner_chunk16_dim16x", ([&] {
            bwd_inner_chunk16_dim16x<scalar_t><<<gridDim, blockDim>>>(
                B_size * H_size * num_chunk, M, N_K, Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(), g_K.data_ptr<float>(),
                DQK.data_ptr<scalar_t>(), DQ.data_ptr<scalar_t>(),
                DK.data_ptr<scalar_t>(), Dg_K.data_ptr<float>());
          }));
  };
  return {DQ, DK, Dg_K};
}

torch::Tensor fwd_cuda(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& g_K) {
  auto QK = torch::empty(
      {Q.size(0), Q.size(1), Q.size(2), Q.size(3), Q.size(3)}, Q.options());
  int B_size = Q.size(0);     // This is the batch size dimension.
  int H_size = Q.size(1);     // This is the head dimension
  int num_chunk = Q.size(2);  // This is the chunk dimension.
  int M = Q.size(-2);         // this is the chunk size
  int N_K = Q.size(-1);       // this is the head_K dim

  dim3 gridDim(B_size * H_size * num_chunk);
  dim3 blockDim(256);
  switch (Q.type().scalarType()) {
    case torch::ScalarType::BFloat16:
      fwd_inner_chunk16_dim16x<bf16><<<gridDim, blockDim>>>(
          B_size * H_size * num_chunk, M, N_K, Q.data_ptr<bf16>(),
          K.data_ptr<bf16>(), g_K.data_ptr<float>(), QK.data_ptr<bf16>());
      break;
    default:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          Q.scalar_type(), "fwd_inner_chunk16_dim16x", ([&] {
            fwd_inner_chunk16_dim16x<scalar_t><<<gridDim, blockDim>>>(
                B_size * H_size * num_chunk, M, N_K, Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(), g_K.data_ptr<float>(),
                QK.data_ptr<scalar_t>());
          }));
  };
  return QK;
}