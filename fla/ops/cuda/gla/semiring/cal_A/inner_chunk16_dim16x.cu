#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void fwd_inner_chunk16_dim16x(int batchSize, int M, int N_K, 
                                     float *Q, float *K, float *G_K, 
                                     float *QK
                                    ) {

  // Batch index
  const uint batchIdx = blockIdx.x;
  // allocate buffer for current block in fast shared mem
  __shared__ float Q_tile[16][16];
  __shared__ float K_tile[16][16];
  __shared__ float G_tile[16][16];
  __shared__ float G_tile_trans[16][16];

  const uint threadCol = threadIdx.x % 16;
  const uint threadRow = threadIdx.x / 16;

  int K_Stride = M * N_K;

  // Adjust the pointers for batch and matrix size
  Q += batchIdx * K_Stride;
  K += batchIdx * K_Stride;
  G_K += batchIdx * K_Stride;
  QK += batchIdx * M * M;
  
  float tmp = 0.0;

  for (int bkIdx = 0; bkIdx < N_K; bkIdx += 16) {
    Q_tile[threadRow][threadCol] = Q[threadRow * N_K + threadCol];
    K_tile[threadRow][threadCol] = K[threadRow * N_K + threadCol];
    float tmp_gk = G_K[threadRow * N_K + threadCol];
    G_tile[threadRow][threadCol] = tmp_gk;
    G_tile_trans[threadCol][threadRow] = tmp_gk;

    __syncthreads();

    Q += 16;
    K += 16;
    G_K += 16;
    
    if(threadCol <= threadRow){
        for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {
            // avoid bank conflict?
            float exp_term = expf(G_tile[threadRow][dotIdx] - G_tile_trans[dotIdx][threadCol]);
            tmp += Q_tile[threadRow][dotIdx] * K_tile[threadCol][dotIdx] * exp_term;
        }
    }
    __syncthreads();    
  }  

  if(threadCol <= threadRow){
    QK[threadRow * M + threadCol] = tmp;
  }  
  else{
    QK[threadRow * M + threadCol] = 0.0;
  }
    
}

void run_fwd_inner_chunk16_dim16x(int batchSize, int M, int N_K, 
                                float *Q, float *K, float *gK, float *QK
                            ) {  
  dim3 gridDim(batchSize); 
  dim3 blockDim(256);
  fwd_inner_chunk16_dim16x<<<gridDim, blockDim>>>(batchSize, M, N_K, Q, K, gK, QK); 
}


__global__ void bwd_inner_chunk16_dim16x(int batchSize, int M, int N_K, 
                                     float *Q, float *K, float *G, 
                                     float *DQK, float *DQ, float *DK, 
                                     float *DG
                                    ) {

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
  QK_tile[threadRow][threadCol] = (threadCol <= threadRow) ? DQK[threadRow * M + threadCol] : 0.0;
  __syncthreads();

  for (int bkIdx = 0; bkIdx < N_K; bkIdx += 16) {
    Q_tile[threadRow][threadCol] = Q[threadRow * N_K + threadCol];
    K_tile[threadRow][threadCol] = K[threadRow * N_K + threadCol];
    float tmp_gk = G[threadRow * N_K + threadCol];
    G_tile[threadRow][threadCol] = tmp_gk;
    // G_tile_trans[threadCol][threadRow] = tmp_gk;

    __syncthreads();

    float threadResults_dK = 0;
    float threadResults_dQ = 0;
    
    for(uint dotIdx = threadRow; dotIdx < 16; dotIdx += 1){
          float tmp =  QK_tile[dotIdx][threadRow] * expf(G_tile[dotIdx][threadCol] - G_tile[threadRow][threadCol]) * Q_tile[dotIdx][threadCol];
          threadResults_dK += tmp;                  
    }
    
    for(uint dotIdx = 0; dotIdx <= threadRow;  dotIdx += 1){
      float tmp = QK_tile[threadRow][dotIdx] * expf(G_tile[threadRow][threadCol] - G_tile[dotIdx][threadCol]) * K_tile[dotIdx][threadCol];                                         
      threadResults_dQ += dotIdx <= threadRow? tmp: 0;                       
    }

    __syncthreads();    
    DQ[threadRow * N_K + threadCol] = threadResults_dQ;
    DK[threadRow * N_K + threadCol] = threadResults_dK;
    DG[threadRow * N_K + threadCol] = threadResults_dQ * Q_tile[threadRow][threadCol] - threadResults_dK * K_tile[threadRow][threadCol];
    Q += 16;
    K += 16;
    G += 16;
    DQ += 16;
    DK += 16;
    DG += 16;
    __syncthreads();
  }  
}

void run_bwd_inner_chunk16_dim16x(int batchSize, int M, int N_K, 
                                float *Q, float *K, float *G, float *DQK,
                                float *DQ, float *DK, float *DG
                            ) {  
  dim3 gridDim(batchSize); 
  dim3 blockDim(256);
  bwd_inner_chunk16_dim16x<<<gridDim, blockDim>>>(batchSize, M, N_K, Q, K, G, DQK, DQ, DK, DG); 
}

