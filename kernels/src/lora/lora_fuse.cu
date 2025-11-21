#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>
using namespace std;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize_sanity_check(int M, int N, int K, int R,
                            float *X, float *A, float *B_down_proj, float *B_up_proj, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  __shared__ float Xs[BM * BK];
  __shared__ float As[BK * BN];

  X += cRow * BM * K;
  A += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint threadRow = threadIdx.x / (BN / TN);
  const uint threadCol = threadIdx.x % (BN / TN);

  // load 128/32 = 4 data to SMEM at a step
  const uint innerRowX = threadIdx.x / (BK / 4);
  const uint innerColX = threadIdx.x % (BK / 4);
  const uint innerRowA = threadIdx.x / (BN / 4);
  const uint innerColA = threadIdx.x % (BN / 4);

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK){
    // populate SMEM cache
    // load into Xs (transposed)
    float4 tmp = 
      reinterpret_cast<float4 *>(&X[innerRowX * K + innerColX * 4])[0];        
    Xs[(innerColX * 4 + 0) * BM + innerRowX] = tmp.x;
    Xs[(innerColX * 4 + 1) * BM + innerRowX] = tmp.y;
    Xs[(innerColX * 4 + 2) * BM + innerRowX] = tmp.z;
    Xs[(innerColX * 4 + 3) * BM + innerRowX] = tmp.w;

    reinterpret_cast<float4 *>(&As[innerRowA * BN + innerColA * 4])[0] = 
      reinterpret_cast<float4 *>(&A[innerRowA * N + innerColA * 4])[0];    

    __syncthreads();

    X += BK;
    A += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; dotIdx++){
      for (uint i = 0; i < TM; i++){        
        regM[i] = Xs[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; i++){
        regN[i] = As[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; resIdxM++){
        for (uint resIdxN = 0; resIdxN < TN; resIdxN++){
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; resIdxM++){
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4){
      float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      tmp.x = threadResults[resIdxM * TN + resIdxN];
      tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
      tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
      tmp.w = threadResults[resIdxM * TN + resIdxN + 3];

      reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;      
    }
  }
}

// sgemmVectorize - Optimized Matrix Multiplication with Projection Operations
//
// This kernel performs the following sequence of operations:
// 1. Multiplying matrix X with matrix A (X * A).
// 2. Multiplying matrix X with the down-projected B matrix (X * B_down_proj).
// 3. Fusing these results with the up-projected B matrix (B_up_proj) to compute the final result (X * B_down_proj * B_up_proj).
//
// It leverages shared memory (SMEM) and registers to optimize memory access patterns
// and perform computations efficiently on the GPU.

template <const int BM, const int BN, const int BR, const int BK, 
          const int TM, const int TN, const int TRM, const int TRN>
__global__ void sgemmVectorize(int M, int N, int K, int R,
                                float *X, float *A, float *B_down_proj, 
                                float *B_up_proj, float *C) {  
  // Get the row and column indices of the current block.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Shared memory declarations:
  __shared__ float Xs[BM * BK];    // Transposed X matrix block
  __shared__ float As[BK * BN];    // A matrix block
  __shared__ float Bds[BK * BR];   // B_down_proj matrix block
  __shared__ float B_inters[BM * BR]; // Internal results of X * B_down_proj
  __shared__ float Bus[BR * BN];   // B_up_proj matrix block

  // Adjust pointers based on block indices
  X += cRow * BM * K;
  A += cCol * BN;  
  C += cRow * BM * N + cCol * BN;

  // Thread-level indices for computation within the block
  const uint threadRow = threadIdx.x / (BN / TN);
  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRowR = threadIdx.x / (BR / TRN);
  const uint threadColR = threadIdx.x % (BR / TRN);

  // Indexing for loading data into shared memory
  const uint innerRowX = threadIdx.x / (BK / 4);
  const uint innerColX = threadIdx.x % (BK / 4);
  const uint innerRowA = threadIdx.x / (BN / 4);
  const uint innerColA = threadIdx.x % (BN / 4);
  const uint innerRowBd = threadIdx.x / (BR / 4);
  const uint innerColBd = threadIdx.x % (BR / 4);

  // Registers to hold intermediate results
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};
  float regRM[TRM] = {0.0};
  float regRN[TRN] = {0.0};
  float threadResultsLora[TRM * TRN] = {0.0};

  // Compute the product: 1. X * A, 2. X * B_down_proj
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
    // Load data into shared memory (SMEM)
    // Load a block of X into Xs
    float4 tmp = reinterpret_cast<float4 *>(&X[innerRowX * K + innerColX * 4])[0];        
    Xs[(innerColX * 4 + 0) * BM + innerRowX] = tmp.x;
    Xs[(innerColX * 4 + 1) * BM + innerRowX] = tmp.y;
    Xs[(innerColX * 4 + 2) * BM + innerRowX] = tmp.z;
    Xs[(innerColX * 4 + 3) * BM + innerRowX] = tmp.w;

    // Load a block of A into As
    reinterpret_cast<float4 *>(&As[innerRowA * BN + innerColA * 4])[0] = 
      reinterpret_cast<float4 *>(&A[innerRowA * N + innerColA * 4])[0];    

    // Load a block of B_down_proj into Bds
    if (innerRowBd < BK) {
      reinterpret_cast<float4 *>(&Bds[innerRowBd * BR + innerColBd * 4])[0] = 
        reinterpret_cast<float4 *>(&B_down_proj[innerRowBd * R + innerColBd * 4])[0];  
    }

    __syncthreads();  // Synchronize threads in the block

    // Update the pointers to the next section of the matrices
    X += BK;
    A += BK * N;
    B_down_proj += BK * R;

    // Multiply X and A,store in threadResults
    // Multiply X and B_down_proj,store in threadResultsLora
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      // Load data from shared memory into registers for matrix multiplication
      for (uint i = 0; i < TM; i++) {
        regM[i] = Xs[dotIdx * BM + threadRow * TM + i];        
      }
      for (uint i = 0; i < TN; i++) {
        regN[i] = As[dotIdx * BN + threadCol * TN + i];
      }
      // Accumulate the results
      for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN++) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }

      // Compute the internal result of X * B_down_proj
      for (uint i = 0; i < TRM; i++) {        
        regRM[i] = Xs[dotIdx * BM + threadRowR * TRM + i];        
      }
      for (uint i = 0; i < TRN; i++) {
        regRN[i] = Bds[dotIdx * BR + threadColR * TRN + i];
      }
      // Accumulate the results for B_down_proj projection
      for (uint resIdxM = 0; resIdxM < TRM; resIdxM++) {
        for (uint resIdxN = 0; resIdxN < TRN; resIdxN++) {
          threadResultsLora[resIdxM * TRN + resIdxN] += regRM[resIdxM] * regRN[resIdxN];
        }
      }
    }
    __syncthreads();  // Synchronize threads in the block
  }

  // Move partial results of X * B_down_proj into shared memory (B_inters)
  for (uint resIdxM = 0; resIdxM < TRM; resIdxM++) {
    for (uint resIdxN = 0; resIdxN < TRN; resIdxN += 4) {
      B_inters[(threadColR * TRN + resIdxN + 0) * BM + (threadRowR * TRM + resIdxM)] = threadResultsLora[resIdxM * TRN + resIdxN];
      B_inters[(threadColR * TRN + resIdxN + 1) * BM + (threadRowR * TRM + resIdxM)] = threadResultsLora[resIdxM * TRN + resIdxN + 1];
      B_inters[(threadColR * TRN + resIdxN + 2) * BM + (threadRowR * TRM + resIdxM)] = threadResultsLora[resIdxM * TRN + resIdxN + 2];
      B_inters[(threadColR * TRN + resIdxN + 3) * BM + (threadRowR * TRM + resIdxM)] = threadResultsLora[resIdxM * TRN + resIdxN + 3];
    }
  }
  __syncthreads();  // Synchronize threads in the block

  // Load B_up_proj into shared memory (Bus)
  B_up_proj += cCol * BN;
  const uint innerRowBu = threadIdx.x / (BN / 8);  // Loading 4 bytes at a time (twice the size)
  const uint innerColBu = threadIdx.x % (BN / 8);
  reinterpret_cast<float4 *>(&Bus[innerRowBu * BN + innerColBu * 8])[0] = 
    reinterpret_cast<float4 *>(&B_up_proj[innerRowBu * N + innerColBu * 8])[0]; 
  reinterpret_cast<float4 *>(&Bus[innerRowBu * BN + innerColBu * 8 + 4])[0] = 
    reinterpret_cast<float4 *>(&B_up_proj[innerRowBu * N + innerColBu * 8 + 4])[0];  
  __syncthreads();  // Synchronize threads in the block

  // Calculate final result of X * B_down_proj * B_up_proj
  for (uint dotIdx = 0; dotIdx < BR; dotIdx++) {
    for (uint i = 0; i < TM; i++) {
      regM[i] = B_inters[dotIdx * BM + threadRow * TM + i];        
    }
    for (uint i = 0; i < TN; i++) {
      regN[i] = Bus[dotIdx * BN + threadCol * TN + i];
    }
    for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
      for (uint resIdxN = 0; resIdxN < TN; resIdxN++) {
        threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
      }
    }
  }
  __syncthreads();  // Synchronize threads in the block

  // Write the result to the output matrix C
  for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      tmp.x = threadResults[resIdxM * TN + resIdxN];
      tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
      tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
      tmp.w = threadResults[resIdxM * TN + resIdxN + 3];

      reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;      
    }
  }
}


void runSgemmVectorize(int M, int N, int K, int R, float *X, float *A,
                       float *B_down_proj, float *B_up_proj, float *C) {
  const uint BR = 16; // assume R is not a very large number
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint TRM = 2;
  const uint TRN = 4;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BR, BK, TM, TN, TRM, TRN>
        <<<gridDim, blockDim>>>(M, N, K, R, X, A, B_down_proj, B_up_proj, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    // const uint BM = 64; //Origin
    // const uint BN = 64; //Origin
    
    const uint BM = 16; //change from 64 to 16 to fit the size of lora
    const uint BN = 16;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BR, BK, TM, TN, TRM, TRN>
        <<<gridDim, blockDim>>>(M, N, K, R, X, A, B_down_proj, B_up_proj, C);
  }
}


void launch_lora_fuse(torch::Tensor X, torch::Tensor A, 
                        torch::Tensor B_down_proj, torch::Tensor B_up_proj, 
                        torch::Tensor C) {
    int M = X.size(0);
    int K = X.size(1);
    int N = A.size(1);    
    int R = B_down_proj.size(1);

    runSgemmVectorize(M, N, K, R, 
                      X.data_ptr<float>(), A.data_ptr<float>(),
                      B_down_proj.data_ptr<float>(), B_up_proj.data_ptr<float>(), 
                      C.data_ptr<float>());    
}