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
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                            float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint threadRow = threadIdx.x / (BN / TN);
  const uint threadCol = threadIdx.x % (BN / TN);

  // load 128/32 = 4 data to SMEM at a step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK){
    // populate SMEM cache
    // load into As (transposed)
    float4 tmp = 
      reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];        
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = 
      reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];    

    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; dotIdx++){
      for (uint i = 0; i < TM; i++){        
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; i++){
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
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
      // tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      // tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      // tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      // tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      tmp.x = threadResults[resIdxM * TN + resIdxN];
      tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
      tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
      tmp.w = threadResults[resIdxM * TN + resIdxN + 3];

      reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;      
    }
  }
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    // const uint BM = 64; //Origin
    // const uint BN = 64; //Origin
    
    const uint BM = 16; //change from 64 to 16 to fit the size of lora
    const uint BN = 16;
    const uint TM = 2;
    const uint TN = 4;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}


void launch_matmul_my_6(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float alpha = 1.0;
    float beta = 0.0;

    runSgemmVectorize(M, N, K, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());    
}